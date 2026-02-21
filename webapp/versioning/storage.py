"""
DataMind AI - Storage Manager Module
Enterprise-grade persistence layer for datasets and outputs

Storage Structure:
storage/
├── datasets/
│   └── <dataset_id>/
│       ├── raw/
│       │   └── uploaded_dataset.<ext>
│       ├── schema.json
│       └── metadata.json
│
└── outputs/
    └── <dataset_id>/
        ├── user_manual.md
        ├── insights.json
        ├── analysis.json
        ├── charts/
        │   ├── bar_*.png
        │   ├── pie_*.png
        │   └── line_*.png
        └── stats.json
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib

from .fingerprint import SHORT_ID_LENGTH, SemanticIDGenerator, DatasetFingerprint


@dataclass
class DatasetMetadata:
    """Metadata for a stored dataset."""
    dataset_id: str
    original_filename: str
    file_type: str
    created_at: str
    row_count: int
    column_count: int
    table_count: int
    file_size_bytes: int
    fingerprint_algorithm: str
    quality_score: Optional[int] = None
    analysis_status: str = "pending"  # pending, completed, failed
    last_accessed: Optional[str] = None
    access_count: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DatasetMetadata':
        return cls(**data)


@dataclass
class OutputManifest:
    """Manifest of generated outputs for a dataset."""
    dataset_id: str
    generated_at: str
    outputs: Dict[str, str]  # output_type -> relative_path
    charts: List[str]
    version: str = "1.0"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OutputManifest':
        return cls(**data)


class StorageManager:
    """
    Manages persistent storage for datasets and their outputs.
    
    Ensures:
    - Every dataset has a unique storage location based on content hash
    - No duplicate storage of identical datasets
    - All outputs are associated with dataset ID
    - Full audit trail and reproducibility
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, base_path: str = None):
        """
        Initialize storage manager.
        
        Args:
            base_path: Base directory for storage (default: ./storage)
        """
        if base_path is None:
            base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'storage')
        
        self.base_path = Path(base_path)
        self.datasets_path = self.base_path / 'datasets'
        self.outputs_path = self.base_path / 'outputs'
        self.index_path = self.base_path / 'index.json'
        
        # Ensure directories exist
        self._initialize_storage()
        
        # Load index
        self._index = self._load_index()
        
        # Auto-migrate legacy hash-based IDs → semantic IDs
        self._migrate_legacy_ids()
        
        # Auto-migrate old semantic IDs (without hash suffix) → new format with suffix
        self._migrate_semantic_ids_add_suffix()
    
    def _initialize_storage(self):
        """Create storage directory structure."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.datasets_path.mkdir(exist_ok=True)
        self.outputs_path.mkdir(exist_ok=True)
    
    def _load_index(self) -> Dict:
        """Load storage index with error recovery."""
        default_index = {
            'version': self.VERSION,
            'created_at': datetime.now().isoformat(),
            'datasets': {},
            'stats': {
                'total_datasets': 0,
                'total_analyses': 0,
                'cache_hits': 0
            }
        }
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                # Validate minimal structure
                if not isinstance(data, dict) or 'datasets' not in data:
                    print('[Storage] Index file has invalid structure, resetting')
                    return default_index
                return data
            except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
                print(f'[Storage] Corrupted index file, resetting: {e}')
                # Back up corrupted file for debugging
                try:
                    backup = self.index_path.with_suffix('.json.bak')
                    shutil.copy2(self.index_path, backup)
                except Exception:
                    pass
                return default_index
        return default_index
    
    def _save_index(self):
        """Persist index to disk atomically (write to temp, then rename)."""
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.base_path), suffix='.tmp', prefix='index_'
            )
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, indent=2, ensure_ascii=False)
            # Atomic rename (on Windows, need to remove target first)
            if os.path.exists(self.index_path):
                os.replace(tmp_path, self.index_path)
            else:
                os.rename(tmp_path, self.index_path)
        except Exception as e:
            print(f'[Storage] Failed to save index: {e}')
            # Clean up temp file if it still exists
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
    
    def _migrate_legacy_ids(self):
        """
        Auto-migrate legacy hex-hash IDs to human-readable semantic IDs.
        
        Legacy IDs are pure hex strings (e.g., '6cc77197020e2dd3' or 64-char SHA-256).
        Converts them to semantic format (e.g., 'DEMOECOMMERCE2024') using:
        - original_filename from index metadata
        - schema.json if available (for table/column-based topic detection)
        """
        legacy_ids = [
            did for did in list(self._index.get('datasets', {}).keys())
            if DatasetFingerprint.is_legacy_hash_id(did)
        ]
        
        if not legacy_ids:
            return
        
        print(f'[Storage] Migrating {len(legacy_ids)} legacy hash IDs to semantic format...')
        sem_gen = SemanticIDGenerator()
        migrated = 0
        
        for old_id in legacy_ids:
            info = self._index['datasets'].get(old_id, {})
            filename = info.get('original_filename', 'unknown.dat')
            
            # Get content hash for the fingerprint suffix
            legacy_content_hash = info.get('content_hash', old_id)
            
            # Try to read schema for better topic detection
            schema_path = self.datasets_path / old_id / 'schema.json'
            if schema_path.exists():
                try:
                    with open(schema_path, 'r', encoding='utf-8') as f:
                        schema = json.load(f)
                    new_id = sem_gen.generate_from_schema(filename, schema,
                                                          content_hash=legacy_content_hash)
                except Exception:
                    new_id = sem_gen.generate(filename, {'tables': []}, '',
                                              content_hash=legacy_content_hash)
            else:
                new_id = sem_gen.generate(filename, {'tables': []}, '',
                                          content_hash=legacy_content_hash)
            
            # Resolve collisions
            final_id = self._resolve_next_id(new_id, exclude_id=old_id)
            
            # Rename directories
            old_ds_dir = self.datasets_path / old_id
            new_ds_dir = self.datasets_path / final_id
            old_out_dir = self.outputs_path / old_id
            new_out_dir = self.outputs_path / final_id
            
            try:
                if old_ds_dir.exists() and not new_ds_dir.exists():
                    old_ds_dir.rename(new_ds_dir)
                elif old_ds_dir.exists() and new_ds_dir.exists():
                    shutil.rmtree(old_ds_dir, ignore_errors=True)
                
                if old_out_dir.exists() and not new_out_dir.exists():
                    old_out_dir.rename(new_out_dir)
                elif old_out_dir.exists() and new_out_dir.exists():
                    shutil.rmtree(old_out_dir, ignore_errors=True)
                
                # Update metadata.json inside the dataset dir
                meta_path = new_ds_dir / 'metadata.json'
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        meta['dataset_id'] = final_id
                        with open(meta_path, 'w', encoding='utf-8') as f:
                            json.dump(meta, f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                
                # Update index: remove old, add new
                entry = self._index['datasets'].pop(old_id)
                entry['content_hash'] = old_id  # Preserve old hash as content_hash
                self._index['datasets'][final_id] = entry
                
                print(f'  [Migrated] {old_id} → {final_id}')
                migrated += 1
                
            except Exception as e:
                print(f'  [Error] Failed to migrate {old_id}: {e}')
        
        if migrated > 0:
            self._save_index()
            print(f'[Storage] Migration complete: {migrated} IDs converted to semantic format')
    
    def _migrate_semantic_ids_add_suffix(self):
        """
        Migrate old-format semantic IDs (without content hash suffix) to new format.
        
        Old format: NETFLIXMOVIE2020   (no content fingerprint)
        New format: NETFLIXMOVIE2020A3F2  (4-char hash suffix for uniqueness)
        
        Detects IDs that:
        - Are NOT legacy hex IDs (already handled)
        - Have a content_hash stored in index
        - Don't already end with a 4-char hex suffix from the content hash
        """
        sem_gen = SemanticIDGenerator()
        suffix_len = sem_gen.HASH_SUFFIX_LENGTH
        
        ids_to_migrate = []
        for did, info in list(self._index.get('datasets', {}).items()):
            # Skip legacy hex IDs (handled by _migrate_legacy_ids)
            if DatasetFingerprint.is_legacy_hash_id(did):
                continue
            
            content_hash = info.get('content_hash', '')
            if not content_hash:
                continue
            
            # Check if ID already has the correct hash suffix
            expected_suffix = content_hash[:suffix_len].upper()
            if did.endswith(expected_suffix):
                continue
            
            # This ID needs the suffix added
            ids_to_migrate.append((did, info, content_hash))
        
        if not ids_to_migrate:
            return
        
        print(f'[Storage] Adding content fingerprint suffix to {len(ids_to_migrate)} semantic IDs...')
        migrated = 0
        
        for old_id, info, content_hash in ids_to_migrate:
            filename = info.get('original_filename', 'unknown.dat')
            
            # Generate new ID with hash suffix
            schema_path = self.datasets_path / old_id / 'schema.json'
            if schema_path.exists():
                try:
                    with open(schema_path, 'r', encoding='utf-8') as f:
                        schema = json.load(f)
                    new_id = sem_gen.generate_from_schema(filename, schema,
                                                          content_hash=content_hash)
                except Exception:
                    new_id = sem_gen.generate(filename, {'tables': []}, '',
                                              content_hash=content_hash)
            else:
                new_id = sem_gen.generate(filename, {'tables': []}, '',
                                          content_hash=content_hash)
            
            # Skip if new ID is same as old (shouldn't happen but be safe)
            if new_id == old_id:
                continue
            
            # Resolve collisions (V2/V3 safety net)
            final_id = self._resolve_next_id(new_id, exclude_id=old_id)
            
            # Rename directories
            old_ds_dir = self.datasets_path / old_id
            new_ds_dir = self.datasets_path / final_id
            old_out_dir = self.outputs_path / old_id
            new_out_dir = self.outputs_path / final_id
            
            try:
                if old_ds_dir.exists() and not new_ds_dir.exists():
                    old_ds_dir.rename(new_ds_dir)
                elif old_ds_dir.exists() and new_ds_dir.exists():
                    shutil.rmtree(old_ds_dir, ignore_errors=True)
                
                if old_out_dir.exists() and not new_out_dir.exists():
                    old_out_dir.rename(new_out_dir)
                elif old_out_dir.exists() and new_out_dir.exists():
                    shutil.rmtree(old_out_dir, ignore_errors=True)
                
                # Update metadata.json
                meta_path = new_ds_dir / 'metadata.json'
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        meta['dataset_id'] = final_id
                        with open(meta_path, 'w', encoding='utf-8') as f:
                            json.dump(meta, f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                
                # Update index
                entry = self._index['datasets'].pop(old_id)
                self._index['datasets'][final_id] = entry
                
                print(f'  [Suffix] {old_id} -> {final_id}')
                migrated += 1
                
            except Exception as e:
                print(f'  [Error] Failed to add suffix to {old_id}: {e}')
        
        if migrated > 0:
            self._save_index()
            print(f'[Storage] Suffix migration complete: {migrated} IDs updated')
    
    def _remove_dataset_dirs(self, dataset_id: str):
        """Remove dataset and output directories for a given ID."""
        ds_dir = self.datasets_path / dataset_id
        out_dir = self.outputs_path / dataset_id
        if ds_dir.exists():
            shutil.rmtree(ds_dir, ignore_errors=True)
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
    
    # =========================================================================
    # Semantic ID Resolution
    # =========================================================================
    
    def resolve_semantic_id(self, semantic_base: str, content_hash: str) -> str:
        """
        Resolve a semantic base ID to a final unique dataset ID.
        
        Logic:
        1. If content_hash already exists → return that dataset's ID (dedup)
        2. If semantic_base is available → use it
        3. If semantic_base is taken by different content → append V2, V3...
        
        Args:
            semantic_base: Base semantic ID (e.g., 'NETFLIXMOVIE2025')
            content_hash: SHA-256 content hash for dedup matching
            
        Returns:
            Final resolved dataset ID
        """
        # Step 1: Dedup — check if this exact content already exists
        existing = self.find_by_content_hash(content_hash)
        if existing:
            return existing
        
        # Step 2: Check if semantic_base is available
        if semantic_base not in self._index['datasets']:
            return semantic_base
        
        # Step 3: Semantic base taken — find next version
        return self._resolve_next_id(semantic_base)
    
    def _resolve_next_id(self, semantic_base: str, exclude_id: str = None) -> str:
        """Find next available version of a semantic ID (V2, V3, etc.)."""
        if semantic_base not in self._index['datasets'] or semantic_base == exclude_id:
            return semantic_base
        
        version = 2
        while True:
            candidate = f"{semantic_base}V{version}"
            if candidate not in self._index['datasets'] or candidate == exclude_id:
                return candidate
            version += 1
    
    def find_by_content_hash(self, content_hash: str) -> Optional[str]:
        """
        Find a dataset ID by its content hash.
        Supports prefix matching to handle legacy short hashes.
        
        Args:
            content_hash: Full or partial content hash to match
            
        Returns:
            dataset_id if found, None otherwise
        """
        if not content_hash:
            return None
        for did, meta in self._index['datasets'].items():
            stored_hash = meta.get('content_hash', '')
            if not stored_hash:
                continue
            # Exact match or prefix match (handles short legacy hashes)
            if (stored_hash == content_hash or 
                content_hash.startswith(stored_hash) or 
                stored_hash.startswith(content_hash)):
                return did
        return None
    
    # =========================================================================
    # Stats Operations
    # =========================================================================

    def increment_stat(self, stat_name: str, amount: int = 1):
        """Safely increment a stat counter and persist."""
        if 'stats' not in self._index:
            self._index['stats'] = {}
        current = self._index['stats'].get(stat_name, 0)
        self._index['stats'][stat_name] = current + amount
        self._save_index()

    # =========================================================================
    # Dataset Operations
    # =========================================================================
    
    def dataset_exists(self, dataset_id: str) -> bool:
        """Check if a dataset with given ID exists in storage."""
        return dataset_id in self._index['datasets']
    
    def get_dataset_path(self, dataset_id: str) -> Path:
        """Get path to dataset storage directory."""
        return self.datasets_path / dataset_id
    
    def get_output_path(self, dataset_id: str) -> Path:
        """Get path to output storage directory."""
        return self.outputs_path / dataset_id
    
    def store_dataset(
        self,
        dataset_id: str,
        source_path: str,
        original_filename: str,
        file_type: str,
        row_count: int,
        column_count: int,
        table_count: int,
        fingerprint_algorithm: str = 'sha256'
    ) -> DatasetMetadata:
        """
        Store a dataset in the persistent storage.
        
        Args:
            dataset_id: Content-based hash ID
            source_path: Path to uploaded file
            original_filename: Original filename
            file_type: Type of file
            row_count: Total rows in dataset
            column_count: Total columns
            table_count: Number of tables/sheets
            fingerprint_algorithm: Algorithm used for fingerprinting
            
        Returns:
            DatasetMetadata for the stored dataset
        """
        dataset_dir = self.get_dataset_path(dataset_id)
        raw_dir = dataset_dir / 'raw'
        
        # Create directories
        dataset_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(exist_ok=True)
        
        # Copy raw file
        ext = original_filename.rsplit('.', 1)[-1] if '.' in original_filename else 'dat'
        raw_file_path = raw_dir / f'dataset.{ext}'
        shutil.copy2(source_path, raw_file_path)
        
        # Get file size
        file_size = os.path.getsize(raw_file_path)
        
        # Create metadata
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            original_filename=original_filename,
            file_type=file_type,
            created_at=datetime.now().isoformat(),
            row_count=row_count,
            column_count=column_count,
            table_count=table_count,
            file_size_bytes=file_size,
            fingerprint_algorithm=fingerprint_algorithm,
            analysis_status="pending"
        )
        
        # Save metadata
        metadata_path = dataset_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Update index
        self._index['datasets'][dataset_id] = {
            'original_filename': original_filename,
            'created_at': metadata.created_at,
            'analysis_status': 'pending',
            'file_type': file_type,
            'content_hash': ''  # Will be set by versioning system
        }
        self._index['stats']['total_datasets'] += 1
        self._save_index()
        
        return metadata
    
    def get_dataset_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Retrieve metadata for a stored dataset."""
        dataset_dir = self.get_dataset_path(dataset_id)
        metadata_path = dataset_dir / 'metadata.json'
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return DatasetMetadata.from_dict(data)
    
    def update_dataset_metadata(self, dataset_id: str, updates: Dict) -> DatasetMetadata:
        """Update metadata for a dataset."""
        metadata = self.get_dataset_metadata(dataset_id)
        if not metadata:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Update fields
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        # Save updated metadata
        dataset_dir = self.get_dataset_path(dataset_id)
        metadata_path = dataset_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Update index
        if dataset_id in self._index['datasets']:
            self._index['datasets'][dataset_id]['analysis_status'] = metadata.analysis_status
        self._save_index()
        
        return metadata
    
    def get_raw_dataset_path(self, dataset_id: str) -> Optional[Path]:
        """Get path to raw dataset file."""
        raw_dir = self.get_dataset_path(dataset_id) / 'raw'
        if not raw_dir.exists():
            return None
        
        # Find the dataset file
        for file_path in raw_dir.iterdir():
            if file_path.is_file() and file_path.name.startswith('dataset'):
                return file_path
        
        return None
    
    def record_access(self, dataset_id: str):
        """Record access to a dataset (for analytics)."""
        metadata = self.get_dataset_metadata(dataset_id)
        if metadata:
            self.update_dataset_metadata(dataset_id, {
                'last_accessed': datetime.now().isoformat(),
                'access_count': metadata.access_count + 1
            })
    
    # =========================================================================
    # Output Operations
    # =========================================================================
    
    def outputs_exist(self, dataset_id: str) -> bool:
        """Check if outputs exist for a dataset."""
        output_dir = self.get_output_path(dataset_id)
        manifest_path = output_dir / 'manifest.json'
        return manifest_path.exists()
    
    def store_analysis_results(self, dataset_id: str, results: Dict) -> str:
        """
        Store analysis results for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            results: Analysis results dictionary
            
        Returns:
            Path to stored analysis file
        """
        output_dir = self.get_output_path(dataset_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_path = output_dir / 'analysis.json'
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        return str(analysis_path)
    
    def store_markdown(self, dataset_id: str, markdown: str) -> str:
        """Store generated markdown documentation."""
        output_dir = self.get_output_path(dataset_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        md_path = output_dir / 'user_manual.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        return str(md_path)
    
    def store_insights(self, dataset_id: str, insights: Dict) -> str:
        """Store generated insights."""
        output_dir = self.get_output_path(dataset_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        insights_path = output_dir / 'insights.json'
        with open(insights_path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False, default=str)
        
        return str(insights_path)
    
    def store_stats(self, dataset_id: str, stats: Dict) -> str:
        """Store statistical analysis."""
        output_dir = self.get_output_path(dataset_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats_path = output_dir / 'stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
        return str(stats_path)
    
    def store_schema(self, dataset_id: str, schema: Dict) -> str:
        """Store dataset schema."""
        dataset_dir = self.get_dataset_path(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        schema_path = dataset_dir / 'schema.json'
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, ensure_ascii=False, default=str)
        
        return str(schema_path)
    
    # =========================================================================
    # Chart Operations
    # =========================================================================
    
    def get_charts_path(self, dataset_id: str) -> Path:
        """Get path to charts directory for a dataset."""
        return self.get_output_path(dataset_id) / 'charts'
    
    def store_chart(self, dataset_id: str, chart_data: bytes, chart_name: str) -> str:
        """
        Store a chart image.
        
        Args:
            dataset_id: Dataset identifier
            chart_data: Chart image bytes (PNG)
            chart_name: Descriptive chart name (e.g., 'bar_revenue_by_month')
            
        Returns:
            Path to stored chart
        """
        charts_dir = self.get_charts_path(dataset_id)
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure .png extension
        if not chart_name.endswith('.png'):
            chart_name = f"{chart_name}.png"
        
        chart_path = charts_dir / chart_name
        with open(chart_path, 'wb') as f:
            f.write(chart_data)
        
        return str(chart_path)
    
    def get_chart(self, dataset_id: str, chart_name: str) -> Optional[bytes]:
        """Retrieve a stored chart."""
        charts_dir = self.get_charts_path(dataset_id)
        
        if not chart_name.endswith('.png'):
            chart_name = f"{chart_name}.png"
        
        chart_path = charts_dir / chart_name
        
        if not chart_path.exists():
            return None
        
        with open(chart_path, 'rb') as f:
            return f.read()
    
    def list_charts(self, dataset_id: str) -> List[str]:
        """List all charts for a dataset."""
        charts_dir = self.get_charts_path(dataset_id)
        
        if not charts_dir.exists():
            return []
        
        return [f.name for f in charts_dir.iterdir() if f.suffix == '.png']
    
    # =========================================================================
    # Output Manifest
    # =========================================================================
    
    def create_output_manifest(self, dataset_id: str, outputs: Dict[str, str]) -> OutputManifest:
        """
        Create output manifest for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            outputs: Dictionary of output_type -> path
            
        Returns:
            OutputManifest object
        """
        charts = self.list_charts(dataset_id)
        
        manifest = OutputManifest(
            dataset_id=dataset_id,
            generated_at=datetime.now().isoformat(),
            outputs=outputs,
            charts=charts
        )
        
        output_dir = self.get_output_path(dataset_id)
        manifest_path = output_dir / 'manifest.json'
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
        
        return manifest
    
    def get_output_manifest(self, dataset_id: str) -> Optional[OutputManifest]:
        """Retrieve output manifest for a dataset."""
        output_dir = self.get_output_path(dataset_id)
        manifest_path = output_dir / 'manifest.json'
        
        if not manifest_path.exists():
            return None
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return OutputManifest.from_dict(data)
    
    # =========================================================================
    # Retrieval Operations
    # =========================================================================
    
    def get_analysis_results(self, dataset_id: str) -> Optional[Dict]:
        """Retrieve stored analysis results."""
        output_dir = self.get_output_path(dataset_id)
        analysis_path = output_dir / 'analysis.json'
        
        if not analysis_path.exists():
            return None
        
        with open(analysis_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_markdown(self, dataset_id: str) -> Optional[str]:
        """Retrieve stored markdown documentation."""
        output_dir = self.get_output_path(dataset_id)
        md_path = output_dir / 'user_manual.md'
        
        if not md_path.exists():
            return None
        
        with open(md_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def get_insights(self, dataset_id: str) -> Optional[Dict]:
        """Retrieve stored insights."""
        output_dir = self.get_output_path(dataset_id)
        insights_path = output_dir / 'insights.json'
        
        if not insights_path.exists():
            return None
        
        with open(insights_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_schema(self, dataset_id: str) -> Optional[Dict]:
        """Retrieve stored schema."""
        dataset_dir = self.get_dataset_path(dataset_id)
        schema_path = dataset_dir / 'schema.json'
        
        if not schema_path.exists():
            return None
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # =========================================================================
    # Housekeeping
    # =========================================================================
    
    def list_all_datasets(self) -> List[Dict]:
        """List all stored datasets."""
        datasets = []
        for dataset_id, info in self._index['datasets'].items():
            datasets.append({
                'dataset_id': dataset_id,
                **info
            })
        return datasets
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        total_size = 0
        for path in self.base_path.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        
        return {
            **self._index['stats'],
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset and its outputs from storage.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            True if deleted, False if not found
        """
        if not self.dataset_exists(dataset_id):
            return False
        
        # Remove dataset directory
        dataset_dir = self.get_dataset_path(dataset_id)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        
        # Remove output directory
        output_dir = self.get_output_path(dataset_id)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        # Update index
        del self._index['datasets'][dataset_id]
        self._index['stats']['total_datasets'] = max(0, self._index['stats']['total_datasets'] - 1)
        self._save_index()
        
        return True
    
    def cleanup_orphaned_outputs(self) -> int:
        """Remove outputs for datasets that no longer exist."""
        removed = 0
        
        for output_dir in self.outputs_path.iterdir():
            if output_dir.is_dir():
                dataset_id = output_dir.name
                if not self.dataset_exists(dataset_id):
                    shutil.rmtree(output_dir)
                    removed += 1
        
        return removed

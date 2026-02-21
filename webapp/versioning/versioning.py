"""
DataMind AI - Dataset Versioning System
Main orchestration layer for idempotent dataset processing

This module provides:
- Automatic content-based dataset identification
- Idempotent analysis execution
- Transparent caching and reuse
- Audit trail and reproducibility
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .fingerprint import DatasetFingerprint, FingerprintResult
from .storage import StorageManager, DatasetMetadata, OutputManifest


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DataMind.Versioning')


@dataclass
class ProcessingResult:
    """Result of dataset processing."""
    dataset_id: str
    is_cached: bool
    analysis_results: Dict
    markdown: Optional[str]
    insights: Optional[Dict]
    charts: List[str]
    processing_time_ms: float
    message: str


class DatasetVersioningSystem:
    """
    Enterprise-grade dataset versioning system.
    
    Workflow:
    1. On upload: Generate content-based fingerprint
    2. Check if dataset ID exists in storage
    3. If exists: Reuse cached analysis results (no recomputation)
    4. If new: Run analysis, store results, generate charts
    
    Guarantees:
    - Same dataset → Same ID → Same outputs (deterministic)
    - Any change → New ID → Fresh analysis
    - Idempotent operations
    """
    
    def __init__(
        self,
        storage_path: str = None,
        hash_algorithm: str = 'sha256'
    ):
        """
        Initialize versioning system.
        
        Args:
            storage_path: Base path for storage
            hash_algorithm: Algorithm for fingerprinting
        """
        self.fingerprinter = DatasetFingerprint(algorithm=hash_algorithm)
        self.storage = StorageManager(base_path=storage_path)
        self._chart_generator = None
        
        logger.info(f"DatasetVersioningSystem initialized with storage at {self.storage.base_path}")
    
    def set_chart_generator(self, generator: Callable):
        """
        Set chart generation function.
        
        Args:
            generator: Function(dataset_id, analysis_results) -> List[chart_paths]
        """
        self._chart_generator = generator
    
    def identify_dataset(self, file_path: str, file_type: str) -> FingerprintResult:
        """
        Generate deterministic ID for a dataset.
        
        This is the core identification function. Same content always
        produces the same ID, regardless of:
        - Filename
        - Upload time
        - Upload order
        
        Args:
            file_path: Path to dataset file
            file_type: Type of file
            
        Returns:
            FingerprintResult with dataset_id
        """
        result = self.fingerprinter.generate_fingerprint(file_path, file_type)
        logger.info(f"Generated dataset ID: {result.dataset_id[:16]}... ({result.row_count} rows, {result.column_count} cols)")
        return result
    
    def process_dataset(
        self,
        file_path: str,
        file_type: str,
        original_filename: str,
        analyzer_factory: Callable,
        markdown_generator: Callable = None,
        force_reprocess: bool = False
    ) -> ProcessingResult:
        """
        Process a dataset with automatic caching.
        
        This is the main entry point. It:
        1. Identifies the dataset by content
        2. Checks cache for existing results
        3. Either returns cached results or runs full analysis
        
        Args:
            file_path: Path to uploaded file
            file_type: Type of file (sqlite, csv, etc.)
            original_filename: Original filename
            analyzer_factory: Function(file_path, file_type) -> analyzer instance
            markdown_generator: Optional function(results) -> markdown string
            force_reprocess: If True, ignore cache and reprocess
            
        Returns:
            ProcessingResult with analysis data
        """
        import time
        start_time = time.time()
        
        # Step 1: Generate content-based ID
        fingerprint = self.identify_dataset(file_path, file_type)
        dataset_id = fingerprint.dataset_id
        
        # Step 2: Check for existing dataset
        if not force_reprocess and self.storage.dataset_exists(dataset_id):
            # Dataset already processed - return cached results
            return self._handle_cache_hit(dataset_id, start_time)
        
        # Step 3: New dataset - run full processing
        return self._handle_new_dataset(
            dataset_id=dataset_id,
            fingerprint=fingerprint,
            file_path=file_path,
            file_type=file_type,
            original_filename=original_filename,
            analyzer_factory=analyzer_factory,
            markdown_generator=markdown_generator,
            start_time=start_time
        )
    
    def _handle_cache_hit(self, dataset_id: str, start_time: float) -> ProcessingResult:
        """Handle case where dataset already exists."""
        import time
        
        logger.info(f"Cache HIT for dataset {dataset_id[:16]}...")
        
        # Update stats
        self.storage._index['stats']['cache_hits'] += 1
        self.storage._save_index()
        
        # Record access
        self.storage.record_access(dataset_id)
        
        # Retrieve cached results
        analysis_results = self.storage.get_analysis_results(dataset_id)
        markdown = self.storage.get_markdown(dataset_id)
        insights = self.storage.get_insights(dataset_id)
        charts = self.storage.list_charts(dataset_id)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ProcessingResult(
            dataset_id=dataset_id,
            is_cached=True,
            analysis_results=analysis_results or {},
            markdown=markdown,
            insights=insights,
            charts=charts,
            processing_time_ms=processing_time,
            message=f"Reused cached analysis (Dataset ID: {dataset_id[:16]}...)"
        )
    
    def _handle_new_dataset(
        self,
        dataset_id: str,
        fingerprint: FingerprintResult,
        file_path: str,
        file_type: str,
        original_filename: str,
        analyzer_factory: Callable,
        markdown_generator: Callable,
        start_time: float
    ) -> ProcessingResult:
        """Handle case where dataset is new."""
        import time
        
        logger.info(f"Processing NEW dataset {dataset_id[:16]}...")
        
        try:
            # Step 1: Store raw dataset
            metadata = self.storage.store_dataset(
                dataset_id=dataset_id,
                source_path=file_path,
                original_filename=original_filename,
                file_type=file_type,
                row_count=fingerprint.row_count,
                column_count=fingerprint.column_count,
                table_count=fingerprint.table_count,
                fingerprint_algorithm=fingerprint.algorithm
            )
            
            # Step 2: Run analysis (using existing AI logic)
            analyzer = analyzer_factory(file_path, file_type)
            analysis_results = analyzer.analyze()
            
            # Close analyzer if it has a close method
            if hasattr(analyzer, 'close'):
                analyzer.close()
            
            # Step 3: Store analysis results
            self.storage.store_analysis_results(dataset_id, analysis_results)
            
            # Step 4: Store schema
            schema = self._extract_schema(analysis_results)
            self.storage.store_schema(dataset_id, schema)
            
            # Step 5: Generate and store markdown
            markdown = None
            if markdown_generator:
                markdown = markdown_generator(analysis_results)
                self.storage.store_markdown(dataset_id, markdown)
            
            # Step 6: Generate insights
            insights = self._generate_insights(analysis_results)
            self.storage.store_insights(dataset_id, insights)
            
            # Step 7: Generate and store charts
            charts = []
            if self._chart_generator:
                charts = self._chart_generator(dataset_id, analysis_results)
            
            # Step 8: Create output manifest
            outputs = {
                'analysis': 'analysis.json',
                'markdown': 'user_manual.md' if markdown else None,
                'insights': 'insights.json',
            }
            self.storage.create_output_manifest(dataset_id, outputs)
            
            # Step 9: Update metadata
            self.storage.update_dataset_metadata(dataset_id, {
                'analysis_status': 'completed',
                'quality_score': analysis_results.get('summary', {}).get('quality_score')
            })
            
            # Update stats
            self.storage._index['stats']['total_analyses'] += 1
            self.storage._save_index()
            
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                dataset_id=dataset_id,
                is_cached=False,
                analysis_results=analysis_results,
                markdown=markdown,
                insights=insights,
                charts=charts,
                processing_time_ms=processing_time,
                message=f"New analysis completed (Dataset ID: {dataset_id[:16]}...)"
            )
            
        except Exception as e:
            # Mark as failed
            self.storage.update_dataset_metadata(dataset_id, {
                'analysis_status': 'failed'
            })
            logger.error(f"Analysis failed for {dataset_id}: {str(e)}")
            raise
    
    def _extract_schema(self, analysis_results: Dict) -> Dict:
        """Extract schema information from analysis results."""
        tables = analysis_results.get('tables', [])
        
        schema = {
            'database_name': analysis_results.get('database_name'),
            'tables': []
        }
        
        for table in tables:
            table_schema = {
                'name': table.get('name'),
                'type': table.get('table_type'),
                'columns': [
                    {
                        'name': col.get('name'),
                        'data_type': col.get('data_type'),
                        'nullable': col.get('nullable'),
                        'is_primary_key': col.get('is_primary_key'),
                        'is_foreign_key': col.get('is_foreign_key'),
                        'references': col.get('references')
                    }
                    for col in table.get('columns', [])
                ]
            }
            schema['tables'].append(table_schema)
        
        schema['relationships'] = analysis_results.get('relationships', [])
        
        return schema
    
    def _generate_insights(self, analysis_results: Dict) -> Dict:
        """Generate insights from analysis results."""
        summary = analysis_results.get('summary', {})
        tables = analysis_results.get('tables', [])
        quality_issues = analysis_results.get('quality_issues', [])
        relationships = analysis_results.get('relationships', [])
        
        insights = {
            'generated_at': datetime.now().isoformat(),
            'overview': {
                'total_tables': summary.get('total_tables', 0),
                'total_columns': summary.get('total_columns', 0),
                'total_rows': summary.get('total_rows', 0),
                'quality_score': summary.get('quality_score', 0)
            },
            'table_insights': [],
            'quality_insights': [],
            'relationship_insights': []
        }
        
        # Table insights
        for table in tables:
            row_count = table.get('row_count', 0)
            col_count = len(table.get('columns', []))
            
            table_insight = {
                'name': table.get('name'),
                'type': table.get('table_type'),
                'size_category': 'large' if row_count > 10000 else 'medium' if row_count > 1000 else 'small',
                'row_count': row_count,
                'column_count': col_count
            }
            insights['table_insights'].append(table_insight)
        
        # Quality insights
        high_severity = [q for q in quality_issues if q.get('severity') == 'high']
        medium_severity = [q for q in quality_issues if q.get('severity') == 'medium']
        
        insights['quality_insights'] = {
            'high_severity_count': len(high_severity),
            'medium_severity_count': len(medium_severity),
            'total_issues': len(quality_issues),
            'top_issues': quality_issues[:5] if quality_issues else []
        }
        
        # Relationship insights
        explicit_rels = [r for r in relationships if r.get('is_explicit', True)]
        implicit_rels = [r for r in relationships if not r.get('is_explicit', True)]
        
        insights['relationship_insights'] = {
            'total_relationships': len(relationships),
            'explicit_relationships': len(explicit_rels),
            'inferred_relationships': len(implicit_rels)
        }
        
        return insights
    
    # =========================================================================
    # Query Interface
    # =========================================================================
    
    def get_dataset_by_id(self, dataset_id: str) -> Optional[Dict]:
        """
        Retrieve dataset info by ID.
        
        Returns:
            Dictionary with metadata and output paths
        """
        if not self.storage.dataset_exists(dataset_id):
            return None
        
        metadata = self.storage.get_dataset_metadata(dataset_id)
        manifest = self.storage.get_output_manifest(dataset_id)
        
        return {
            'metadata': metadata.to_dict() if metadata else None,
            'manifest': manifest.to_dict() if manifest else None,
            'raw_path': str(self.storage.get_raw_dataset_path(dataset_id)),
            'output_path': str(self.storage.get_output_path(dataset_id))
        }
    
    def get_analysis(self, dataset_id: str) -> Optional[Dict]:
        """Retrieve analysis results by dataset ID."""
        return self.storage.get_analysis_results(dataset_id)
    
    def get_markdown(self, dataset_id: str) -> Optional[str]:
        """Retrieve markdown documentation by dataset ID."""
        return self.storage.get_markdown(dataset_id)
    
    def get_chart(self, dataset_id: str, chart_name: str) -> Optional[bytes]:
        """Retrieve chart by dataset ID and name."""
        return self.storage.get_chart(dataset_id, chart_name)
    
    def list_charts(self, dataset_id: str) -> List[str]:
        """List all charts for a dataset."""
        return self.storage.list_charts(dataset_id)
    
    # =========================================================================
    # Admin Interface
    # =========================================================================
    
    def list_all_datasets(self) -> List[Dict]:
        """List all stored datasets."""
        return self.storage.list_all_datasets()
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        return self.storage.get_storage_stats()
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and all its outputs."""
        return self.storage.delete_dataset(dataset_id)
    
    def verify_integrity(self, dataset_id: str) -> Dict:
        """
        Verify integrity of a stored dataset.
        
        Re-computes fingerprint and compares to stored ID.
        """
        raw_path = self.storage.get_raw_dataset_path(dataset_id)
        if not raw_path:
            return {
                'valid': False,
                'error': 'Raw dataset not found'
            }
        
        metadata = self.storage.get_dataset_metadata(dataset_id)
        if not metadata:
            return {
                'valid': False,
                'error': 'Metadata not found'
            }
        
        # Re-compute fingerprint
        current_fingerprint = self.fingerprinter.generate_fingerprint(
            str(raw_path),
            metadata.file_type
        )
        
        return {
            'valid': current_fingerprint.dataset_id == dataset_id,
            'stored_id': dataset_id,
            'computed_id': current_fingerprint.dataset_id,
            'match': current_fingerprint.dataset_id == dataset_id
        }

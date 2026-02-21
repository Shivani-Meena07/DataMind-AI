"""
DataMind AI - Dataset Fingerprinting Module
Generates deterministic content-based identifiers for datasets

This module ensures:
- Same dataset → Same ID (idempotent)
- Any change (single cell, row, column) → Different ID
- Column order independent
- Row order independent
"""

import hashlib
import json
import sqlite3
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from io import StringIO

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class FingerprintResult:
    """Result of fingerprint generation."""
    dataset_id: str
    algorithm: str
    row_count: int
    column_count: int
    table_count: int
    canonical_signature: str


class DatasetFingerprint:
    """
    Generates cryptographic fingerprints for datasets.
    
    The fingerprinting process:
    1. Extract all tables/sheets from the dataset
    2. For each table:
       - Sort columns alphabetically
       - Sort rows lexicographically
       - Normalize NULL values
       - Normalize data types
    3. Generate canonical JSON representation
    4. Compute SHA-256 hash
    
    This ensures deterministic, content-based identification.
    """
    
    NULL_CANONICAL = "__NULL__"
    NAN_CANONICAL = "__NAN__"
    EMPTY_CANONICAL = "__EMPTY__"
    
    def __init__(self, algorithm: str = 'sha256'):
        """
        Initialize fingerprinter.
        
        Args:
            algorithm: Hash algorithm ('sha256', 'sha384', 'sha512', 'blake2b')
        """
        self.algorithm = algorithm
        
    def generate_fingerprint(self, file_path: str, file_type: str) -> FingerprintResult:
        """
        Generate fingerprint for a dataset file.
        
        Args:
            file_path: Path to dataset file
            file_type: Type of file ('sqlite', 'csv', 'excel', 'json', etc.)
            
        Returns:
            FingerprintResult with dataset_id and metadata
        """
        # Extract canonical data structure
        canonical_data = self._extract_canonical_data(file_path, file_type)
        
        # Generate canonical JSON string
        canonical_json = self._to_canonical_json(canonical_data)
        
        # Compute hash
        dataset_id = self._compute_hash(canonical_json)
        
        # Get counts
        total_rows = sum(len(table['rows']) for table in canonical_data['tables'])
        total_cols = sum(len(table['columns']) for table in canonical_data['tables'])
        
        return FingerprintResult(
            dataset_id=dataset_id,
            algorithm=self.algorithm,
            row_count=total_rows,
            column_count=total_cols,
            table_count=len(canonical_data['tables']),
            canonical_signature=canonical_json[:200] + "..." if len(canonical_json) > 200 else canonical_json
        )
    
    def _extract_canonical_data(self, file_path: str, file_type: str) -> Dict:
        """Extract data in canonical form from file."""
        if file_type == 'sqlite':
            return self._extract_sqlite(file_path)
        elif file_type == 'csv':
            return self._extract_csv(file_path)
        elif file_type == 'tsv':
            return self._extract_tsv(file_path)
        elif file_type == 'txt':
            return self._extract_txt(file_path)
        elif file_type == 'excel':
            return self._extract_excel(file_path)
        elif file_type == 'json':
            return self._extract_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _extract_sqlite(self, file_path: str) -> Dict:
        """Extract canonical data from SQLite database."""
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        
        # Get all table names (sorted for determinism)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        table_names = [row[0] for row in cursor.fetchall()]
        
        tables = []
        for table_name in table_names:
            # Get columns (sorted alphabetically)
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            columns = sorted([col[1] for col in columns_info])
            
            # Get all data, sorted by all columns for row-order independence
            col_list = ', '.join(f'[{c}]' for c in columns)
            order_by = ', '.join(f'[{c}]' for c in columns)
            cursor.execute(f"SELECT {col_list} FROM [{table_name}] ORDER BY {order_by}")
            
            rows = []
            for row in cursor.fetchall():
                canonical_row = [self._normalize_value(v) for v in row]
                rows.append(canonical_row)
            
            tables.append({
                'name': table_name,
                'columns': columns,
                'rows': rows
            })
        
        conn.close()
        
        return {'tables': tables}
    
    def _extract_csv(self, file_path: str) -> Dict:
        """Extract canonical data from CSV file."""
        if PANDAS_AVAILABLE:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except:
                df = pd.read_csv(file_path, encoding='latin-1')
            return self._dataframe_to_canonical(Path(file_path).stem, df)
        else:
            return self._csv_pure_extract(file_path)
    
    def _extract_tsv(self, file_path: str) -> Dict:
        """Extract canonical data from TSV file."""
        if PANDAS_AVAILABLE:
            try:
                df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            except:
                df = pd.read_csv(file_path, sep='\t', encoding='latin-1')
            return self._dataframe_to_canonical(Path(file_path).stem, df)
        else:
            return self._csv_pure_extract(file_path, delimiter='\t')
    
    def _extract_txt(self, file_path: str) -> Dict:
        """Extract canonical data from text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
        
        if '\t' in first_line:
            return self._extract_tsv(file_path)
        else:
            return self._extract_csv(file_path)
    
    def _extract_excel(self, file_path: str) -> Dict:
        """Extract canonical data from Excel file."""
        if not PANDAS_AVAILABLE:
            raise ValueError("pandas required for Excel files")
        
        excel_file = pd.ExcelFile(file_path)
        tables = []
        
        for sheet_name in sorted(excel_file.sheet_names):
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            if not df.empty:
                table_data = self._dataframe_to_canonical(sheet_name, df)
                tables.extend(table_data['tables'])
        
        return {'tables': tables}
    
    def _extract_json(self, file_path: str) -> Dict:
        """Extract canonical data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tables = []
        
        if isinstance(data, list):
            # Array of objects
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(data)
                return self._dataframe_to_canonical(Path(file_path).stem, df)
            else:
                columns = sorted(list(data[0].keys())) if data else []
                rows = self._normalize_json_rows(data, columns)
                tables.append({
                    'name': Path(file_path).stem,
                    'columns': columns,
                    'rows': rows
                })
        elif isinstance(data, dict):
            # Object with multiple arrays
            for key in sorted(data.keys()):
                value = data[key]
                if isinstance(value, list) and value:
                    if isinstance(value[0], dict):
                        if PANDAS_AVAILABLE:
                            df = pd.DataFrame(value)
                            table_data = self._dataframe_to_canonical(key, df)
                            tables.extend(table_data['tables'])
                        else:
                            columns = sorted(list(value[0].keys()))
                            rows = self._normalize_json_rows(value, columns)
                            tables.append({
                                'name': key,
                                'columns': columns,
                                'rows': rows
                            })
        
        return {'tables': tables}
    
    def _dataframe_to_canonical(self, name: str, df: 'pd.DataFrame') -> Dict:
        """Convert pandas DataFrame to canonical form."""
        # Sort columns alphabetically
        columns = sorted(df.columns.tolist())
        df_sorted = df[columns].copy()
        
        # Sort rows by all columns for row-order independence
        try:
            df_sorted = df_sorted.sort_values(by=columns).reset_index(drop=True)
        except:
            # If sorting fails (mixed types), convert to string first
            df_str = df_sorted.astype(str)
            sort_order = df_str.sort_values(by=columns).index
            df_sorted = df_sorted.loc[sort_order].reset_index(drop=True)
        
        # Convert to rows with normalized values
        rows = []
        for _, row in df_sorted.iterrows():
            canonical_row = [self._normalize_value(row[col]) for col in columns]
            rows.append(canonical_row)
        
        columns = [str(c) for c in columns]  # Ensure string column names
        
        return {
            'tables': [{
                'name': name,
                'columns': columns,
                'rows': rows
            }]
        }
    
    def _csv_pure_extract(self, file_path: str, delimiter: str = ',') -> Dict:
        """Extract CSV without pandas."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            columns = sorted(reader.fieldnames or [])
            
            rows = []
            for row in reader:
                canonical_row = [self._normalize_value(row.get(col)) for col in columns]
                rows.append(canonical_row)
        
        # Sort rows
        rows.sort(key=lambda r: tuple(str(v) for v in r))
        
        return {
            'tables': [{
                'name': Path(file_path).stem,
                'columns': columns,
                'rows': rows
            }]
        }
    
    def _normalize_json_rows(self, data: List[Dict], columns: List[str]) -> List[List]:
        """Normalize JSON array of objects to rows."""
        rows = []
        for item in data:
            row = [self._normalize_value(item.get(col)) for col in columns]
            rows.append(row)
        
        # Sort for row-order independence
        rows.sort(key=lambda r: tuple(str(v) for v in r))
        return rows
    
    def _normalize_value(self, value: Any) -> str:
        """
        Normalize a value to canonical string representation.
        
        This ensures:
        - NULL/None → "__NULL__"
        - NaN → "__NAN__"
        - Empty string → "__EMPTY__"
        - Numbers → Consistent string representation
        - Booleans → "true"/"false"
        """
        if value is None:
            return self.NULL_CANONICAL
        
        # Handle pandas NA/NaN
        if PANDAS_AVAILABLE:
            try:
                if pd.isna(value):
                    return self.NAN_CANONICAL
            except (ValueError, TypeError):
                pass
        
        # Handle numpy types
        try:
            import numpy as np
            if isinstance(value, np.bool_):
                return "true" if value else "false"
            if isinstance(value, (np.integer, np.floating)):
                if np.isnan(value) or np.isinf(value):
                    return self.NAN_CANONICAL
                return str(value)
        except ImportError:
            pass
        
        # Standard types
        if isinstance(value, bool):
            return "true" if value else "false"
        
        if isinstance(value, (int, float)):
            if value != value:  # NaN check
                return self.NAN_CANONICAL
            return str(value)
        
        if isinstance(value, str):
            if value == '':
                return self.EMPTY_CANONICAL
            return value.strip()
        
        # Fallback: convert to string
        return str(value)
    
    def _to_canonical_json(self, data: Dict) -> str:
        """
        Convert to deterministic JSON string.
        
        Uses sorted keys and consistent formatting for reproducibility.
        """
        return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    
    def _compute_hash(self, data: str) -> str:
        """Compute cryptographic hash of data string."""
        if self.algorithm == 'sha256':
            hasher = hashlib.sha256()
        elif self.algorithm == 'sha384':
            hasher = hashlib.sha384()
        elif self.algorithm == 'sha512':
            hasher = hashlib.sha512()
        elif self.algorithm == 'blake2b':
            hasher = hashlib.blake2b(digest_size=32)
        else:
            hasher = hashlib.sha256()
        
        hasher.update(data.encode('utf-8'))
        return hasher.hexdigest()
    
    def verify_fingerprint(self, file_path: str, file_type: str, expected_id: str) -> bool:
        """
        Verify that a file matches an expected fingerprint.
        
        Args:
            file_path: Path to dataset file
            file_type: Type of file
            expected_id: Expected dataset ID
            
        Returns:
            True if fingerprint matches, False otherwise
        """
        result = self.generate_fingerprint(file_path, file_type)
        return result.dataset_id == expected_id
    
    def compute_incremental_hash(self, base_id: str, new_data: str) -> str:
        """
        Compute hash that incorporates base ID and new data.
        Useful for derived datasets.
        """
        combined = f"{base_id}:{new_data}"
        return self._compute_hash(combined)

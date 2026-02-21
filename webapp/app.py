"""
DataMind AI - Web Application
A beautiful, no-API-key-needed database documentation generator
"""

import os
import json
import sqlite3
import tempfile
import csv
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
from flask.json.provider import DefaultJSONProvider
from werkzeug.utils import secure_filename
import numpy as np

# Import versioning system
from versioning import DatasetVersioningSystem, ChartGenerator, create_chart_generator

# Custom JSON encoder for numpy types
class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd is not None and hasattr(pd, 'isna') and pd.isna(obj):
            return None
        return super().default(obj)

app = Flask(__name__)
app.json_provider_class = CustomJSONProvider
app.json = CustomJSONProvider(app)
app.secret_key = 'datamind-secret-key-2024'
CORS(app)

# ============================================================================
# Dataset Versioning System (Content-Based Identity & Persistence)
# ============================================================================

# Initialize versioning system
STORAGE_PATH = os.path.join(os.path.dirname(__file__), 'storage')
versioning_system = DatasetVersioningSystem(storage_path=STORAGE_PATH)

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {
    # SQLite databases
    'db', 'sqlite', 'sqlite3',
    # CSV files
    'csv', 'tsv', 'txt',
    # Excel files
    'xlsx', 'xls', 'xlsm',
    # JSON files
    'json',
    # SQL dump files
    'sql'
}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    references: Optional[str] = None
    sample_values: List[Any] = None
    null_count: int = 0
    unique_count: int = 0
    total_count: int = 0
    min_value: Any = None
    max_value: Any = None
    semantic_type: str = "unknown"
    business_description: str = ""

@dataclass 
class TableInfo:
    name: str
    columns: List[ColumnInfo]
    row_count: int
    primary_keys: List[str]
    foreign_keys: List[Dict]
    indexes: List[str]
    table_type: str = "unknown"  # fact, dimension, bridge, lookup
    business_description: str = ""
    
@dataclass
class Relationship:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str  # one-to-one, one-to-many, many-to-many
    is_explicit: bool = True

@dataclass
class DataQualityIssue:
    table: str
    column: str
    issue_type: str
    severity: str  # high, medium, low
    description: str
    recommendation: str

# ============================================================================
# JSON Sanitization Helper
# ============================================================================

def sanitize_for_json(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_for_json(item) for item in obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if PANDAS_AVAILABLE:
        if pd.isna(obj):
            return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)):
        return obj
    # Fallback: convert to string
    try:
        return str(obj)
    except:
        return None

# ============================================================================
# Smart Analyzer (No LLM Required)
# ============================================================================

class SmartAnalyzer:
    """Rule-based semantic analysis - no API keys needed!"""
    
    # Semantic type patterns
    SEMANTIC_PATTERNS = {
        'identifier': [r'_id$', r'^id$', r'_pk$', r'_key$', r'_code$'],
        'name': [r'_name$', r'^name$', r'_title$', r'first_name', r'last_name', r'full_name'],
        'email': [r'email', r'e_mail', r'mail_address'],
        'phone': [r'phone', r'mobile', r'cell', r'tel_', r'telephone'],
        'address': [r'address', r'street', r'city', r'state', r'country', r'zip', r'postal', r'region'],
        'date': [r'_date$', r'_at$', r'created', r'updated', r'modified', r'timestamp', r'_time$'],
        'amount': [r'amount', r'price', r'cost', r'total', r'sum', r'value', r'fee', r'payment'],
        'quantity': [r'quantity', r'qty', r'count', r'number', r'num_'],
        'percentage': [r'percent', r'rate', r'ratio', r'pct'],
        'status': [r'status', r'state', r'is_', r'has_', r'flag', r'active', r'enabled'],
        'description': [r'description', r'desc', r'comment', r'note', r'remarks', r'text'],
        'url': [r'url', r'link', r'website', r'uri', r'href'],
        'category': [r'category', r'type', r'class', r'group', r'segment'],
        'score': [r'score', r'rating', r'rank', r'weight', r'priority'],
        'coordinate': [r'lat', r'lng', r'longitude', r'latitude', r'geo', r'coord'],
    }
    
    # Table type patterns
    TABLE_TYPE_PATTERNS = {
        'fact': [r'order', r'transaction', r'sale', r'payment', r'event', r'log', r'history'],
        'dimension': [r'customer', r'product', r'user', r'employee', r'store', r'vendor', r'seller'],
        'bridge': [r'_item', r'_line', r'_detail', r'_product', r'mapping'],
        'lookup': [r'status', r'type', r'category', r'translation', r'config', r'setting'],
    }
    
    # Business description templates
    TABLE_DESCRIPTIONS = {
        'customer': 'Stores customer information including contact details and demographics',
        'order': 'Contains order transactions with references to customers and order details',
        'product': 'Product catalog with descriptions, categories, and specifications',
        'user': 'User account information for system access and authentication',
        'payment': 'Payment transactions and financial records',
        'seller': 'Seller/vendor information in the marketplace',
        'review': 'Customer reviews and ratings for products or services',
        'geolocation': 'Geographic location data with coordinates',
        'category': 'Product or item categorization hierarchy',
        'item': 'Line items or detailed records within transactions',
    }
    
    COLUMN_DESCRIPTIONS = {
        'id': 'Unique identifier for the record',
        'name': 'Display name or label',
        'email': 'Email address for communication',
        'phone': 'Contact phone number',
        'address': 'Physical or mailing address',
        'city': 'City name for geographic location',
        'state': 'State or province',
        'country': 'Country name or code',
        'zip': 'Postal/ZIP code',
        'date': 'Date of the event or record',
        'created': 'Timestamp when record was created',
        'updated': 'Timestamp of last modification',
        'status': 'Current status or state',
        'price': 'Monetary price value',
        'amount': 'Numerical amount or total',
        'quantity': 'Count or quantity',
        'description': 'Detailed description text',
        'category': 'Classification category',
        'type': 'Type or classification',
        'score': 'Numerical score or rating',
        'weight': 'Weight measurement or priority',
        'latitude': 'Geographic latitude coordinate',
        'longitude': 'Geographic longitude coordinate',
    }
    
    def infer_semantic_type(self, column_name: str, data_type: str, samples: List) -> str:
        """Infer semantic type from column name and data."""
        col_lower = column_name.lower()
        
        for semantic_type, patterns in self.SEMANTIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, col_lower):
                    return semantic_type
        
        # Check data type
        if 'int' in data_type.lower() and col_lower.endswith('_id'):
            return 'identifier'
        if 'date' in data_type.lower() or 'time' in data_type.lower():
            return 'date'
        if 'decimal' in data_type.lower() or 'float' in data_type.lower():
            if any(kw in col_lower for kw in ['price', 'amount', 'cost', 'total']):
                return 'amount'
            return 'numeric'
        
        return 'text' if 'char' in data_type.lower() or 'text' in data_type.lower() else 'unknown'
    
    def infer_table_type(self, table_name: str, columns: List[ColumnInfo], fk_count: int) -> str:
        """Infer table type (fact, dimension, bridge, lookup)."""
        table_lower = table_name.lower()
        
        for table_type, patterns in self.TABLE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, table_lower):
                    return table_type
        
        # Heuristics
        pk_count = sum(1 for c in columns if c.is_primary_key)
        
        if fk_count >= 2 and pk_count <= 1:
            return 'fact'
        elif fk_count == 0 and pk_count == 1:
            return 'dimension'
        elif fk_count >= 2 and pk_count >= 2:
            return 'bridge'
        
        return 'dimension'
    
    def generate_table_description(self, table_name: str, table_type: str, columns: List[ColumnInfo]) -> str:
        """Generate business description for table."""
        table_lower = table_name.lower()
        
        # Check for known patterns
        for keyword, desc in self.TABLE_DESCRIPTIONS.items():
            if keyword in table_lower:
                return desc
        
        # Generate based on table type
        col_names = ', '.join([c.name for c in columns[:5]])
        
        if table_type == 'fact':
            return f"Transactional table storing business events with columns: {col_names}"
        elif table_type == 'dimension':
            return f"Master data table containing reference information with columns: {col_names}"
        elif table_type == 'bridge':
            return f"Junction table linking multiple entities together"
        else:
            return f"Data table with columns: {col_names}"
    
    def generate_column_description(self, column: ColumnInfo, table_name: str) -> str:
        """Generate business description for column."""
        col_lower = column.name.lower()
        
        # Check for known patterns
        for keyword, desc in self.COLUMN_DESCRIPTIONS.items():
            if keyword in col_lower:
                return desc
        
        # Generate based on semantic type
        descriptions = {
            'identifier': f'Unique identifier linking to {column.references or "related records"}',
            'name': 'Name or label for display purposes',
            'email': 'Email address for electronic communication',
            'phone': 'Phone number for contact',
            'address': 'Physical or mailing address component',
            'date': 'Date/time value for temporal tracking',
            'amount': 'Monetary or numeric amount',
            'quantity': 'Count or quantity value',
            'percentage': 'Percentage or ratio value',
            'status': 'Status flag or indicator',
            'description': 'Free-text description or notes',
            'url': 'Web link or URL reference',
            'category': 'Classification or categorization',
            'score': 'Numeric score, rating, or rank',
            'coordinate': 'Geographic coordinate value',
        }
        
        return descriptions.get(column.semantic_type, f'Data field of type {column.data_type}')


# ============================================================================
# Database Analyzer
# ============================================================================

class DatabaseAnalyzer:
    """Analyzes SQLite databases and generates documentation."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.smart_analyzer = SmartAnalyzer()
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.quality_issues: List[DataQualityIssue] = []
    
    def analyze(self) -> Dict:
        """Run full analysis and return results."""
        self._scan_schema()
        self._profile_data()
        self._analyze_relationships()
        self._detect_quality_issues()
        self._generate_business_context()
        
        return self._compile_results()
    
    def _scan_schema(self):
        """Scan database schema."""
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        table_names = [row[0] for row in cursor.fetchall()]
        
        for table_name in table_names:
            # Get columns
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_raw = cursor.fetchall()
            
            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            fks = cursor.fetchall()
            fk_columns = {fk[3]: {'table': fk[2], 'column': fk[4]} for fk in fks}
            
            # Get indexes
            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = [idx[1] for idx in cursor.fetchall()]
            
            columns = []
            pks = []
            
            for col in columns_raw:
                col_name = col[1]
                is_pk = bool(col[5])
                is_fk = col_name in fk_columns
                
                if is_pk:
                    pks.append(col_name)
                
                column = ColumnInfo(
                    name=col_name,
                    data_type=col[2] or 'TEXT',
                    nullable=not bool(col[3]),
                    is_primary_key=is_pk,
                    is_foreign_key=is_fk,
                    references=f"{fk_columns[col_name]['table']}.{fk_columns[col_name]['column']}" if is_fk else None,
                    sample_values=[]
                )
                columns.append(column)
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            table = TableInfo(
                name=table_name,
                columns=columns,
                row_count=row_count,
                primary_keys=pks,
                foreign_keys=[{'column': k, 'references': v} for k, v in fk_columns.items()],
                indexes=indexes
            )
            self.tables.append(table)
    
    def _profile_data(self):
        """Profile data in each table."""
        cursor = self.conn.cursor()
        
        for table in self.tables:
            for column in table.columns:
                try:
                    # Get sample values
                    cursor.execute(f"SELECT DISTINCT [{column.name}] FROM [{table.name}] WHERE [{column.name}] IS NOT NULL LIMIT 10")
                    column.sample_values = [row[0] for row in cursor.fetchall()]
                    
                    # Get null count
                    cursor.execute(f"SELECT COUNT(*) FROM [{table.name}] WHERE [{column.name}] IS NULL")
                    column.null_count = cursor.fetchone()[0]
                    
                    # Get unique count
                    cursor.execute(f"SELECT COUNT(DISTINCT [{column.name}]) FROM [{table.name}]")
                    column.unique_count = cursor.fetchone()[0]
                    
                    column.total_count = table.row_count
                    
                    # Get min/max for numeric columns
                    if column.data_type.upper() in ['INTEGER', 'REAL', 'NUMERIC', 'FLOAT', 'DECIMAL']:
                        cursor.execute(f"SELECT MIN([{column.name}]), MAX([{column.name}]) FROM [{table.name}]")
                        result = cursor.fetchone()
                        column.min_value = result[0]
                        column.max_value = result[1]
                    
                    # Infer semantic type
                    column.semantic_type = self.smart_analyzer.infer_semantic_type(
                        column.name, 
                        column.data_type, 
                        column.sample_values
                    )
                except Exception as e:
                    print(f"Error profiling {table.name}.{column.name}: {e}")
    
    def _analyze_relationships(self):
        """Analyze table relationships."""
        # Explicit foreign keys
        for table in self.tables:
            for fk in table.foreign_keys:
                ref_parts = fk['references']['table'], fk['references']['column']
                
                rel = Relationship(
                    from_table=table.name,
                    from_column=fk['column'],
                    to_table=ref_parts[0],
                    to_column=ref_parts[1],
                    relationship_type='many-to-one',
                    is_explicit=True
                )
                self.relationships.append(rel)
        
        # Detect implicit relationships (by naming convention)
        table_names = {t.name.lower(): t.name for t in self.tables}
        
        for table in self.tables:
            for column in table.columns:
                if column.is_foreign_key:
                    continue
                
                col_lower = column.name.lower()
                
                # Check for _id suffix
                if col_lower.endswith('_id'):
                    potential_table = col_lower[:-3]
                    
                    # Try plural/singular variations
                    for variation in [potential_table, potential_table + 's', potential_table + 'es', potential_table[:-1] if potential_table.endswith('s') else potential_table]:
                        if variation in table_names and variation != table.name.lower():
                            rel = Relationship(
                                from_table=table.name,
                                from_column=column.name,
                                to_table=table_names[variation],
                                to_column='id',
                                relationship_type='many-to-one',
                                is_explicit=False
                            )
                            self.relationships.append(rel)
                            break
    
    def _detect_quality_issues(self):
        """Detect data quality issues."""
        for table in self.tables:
            for column in table.columns:
                # High null percentage
                if column.total_count > 0:
                    null_pct = (column.null_count / column.total_count) * 100
                    
                    if null_pct > 50:
                        self.quality_issues.append(DataQualityIssue(
                            table=table.name,
                            column=column.name,
                            issue_type='high_null_rate',
                            severity='high' if null_pct > 80 else 'medium',
                            description=f'{null_pct:.1f}% of values are NULL',
                            recommendation='Consider making this column nullable or providing default values'
                        ))
                    
                    # Low uniqueness for ID columns
                    if column.semantic_type == 'identifier' and column.unique_count < column.total_count * 0.5:
                        self.quality_issues.append(DataQualityIssue(
                            table=table.name,
                            column=column.name,
                            issue_type='low_cardinality',
                            severity='medium',
                            description=f'Identifier column has only {column.unique_count} unique values',
                            recommendation='Verify if this column should be a unique identifier'
                        ))
            
            # Empty table
            if table.row_count == 0:
                self.quality_issues.append(DataQualityIssue(
                    table=table.name,
                    column='*',
                    issue_type='empty_table',
                    severity='low',
                    description='Table contains no data',
                    recommendation='Verify if this table should contain data'
                ))
    
    def _generate_business_context(self):
        """Generate business descriptions using smart analyzer."""
        for table in self.tables:
            fk_count = len(table.foreign_keys)
            table.table_type = self.smart_analyzer.infer_table_type(table.name, table.columns, fk_count)
            table.business_description = self.smart_analyzer.generate_table_description(
                table.name, table.table_type, table.columns
            )
            
            for column in table.columns:
                column.business_description = self.smart_analyzer.generate_column_description(column, table.name)
    
    def _compile_results(self) -> Dict:
        """Compile all analysis results."""
        return {
            'database_name': Path(self.db_path).stem,
            'analysis_date': datetime.now().isoformat(),
            'summary': {
                'total_tables': len(self.tables),
                'total_columns': sum(len(t.columns) for t in self.tables),
                'total_rows': sum(t.row_count for t in self.tables),
                'total_relationships': len(self.relationships),
                'quality_issues': len(self.quality_issues),
                'quality_score': self._calculate_quality_score()
            },
            'tables': [self._table_to_dict(t) for t in self.tables],
            'relationships': [asdict(r) for r in self.relationships],
            'quality_issues': [asdict(q) for q in self.quality_issues],
            'sample_queries': self._generate_sample_queries()
        }
    
    def _calculate_quality_score(self) -> int:
        """Calculate overall quality score (0-100)."""
        if not self.tables:
            return 0
        
        score = 100
        
        # Deduct for quality issues
        for issue in self.quality_issues:
            if issue.severity == 'high':
                score -= 10
            elif issue.severity == 'medium':
                score -= 5
            else:
                score -= 2
        
        return max(0, min(100, score))
    
    def _table_to_dict(self, table: TableInfo) -> Dict:
        """Convert table to dictionary."""
        return {
            'name': table.name,
            'row_count': table.row_count,
            'table_type': table.table_type,
            'business_description': table.business_description,
            'primary_keys': table.primary_keys,
            'indexes': table.indexes,
            'columns': [
                {
                    'name': c.name,
                    'data_type': c.data_type,
                    'nullable': c.nullable,
                    'is_primary_key': c.is_primary_key,
                    'is_foreign_key': c.is_foreign_key,
                    'references': c.references,
                    'semantic_type': c.semantic_type,
                    'business_description': c.business_description,
                    'null_count': c.null_count,
                    'unique_count': c.unique_count,
                    'total_count': c.total_count,
                    'null_percentage': round((c.null_count / c.total_count * 100), 1) if c.total_count > 0 else 0,
                    'sample_values': c.sample_values[:5] if c.sample_values else []
                }
                for c in table.columns
            ]
        }
    
    def _generate_sample_queries(self) -> List[Dict]:
        """Generate sample SQL queries."""
        queries = []
        
        # Find fact tables and dimension tables
        fact_tables = [t for t in self.tables if t.table_type == 'fact']
        dim_tables = [t for t in self.tables if t.table_type == 'dimension']
        
        # Basic SELECT for each table
        for table in self.tables[:3]:
            queries.append({
                'title': f'View {table.name} data',
                'description': f'Retrieve sample records from {table.name}',
                'sql': f'SELECT * FROM {table.name} LIMIT 10;'
            })
        
        # Join queries
        for rel in self.relationships[:3]:
            queries.append({
                'title': f'Join {rel.from_table} with {rel.to_table}',
                'description': f'Combine data from related tables',
                'sql': f'''SELECT a.*, b.*
FROM {rel.from_table} a
JOIN {rel.to_table} b ON a.{rel.from_column} = b.{rel.to_column}
LIMIT 10;'''
            })
        
        # Aggregation queries
        for table in fact_tables[:2]:
            amount_cols = [c for c in table.columns if c.semantic_type == 'amount']
            if amount_cols:
                queries.append({
                    'title': f'Aggregate {table.name}',
                    'description': f'Summary statistics for {table.name}',
                    'sql': f'''SELECT 
    COUNT(*) as total_records,
    SUM({amount_cols[0].name}) as total_amount,
    AVG({amount_cols[0].name}) as avg_amount
FROM {table.name};'''
                })
        
        return queries
    
    def close(self):
        """Close database connection."""
        self.conn.close()


# ============================================================================
# CSV/Excel/JSON Analyzer
# ============================================================================

class FileDataAnalyzer:
    """Analyzes CSV, Excel, and JSON files and generates documentation."""
    
    def __init__(self, file_path: str, file_type: str):
        self.file_path = file_path
        self.file_type = file_type
        self.smart_analyzer = SmartAnalyzer()
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.quality_issues: List[DataQualityIssue] = []
        self.dataframes: Dict[str, Any] = {}
    
    def analyze(self) -> Dict:
        """Run full analysis and return results."""
        self._load_data()
        self._analyze_structure()
        self._profile_data()
        self._detect_relationships()
        self._detect_quality_issues()
        self._generate_business_context()
        
        return self._compile_results()
    
    def _load_data(self):
        """Load data from file."""
        if self.file_type == 'csv':
            self._load_csv()
        elif self.file_type == 'tsv':
            self._load_tsv()
        elif self.file_type == 'excel':
            self._load_excel()
        elif self.file_type == 'json':
            self._load_json()
        elif self.file_type == 'txt':
            self._load_txt()
    
    def _load_csv(self):
        """Load CSV file."""
        if PANDAS_AVAILABLE:
            try:
                df = pd.read_csv(self.file_path, encoding='utf-8')
            except:
                df = pd.read_csv(self.file_path, encoding='latin-1')
            self.dataframes[Path(self.file_path).stem] = df
        else:
            self._load_csv_pure()
    
    def _load_csv_pure(self):
        """Load CSV without pandas."""
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.dataframes[Path(self.file_path).stem] = {
                'columns': reader.fieldnames or [],
                'rows': rows
            }
    
    def _load_tsv(self):
        """Load TSV file."""
        if PANDAS_AVAILABLE:
            try:
                df = pd.read_csv(self.file_path, sep='\t', encoding='utf-8')
            except:
                df = pd.read_csv(self.file_path, sep='\t', encoding='latin-1')
            self.dataframes[Path(self.file_path).stem] = df
        else:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows = list(reader)
                self.dataframes[Path(self.file_path).stem] = {
                    'columns': reader.fieldnames or [],
                    'rows': rows
                }
    
    def _load_txt(self):
        """Load text file (try CSV or TSV)."""
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            f.seek(0)
            
            if '\t' in first_line:
                self._load_tsv()
            else:
                self._load_csv()
    
    def _load_excel(self):
        """Load Excel file."""
        if PANDAS_AVAILABLE and EXCEL_AVAILABLE:
            excel_file = pd.ExcelFile(self.file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if not df.empty:
                    self.dataframes[sheet_name] = df
        else:
            raise ValueError("pandas and openpyxl required for Excel files. Install with: pip install pandas openpyxl")
    
    def _load_json(self):
        """Load JSON file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Array of objects
            if PANDAS_AVAILABLE:
                self.dataframes[Path(self.file_path).stem] = pd.DataFrame(data)
            else:
                self.dataframes[Path(self.file_path).stem] = {
                    'columns': list(data[0].keys()) if data else [],
                    'rows': data
                }
        elif isinstance(data, dict):
            # Object with multiple arrays
            for key, value in data.items():
                if isinstance(value, list) and value:
                    if isinstance(value[0], dict):
                        if PANDAS_AVAILABLE:
                            self.dataframes[key] = pd.DataFrame(value)
                        else:
                            self.dataframes[key] = {
                                'columns': list(value[0].keys()),
                                'rows': value
                            }
    
    def _analyze_structure(self):
        """Analyze data structure."""
        for table_name, data in self.dataframes.items():
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                columns = self._analyze_pandas_df(data, table_name)
                row_count = len(data)
            else:
                columns = self._analyze_dict_data(data, table_name)
                row_count = len(data.get('rows', []))
            
            table = TableInfo(
                name=table_name,
                columns=columns,
                row_count=row_count,
                primary_keys=[],
                foreign_keys=[],
                indexes=[]
            )
            self.tables.append(table)
    
    def _analyze_pandas_df(self, df, table_name: str) -> List[ColumnInfo]:
        """Analyze pandas DataFrame columns."""
        columns = []
        
        for col_name in df.columns:
            dtype = str(df[col_name].dtype)
            
            # Map pandas dtypes to SQL types
            if 'int' in dtype:
                sql_type = 'INTEGER'
            elif 'float' in dtype:
                sql_type = 'REAL'
            elif 'datetime' in dtype:
                sql_type = 'DATETIME'
            elif 'bool' in dtype:
                sql_type = 'BOOLEAN'
            else:
                sql_type = 'TEXT'
            
            # Get sample values
            samples = df[col_name].dropna().head(10).tolist()
            
            # Check for potential primary key
            is_pk = (col_name.lower() in ['id', 'pk'] or 
                     col_name.lower().endswith('_id') and 
                     df[col_name].nunique() == len(df))
            
            # Check for potential foreign key
            is_fk = (col_name.lower().endswith('_id') and 
                     col_name.lower() not in ['id'] and 
                     not is_pk)
            
            column = ColumnInfo(
                name=str(col_name),
                data_type=sql_type,
                nullable=df[col_name].isnull().any(),
                is_primary_key=is_pk,
                is_foreign_key=is_fk,
                references=None,
                sample_values=samples,
                null_count=int(df[col_name].isnull().sum()),
                unique_count=int(df[col_name].nunique()),
                total_count=len(df)
            )
            columns.append(column)
        
        return columns
    
    def _analyze_dict_data(self, data: Dict, table_name: str) -> List[ColumnInfo]:
        """Analyze dictionary-based data."""
        columns = []
        col_names = data.get('columns', [])
        rows = data.get('rows', [])
        
        for col_name in col_names:
            # Get values for this column
            values = [row.get(col_name) for row in rows]
            non_null_values = [v for v in values if v is not None and v != '']
            
            # Infer type
            sql_type = self._infer_type(non_null_values)
            
            # Sample values
            samples = non_null_values[:10]
            
            column = ColumnInfo(
                name=str(col_name),
                data_type=sql_type,
                nullable=len(non_null_values) < len(values),
                is_primary_key=col_name.lower() in ['id', 'pk'],
                is_foreign_key=col_name.lower().endswith('_id') and col_name.lower() != 'id',
                references=None,
                sample_values=samples,
                null_count=len(values) - len(non_null_values),
                unique_count=len(set(values)),
                total_count=len(values)
            )
            columns.append(column)
        
        return columns
    
    def _infer_type(self, values: List) -> str:
        """Infer SQL type from values."""
        if not values:
            return 'TEXT'
        
        # Try integer
        try:
            for v in values[:100]:
                if v is not None:
                    int(v)
            return 'INTEGER'
        except:
            pass
        
        # Try float
        try:
            for v in values[:100]:
                if v is not None:
                    float(v)
            return 'REAL'
        except:
            pass
        
        return 'TEXT'
    
    def _profile_data(self):
        """Profile data in each table."""
        for table in self.tables:
            data = self.dataframes.get(table.name)
            
            if data is None:
                continue
                
            for column in table.columns:
                if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                    col_data = data[column.name]
                    
                    # Get min/max for numeric columns
                    if column.data_type in ['INTEGER', 'REAL']:
                        try:
                            column.min_value = col_data.min()
                            column.max_value = col_data.max()
                        except:
                            pass
                    
                    # Infer semantic type
                    samples = column.sample_values or []
                    column.semantic_type = self.smart_analyzer.infer_semantic_type(
                        column.name, column.data_type, samples
                    )
    
    def _detect_relationships(self):
        """Detect relationships between tables."""
        if len(self.tables) < 2:
            return
        
        table_names = {t.name.lower(): t.name for t in self.tables}
        
        for table in self.tables:
            for column in table.columns:
                if column.is_foreign_key:
                    col_lower = column.name.lower()
                    
                    if col_lower.endswith('_id'):
                        potential_table = col_lower[:-3]
                        
                        for variation in [potential_table, potential_table + 's', potential_table + 'es']:
                            if variation in table_names:
                                rel = Relationship(
                                    from_table=table.name,
                                    from_column=column.name,
                                    to_table=table_names[variation],
                                    to_column='id',
                                    relationship_type='many-to-one',
                                    is_explicit=False
                                )
                                self.relationships.append(rel)
                                break
    
    def _detect_quality_issues(self):
        """Detect data quality issues."""
        for table in self.tables:
            for column in table.columns:
                if column.total_count > 0:
                    null_pct = (column.null_count / column.total_count) * 100
                    
                    if null_pct > 50:
                        self.quality_issues.append(DataQualityIssue(
                            table=table.name,
                            column=column.name,
                            issue_type='high_null_rate',
                            severity='high' if null_pct > 80 else 'medium',
                            description=f'{null_pct:.1f}% of values are NULL/empty',
                            recommendation='Consider filling missing values or removing column'
                        ))
            
            if table.row_count == 0:
                self.quality_issues.append(DataQualityIssue(
                    table=table.name,
                    column='*',
                    issue_type='empty_table',
                    severity='low',
                    description='Table/sheet contains no data',
                    recommendation='Verify if data should exist'
                ))
    
    def _generate_business_context(self):
        """Generate business descriptions."""
        for table in self.tables:
            fk_count = sum(1 for c in table.columns if c.is_foreign_key)
            table.table_type = self.smart_analyzer.infer_table_type(table.name, table.columns, fk_count)
            table.business_description = self.smart_analyzer.generate_table_description(
                table.name, table.table_type, table.columns
            )
            
            for column in table.columns:
                column.business_description = self.smart_analyzer.generate_column_description(column, table.name)
    
    def _compile_results(self) -> Dict:
        """Compile all analysis results."""
        return {
            'database_name': Path(self.file_path).stem,
            'file_type': self.file_type.upper(),
            'analysis_date': datetime.now().isoformat(),
            'summary': {
                'total_tables': len(self.tables),
                'total_columns': sum(len(t.columns) for t in self.tables),
                'total_rows': sum(t.row_count for t in self.tables),
                'total_relationships': len(self.relationships),
                'quality_issues': len(self.quality_issues),
                'quality_score': self._calculate_quality_score()
            },
            'tables': [self._table_to_dict(t) for t in self.tables],
            'relationships': [asdict(r) for r in self.relationships],
            'quality_issues': [asdict(q) for q in self.quality_issues],
            'sample_queries': self._generate_sample_queries()
        }
    
    def _calculate_quality_score(self) -> int:
        """Calculate overall quality score."""
        if not self.tables:
            return 0
        
        score = 100
        for issue in self.quality_issues:
            if issue.severity == 'high':
                score -= 10
            elif issue.severity == 'medium':
                score -= 5
            else:
                score -= 2
        
        return max(0, min(100, score))
    
    def _table_to_dict(self, table: TableInfo) -> Dict:
        """Convert table to dictionary."""
        return {
            'name': table.name,
            'row_count': table.row_count,
            'table_type': table.table_type,
            'business_description': table.business_description,
            'primary_keys': table.primary_keys,
            'indexes': table.indexes,
            'columns': [
                {
                    'name': c.name,
                    'data_type': c.data_type,
                    'nullable': c.nullable,
                    'is_primary_key': c.is_primary_key,
                    'is_foreign_key': c.is_foreign_key,
                    'references': c.references,
                    'semantic_type': c.semantic_type,
                    'business_description': c.business_description,
                    'null_count': c.null_count,
                    'unique_count': c.unique_count,
                    'total_count': c.total_count,
                    'null_percentage': round((c.null_count / c.total_count * 100), 1) if c.total_count > 0 else 0,
                    'sample_values': c.sample_values[:5] if c.sample_values else []
                }
                for c in table.columns
            ]
        }
    
    def _generate_sample_queries(self) -> List[Dict]:
        """Generate sample queries."""
        queries = []
        
        for table in self.tables[:3]:
            queries.append({
                'title': f'View {table.name} data',
                'description': f'Retrieve sample records from {table.name}',
                'sql': f'SELECT * FROM "{table.name}" LIMIT 10;'
            })
        
        for rel in self.relationships[:2]:
            queries.append({
                'title': f'Join {rel.from_table} with {rel.to_table}',
                'description': f'Combine data from related tables',
                'sql': f'''SELECT a.*, b.*
FROM "{rel.from_table}" a
JOIN "{rel.to_table}" b ON a.{rel.from_column} = b.{rel.to_column}
LIMIT 10;'''
            })
        
        return queries
    
    def close(self):
        """Clean up resources."""
        self.dataframes.clear()


# ============================================================================
# File Type Detection
# ============================================================================

def get_file_type(filename: str) -> str:
    """Detect file type from extension."""
    ext = filename.lower().split('.')[-1]
    
    if ext in ['db', 'sqlite', 'sqlite3']:
        return 'sqlite'
    elif ext == 'csv':
        return 'csv'
    elif ext == 'tsv':
        return 'tsv'
    elif ext == 'txt':
        return 'txt'
    elif ext in ['xlsx', 'xls', 'xlsm']:
        return 'excel'
    elif ext == 'json':
        return 'json'
    else:
        return 'unknown'


def get_analyzer(file_path: str, file_type: str):
    """Get appropriate analyzer for file type."""
    if file_type == 'sqlite':
        return DatabaseAnalyzer(file_path)
    elif file_type in ['csv', 'tsv', 'txt', 'excel', 'json']:
        return FileDataAnalyzer(file_path, file_type)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


# ============================================================================
# Demo Data Generator
# ============================================================================

def create_demo_database() -> str:
    """Create a demo Olist-style database."""
    demo_path = os.path.join(UPLOAD_FOLDER, 'demo_ecommerce.db')
    
    conn = sqlite3.connect(demo_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript('''
        DROP TABLE IF EXISTS order_items;
        DROP TABLE IF EXISTS order_payments;
        DROP TABLE IF EXISTS order_reviews;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS sellers;
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS geolocation;
        DROP TABLE IF EXISTS product_category;
        
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            customer_unique_id TEXT,
            customer_zip_code TEXT,
            customer_city TEXT,
            customer_state TEXT
        );
        
        CREATE TABLE sellers (
            seller_id TEXT PRIMARY KEY,
            seller_zip_code TEXT,
            seller_city TEXT,
            seller_state TEXT
        );
        
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            product_category_name TEXT,
            product_name_length INTEGER,
            product_description_length INTEGER,
            product_photos_qty INTEGER,
            product_weight_g REAL,
            product_length_cm REAL,
            product_height_cm REAL,
            product_width_cm REAL
        );
        
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT,
            order_status TEXT,
            order_purchase_timestamp TEXT,
            order_approved_at TEXT,
            order_delivered_carrier_date TEXT,
            order_delivered_customer_date TEXT,
            order_estimated_delivery_date TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
        
        CREATE TABLE order_items (
            order_id TEXT,
            order_item_id INTEGER,
            product_id TEXT,
            seller_id TEXT,
            shipping_limit_date TEXT,
            price REAL,
            freight_value REAL,
            PRIMARY KEY (order_id, order_item_id),
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id),
            FOREIGN KEY (seller_id) REFERENCES sellers(seller_id)
        );
        
        CREATE TABLE order_payments (
            order_id TEXT,
            payment_sequential INTEGER,
            payment_type TEXT,
            payment_installments INTEGER,
            payment_value REAL,
            PRIMARY KEY (order_id, payment_sequential),
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
        
        CREATE TABLE order_reviews (
            review_id TEXT PRIMARY KEY,
            order_id TEXT,
            review_score INTEGER,
            review_comment_title TEXT,
            review_comment_message TEXT,
            review_creation_date TEXT,
            review_answer_timestamp TEXT,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
        
        CREATE TABLE geolocation (
            geolocation_zip_code TEXT,
            geolocation_lat REAL,
            geolocation_lng REAL,
            geolocation_city TEXT,
            geolocation_state TEXT
        );
        
        CREATE TABLE product_category (
            product_category_name TEXT PRIMARY KEY,
            product_category_name_english TEXT
        );
    ''')
    
    # Insert sample data
    import random
    import string
    
    def random_id():
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=32))
    
    cities = ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Brasília', 'Salvador']
    states = ['SP', 'RJ', 'MG', 'DF', 'BA']
    categories = ['electronics', 'furniture', 'clothing', 'sports', 'books', 'toys', 'beauty']
    statuses = ['delivered', 'shipped', 'processing', 'canceled']
    payment_types = ['credit_card', 'boleto', 'voucher', 'debit_card']
    
    # Customers
    customer_ids = []
    for i in range(100):
        cid = random_id()
        customer_ids.append(cid)
        cursor.execute(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
            (cid, random_id(), f'{random.randint(10000, 99999)}', random.choice(cities), random.choice(states))
        )
    
    # Sellers
    seller_ids = []
    for i in range(30):
        sid = random_id()
        seller_ids.append(sid)
        cursor.execute(
            "INSERT INTO sellers VALUES (?, ?, ?, ?)",
            (sid, f'{random.randint(10000, 99999)}', random.choice(cities), random.choice(states))
        )
    
    # Products
    product_ids = []
    for i in range(50):
        pid = random_id()
        product_ids.append(pid)
        cursor.execute(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, random.choice(categories), random.randint(10, 50), random.randint(100, 500),
             random.randint(1, 5), random.uniform(100, 5000), random.uniform(10, 100),
             random.uniform(5, 50), random.uniform(5, 50))
        )
    
    # Orders & related
    for i in range(200):
        oid = random_id()
        cursor.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (oid, random.choice(customer_ids), random.choice(statuses),
             f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
             f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
             f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
             f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
             f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}')
        )
        
        # Order items
        for j in range(random.randint(1, 3)):
            cursor.execute(
                "INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?, ?)",
                (oid, j+1, random.choice(product_ids), random.choice(seller_ids),
                 f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
                 round(random.uniform(50, 500), 2), round(random.uniform(10, 50), 2))
            )
        
        # Payment
        cursor.execute(
            "INSERT INTO order_payments VALUES (?, ?, ?, ?, ?)",
            (oid, 1, random.choice(payment_types), random.randint(1, 12), round(random.uniform(50, 1000), 2))
        )
        
        # Review
        if random.random() > 0.3:
            cursor.execute(
                "INSERT INTO order_reviews VALUES (?, ?, ?, ?, ?, ?, ?)",
                (random_id(), oid, random.randint(1, 5), '', '', 
                 f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
                 f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}')
            )
    
    # Categories
    for cat in categories:
        cursor.execute(
            "INSERT INTO product_category VALUES (?, ?)",
            (cat, cat.title())
        )
    
    # Geolocation
    for i in range(50):
        cursor.execute(
            "INSERT INTO geolocation VALUES (?, ?, ?, ?, ?)",
            (f'{random.randint(10000, 99999)}', 
             round(random.uniform(-23.5, -22.5), 6),
             round(random.uniform(-46.5, -43.5), 6),
             random.choice(cities), random.choice(states))
        )
    
    conn.commit()
    conn.close()
    
    return demo_path


# ============================================================================
# Flask Routes
# ============================================================================

# Error handlers to always return JSON for API routes
@app.errorhandler(400)
def bad_request(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Bad request', 'details': str(e)}), 400
    return e

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return e

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(500)
def server_error(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500
    return e


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_database():
    """Handle file upload (SQLite, CSV, Excel, JSON) with content-based versioning."""
    print(f"=== UPLOAD ENDPOINT CALLED ===")
    try:
        if 'file' not in request.files:
            print("ERROR: No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get file extension
        ext = file.filename.lower().split('.')[-1]
        
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                'error': f'Invalid file type ".{ext}". Supported: SQLite (.db, .sqlite), CSV (.csv, .tsv), Excel (.xlsx, .xls), JSON (.json)'
            }), 400
        
        # Check for pandas/openpyxl for Excel files
        if ext in ['xlsx', 'xls', 'xlsm'] and not (PANDAS_AVAILABLE and EXCEL_AVAILABLE):
            return jsonify({
                'error': 'Excel files require pandas and openpyxl. Install with: pip install pandas openpyxl'
            }), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Store file path and type
        file_type = get_file_type(filename)
        session['db_path'] = filepath
        session['file_type'] = file_type
        session['original_filename'] = file.filename
        
        print(f"File saved to: {filepath}")
        print(f"Session after upload: {dict(session)}")
        
        # Generate content-based fingerprint for the dataset
        try:
            fingerprint = versioning_system.identify_dataset(filepath, file_type)
            session['dataset_id'] = fingerprint.dataset_id
            
            # Check if this dataset was already analyzed
            is_cached = versioning_system.storage.dataset_exists(fingerprint.dataset_id)
            
            return jsonify({
                'success': True, 
                'message': f'{file_type.upper()} file uploaded successfully',
                'file_type': file_type,
                'dataset_id': fingerprint.dataset_id[:16] + '...',
                'is_cached': is_cached,
                'cache_message': 'Dataset recognized - cached analysis available!' if is_cached else 'New dataset - will analyze fresh'
            })
        except Exception as fp_error:
            # If fingerprinting fails, continue without versioning
            print(f"Fingerprinting warning: {fp_error}")
            return jsonify({
                'success': True, 
                'message': f'{file_type.upper()} file uploaded successfully',
                'file_type': file_type,
                'is_cached': False
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/demo', methods=['POST'])
def use_demo():
    """Use demo database."""
    try:
        demo_path = create_demo_database()
        session['db_path'] = demo_path
        session['file_type'] = 'sqlite'
        session['original_filename'] = 'demo_ecommerce.db'
        
        # Generate fingerprint for demo database
        try:
            fingerprint = versioning_system.identify_dataset(demo_path, 'sqlite')
            session['dataset_id'] = fingerprint.dataset_id
        except:
            pass
        
        return jsonify({'success': True, 'message': 'Demo database created'})
    except Exception as e:
        return jsonify({'error': f'Demo creation failed: {str(e)}'}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze the uploaded file with automatic caching.
    
    Uses content-based fingerprinting:
    - Same dataset = reuse cached analysis (no recomputation)
    - Different dataset = fresh analysis and cache
    """
    print(f"=== ANALYZE ENDPOINT CALLED ===")
    print(f"Session data: {dict(session)}")
    
    db_path = session.get('db_path')
    file_type = session.get('file_type', 'sqlite')
    original_filename = session.get('original_filename', 'unknown')
    
    print(f"db_path: {db_path}")
    print(f"file_type: {file_type}")
    print(f"File exists: {os.path.exists(db_path) if db_path else 'No path'}")
    
    if not db_path or not os.path.exists(db_path):
        print("ERROR: No file found in session!")
        return jsonify({'error': 'No file uploaded. Please upload a file first.'}), 400
    
    try:
        # Use versioning system for idempotent processing
        dataset_id = session.get('dataset_id')
        
        # Check if we have cached results
        if dataset_id and versioning_system.storage.dataset_exists(dataset_id):
            # CACHE HIT - Reuse stored results
            cached_results = versioning_system.storage.get_analysis_results(dataset_id)
            
            if cached_results and len(cached_results) > 2:
                print(f"CACHE HIT: Found valid cached results for {dataset_id[:16]}")
                # Update access stats
                versioning_system.storage.record_access(dataset_id)
                versioning_system.storage._index['stats']['cache_hits'] += 1
                versioning_system.storage._save_index()
                
                # Sanitize for JSON
                results = sanitize_for_json(cached_results)
                session['analysis_results'] = results
                
                # Add dataset_id to top level for charts API
                results['dataset_id'] = dataset_id
                
                # Add cache info to response
                results['_cache_info'] = {
                    'is_cached': True,
                    'dataset_id': dataset_id[:16] + '...',
                    'message': 'Reused cached analysis - no recomputation needed!'
                }
                
                return jsonify(results)
            else:
                print(f"CACHE MISS: Dataset {dataset_id[:16]} exists in index but no analysis data - running fresh")
        
        # CACHE MISS - Run fresh analysis
        # Create analyzer factory for versioning system
        def analyzer_factory(path, ftype):
            return get_analyzer(path, ftype)
        
        # Process dataset with versioning
        processing_result = versioning_system.process_dataset(
            file_path=db_path,
            file_type=file_type,
            original_filename=original_filename,
            analyzer_factory=analyzer_factory,
            markdown_generator=generate_markdown
        )
        
        # Store dataset ID in session
        session['dataset_id'] = processing_result.dataset_id
        
        # Sanitize results for JSON serialization
        results = sanitize_for_json(processing_result.analysis_results)
        session['analysis_results'] = results
        
        # Add dataset_id to top level for charts API
        results['dataset_id'] = processing_result.dataset_id
        
        # Add versioning info to response
        results['_cache_info'] = {
            'is_cached': processing_result.is_cached,
            'dataset_id': processing_result.dataset_id[:16] + '...',
            'processing_time_ms': round(processing_result.processing_time_ms, 2),
            'message': processing_result.message
        }
        
        return jsonify(results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Fallback: Run analysis without versioning
        try:
            analyzer = get_analyzer(db_path, file_type)
            results = analyzer.analyze()
            analyzer.close()
            results = sanitize_for_json(results)
            session['analysis_results'] = results
            return jsonify(results)
        except Exception as inner_e:
            return jsonify({'error': str(inner_e)}), 500


@app.route('/api/export/markdown', methods=['GET'])
def export_markdown():
    """Export documentation as Markdown."""
    results = session.get('analysis_results')
    
    if not results:
        return jsonify({'error': 'No analysis results. Please analyze a database first.'}), 400
    
    markdown = generate_markdown(results)
    
    return jsonify({'markdown': markdown})


@app.route('/api/download/markdown', methods=['GET'])
def download_markdown():
    """Download documentation as Markdown file."""
    results = session.get('analysis_results')
    
    if not results:
        return jsonify({'error': 'No analysis results'}), 400
    
    markdown = generate_markdown(results)
    
    # Save to temp file
    filepath = os.path.join(UPLOAD_FOLDER, f"{results['database_name']}_documentation.md")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    return send_file(filepath, as_attachment=True, download_name=f"{results['database_name']}_documentation.md")


def generate_markdown(results: Dict) -> str:
    """Generate Markdown documentation from analysis results."""
    md = []
    
    # Header
    md.append(f"# 📊 {results['database_name']} - Database User Manual")
    md.append(f"\n*Generated by DataMind AI on {results['analysis_date'][:10]}*\n")
    
    # Executive Summary
    md.append("## 📋 Executive Summary\n")
    summary = results['summary']
    md.append(f"This documentation provides a comprehensive overview of the **{results['database_name']}** database.\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Total Tables | {summary['total_tables']} |")
    md.append(f"| Total Columns | {summary['total_columns']} |")
    md.append(f"| Total Rows | {summary['total_rows']:,} |")
    md.append(f"| Relationships | {summary['total_relationships']} |")
    md.append(f"| Quality Score | {summary['quality_score']}/100 |")
    md.append("")
    
    # Table of Contents
    md.append("## 📑 Table of Contents\n")
    for i, table in enumerate(results['tables'], 1):
        md.append(f"{i}. [{table['name']}](#{table['name'].lower().replace('_', '-')})")
    md.append("")
    
    # Entity Descriptions
    md.append("---\n## 📦 Entity Descriptions\n")
    
    for table in results['tables']:
        md.append(f"### {table['name']}\n")
        md.append(f"**Type:** {table['table_type'].title()} | **Rows:** {table['row_count']:,}\n")
        md.append(f"**Description:** {table['business_description']}\n")
        
        # Columns table
        md.append("| Column | Type | Nullable | Key | Description |")
        md.append("|--------|------|----------|-----|-------------|")
        
        for col in table['columns']:
            key = ""
            if col['is_primary_key']:
                key = "🔑 PK"
            elif col['is_foreign_key']:
                key = f"🔗 FK → {col['references']}"
            
            md.append(f"| {col['name']} | {col['data_type']} | {'Yes' if col['nullable'] else 'No'} | {key} | {col['business_description']} |")
        
        md.append("")
    
    # Relationships
    md.append("---\n## 🔗 Relationships\n")
    
    if results['relationships']:
        md.append("| From Table | Column | To Table | Column | Type |")
        md.append("|------------|--------|----------|--------|------|")
        
        for rel in results['relationships']:
            explicit = "✓" if rel['is_explicit'] else "~"
            md.append(f"| {rel['from_table']} | {rel['from_column']} | {rel['to_table']} | {rel['to_column']} | {rel['relationship_type']} {explicit} |")
    else:
        md.append("*No relationships detected.*")
    
    md.append("")
    
    # Data Quality
    md.append("---\n## 🔍 Data Quality Report\n")
    md.append(f"**Overall Quality Score: {summary['quality_score']}/100**\n")
    
    if results['quality_issues']:
        md.append("### Issues Found\n")
        md.append("| Table | Column | Issue | Severity | Recommendation |")
        md.append("|-------|--------|-------|----------|----------------|")
        
        for issue in results['quality_issues']:
            severity_icon = "🔴" if issue['severity'] == 'high' else "🟡" if issue['severity'] == 'medium' else "🟢"
            md.append(f"| {issue['table']} | {issue['column']} | {issue['description']} | {severity_icon} {issue['severity']} | {issue['recommendation']} |")
    else:
        md.append("✅ **No quality issues detected!**")
    
    md.append("")
    
    # Sample Queries
    md.append("---\n## 📝 Sample Queries\n")
    
    for query in results['sample_queries']:
        md.append(f"### {query['title']}\n")
        md.append(f"{query['description']}\n")
        md.append(f"```sql\n{query['sql']}\n```\n")
    
    # Footer
    md.append("---\n*Generated by DataMind AI - No API Key Required* 🚀")
    
    return "\n".join(md)


# ============================================================================
# Dataset Versioning API Routes
# ============================================================================

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List all stored datasets."""
    try:
        datasets = versioning_system.list_all_datasets()
        return jsonify({
            'success': True,
            'datasets': datasets,
            'count': len(datasets)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def get_dataset_info(dataset_id):
    """Get information about a specific dataset."""
    try:
        info = versioning_system.get_dataset_by_id(dataset_id)
        if not info:
            return jsonify({'error': 'Dataset not found'}), 404
        return jsonify({'success': True, **info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/<dataset_id>/analysis', methods=['GET'])
def get_dataset_analysis(dataset_id):
    """Get analysis results for a dataset by ID."""
    try:
        results = versioning_system.get_analysis(dataset_id)
        if not results:
            return jsonify({'error': 'Analysis not found'}), 404
        return jsonify(sanitize_for_json(results))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/<dataset_id>/markdown', methods=['GET'])
def get_dataset_markdown(dataset_id):
    """Get markdown documentation for a dataset by ID."""
    try:
        markdown = versioning_system.get_markdown(dataset_id)
        if not markdown:
            return jsonify({'error': 'Documentation not found'}), 404
        return jsonify({'markdown': markdown})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/<dataset_id>/charts', methods=['GET'])
def list_dataset_charts(dataset_id):
    """List all charts for a dataset."""
    try:
        charts = versioning_system.list_charts(dataset_id)
        charts_dir = versioning_system.storage.get_charts_path(dataset_id)
        return jsonify({
            'success': True,
            'charts': charts,
            'charts_path': str(charts_dir)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/<dataset_id>/charts/<chart_name>', methods=['GET'])
def get_chart(dataset_id, chart_name):
    """Get a specific chart image."""
    try:
        chart_data = versioning_system.get_chart(dataset_id, chart_name)
        if not chart_data:
            return jsonify({'error': 'Chart not found'}), 404
        
        return send_file(
            io.BytesIO(chart_data),
            mimetype='image/png',
            as_attachment=False,
            download_name=f"{chart_name}.png"
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/<dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """Delete a dataset and all its outputs."""
    try:
        success = versioning_system.delete_dataset(dataset_id)
        if not success:
            return jsonify({'error': 'Dataset not found'}), 404
        return jsonify({'success': True, 'message': 'Dataset deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/versioning/stats', methods=['GET'])
def versioning_stats():
    """Get versioning system statistics."""
    try:
        stats = versioning_system.get_stats()
        return jsonify({'success': True, **stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/<dataset_id>/verify', methods=['GET'])
def verify_dataset_integrity(dataset_id):
    """Verify integrity of a stored dataset."""
    try:
        result = versioning_system.verify_integrity(dataset_id)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Generate Charts On Demand
# ============================================================================

@app.route('/api/datasets/<dataset_id>/generate-charts', methods=['POST'])
def generate_charts_for_dataset(dataset_id):
    """Generate charts for a dataset."""
    try:
        # Get analysis results
        results = versioning_system.get_analysis(dataset_id)
        if not results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Create chart generator
        chart_gen = create_chart_generator(versioning_system.storage)
        
        # Generate all charts
        generated = chart_gen.generate_all_charts(dataset_id, results)
        
        return jsonify({
            'success': True,
            'charts_generated': generated,
            'count': len(generated)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Ensure templates directory exists
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Ensure storage directory exists
    os.makedirs(STORAGE_PATH, exist_ok=True)
    
    print("=" * 60)
    print("DataMind AI - Dataset Versioning System Active")
    print(f"Storage path: {STORAGE_PATH}")
    print("=" * 60)
    
    app.run(debug=True, port=5000)

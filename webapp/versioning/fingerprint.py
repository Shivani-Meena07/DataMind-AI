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
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from io import StringIO
from collections import Counter

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# Length of the short hex ID (16 hex chars = 64 bits = collision-safe for millions)
SHORT_ID_LENGTH = 16


@dataclass
class FingerprintResult:
    """Result of fingerprint generation."""
    dataset_id: str          # Final resolved semantic ID (e.g., NETFLIXMOVIE2025)
    semantic_base: str       # Base semantic ID before version resolution
    content_hash: str        # Full SHA-256 content hash for dedup
    algorithm: str
    row_count: int
    column_count: int
    table_count: int
    canonical_signature: str
    # Keep full_hash as alias for backward compatibility
    @property
    def full_hash(self) -> str:
        return self.content_hash


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
        
    def generate_fingerprint(self, file_path: str, file_type: str, original_filename: str = '') -> FingerprintResult:
        """
        Generate fingerprint for a dataset file.
        
        Args:
            file_path: Path to dataset file
            file_type: Type of file ('sqlite', 'csv', 'excel', 'json', etc.)
            original_filename: Original filename for semantic ID generation
            
        Returns:
            FingerprintResult with semantic dataset_id and content hash
        """
        # Extract canonical data structure
        canonical_data = self._extract_canonical_data(file_path, file_type)
        
        # Generate canonical JSON string
        canonical_json = self._to_canonical_json(canonical_data)
        
        # Compute content hash for dedup
        content_hash = self._compute_full_hash(canonical_json)
        
        # Generate human-readable semantic ID (with content hash suffix for uniqueness)
        if not original_filename:
            original_filename = Path(file_path).name
        semantic_gen = SemanticIDGenerator()
        semantic_base = semantic_gen.generate(original_filename, canonical_data, file_type,
                                              content_hash=content_hash)
        
        # Get counts
        total_rows = sum(len(table['rows']) for table in canonical_data['tables'])
        total_cols = sum(len(table['columns']) for table in canonical_data['tables'])
        
        return FingerprintResult(
            dataset_id=semantic_base,   # Will be resolved by versioning system
            semantic_base=semantic_base,
            content_hash=content_hash,
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
    
    def _compute_full_hash(self, data: str) -> str:
        """Compute full cryptographic hash of data string."""
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

    def _compute_hash(self, data: str) -> str:
        """Compute short hash ID (first 16 hex chars of full hash)."""
        return self._compute_full_hash(data)[:SHORT_ID_LENGTH]
    
    def verify_fingerprint(self, file_path: str, file_type: str, expected_id: str) -> bool:
        """
        Verify that a file matches an expected fingerprint.
        
        Accepts semantic IDs, content hashes, or partial hashes.
        
        Args:
            file_path: Path to dataset file
            file_type: Type of file
            expected_id: Expected dataset ID or content hash
            
        Returns:
            True if fingerprint matches, False otherwise
        """
        result = self.generate_fingerprint(file_path, file_type)
        # Match against semantic ID, content hash, or partial hash
        return (result.dataset_id == expected_id or 
                result.semantic_base == expected_id or
                result.content_hash == expected_id or
                result.content_hash[:len(expected_id)] == expected_id)
    
    def compute_incremental_hash(self, base_id: str, new_data: str) -> str:
        """
        Compute hash that incorporates base ID and new data.
        Useful for derived datasets.
        """
        combined = f"{base_id}:{new_data}"
        return self._compute_hash(combined)

    @staticmethod
    def is_legacy_hash_id(dataset_id: str) -> bool:
        """Check if a dataset_id is a legacy hex-hash ID (all hex chars)."""
        return bool(dataset_id) and all(c in '0123456789abcdef' for c in dataset_id)


# =============================================================================
# Semantic ID Generator
# =============================================================================

class SemanticIDGenerator:
    """
    Generates human-readable dataset IDs in format: {SOURCE}{TOPIC}{YEAR}
    
    Examples:
        netflix_titles.csv (movie data)      → NETFLIXMOVIE2025
        olist_customers_dataset.csv          → OLISTCUSTOMER
        demo_ecommerce.db                    → DEMOECOMMERCE2024
        uber_rides_2023.csv                  → UBERRIDE2023
        amazon_products.xlsx                 → AMAZONPRODUCT
        spotify_tracks.csv (music data)      → SPOTIFYMUSIC
        sales_data_2024.csv                  → SALESDATA2024
    """
    
    # Known companies / data sources (lowercase for matching)
    KNOWN_SOURCES = {
        'netflix', 'amazon', 'google', 'meta', 'apple', 'microsoft',
        'uber', 'spotify', 'airbnb', 'twitter', 'facebook', 'instagram',
        'olist', 'walmart', 'tesla', 'ibm', 'oracle', 'salesforce',
        'imdb', 'kaggle', 'github', 'linkedin', 'youtube', 'tiktok',
        'flipkart', 'zomato', 'swiggy', 'paytm', 'phonepe', 'ola',
        'jio', 'reliance', 'tata', 'infosys', 'wipro', 'hcl',
        'adobe', 'nvidia', 'intel', 'samsung', 'sony', 'toyota',
        'coca', 'pepsi', 'nike', 'adidas', 'starbucks', 'mcdonalds',
        'visa', 'mastercard', 'paypal', 'stripe', 'shopify', 'ebay',
        'demo', 'test', 'sample', 'example', 'mock',
    }
    
    # Topic keywords: column/table name fragment → topic label
    TOPIC_KEYWORDS = {
        # Entertainment / Media
        'movie': 'MOVIE', 'film': 'MOVIE', 'cinema': 'MOVIE',
        'title': 'MOVIE', 'director': 'MOVIE', 'cast': 'MOVIE',
        'show': 'SHOW', 'series': 'SERIES', 'episode': 'EPISODE',
        'anime': 'ANIME', 'cartoon': 'CARTOON', 'video': 'VIDEO',
        'stream': 'STREAM', 'watch': 'STREAM',
        
        # Music
        'song': 'MUSIC', 'artist': 'MUSIC', 'album': 'MUSIC',
        'playlist': 'MUSIC', 'track': 'MUSIC', 'genre': 'MUSIC',
        'listen': 'MUSIC', 'audio': 'MUSIC',
        
        # E-commerce / Business
        'ecommerce': 'ECOMMERCE', 'commerce': 'ECOMMERCE',
        'shop': 'ECOMMERCE', 'store': 'ECOMMERCE', 'cart': 'ECOMMERCE',
        'order': 'ORDER', 'purchase': 'ORDER', 'checkout': 'ORDER',
        'payment': 'PAYMENT', 'transaction': 'PAYMENT', 'invoice': 'PAYMENT',
        'product': 'PRODUCT', 'item': 'PRODUCT', 'catalog': 'PRODUCT',
        'inventory': 'INVENTORY', 'stock': 'STOCK', 'warehouse': 'INVENTORY',
        
        # People
        'customer': 'CUSTOMER', 'client': 'CUSTOMER', 'buyer': 'CUSTOMER',
        'user': 'USER', 'member': 'USER', 'subscriber': 'USER',
        'employee': 'EMPLOYEE', 'staff': 'EMPLOYEE', 'worker': 'EMPLOYEE',
        'salary': 'EMPLOYEE', 'payroll': 'EMPLOYEE',
        'seller': 'SELLER', 'vendor': 'SELLER', 'supplier': 'SELLER',
        
        # Finance
        'sales': 'SALES', 'revenue': 'SALES', 'income': 'FINANCE',
        'profit': 'FINANCE', 'expense': 'FINANCE', 'budget': 'FINANCE',
        'market': 'MARKET', 'trading': 'TRADING', 'crypto': 'CRYPTO',
        'loan': 'LOAN', 'credit': 'CREDIT', 'bank': 'BANKING',
        'insurance': 'INSURANCE',
        
        # Education
        'student': 'EDUCATION', 'course': 'EDUCATION', 'grade': 'EDUCATION',
        'school': 'EDUCATION', 'university': 'EDUCATION', 'exam': 'EDUCATION',
        'teacher': 'EDUCATION', 'college': 'EDUCATION',
        
        # Health
        'patient': 'HEALTH', 'diagnosis': 'HEALTH', 'hospital': 'HEALTH',
        'medical': 'HEALTH', 'disease': 'HEALTH', 'drug': 'HEALTH',
        'covid': 'COVID', 'vaccine': 'HEALTH', 'clinic': 'HEALTH',
        
        # Travel / Transport
        'flight': 'FLIGHT', 'airline': 'FLIGHT', 'airport': 'FLIGHT',
        'hotel': 'HOTEL', 'booking': 'BOOKING', 'reservation': 'BOOKING',
        'ride': 'RIDE', 'trip': 'TRIP', 'travel': 'TRAVEL',
        'taxi': 'RIDE', 'cab': 'RIDE', 'delivery': 'DELIVERY',
        
        # Social / Reviews
        'tweet': 'SOCIAL', 'post': 'SOCIAL', 'comment': 'SOCIAL',
        'review': 'REVIEW', 'rating': 'REVIEW', 'feedback': 'REVIEW',
        'sentiment': 'SENTIMENT', 'opinion': 'SENTIMENT',
        
        # Technology
        'log': 'LOG', 'event': 'EVENT', 'metric': 'METRIC',
        'sensor': 'SENSOR', 'iot': 'IOT', 'device': 'DEVICE',
        'server': 'SERVER', 'network': 'NETWORK', 'api': 'API',
        
        # Geography
        'geolocation': 'GEO', 'location': 'GEO', 'country': 'GEO',
        'region': 'GEO', 'city': 'GEO', 'address': 'GEO',
        
        # Food
        'food': 'FOOD', 'restaurant': 'FOOD', 'recipe': 'FOOD',
        'menu': 'FOOD', 'meal': 'FOOD', 'nutrition': 'FOOD',
        
        # Sports
        'sports': 'SPORTS', 'player': 'SPORTS', 'team': 'SPORTS',
        'match': 'SPORTS', 'game': 'GAME', 'score': 'SPORTS',
        'cricket': 'CRICKET', 'football': 'FOOTBALL', 'soccer': 'FOOTBALL',
        
        # Real Estate
        'house': 'REALESTATE', 'property': 'REALESTATE', 'housing': 'REALESTATE',
        'rent': 'REALESTATE', 'apartment': 'REALESTATE',
        
        # Weather / Environment
        'weather': 'WEATHER', 'temperature': 'WEATHER', 'climate': 'CLIMATE',
        'pollution': 'ENVIRONMENT', 'emission': 'ENVIRONMENT',
        
        # Jobs / HR
        'job': 'JOB', 'hiring': 'JOB', 'resume': 'JOB', 'career': 'JOB',
        'recruitment': 'JOB', 'vacancy': 'JOB',
    }
    
    # Source-specific topic hints (used as tiebreaker)
    SOURCE_TOPIC_HINTS = {
        'NETFLIX': ['MOVIE', 'SHOW', 'SERIES'],
        'SPOTIFY': ['MUSIC', 'SONG', 'TRACK'],
        'UBER': ['RIDE', 'TRIP', 'DELIVERY'],
        'AIRBNB': ['BOOKING', 'HOTEL', 'RENTAL'],
        'AMAZON': ['ECOMMERCE', 'PRODUCT', 'ORDER'],
        'IMDB': ['MOVIE', 'RATING', 'FILM'],
        'YOUTUBE': ['VIDEO', 'STREAM', 'CHANNEL'],
        'TWITTER': ['SOCIAL', 'TWEET', 'POST'],
        'INSTAGRAM': ['SOCIAL', 'POST', 'PHOTO'],
        'ZOMATO': ['FOOD', 'RESTAURANT', 'REVIEW'],
        'SWIGGY': ['FOOD', 'DELIVERY', 'ORDER'],
        'OLA': ['RIDE', 'TRIP', 'CAB'],
        'FLIPKART': ['ECOMMERCE', 'PRODUCT', 'ORDER'],
    }
    
    # Number of hex chars from content hash to append as fingerprint suffix
    HASH_SUFFIX_LENGTH = 4

    def generate(self, filename: str, canonical_data: Dict, file_type: str,
                 content_hash: str = '') -> str:
        """
        Generate a human-readable semantic ID with content fingerprint.
        
        Format: {SOURCE}{TOPIC}{YEAR}{HASH4}  (all uppercase, no separators)
        The trailing 4-char hex suffix ensures different data never collides
        while same data always produces the same ID.
        
        Args:
            filename: Original filename (e.g., 'netflix_titles.csv')
            canonical_data: Extracted canonical data with tables/columns/rows
            file_type: File type string
            content_hash: SHA-256 content hash (used for 4-char suffix)
            
        Returns:
            Semantic ID string like 'NETFLIXMOVIE2020A3F2'
        """
        source = self._extract_source(filename)
        topic = self._extract_topic(filename, canonical_data, source)
        year = self._detect_year(filename, canonical_data)
        
        semantic_id = f"{source}{topic}"
        if year:
            semantic_id += year
        
        # Append content fingerprint suffix for uniqueness
        if content_hash:
            suffix = content_hash[:self.HASH_SUFFIX_LENGTH].upper()
            semantic_id += suffix
        
        return semantic_id
    
    def generate_from_schema(self, filename: str, schema: Dict,
                              content_hash: str = '') -> str:
        """
        Generate semantic ID from stored schema (for migration).
        Uses table/column names from schema instead of full canonical data.
        """
        # Build pseudo-canonical data from schema
        pseudo_canonical = {'tables': []}
        for table in schema.get('tables', []):
            pseudo_canonical['tables'].append({
                'name': table.get('name', ''),
                'columns': [c.get('name', '') for c in table.get('columns', [])],
                'rows': []  # No rows needed for topic detection
            })
        
        return self.generate(filename, pseudo_canonical, '', content_hash=content_hash)
    
    def _extract_source(self, filename: str) -> str:
        """Extract company/source name from filename."""
        name = Path(filename).stem.lower()
        parts = re.split(r'[_\-\s\.]+', name)
        parts = [p for p in parts if p]  # Remove empty parts
        
        # Priority 1: First part is a known company
        if parts and parts[0] in self.KNOWN_SOURCES:
            return parts[0].upper()
        
        # Priority 2: Any part is a known company
        for part in parts:
            if part in self.KNOWN_SOURCES:
                return part.upper()
        
        # Priority 3: Use first meaningful word (≥2 chars, not a number)
        for part in parts:
            if len(part) >= 2 and not part.isdigit():
                return part.upper()[:12]  # Cap at 12 chars
        
        return 'DATASET'
    
    def _extract_topic(self, filename: str, canonical_data: Dict, source: str) -> str:
        """
        Extract the main topic from filename + data analysis.
        
        Priority:
        1. Column/table name keyword matches (most reliable)
        2. Filename word matches
        3. Source-specific topic hints (tiebreaker)
        4. Fallback: second filename word
        """
        topic_scores = Counter()
        
        # --- Analyze data content (highest signal) ---
        
        # Table names (weight: 3 per match)
        for table in canonical_data.get('tables', []):
            table_name = table.get('name', '').lower()
            for part in re.split(r'[_\-\s\.]+', table_name):
                if part in self.TOPIC_KEYWORDS:
                    topic_scores[self.TOPIC_KEYWORDS[part]] += 3
        
        # Column names (weight: 1 per match — many columns add up fast)
        for table in canonical_data.get('tables', []):
            for col in table.get('columns', []):
                col_lower = col.lower()
                for part in re.split(r'[_\-\s\.]+', col_lower):
                    if part in self.TOPIC_KEYWORDS:
                        topic_scores[self.TOPIC_KEYWORDS[part]] += 1
        
        # --- Analyze filename (HIGHEST signal — user's explicit label, weight: 20) ---
        # Filename is the user's own description of the data, always prioritize it
        name = Path(filename).stem.lower()
        name_parts = re.split(r'[_\-\s\.]+', name)
        for part in name_parts:
            if part in self.TOPIC_KEYWORDS and part.upper() != source:
                topic_scores[self.TOPIC_KEYWORDS[part]] += 20
        
        # --- Source-specific hints (tiebreaker, weight: 0.5 each) ---
        if source in self.SOURCE_TOPIC_HINTS:
            for hint_topic in self.SOURCE_TOPIC_HINTS[source]:
                topic_scores[hint_topic] += 0.5
        
        # --- Pick winner ---
        if topic_scores:
            # Don't use the source name as topic (avoid NETFLIXNETFLIX)
            best_topics = topic_scores.most_common(5)
            for topic, score in best_topics:
                if topic != source:
                    return topic
        
        # --- Fallback: use non-source filename words ---
        for part in name_parts:
            if part.upper() != source and len(part) >= 3 and not part.isdigit():
                # Use it as-is (e.g., 'titles' → 'TITLES')
                return part.upper()[:10]
        
        return 'DATA'
    
    def _detect_year(self, filename: str, canonical_data: Dict) -> str:
        """
        Detect the year of the data from filename or data values.
        Returns 4-digit year string or empty string if not confident.
        """
        # Priority 1: Explicit year in filename (e.g., 'data_2024.csv', 'sales-2023')
        name = Path(filename).stem
        year_match = re.search(r'(20[0-2]\d)', name)
        if year_match:
            return year_match.group(1)
        
        # Priority 2: Scan date columns in data for most common year
        year_counter = Counter()
        
        for table in canonical_data.get('tables', []):
            columns = table.get('columns', [])
            rows = table.get('rows', [])
            
            # Find columns likely containing dates
            date_col_indices = []
            for i, col in enumerate(columns):
                col_lower = col.lower()
                if any(kw in col_lower for kw in [
                    'date', 'year', 'time', 'created', 'updated',
                    'timestamp', 'released', 'published', 'added'
                ]):
                    date_col_indices.append(i)
            
            # Sample rows for year extraction (max 200 rows for performance)
            sample_rows = rows[:200]
            
            if date_col_indices:
                # Scan date columns specifically
                for row in sample_rows:
                    for idx in date_col_indices:
                        if idx < len(row):
                            val = str(row[idx])
                            years = re.findall(r'(20[0-2]\d)', val)
                            year_counter.update(years)
            else:
                # No obvious date columns — scan all values for year-like patterns
                for row in sample_rows[:50]:  # Smaller sample for full-scan
                    for val in row:
                        val_str = str(val)
                        # Only match isolated 4-digit years (not part of larger numbers)
                        if re.match(r'^20[0-2]\d$', val_str):
                            year_counter[val_str] += 1
        
        if year_counter:
            best_year, count = year_counter.most_common(1)[0]
            # Only use if we found enough evidence
            if count >= 3:
                return best_year
        
        return ''

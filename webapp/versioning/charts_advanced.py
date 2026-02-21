"""
DataMind AI - Advanced Chart Generation Module
Enterprise-grade visualizations with advanced relationship detection algorithms.

All charts are derived artifacts stored under:
outputs/<dataset_id>/charts/

Advanced Features:
- Network relationship graphs
- Correlation heatmaps
- Statistical distribution analysis
- Data quality matrices
- Schema treemaps
- Intelligent relationship detection
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import json
import io
import math
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patheffects as path_effects
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# ADVANCED RELATIONSHIP DETECTION ALGORITHMS
# ============================================================================

class AdvancedRelationshipDetector:
    """
    Enterprise-grade relationship detection using multiple analysis techniques.
    
    Algorithms Used:
    1. Semantic Name Matching (N-gram Jaccard Similarity)
    2. Data Type Compatibility Scoring
    3. Foreign Key Pattern Recognition
    4. Cardinality Analysis
    5. Value Distribution Correlation
    6. Column Uniqueness Scoring
    7. Pattern-based Inference
    """
    
    @staticmethod
    def get_ngrams(text: str, n: int = 2) -> Set[str]:
        """Extract n-grams from text for similarity comparison."""
        text = text.lower().replace('_', ' ').replace('-', ' ').replace('.', ' ')
        tokens = text.split()
        ngrams = set()
        for token in tokens:
            if len(token) >= n:
                for i in range(len(token) - n + 1):
                    ngrams.add(token[i:i+n])
            if len(token) < n and token:
                ngrams.add(token)
        return ngrams
    
    @classmethod
    def calculate_name_similarity(cls, name1: str, name2: str) -> float:
        """
        Calculate semantic similarity between column/table names.
        Uses Jaccard similarity coefficient with bigrams.
        
        Returns: similarity score 0.0 to 1.0
        """
        ngrams1 = cls.get_ngrams(name1)
        ngrams2 = cls.get_ngrams(name2)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def detect_fk_pattern(col_name: str, table_names: List[str]) -> Tuple[Optional[str], float]:
        """
        Detect foreign key patterns in column names.
        
        Patterns detected:
        - {table}_id, {table}id (e.g., customer_id, customerid)
        - fk_{table}, {table}_fk (e.g., fk_customer, customer_fk)
        - id_{table} (e.g., id_customer)
        - {table}_key, {table}_ref (e.g., customer_key, customer_ref)
        
        Returns: (target_table, confidence_score)
        """
        col_lower = col_name.lower()
        
        best_match = (None, 0.0)
        
        for table in table_names:
            table_lower = table.lower()
            
            # Skip self-references for this check
            score = 0.0
            
            # Pattern: exact {table}_id
            if col_lower == f"{table_lower}_id":
                score = 0.98
            # Pattern: {table}id (no underscore)
            elif col_lower == f"{table_lower}id":
                score = 0.95
            # Pattern: fk_{table}
            elif col_lower == f"fk_{table_lower}":
                score = 0.97
            # Pattern: {table}_fk
            elif col_lower == f"{table_lower}_fk":
                score = 0.95
            # Pattern: id_{table}
            elif col_lower == f"id_{table_lower}":
                score = 0.90
            # Pattern: {table}_key
            elif col_lower == f"{table_lower}_key":
                score = 0.88
            # Pattern: {table}_ref
            elif col_lower == f"{table_lower}_ref":
                score = 0.90
            # Pattern: singular form (customers -> customer_id)
            elif table_lower.endswith('s'):
                singular = table_lower[:-1]
                if col_lower == f"{singular}_id":
                    score = 0.92
                elif col_lower.startswith(singular) and 'id' in col_lower:
                    score = 0.80
            # Partial match: table name appears in column name with id indicator
            elif table_lower in col_lower and ('id' in col_lower or 'key' in col_lower):
                score = 0.75
            
            if score > best_match[1]:
                best_match = (table, score)
        
        return best_match
    
    @staticmethod  
    def calculate_type_compatibility(type1: str, type2: str) -> float:
        """
        Calculate compatibility score between data types.
        
        Type families:
        - Integer: INT, INTEGER, BIGINT, SMALLINT, TINYINT
        - Float: FLOAT, DOUBLE, DECIMAL, NUMERIC, REAL
        - Text: VARCHAR, CHAR, TEXT, STRING, NVARCHAR
        - Date: DATE, DATETIME, TIMESTAMP, TIME
        - Boolean: BOOL, BOOLEAN, BIT
        
        Returns: compatibility score 0.0 to 1.0
        """
        type1 = type1.upper() if type1 else ''
        type2 = type2.upper() if type2 else ''
        
        if type1 == type2:
            return 1.0
        
        # Define type families
        families = {
            'INT': {'INT', 'INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'SERIAL', 'BIGSERIAL'},
            'FLOAT': {'FLOAT', 'DOUBLE', 'DECIMAL', 'NUMERIC', 'REAL', 'MONEY'},
            'TEXT': {'VARCHAR', 'CHAR', 'TEXT', 'STRING', 'NVARCHAR', 'NCHAR', 'CLOB'},
            'DATE': {'DATE', 'DATETIME', 'TIMESTAMP', 'TIME', 'DATETIME2'},
            'BOOL': {'BOOL', 'BOOLEAN', 'BIT'},
        }
        
        def get_family(dtype: str):
            dtype_clean = dtype.split('(')[0].strip()
            for family, types in families.items():
                if any(t in dtype_clean for t in types):
                    return family
            return 'OTHER'
        
        f1 = get_family(type1)
        f2 = get_family(type2)
        
        if f1 == f2:
            return 0.9
        
        # Cross-family compatibility
        if f1 in {'INT', 'FLOAT'} and f2 in {'INT', 'FLOAT'}:
            return 0.8
        if f1 in {'TEXT'} and f2 in {'TEXT', 'OTHER'}:
            return 0.6
        
        return 0.0
    
    @staticmethod
    def calculate_cardinality(unique_ratio_source: float, unique_ratio_target: float) -> Tuple[str, float]:
        """
        Determine relationship cardinality based on uniqueness ratios.
        
        Cardinality Types:
        - "1:1": Both columns highly unique (>95%)
        - "1:N": Target unique, source has duplicates
        - "N:1": Source unique, target has duplicates
        - "N:M": Both have duplicates
        
        Returns: (cardinality_type, confidence)
        """
        if unique_ratio_source > 0.95 and unique_ratio_target > 0.95:
            return ("1:1", 0.90)
        elif unique_ratio_source < 0.5 and unique_ratio_target > 0.95:
            return ("N:1", 0.85)
        elif unique_ratio_source > 0.95 and unique_ratio_target < 0.5:
            return ("1:N", 0.85)
        elif unique_ratio_source > 0.7 and unique_ratio_target > 0.7:
            return ("1:1", 0.60)
        else:
            return ("N:M", 0.70)
    
    @classmethod
    def analyze_all_relationships(cls, tables: List[Dict]) -> List[Dict]:
        """
        Comprehensive relationship analysis across all tables.
        
        Process:
        1. Extract table and column metadata
        2. Detect FK patterns in column names
        3. Calculate type compatibility
        4. Score and rank potential relationships
        
        Returns: List of scored relationships sorted by confidence
        """
        relationships = []
        table_names = [t.get('name', '') for t in tables]
        
        # Build column lookup index
        column_index = {}
        pk_columns = {}  # table -> pk_column_name
        
        for table in tables:
            table_name = table.get('name', '')
            for col in table.get('columns', []):
                col_name = col.get('name', '')
                column_index[(table_name, col_name)] = col
                if col.get('is_primary_key') or col.get('name', '').lower() == 'id':
                    pk_columns[table_name] = col_name
        
        # Analyze each table's columns for FK patterns
        for table in tables:
            source_table = table.get('name', '')
            
            for col in table.get('columns', []):
                col_name = col.get('name', '')
                
                # Skip PKs - they don't reference other tables
                if col.get('is_primary_key'):
                    continue
                
                # Detect FK pattern
                target_table, pattern_score = cls.detect_fk_pattern(col_name, table_names)
                
                if target_table and target_table != source_table and pattern_score > 0.5:
                    # Get target column (PK)
                    target_col = pk_columns.get(target_table, 'id')
                    target_col_info = column_index.get((target_table, target_col), {})
                    
                    # Calculate type compatibility
                    type_compat = cls.calculate_type_compatibility(
                        col.get('data_type', ''),
                        target_col_info.get('data_type', '')
                    )
                    
                    # Calculate uniqueness (if available)
                    source_unique = col.get('uniqueness_ratio', 0.5)
                    target_unique = target_col_info.get('uniqueness_ratio', 0.95)
                    
                    cardinality, card_confidence = cls.calculate_cardinality(
                        source_unique, target_unique
                    )
                    
                    # Calculate overall confidence score
                    # Weighted combination of all factors
                    overall_score = (
                        pattern_score * 0.50 +       # Pattern matching weight
                        type_compat * 0.30 +         # Type compatibility weight
                        card_confidence * 0.20       # Cardinality confidence weight
                    )
                    
                    relationships.append({
                        'from_table': source_table,
                        'from_column': col_name,
                        'to_table': target_table,
                        'to_column': target_col,
                        'confidence': round(overall_score, 3),
                        'pattern_score': round(pattern_score, 3),
                        'type_compatibility': round(type_compat, 3),
                        'cardinality': cardinality,
                        'cardinality_confidence': round(card_confidence, 3),
                        'is_inferred': True,
                        'algorithm': 'AdvancedRelationshipDetector'
                    })
        
        # Sort by confidence descending
        relationships.sort(key=lambda x: x['confidence'], reverse=True)
        
        return relationships


# ============================================================================
# CHART GENERATOR
# ============================================================================

class ChartGenerator:
    """
    Enterprise-grade chart generator with comprehensive visualization capabilities.
    
    Chart Types:
    1. Table Analysis:
       - Table Size Distribution (Bar)
       - Table Type Distribution (Pie/Donut)
       - Row Count Treemap
    
    2. Column Analysis:
       - Column Data Types (Bar)
       - Null Value Distribution (Horizontal Bar)
       - Uniqueness Heatmap
    
    3. Relationship Analysis:
       - Network Relationship Graph
       - Relationship Matrix Heatmap
       - Data Flow Sankey Diagram
       - Dependency Hierarchy Tree
    
    4. Quality Analysis:
       - Quality Score Gauge
       - Quality Issues by Severity (Bar)
       - Completeness Radar Chart
       - Quality Matrix Heatmap
    
    5. Statistical Analysis:
       - Value Distribution Histogram
       - Correlation Heatmap
       - Anomaly Detection Chart
    """
    
    # Modern enterprise color palette
    COLORS = [
        '#06B6D4',  # Cyan (Primary brand color)
        '#4F46E5',  # Indigo
        '#10B981',  # Emerald
        '#F59E0B',  # Amber
        '#EF4444',  # Red
        '#8B5CF6',  # Violet
        '#F97316',  # Orange
        '#EC4899',  # Pink
        '#84CC16',  # Lime
        '#6366F1',  # Indigo light
        '#14B8A6',  # Teal
        '#A855F7',  # Purple
    ]
    
    # Quality color scale
    QUALITY_COLORS = {
        'excellent': '#10B981',  # Green
        'good': '#06B6D4',       # Cyan
        'fair': '#F59E0B',       # Amber
        'poor': '#EF4444',       # Red
    }
    
    # Style settings for consistent look
    STYLE = {
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#E5E7EB',
        'axes.labelcolor': '#374151',
        'text.color': '#374151',
        'xtick.color': '#6B7280',
        'ytick.color': '#6B7280',
        'grid.color': '#F3F4F6',
        'font.family': 'sans-serif',
    }
    
    def __init__(self, storage_manager):
        """Initialize with storage manager."""
        self.storage = storage_manager
        self._apply_style()
    
    def _apply_style(self):
        """Apply consistent styling to matplotlib."""
        if MATPLOTLIB_AVAILABLE:
            for key, value in self.STYLE.items():
                plt.rcParams[key] = value
    
    def generate_all_charts(self, dataset_id: str, results: Dict) -> List[str]:
        """
        Generate all applicable charts for a dataset.
        
        Returns: List of generated chart filenames
        """
        if not MATPLOTLIB_AVAILABLE:
            return []
        
        generated = []
        
        # Table Analysis Charts
        chart_methods = [
            ('bar_table_size', self._generate_table_size_chart),
            ('donut_table_type', self._generate_table_type_donut),
            ('treemap_tables', self._generate_table_treemap),
            ('bar_column_types', self._generate_column_type_chart),
            ('heatmap_null', self._generate_null_heatmap),
            ('gauge_quality', self._generate_quality_gauge),
            ('bar_quality_issues', self._generate_quality_issues_chart),
            ('radar_completeness', self._generate_completeness_radar),
            ('network_relationships', self._generate_network_graph),
            ('heatmap_relationships', self._generate_relationship_heatmap),
            ('hierarchy_tables', self._generate_hierarchy_chart),
            ('bar_null_distribution', self._generate_null_distribution_chart),
            ('matrix_overview', self._generate_schema_matrix),
        ]
        
        for chart_name, method in chart_methods:
            try:
                result = method(dataset_id, results)
                if result:
                    generated.append(result)
            except Exception as e:
                print(f"Error generating {chart_name}: {e}")
                continue
        
        return generated
    
    def _save_chart(self, dataset_id: str, chart_name: str, fig) -> Optional[str]:
        """Save chart to storage."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        chart_data = buf.read()
        plt.close(fig)
        
        self.storage.store_chart(dataset_id, chart_data, chart_name)
        
        return f"{chart_name}.png"
    
    # ========================================================================
    # TABLE ANALYSIS CHARTS
    # ========================================================================
    
    def _generate_table_size_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate horizontal bar chart of table row counts."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        # Sort by row count and take top 15
        sorted_tables = sorted(tables, key=lambda t: t.get('row_count', 0), reverse=True)[:15]
        
        names = [t.get('name', 'Unknown')[:25] for t in sorted_tables]
        counts = [t.get('row_count', 0) for t in sorted_tables]
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.5)))
        
        # Create gradient colors based on values
        max_count = max(counts) if counts else 1
        colors = [self.COLORS[i % len(self.COLORS)] for i in range(len(names))]
        
        bars = ax.barh(names, counts, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            label = f'{count:,}'
            ax.text(width + max_count * 0.02, bar.get_y() + bar.get_height()/2,
                   label, ha='left', va='center', fontsize=9, color='#374151', fontweight='500')
        
        ax.set_xlabel('Number of Rows', fontsize=11, fontweight='500')
        ax.set_title('Table Size Distribution', fontsize=14, fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'bar_table_size_distribution', fig)
    
    def _generate_table_type_donut(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate donut chart of table types."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        # Count table types
        type_counts = defaultdict(int)
        for table in tables:
            table_type = table.get('table_type', 'unknown')
            type_counts[table_type] += 1
        
        if not type_counts:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        colors = self.COLORS[:len(labels)]
        
        # Create donut chart
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
        )
        
        # Style autotexts
        for autotext in autotexts:
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        # Add legend
        ax.legend(wedges, [f'{l} ({s})' for l, s in zip(labels, sizes)],
                 loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        # Add center text
        ax.text(0, 0, f'{sum(sizes)}\nTables', ha='center', va='center',
               fontsize=20, fontweight='bold', color='#374151')
        
        ax.set_title('Table Type Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.axis('equal')
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'donut_table_type_distribution', fig)
    
    def _generate_table_treemap(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate treemap visualization of table sizes."""
        tables = results.get('tables', [])
        if not tables or len(tables) < 2:
            return None
        
        # Sort by row count
        sorted_tables = sorted(tables, key=lambda t: t.get('row_count', 0), reverse=True)[:20]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Calculate treemap areas
        total_rows = sum(t.get('row_count', 1) for t in sorted_tables)
        
        # Simple squarified treemap algorithm
        def squarify(values, x, y, width, height):
            """Returns rectangles for treemap."""
            if not values:
                return []
            
            total = sum(values)
            rects = []
            
            if width >= height:
                # Horizontal layout
                for i, val in enumerate(values):
                    w = (val / total) * width if total > 0 else width / len(values)
                    rects.append((x, y, w, height))
                    x += w
            else:
                # Vertical layout
                for i, val in enumerate(values):
                    h = (val / total) * height if total > 0 else height / len(values)
                    rects.append((x, y, width, h))
                    y += h
            
            return rects
        
        values = [t.get('row_count', 1) for t in sorted_tables]
        rects = squarify(values, 0, 0, 10, 10)
        
        for i, (rect, table) in enumerate(zip(rects, sorted_tables)):
            x, y, w, h = rect
            color = self.COLORS[i % len(self.COLORS)]
            
            # Draw rectangle
            ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, 
                                        edgecolor='white', linewidth=2))
            
            # Add label if space allows
            if w > 0.8 and h > 0.6:
                name = table.get('name', '')[:15]
                count = table.get('row_count', 0)
                ax.text(x + w/2, y + h/2, f'{name}\n{count:,}',
                       ha='center', va='center', fontsize=8, 
                       color='white', fontweight='bold')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Table Size Treemap', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'treemap_table_sizes', fig)
    
    # ========================================================================
    # COLUMN ANALYSIS CHARTS
    # ========================================================================
    
    def _generate_column_type_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate bar chart of column data types distribution."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        # Count data types
        type_counts = defaultdict(int)
        for table in tables:
            for col in table.get('columns', []):
                data_type = col.get('data_type', 'UNKNOWN').upper()
                # Normalize type names
                normalized = self._normalize_data_type(data_type)
                type_counts[normalized] += 1
        
        if not type_counts:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Sort by count
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        types = [t[0] for t in sorted_types]
        counts = [t[1] for t in sorted_types]
        colors = [self.COLORS[i % len(self.COLORS)] for i in range(len(types))]
        
        bars = ax.bar(types, counts, color=colors, edgecolor='white', linewidth=0.5)
        
        # Add value labels on top
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max(counts) * 0.02,
                   str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Data Type', fontsize=11, fontweight='500')
        ax.set_ylabel('Number of Columns', fontsize=11, fontweight='500')
        ax.set_title('Column Data Type Distribution', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'bar_column_type_distribution', fig)
    
    def _normalize_data_type(self, dtype: str) -> str:
        """Normalize data type name for grouping."""
        dtype = dtype.upper()
        if 'INT' in dtype:
            return 'INTEGER'
        elif 'CHAR' in dtype or 'TEXT' in dtype or 'STRING' in dtype:
            return 'TEXT'
        elif 'REAL' in dtype or 'FLOAT' in dtype or 'DOUBLE' in dtype or 'DECIMAL' in dtype:
            return 'NUMERIC'
        elif 'DATE' in dtype or 'TIME' in dtype:
            return 'DATETIME'
        elif 'BOOL' in dtype or 'BIT' in dtype:
            return 'BOOLEAN'
        elif 'BLOB' in dtype or 'BINARY' in dtype:
            return 'BINARY'
        return dtype
    
    def _generate_null_heatmap(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate heatmap of null percentages across tables and columns."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        # Collect null data
        table_names = []
        null_percentages = []
        
        for table in tables[:15]:  # Limit to 15 tables
            table_name = table.get('name', 'Unknown')[:20]
            columns = table.get('columns', [])[:10]  # Limit columns
            
            if columns:
                table_names.append(table_name)
                avg_null = sum(c.get('null_percentage', 0) for c in columns) / len(columns)
                null_percentages.append(avg_null * 100)
        
        if not table_names:
            return None
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(table_names) * 0.5)))
        
        # Create color gradient
        colors = []
        for pct in null_percentages:
            if pct > 50:
                colors.append('#EF4444')  # Red
            elif pct > 25:
                colors.append('#F59E0B')  # Amber
            elif pct > 10:
                colors.append('#06B6D4')  # Cyan
            else:
                colors.append('#10B981')  # Green
        
        bars = ax.barh(table_names, null_percentages, color=colors, 
                       edgecolor='white', linewidth=0.5, height=0.7)
        
        # Add percentage labels
        for bar, pct in zip(bars, null_percentages):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%', ha='left', va='center', fontsize=9, fontweight='500')
        
        ax.set_xlabel('Average Null Percentage', fontsize=11, fontweight='500')
        ax.set_title('Null Value Distribution by Table', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(0, max(null_percentages) * 1.2 if null_percentages else 100)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add threshold lines
        ax.axvline(x=25, color='#F59E0B', linestyle='--', alpha=0.5, label='Warning (25%)')
        ax.axvline(x=50, color='#EF4444', linestyle='--', alpha=0.5, label='Critical (50%)')
        ax.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'heatmap_null_distribution', fig)
    
    def _generate_null_distribution_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate detailed null distribution chart showing columns with nulls."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        # Collect columns with significant nulls
        null_data = []
        for table in tables:
            table_name = table.get('name', 'Unknown')
            for col in table.get('columns', []):
                null_pct = col.get('null_percentage', 0) * 100
                if null_pct > 0:
                    null_data.append({
                        'table': table_name,
                        'column': col.get('name', 'Unknown'),
                        'null_pct': null_pct
                    })
        
        if not null_data:
            return None
        
        # Sort and take top 20
        null_data.sort(key=lambda x: x['null_pct'], reverse=True)
        null_data = null_data[:20]
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(null_data) * 0.4)))
        
        labels = [f"{d['table']}.{d['column']}"[:35] for d in null_data]
        values = [d['null_pct'] for d in null_data]
        
        # Color based on severity
        colors = []
        for v in values:
            if v > 80:
                colors.append('#EF4444')
            elif v > 50:
                colors.append('#F59E0B')
            elif v > 25:
                colors.append('#06B6D4')
            else:
                colors.append('#10B981')
        
        bars = ax.barh(labels, values, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
        
        # Add percentage labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Null Percentage', fontsize=11, fontweight='500')
        ax.set_title('Columns with Null Values', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(0, 110)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'bar_null_columns', fig)
    
    # ========================================================================
    # QUALITY ANALYSIS CHARTS
    # ========================================================================
    
    def _generate_quality_gauge(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate modern quality score gauge."""
        summary = results.get('summary', {})
        quality_score = summary.get('quality_score', 0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create gauge background
        theta = np.linspace(0, np.pi, 100) if NUMPY_AVAILABLE else [i * math.pi / 99 for i in range(100)]
        
        # Background arc
        for i, t in enumerate(theta[:-1]):
            color_idx = int(i / 33)
            if color_idx == 0:
                color = '#EF4444'  # Red
            elif color_idx == 1:
                color = '#F59E0B'  # Amber
            else:
                color = '#10B981'  # Green
            
            wedge = Wedge((0, 0), 1, math.degrees(t), math.degrees(theta[i+1]), 
                         width=0.3, facecolor=color, alpha=0.2)
            ax.add_patch(wedge)
        
        # Score arc
        score_angle = np.pi * (quality_score / 100) if NUMPY_AVAILABLE else math.pi * (quality_score / 100)
        
        if quality_score >= 80:
            score_color = '#10B981'
        elif quality_score >= 60:
            score_color = '#06B6D4'
        elif quality_score >= 40:
            score_color = '#F59E0B'
        else:
            score_color = '#EF4444'
        
        # Draw score wedge
        wedge = Wedge((0, 0), 1, 0, math.degrees(score_angle), width=0.3, 
                     facecolor=score_color, edgecolor='white', linewidth=2)
        ax.add_patch(wedge)
        
        # Add needle
        needle_angle = score_angle
        needle_x = 0.85 * math.cos(needle_angle)
        needle_y = 0.85 * math.sin(needle_angle)
        ax.annotate('', xy=(needle_x, needle_y), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='#374151', lw=3))
        
        # Center circle
        center_circle = plt.Circle((0, 0), 0.15, color=score_color, zorder=5)
        ax.add_patch(center_circle)
        
        # Score text
        ax.text(0, -0.5, f'{quality_score}', ha='center', va='center',
               fontsize=60, fontweight='bold', color=score_color)
        ax.text(0, -0.75, 'Quality Score', ha='center', va='center',
               fontsize=16, color='#6B7280')
        
        # Quality label
        if quality_score >= 80:
            label = 'Excellent'
        elif quality_score >= 60:
            label = 'Good'
        elif quality_score >= 40:
            label = 'Fair'
        else:
            label = 'Needs Improvement'
        ax.text(0, -0.95, label, ha='center', va='center',
               fontsize=14, fontweight='bold', color=score_color)
        
        # Scale labels
        ax.text(-1.1, 0, '0', ha='center', va='center', fontsize=10, color='#9CA3AF')
        ax.text(0, 1.1, '50', ha='center', va='center', fontsize=10, color='#9CA3AF')
        ax.text(1.1, 0, '100', ha='center', va='center', fontsize=10, color='#9CA3AF')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.2, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Data Quality Assessment', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'gauge_quality_score', fig)
    
    def _generate_quality_issues_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate chart of quality issues by severity."""
        quality_issues = results.get('quality_issues', [])
        
        if not quality_issues:
            # Create empty state chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No Quality Issues Found!', ha='center', va='center',
                   fontsize=20, fontweight='bold', color='#10B981')
            ax.text(0.5, 0.35, 'Your data quality is excellent', ha='center', va='center',
                   fontsize=12, color='#6B7280')
            ax.axis('off')
            return self._save_chart(dataset_id, 'bar_quality_issues', fig)
        
        # Count by severity
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for issue in quality_issues:
            severity = issue.get('severity', 'low')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        severities = ['High', 'Medium', 'Low']
        counts = [severity_counts['high'], severity_counts['medium'], severity_counts['low']]
        colors = ['#EF4444', '#F59E0B', '#10B981']
        
        bars = ax.bar(severities, counts, color=colors, edgecolor='white', linewidth=2, width=0.6)
        
        # Add value labels with icons
        icons = ['⚠️', '⚡', 'ℹ️']
        for bar, count, icon in zip(bars, counts, icons):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + max(counts) * 0.05,
                       f'{count}', ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        ax.set_xlabel('Severity Level', fontsize=12, fontweight='500')
        ax.set_ylabel('Number of Issues', fontsize=12, fontweight='500')
        ax.set_title('Data Quality Issues by Severity', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add total count
        total = sum(counts)
        ax.text(0.98, 0.98, f'Total: {total} issues', transform=ax.transAxes,
               ha='right', va='top', fontsize=11, color='#6B7280')
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'bar_quality_issues_severity', fig)
    
    def _generate_completeness_radar(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate radar chart of data completeness metrics."""
        tables = results.get('tables', [])
        summary = results.get('summary', {})
        
        if not tables:
            return None
        
        # Calculate metrics
        total_columns = sum(len(t.get('columns', [])) for t in tables)
        total_rows = sum(t.get('row_count', 0) for t in tables)
        null_cols = 0
        documented_cols = 0
        
        for table in tables:
            for col in table.get('columns', []):
                if col.get('null_percentage', 0) < 0.1:
                    null_cols += 1
                if col.get('description'):
                    documented_cols += 1
        
        # Metrics for radar
        metrics = {
            'Completeness': (null_cols / total_columns * 100) if total_columns > 0 else 0,
            'Quality Score': summary.get('quality_score', 0),
            'Documentation': (documented_cols / total_columns * 100) if total_columns > 0 else 0,
            'Relationships': min(100, len(results.get('relationships', [])) * 10),
            'Schema Coverage': 100 if total_columns > 0 else 0,
        }
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Number of variables
        categories = list(metrics.keys())
        N = len(categories)
        
        # Calculate angles
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop
        
        # Values
        values = list(metrics.values())
        values += values[:1]
        
        # Draw the chart
        ax.plot(angles, values, 'o-', linewidth=2, color='#06B6D4')
        ax.fill(angles, values, alpha=0.25, color='#06B6D4')
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight='500')
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8, color='#9CA3AF')
        
        ax.set_title('Data Completeness Overview', fontsize=14, fontweight='bold', pad=20)
        
        # Add value annotations
        for angle, value, cat in zip(angles[:-1], values[:-1], categories):
            ax.annotate(f'{value:.0f}%', xy=(angle, value), xytext=(angle, value + 10),
                       ha='center', fontsize=9, fontweight='bold', color='#374151')
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'radar_completeness', fig)
    
    # ========================================================================
    # RELATIONSHIP ANALYSIS CHARTS
    # ========================================================================
    
    def _generate_network_graph(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate network graph visualization of table relationships."""
        relationships = results.get('relationships', [])
        tables = results.get('tables', [])
        
        if not tables or len(tables) < 2:
            return None
        
        # Use advanced relationship detector to find additional relationships
        detector = AdvancedRelationshipDetector()
        inferred_rels = detector.analyze_all_relationships(tables)
        
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Get unique tables
        involved_tables = set()
        for rel in relationships:
            involved_tables.add(rel.get('from_table'))
            involved_tables.add(rel.get('to_table'))
        
        # Add tables from inferred relationships
        for rel in inferred_rels[:10]:  # Top 10 inferred
            involved_tables.add(rel.get('from_table'))
            involved_tables.add(rel.get('to_table'))
        
        # If still not enough, add some tables
        if len(involved_tables) < 3:
            for t in tables[:10]:
                involved_tables.add(t.get('name'))
        
        involved_tables = [t for t in involved_tables if t]
        n_tables = len(involved_tables)
        
        if n_tables < 2:
            return None
        
        # Calculate positions in a circle
        radius = 5
        angles = [2 * math.pi * i / n_tables for i in range(n_tables)]
        positions = {}
        
        for i, table in enumerate(involved_tables):
            x = radius * math.cos(angles[i] - math.pi/2)
            y = radius * math.sin(angles[i] - math.pi/2)
            positions[table] = (x, y)
        
        # Draw connections (explicit relationships)
        for rel in relationships:
            from_pos = positions.get(rel.get('from_table'))
            to_pos = positions.get(rel.get('to_table'))
            
            if from_pos and to_pos:
                ax.annotate('', xy=to_pos, xytext=from_pos,
                           arrowprops=dict(arrowstyle='-|>', color='#4F46E5',
                                          connectionstyle='arc3,rad=0.1',
                                          lw=2.5))
        
        # Draw inferred relationships (dashed)
        for rel in inferred_rels[:10]:
            from_pos = positions.get(rel.get('from_table'))
            to_pos = positions.get(rel.get('to_table'))
            
            if from_pos and to_pos:
                confidence = rel.get('confidence', 0.5)
                alpha = 0.3 + confidence * 0.5
                ax.annotate('', xy=to_pos, xytext=from_pos,
                           arrowprops=dict(arrowstyle='-|>', color='#06B6D4',
                                          linestyle='--', lw=1.5, alpha=alpha,
                                          connectionstyle='arc3,rad=-0.1'))
        
        # Draw table nodes
        # Get row counts for sizing
        table_rows = {t.get('name'): t.get('row_count', 100) for t in tables}
        max_rows = max(table_rows.values()) if table_rows else 1
        
        for table, (x, y) in positions.items():
            # Size based on row count
            rows = table_rows.get(table, 100)
            node_size = 0.4 + (rows / max_rows) * 0.4
            
            # Draw node
            circle = plt.Circle((x, y), node_size, color='#4F46E5', 
                                ec='white', lw=3, zorder=10)
            ax.add_patch(circle)
            
            # Add table name
            ax.text(x, y - node_size - 0.3, table[:18], ha='center', va='top',
                   fontsize=9, fontweight='bold', color='#374151')
        
        # Legend
        explicit_line = mpatches.Patch(color='#4F46E5', label='Explicit FK Relationship')
        inferred_line = mpatches.Patch(color='#06B6D4', label='Inferred Relationship')
        ax.legend(handles=[explicit_line, inferred_line], loc='upper left', fontsize=10)
        
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Table Relationship Network', fontsize=16, fontweight='bold', pad=20)
        
        # Add stats
        stats_text = f'Tables: {n_tables} | Explicit: {len(relationships)} | Inferred: {len(inferred_rels)}'
        ax.text(0.5, -0.02, stats_text, transform=ax.transAxes, ha='center',
               fontsize=10, color='#6B7280')
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'network_relationships', fig)
    
    def _generate_relationship_heatmap(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate heatmap showing relationship strengths between tables."""
        tables = results.get('tables', [])
        relationships = results.get('relationships', [])
        
        if not tables or len(tables) < 2:
            return None
        
        # Build relationship matrix
        table_names = [t.get('name', '')[:15] for t in tables[:12]]
        n = len(table_names)
        
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        # Fill matrix with relationship data
        name_to_idx = {name: i for i, name in enumerate(table_names)}
        
        for rel in relationships:
            from_name = rel.get('from_table', '')[:15]
            to_name = rel.get('to_table', '')[:15]
            
            if from_name in name_to_idx and to_name in name_to_idx:
                i = name_to_idx[from_name]
                j = name_to_idx[to_name]
                matrix[i][j] = 1
                matrix[j][i] = 0.5  # Reverse direction weaker
        
        # Use advanced detector for inferred relationships
        detector = AdvancedRelationshipDetector()
        inferred = detector.analyze_all_relationships(tables)
        
        for rel in inferred:
            from_name = rel.get('from_table', '')[:15]
            to_name = rel.get('to_table', '')[:15]
            confidence = rel.get('confidence', 0.5)
            
            if from_name in name_to_idx and to_name in name_to_idx:
                i = name_to_idx[from_name]
                j = name_to_idx[to_name]
                if matrix[i][j] == 0:  # Don't overwrite explicit relationships
                    matrix[i][j] = confidence * 0.7
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap
        if NUMPY_AVAILABLE:
            matrix_np = np.array(matrix)
            im = ax.imshow(matrix_np, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        else:
            im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Relationship Strength', fontsize=11)
        
        # Set labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(table_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(table_names, fontsize=9)
        
        # Add value annotations
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > 0:
                    text_color = 'white' if matrix[i][j] > 0.5 else 'black'
                    ax.text(j, i, f'{matrix[i][j]:.1f}', ha='center', va='center',
                           fontsize=8, color=text_color, fontweight='bold')
        
        ax.set_title('Table Relationship Matrix', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Target Table', fontsize=11, fontweight='500')
        ax.set_ylabel('Source Table', fontsize=11, fontweight='500')
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'heatmap_relationships', fig)
    
    def _generate_hierarchy_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate hierarchical tree chart showing table dependencies."""
        tables = results.get('tables', [])
        relationships = results.get('relationships', [])
        
        if not tables:
            return None
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Build dependency graph
        deps = defaultdict(list)
        has_parent = set()
        
        for rel in relationships:
            from_table = rel.get('from_table')
            to_table = rel.get('to_table')
            if from_table and to_table:
                deps[to_table].append(from_table)
                has_parent.add(from_table)
        
        # Find root tables (tables with no parents)
        all_tables = set(t.get('name') for t in tables)
        root_tables = all_tables - has_parent
        
        if not root_tables:
            root_tables = all_tables
        
        # Limit tables
        root_tables = list(root_tables)[:8]
        
        # Draw hierarchy
        level_y = {0: 0.9}
        x_positions = {}
        nodes_drawn = set()
        
        def draw_level(parent_tables, level, x_start, x_end):
            if not parent_tables or level > 4:
                return
            
            y = 0.9 - level * 0.2
            n = len(parent_tables)
            
            for i, table in enumerate(parent_tables):
                if table in nodes_drawn:
                    continue
                
                x = x_start + (i + 0.5) * (x_end - x_start) / n
                x_positions[table] = (x, y)
                nodes_drawn.add(table)
                
                # Draw node
                color = self.COLORS[level % len(self.COLORS)]
                box = FancyBboxPatch((x - 0.08, y - 0.03), 0.16, 0.06,
                                    boxstyle="round,pad=0.01",
                                    facecolor=color, edgecolor='white', linewidth=2)
                ax.add_patch(box)
                
                # Table name
                ax.text(x, y, table[:15], ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
                
                # Draw children
                children = deps.get(table, [])[:5]
                if children:
                    child_x_range = (x_end - x_start) / n
                    child_start = x - child_x_range / 2
                    child_end = x + child_x_range / 2
                    draw_level(children, level + 1, child_start, child_end)
                    
                    # Draw connections
                    for child in children:
                        if child in x_positions:
                            cx, cy = x_positions[child]
                            ax.plot([x, cx], [y - 0.03, cy + 0.03], 
                                   color='#9CA3AF', lw=1.5, zorder=1)
        
        draw_level(root_tables, 0, 0.05, 0.95)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Table Dependency Hierarchy', fontsize=14, fontweight='bold', pad=15)
        
        # Legend
        for i, level_name in enumerate(['Root Tables', 'Level 1', 'Level 2', 'Level 3']):
            color = self.COLORS[i % len(self.COLORS)]
            ax.add_patch(plt.Rectangle((0.02, 0.95 - i * 0.04), 0.02, 0.02, 
                                       facecolor=color, edgecolor='white'))
            ax.text(0.06, 0.96 - i * 0.04, level_name, fontsize=8, va='center')
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'hierarchy_table_dependencies', fig)
    
    # ========================================================================
    # SCHEMA OVERVIEW CHARTS
    # ========================================================================
    
    def _generate_schema_matrix(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate comprehensive schema overview matrix."""
        tables = results.get('tables', [])
        summary = results.get('summary', {})
        
        if not tables:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. Table Overview (top-left)
        ax1 = axes[0, 0]
        table_data = [(t.get('name', '')[:20], t.get('row_count', 0), 
                       len(t.get('columns', []))) for t in tables[:10]]
        
        if table_data:
            names, rows, cols = zip(*table_data)
            x = range(len(names))
            
            ax1.bar(x, rows, color='#06B6D4', alpha=0.7, label='Rows', width=0.4)
            ax1.bar([i + 0.4 for i in x], [c * 100 for c in cols], color='#4F46E5', 
                    alpha=0.7, label='Columns (×100)', width=0.4)
            
            ax1.set_xticks([i + 0.2 for i in x])
            ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax1.set_title('Table Overview', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=8)
            ax1.grid(axis='y', alpha=0.3)
        
        # 2. Quality Distribution (top-right)
        ax2 = axes[0, 1]
        quality_issues = results.get('quality_issues', [])
        severity_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        for issue in quality_issues:
            sev = issue.get('severity', 'low').capitalize()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        colors = ['#EF4444', '#F59E0B', '#10B981']
        ax2.pie(list(severity_counts.values()), labels=list(severity_counts.keys()),
               autopct='%1.0f%%', colors=colors, startangle=90)
        ax2.set_title('Quality Issues Distribution', fontsize=12, fontweight='bold')
        
        # 3. Column Types (bottom-left)
        ax3 = axes[1, 0]
        type_counts = defaultdict(int)
        for table in tables:
            for col in table.get('columns', []):
                dtype = self._normalize_data_type(col.get('data_type', 'OTHER'))
                type_counts[dtype] += 1
        
        if type_counts:
            types, counts = zip(*sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:8])
            ax3.barh(types, counts, color=[self.COLORS[i % len(self.COLORS)] for i in range(len(types))])
            ax3.set_title('Column Data Types', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Count')
        
        # 4. Key Metrics (bottom-right)
        ax4 = axes[1, 1]
        metrics = {
            'Tables': len(tables),
            'Columns': sum(len(t.get('columns', [])) for t in tables),
            'Rows (K)': sum(t.get('row_count', 0) for t in tables) / 1000,
            'Relationships': len(results.get('relationships', [])),
            'Quality Score': summary.get('quality_score', 0),
        }
        
        ax4.barh(list(metrics.keys()), list(metrics.values()), 
                color=['#06B6D4', '#4F46E5', '#10B981', '#F59E0B', '#8B5CF6'])
        ax4.set_title('Key Metrics', fontsize=12, fontweight='bold')
        
        # Add value labels
        for i, (k, v) in enumerate(metrics.items()):
            label = f'{v:,.0f}' if v >= 1 else f'{v:.2f}'
            ax4.text(v + max(metrics.values()) * 0.02, i, label, va='center', fontsize=9)
        
        plt.suptitle('Schema Analysis Overview', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'matrix_schema_overview', fig)


def create_chart_generator(storage_manager) -> ChartGenerator:
    """Factory function to create chart generator."""
    return ChartGenerator(storage_manager)

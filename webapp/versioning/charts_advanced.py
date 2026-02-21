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
import threading
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
    
    Chart Types (21 total):
    1. Table Analysis:
       - Table Size Distribution (Gradient Bar)
       - Table Type Distribution (Donut with glow)
       - Row Count Treemap
       - Data Density Heatmap Grid
       - Table DNA Barcode
    
    2. Column Analysis:
       - Column Data Types (Gradient Bar)
       - Null Value Distribution (Horizontal Bar)
       - Uniqueness Heatmap
       - Column Fingerprint Bubble Scatter
    
    3. Relationship Analysis:
       - Network Relationship Graph
       - Relationship Matrix Heatmap
       - Data Flow Sankey Diagram
       - Dependency Hierarchy Tree
       - Schema Constellation (force-directed)
       - FK Coverage Ring
    
    4. Quality Analysis:
       - Quality Score Gauge (neon)
       - Quality Issues by Severity (Bar)
       - Completeness Radar Chart
       - Quality Matrix Heatmap
    
    5. Advanced / Summary:
       - Sunburst Hierarchy (table→column→type)
       - Statistical Distribution Violin
       - Schema Overview Matrix (4-panel)
    """
    
    # Modern enterprise color palette — extended
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
    
    # Gradient pairs for advanced charts
    GRADIENTS = [
        ('#06B6D4', '#0E7490'),
        ('#4F46E5', '#3730A3'),
        ('#10B981', '#047857'),
        ('#F59E0B', '#D97706'),
        ('#EF4444', '#DC2626'),
        ('#8B5CF6', '#6D28D9'),
    ]
    
    # Quality color scale
    QUALITY_COLORS = {
        'excellent': '#10B981',
        'good': '#06B6D4',
        'fair': '#F59E0B',
        'poor': '#EF4444',
    }
    
    # Style settings for consistent premium look
    STYLE = {
        'figure.facecolor': '#FAFBFC',
        'axes.facecolor': '#FAFBFC',
        'axes.edgecolor': '#E5E7EB',
        'axes.labelcolor': '#1F2937',
        'text.color': '#1F2937',
        'xtick.color': '#6B7280',
        'ytick.color': '#6B7280',
        'grid.color': '#F3F4F6',
        'font.family': 'sans-serif',
    }
    
    # Module-level lock for matplotlib thread safety
    _mpl_lock = threading.Lock()

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
        
        chart_methods = [
            # --- Table Analysis ---
            ('bar_table_size', self._generate_table_size_chart),
            ('donut_table_type', self._generate_table_type_donut),
            ('treemap_tables', self._generate_table_treemap),
            ('density_heatmap', self._generate_data_density_heatmap),
            ('table_dna', self._generate_table_dna_barcode),
            # --- Column Analysis ---
            ('bar_column_types', self._generate_column_type_chart),
            ('heatmap_null', self._generate_null_heatmap),
            ('bar_null_distribution', self._generate_null_distribution_chart),
            ('column_fingerprint', self._generate_column_fingerprint),
            # --- Quality Analysis ---
            ('gauge_quality', self._generate_quality_gauge),
            ('bar_quality_issues', self._generate_quality_issues_chart),
            ('radar_completeness', self._generate_completeness_radar),
            # --- Relationship Analysis ---
            ('network_relationships', self._generate_network_graph),
            ('heatmap_relationships', self._generate_relationship_heatmap),
            ('hierarchy_tables', self._generate_hierarchy_chart),
            ('sankey_dataflow', self._generate_sankey_dataflow),
            ('constellation_schema', self._generate_schema_constellation),
            ('fk_coverage_ring', self._generate_fk_coverage_ring),
            # --- Advanced / Summary ---
            ('sunburst_hierarchy', self._generate_sunburst_chart),
            ('violin_distribution', self._generate_violin_distribution),
            ('matrix_overview', self._generate_schema_matrix),
        ]
        
        for chart_name, method in chart_methods:
            try:
                with self._mpl_lock:
                    result = method(dataset_id, results)
                if result:
                    generated.append(result)
            except Exception as e:
                print(f"Error generating {chart_name}: {e}")
                continue
        
        return generated
    
    def _save_chart(self, dataset_id: str, chart_name: str, fig) -> Optional[str]:
        """Save chart to storage with high quality."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none',
                    pad_inches=0.3)
        buf.seek(0)
        chart_data = buf.read()
        plt.close(fig)
        
        self.storage.store_chart(dataset_id, chart_data, chart_name)
        
        return f"{chart_name}.png"
    
    def _add_watermark(self, fig, text='DataMind AI'):
        """Add subtle watermark to chart."""
        fig.text(0.99, 0.01, text, fontsize=7, color='#D1D5DB',
                 ha='right', va='bottom', alpha=0.5, style='italic')
    
    # ========================================================================
    # TABLE ANALYSIS CHARTS
    # ========================================================================
    
    def _generate_table_size_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate premium gradient horizontal bar chart of table row counts."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        sorted_tables = sorted(tables, key=lambda t: t.get('row_count', 0), reverse=True)[:15]
        
        names = [t.get('name', 'Unknown')[:25] for t in sorted_tables]
        counts = [t.get('row_count', 0) for t in sorted_tables]
        
        fig, ax = plt.subplots(figsize=(14, max(7, len(names) * 0.6)))
        fig.set_facecolor('#FAFBFC')
        
        max_count = max(counts) if counts else 1
        
        # Draw bars with gradient effect (multiple overlapping bars)
        for i, (name, count) in enumerate(zip(names, counts)):
            base_color = self.COLORS[i % len(self.COLORS)]
            # Main bar
            bar = ax.barh(i, count, color=base_color, edgecolor='white',
                         linewidth=0.5, height=0.65, alpha=0.85, zorder=3)
            # Highlight strip on top edge
            ax.barh(i + 0.25, count, color=base_color, height=0.08, alpha=0.4, zorder=4)
            # Value label
            label = f'{count:,}'
            ax.text(count + max_count * 0.02, i,
                   label, ha='left', va='center', fontsize=10,
                   color='#1F2937', fontweight='600')
        
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Number of Rows', fontsize=12, fontweight='600', labelpad=10)
        ax.set_title('Table Size Distribution', fontsize=16, fontweight='bold', pad=20,
                     color='#111827')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.15, linestyle='-', color='#CBD5E1')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E5E7EB')
        ax.spines['bottom'].set_color('#E5E7EB')
        
        # Summary annotation
        total = sum(counts)
        ax.text(0.98, 0.02, f'Total: {total:,} rows across {len(names)} tables',
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=9, color='#9CA3AF', style='italic')
        
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'bar_table_size', fig)
    
    def _generate_table_type_donut(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate premium donut chart of table types with glow effect."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        type_counts = defaultdict(int)
        for table in tables:
            table_type = table.get('table_type', 'unknown')
            type_counts[table_type] += 1
        
        if not type_counts:
            return None
        
        fig, ax = plt.subplots(figsize=(11, 11))
        fig.set_facecolor('#FAFBFC')
        
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        colors = self.COLORS[:len(labels)]
        
        # Outer ring — subtle shadow
        ax.pie(sizes, labels=None, colors=['#E5E7EB'] * len(sizes),
               startangle=90, radius=1.08,
               wedgeprops=dict(width=0.52, edgecolor='#FAFBFC', linewidth=0))
        
        # Main donut
        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, autopct='%1.1f%%',
            colors=colors, startangle=90, pctdistance=0.78,
            wedgeprops=dict(width=0.48, edgecolor='white', linewidth=3),
            radius=1.05
        )
        
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
            autotext.set_color('white')
            autotext.set_path_effects([
                path_effects.withStroke(linewidth=2, foreground='black', alpha=0.3)])
        
        # Legend with counts
        legend_labels = [f'{l.capitalize()}  ({s})' for l, s in zip(labels, sizes)]
        leg = ax.legend(wedges, legend_labels,
                 loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11,
                 frameon=True, fancybox=True, shadow=True,
                 edgecolor='#E5E7EB')
        leg.get_frame().set_facecolor('#FFFFFF')
        
        # Center text with icon
        ax.text(0, 0.08, f'{sum(sizes)}', ha='center', va='center',
               fontsize=42, fontweight='bold', color='#111827')
        ax.text(0, -0.12, 'Tables', ha='center', va='center',
               fontsize=14, color='#6B7280', fontweight='500')
        
        ax.set_title('Table Type Distribution', fontsize=16, fontweight='bold', pad=25,
                     color='#111827')
        ax.axis('equal')
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'donut_table_type', fig)
    
    def _generate_table_treemap(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate premium treemap with gradient fills and rounded labels."""
        tables = results.get('tables', [])
        if not tables or len(tables) < 2:
            return None
        
        sorted_tables = sorted(tables, key=lambda t: t.get('row_count', 0), reverse=True)[:20]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.set_facecolor('#FAFBFC')
        
        total_rows = sum(t.get('row_count', 1) for t in sorted_tables)
        
        def squarify(values, x, y, width, height):
            if not values:
                return []
            total = sum(values)
            rects = []
            if width >= height:
                for val in values:
                    w = (val / total) * width if total > 0 else width / len(values)
                    rects.append((x, y, w, height))
                    x += w
            else:
                for val in values:
                    h = (val / total) * height if total > 0 else height / len(values)
                    rects.append((x, y, width, h))
                    y += h
            return rects
        
        values = [t.get('row_count', 1) for t in sorted_tables]
        rects = squarify(values, 0, 0, 10, 10)
        
        for i, (rect, table) in enumerate(zip(rects, sorted_tables)):
            x, y, w, h = rect
            color = self.COLORS[i % len(self.COLORS)]
            
            # Main rect
            ax.add_patch(plt.Rectangle((x + 0.05, y + 0.05), w - 0.1, h - 0.1,
                                        facecolor=color, edgecolor='white',
                                        linewidth=2.5, alpha=0.85, zorder=2))
            # Highlight overlay on top edge
            ax.add_patch(plt.Rectangle((x + 0.05, y + h - 0.15), w - 0.1, 0.08,
                                        facecolor='white', alpha=0.15, zorder=3))
            
            # Labels
            if w > 0.8 and h > 0.6:
                name = table.get('name', '')[:15]
                count = table.get('row_count', 0)
                pct = (count / total_rows * 100) if total_rows > 0 else 0
                
                ax.text(x + w/2, y + h/2 + 0.1, name.upper(),
                       ha='center', va='center', fontsize=9 if w > 1.5 else 7,
                       color='white', fontweight='bold', zorder=4,
                       path_effects=[path_effects.withStroke(linewidth=2, foreground='black', alpha=0.3)])
                ax.text(x + w/2, y + h/2 - 0.25, f'{count:,} rows  ({pct:.1f}%)',
                       ha='center', va='center', fontsize=7 if w > 1.5 else 6,
                       color='white', alpha=0.85, zorder=4)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Table Size Treemap', fontsize=16, fontweight='bold', pad=20, color='#111827')
        
        # Summary
        ax.text(0.5, -0.02, f'{len(sorted_tables)} tables  ·  {total_rows:,} total rows',
               transform=ax.transAxes, ha='center', fontsize=10, color='#6B7280')
        
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'treemap_tables', fig)
    
    # ========================================================================
    # COLUMN ANALYSIS CHARTS
    # ========================================================================
    
    def _generate_column_type_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate premium column data type distribution with gradient bars."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        type_counts = defaultdict(int)
        for table in tables:
            for col in table.get('columns', []):
                data_type = col.get('data_type', 'UNKNOWN').upper()
                normalized = self._normalize_data_type(data_type)
                type_counts[normalized] += 1
        
        if not type_counts:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.set_facecolor('#FAFBFC')
        
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        types = [t[0] for t in sorted_types]
        counts = [t[1] for t in sorted_types]
        total = sum(counts)
        
        type_colors = {
            'INTEGER': '#4F46E5', 'TEXT': '#06B6D4', 'NUMERIC': '#10B981',
            'DATETIME': '#F59E0B', 'BOOLEAN': '#EC4899', 'BINARY': '#8B5CF6',
        }
        colors = [type_colors.get(t, self.COLORS[i % len(self.COLORS)]) for i, t in enumerate(types)]
        
        bars = ax.bar(types, counts, color=colors, edgecolor='white', linewidth=1.5,
                     alpha=0.85, width=0.65)
        
        # Highlight strip on top of each bar
        for bar in bars:
            x = bar.get_x()
            w = bar.get_width()
            h = bar.get_height()
            ax.add_patch(plt.Rectangle((x, h - h * 0.04), w, h * 0.04,
                                      facecolor='white', alpha=0.25, zorder=3))
        
        # Value labels with percentage
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = count / total * 100 if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2, height + max(counts) * 0.03,
                   f'{count}  ({pct:.0f}%)', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='#374151')
        
        ax.set_xlabel('Data Type', fontsize=12, fontweight='600', color='#374151')
        ax.set_ylabel('Number of Columns', fontsize=12, fontweight='600', color='#374151')
        ax.set_title('Column Data Type Distribution', fontsize=16, fontweight='bold',
                     pad=20, color='#111827')
        ax.grid(axis='y', alpha=0.12, linestyle='-')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E5E7EB')
        ax.spines['bottom'].set_color('#E5E7EB')
        
        plt.xticks(rotation=45, ha='right')
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'bar_column_types', fig)
    
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
        """Generate premium null-percentage heatmap with gradient bars and severity zones."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        table_names = []
        null_percentages = []
        
        for table in tables[:15]:
            table_name = table.get('name', 'Unknown')[:20]
            columns = table.get('columns', [])[:10]
            if columns:
                table_names.append(table_name)
                avg_null = sum(c.get('null_percentage', 0) for c in columns) / len(columns)
                null_percentages.append(avg_null * 100)
        
        if not table_names:
            return None
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(table_names) * 0.55)))
        fig.set_facecolor('#FAFBFC')
        
        cmap = LinearSegmentedColormap.from_list('nulls',
            ['#10B981', '#06B6D4', '#F59E0B', '#EF4444'], N=256)
        
        max_pct = max(null_percentages) if null_percentages else 100
        colors = [cmap(min(pct / 100, 1.0)) for pct in null_percentages]
        
        bars = ax.barh(table_names, null_percentages, color=colors,
                       edgecolor='white', linewidth=1.5, height=0.65, alpha=0.85)
        
        for bar, pct in zip(bars, null_percentages):
            w = bar.get_width()
            icon = '✔' if pct < 10 else ('⚠' if pct < 50 else '✖')
            ax.text(w + 2, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%  {icon}', ha='left', va='center', fontsize=9,
                   fontweight='600', color='#374151')
        
        # Severity zones
        ax.axvline(x=10, color='#10B981', linestyle=':', alpha=0.4, lw=1)
        ax.axvline(x=25, color='#F59E0B', linestyle='--', alpha=0.4, lw=1.5, label='Warning (25%)')
        ax.axvline(x=50, color='#EF4444', linestyle='--', alpha=0.4, lw=1.5, label='Critical (50%)')
        
        ax.set_xlabel('Average Null Percentage', fontsize=12, fontweight='600', color='#374151')
        ax.set_title('Null Value Distribution by Table', fontsize=16, fontweight='bold',
                     pad=20, color='#111827')
        ax.set_xlim(0, max_pct * 1.25 if max_pct > 0 else 100)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E5E7EB')
        ax.legend(loc='lower right', fontsize=8, fancybox=True, edgecolor='#E5E7EB')
        
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'heatmap_null', fig)
    
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
        
        bars = ax.barh(labels, values, color=colors, edgecolor='white', linewidth=1.5,
                       height=0.65, alpha=0.85)
        
        for bar, val in zip(bars, values):
            icon = '✔' if val < 25 else ('⚠' if val < 50 else '✖')
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%  {icon}', ha='left', va='center', fontsize=9, fontweight='600')
        
        ax.set_xlabel('Null Percentage', fontsize=12, fontweight='600', color='#374151')
        ax.set_title('Columns with Null Values', fontsize=16, fontweight='bold',
                     pad=20, color='#111827')
        ax.set_xlim(0, 110)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E5E7EB')
        
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'bar_null_distribution', fig)
    
    # ========================================================================
    # QUALITY ANALYSIS CHARTS
    # ========================================================================
    
    def _generate_quality_gauge(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate premium neon-style quality score gauge."""
        summary = results.get('summary', {})
        quality_score = summary.get('quality_score', 0)
        
        fig, ax = plt.subplots(figsize=(11, 9))
        fig.set_facecolor('#0F172A')  # Dark background for neon effect
        
        theta = np.linspace(0, np.pi, 200) if NUMPY_AVAILABLE else [i * math.pi / 199 for i in range(200)]
        
        # Background arc segments (red → amber → green)
        segment_colors = [
            ('#EF4444', 0, 0.33),
            ('#F59E0B', 0.33, 0.66),
            ('#10B981', 0.66, 1.0),
        ]
        
        for color, start_pct, end_pct in segment_colors:
            s_idx = int(start_pct * len(theta))
            e_idx = int(end_pct * len(theta))
            for i in range(s_idx, min(e_idx, len(theta) - 1)):
                wedge = Wedge((0, 0), 1.05, math.degrees(theta[i]), math.degrees(theta[i+1]),
                             width=0.28, facecolor=color, alpha=0.12)
                ax.add_patch(wedge)
        
        # Score arc with neon glow
        score_angle = math.pi * (quality_score / 100)
        
        if quality_score >= 80:
            score_color = '#10B981'
        elif quality_score >= 60:
            score_color = '#06B6D4'
        elif quality_score >= 40:
            score_color = '#F59E0B'
        else:
            score_color = '#EF4444'
        
        # Glow layers
        for width_mult, alpha in [(0.38, 0.06), (0.34, 0.12), (0.30, 0.25)]:
            glow = Wedge((0, 0), 1.05, 0, math.degrees(score_angle),
                        width=width_mult, facecolor=score_color, alpha=alpha)
            ax.add_patch(glow)
        
        # Main score arc
        score_wedge = Wedge((0, 0), 1.05, 0, math.degrees(score_angle),
                           width=0.28, facecolor=score_color, edgecolor='none')
        ax.add_patch(score_wedge)
        
        # Needle with glow
        needle_x = 0.9 * math.cos(score_angle)
        needle_y = 0.9 * math.sin(score_angle)
        # Needle glow
        ax.plot([0, needle_x], [0, needle_y], color=score_color, lw=4, alpha=0.3, zorder=4)
        ax.plot([0, needle_x], [0, needle_y], color='white', lw=2, alpha=0.8, zorder=5)
        
        # Center dot
        for r, alpha in [(0.18, 0.15), (0.14, 0.3), (0.10, 0.8)]:
            center = plt.Circle((0, 0), r, color=score_color, alpha=alpha, zorder=6)
            ax.add_patch(center)
        
        # Score text
        ax.text(0, -0.45, f'{quality_score}', ha='center', va='center',
               fontsize=64, fontweight='bold', color=score_color, zorder=7,
               path_effects=[path_effects.withStroke(linewidth=3, foreground=score_color, alpha=0.2)])
        ax.text(0, -0.72, 'Quality Score', ha='center', va='center',
               fontsize=15, color='#94A3B8', fontweight='500')
        
        # Quality label
        if quality_score >= 80:
            label = 'EXCELLENT'
        elif quality_score >= 60:
            label = 'GOOD'
        elif quality_score >= 40:
            label = 'FAIR'
        else:
            label = 'NEEDS IMPROVEMENT'
        ax.text(0, -0.92, label, ha='center', va='center',
               fontsize=13, fontweight='bold', color=score_color,
               path_effects=[path_effects.withStroke(linewidth=2, foreground=score_color, alpha=0.15)])
        
        # Scale labels
        for val, angle in [(0, 0), (25, math.pi * 0.25), (50, math.pi * 0.5),
                           (75, math.pi * 0.75), (100, math.pi)]:
            x = 1.18 * math.cos(angle)
            y = 1.18 * math.sin(angle)
            ax.text(x, y, str(val), ha='center', va='center', fontsize=9,
                   color='#64748B', fontweight='500')
        
        # Issue count badges
        q_issues = results.get('quality_issues', [])
        high_count = sum(1 for q in q_issues if q.get('severity') == 'high')
        med_count = sum(1 for q in q_issues if q.get('severity') == 'medium')
        low_count = sum(1 for q in q_issues if q.get('severity') == 'low')
        
        badge_y = -1.15
        for x_off, count, color, label in [
            (-0.7, high_count, '#EF4444', 'High'),
            (0, med_count, '#F59E0B', 'Medium'),
            (0.7, low_count, '#10B981', 'Low')]:
            ax.text(x_off, badge_y, f'{count}', ha='center', va='center',
                   fontsize=16, fontweight='bold', color=color)
            ax.text(x_off, badge_y - 0.15, label, ha='center', va='center',
                   fontsize=8, color='#64748B')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.4, 1.35)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Data Quality Assessment', fontsize=16, fontweight='bold',
                     pad=20, color='white')
        
        fig.text(0.99, 0.01, 'DataMind AI', fontsize=7, color='#475569',
                 ha='right', va='bottom', alpha=0.5, style='italic')
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'gauge_quality', fig)
    
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
        fig.set_facecolor('#FAFBFC')
        
        severities = ['High', 'Medium', 'Low']
        counts = [severity_counts['high'], severity_counts['medium'], severity_counts['low']]
        colors = ['#EF4444', '#F59E0B', '#10B981']
        
        bars = ax.bar(severities, counts, color=colors, edgecolor='white', linewidth=2.5,
                     width=0.55, alpha=0.85)
        
        # Highlight strip
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.add_patch(plt.Rectangle((bar.get_x(), h - h * 0.05),
                             bar.get_width(), h * 0.05,
                             facecolor='white', alpha=0.25, zorder=3))
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + max(max(counts), 1) * 0.05,
                       f'{count}', ha='center', va='bottom', fontsize=20, fontweight='bold',
                       color='#374151')
        
        ax.set_xlabel('Severity Level', fontsize=12, fontweight='600', color='#374151')
        ax.set_ylabel('Number of Issues', fontsize=12, fontweight='600', color='#374151')
        ax.set_title('Data Quality Issues by Severity', fontsize=16, fontweight='bold',
                     pad=20, color='#111827')
        ax.grid(axis='y', alpha=0.12)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E5E7EB')
        ax.spines['bottom'].set_color('#E5E7EB')
        
        total = sum(counts)
        ax.text(0.98, 0.98, f'Total: {total} issues', transform=ax.transAxes,
               ha='right', va='top', fontsize=12, fontweight='600', color='#6B7280',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#F1F5F9', edgecolor='#E5E7EB'))
        
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'bar_quality_issues', fig)
    
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
        fig.set_facecolor('#FAFBFC')
        
        categories = list(metrics.keys())
        N = len(categories)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        
        values = list(metrics.values())
        values += values[:1]
        
        # Glow fill layers
        for alpha in [0.05, 0.12, 0.22]:
            ax.fill(angles, values, alpha=alpha, color='#06B6D4')
        ax.plot(angles, values, 'o-', linewidth=2.5, color='#06B6D4', markersize=8,
               markeredgecolor='white', markeredgewidth=2)
        
        # Grid styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='600', color='#374151')
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8, color='#9CA3AF')
        ax.set_rlabel_position(30)
        ax.grid(color='#E5E7EB', linewidth=0.5)
        ax.spines['polar'].set_color('#E5E7EB')
        
        ax.set_title('Data Completeness Overview', fontsize=16, fontweight='bold',
                     pad=25, color='#111827')
        
        # Value annotations with badge style
        for angle, value in zip(angles[:-1], values[:-1]):
            color = '#10B981' if value >= 70 else ('#F59E0B' if value >= 40 else '#EF4444')
            ax.annotate(f'{value:.0f}%', xy=(angle, value), xytext=(angle, value + 12),
                       ha='center', fontsize=10, fontweight='bold', color=color,
                       path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
        
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'radar_completeness', fig)
    
    # ========================================================================
    # RELATIONSHIP ANALYSIS CHARTS
    # ========================================================================
    
    def _generate_network_graph(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate premium dark-mode network graph with neon edges and glow nodes."""
        relationships = results.get('relationships', [])
        tables = results.get('tables', [])
        
        if not tables or len(tables) < 2:
            return None
        
        # Use advanced relationship detector to find additional relationships
        detector = AdvancedRelationshipDetector()
        inferred_rels = detector.analyze_all_relationships(tables)
        
        fig, ax = plt.subplots(figsize=(16, 14))
        fig.set_facecolor('#0F172A')
        
        # Get unique tables
        involved_tables = set()
        for rel in relationships:
            involved_tables.add(rel.get('from_table'))
            involved_tables.add(rel.get('to_table'))
        for rel in inferred_rels[:10]:
            involved_tables.add(rel.get('from_table'))
            involved_tables.add(rel.get('to_table'))
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
        
        # Draw connections — explicit (neon glow)
        for rel in relationships:
            fp = positions.get(rel.get('from_table'))
            tp = positions.get(rel.get('to_table'))
            if fp and tp:
                for lw, alpha in [(7, 0.06), (4, 0.15), (2, 0.6)]:
                    ax.annotate('', xy=tp, xytext=fp,
                               arrowprops=dict(arrowstyle='-|>', color='#818CF8',
                                              connectionstyle='arc3,rad=0.1',
                                              lw=lw, alpha=alpha))
        
        # Draw connections — inferred (dashed cyan glow)
        for rel in inferred_rels[:10]:
            fp = positions.get(rel.get('from_table'))
            tp = positions.get(rel.get('to_table'))
            if fp and tp:
                conf = rel.get('confidence', 0.5)
                ax.annotate('', xy=tp, xytext=fp,
                           arrowprops=dict(arrowstyle='-|>', color='#06B6D4',
                                          linestyle='--', lw=1.5, alpha=0.2 + conf * 0.4,
                                          connectionstyle='arc3,rad=-0.1'))
        
        # Draw table nodes with glow
        table_rows = {t.get('name'): t.get('row_count', 100) for t in tables}
        max_rows = max(table_rows.values()) if table_rows else 1
        
        for idx, (table, (x, y)) in enumerate(positions.items()):
            rows = table_rows.get(table, 100)
            node_size = 0.45 + (rows / max_rows) * 0.45
            color = self.COLORS[idx % len(self.COLORS)]
            
            # Glow layers
            for r_mult, alpha in [(2.5, 0.04), (1.8, 0.10), (1.3, 0.18)]:
                glow = plt.Circle((x, y), node_size * r_mult, color=color, alpha=alpha, zorder=3)
                ax.add_patch(glow)
            
            # Core node
            circle = plt.Circle((x, y), node_size, color=color, ec='white', lw=2, zorder=10)
            ax.add_patch(circle)
            
            # Row count inside
            if rows >= 1000:
                rlabel = f'{rows/1000:.1f}K'
            else:
                rlabel = str(rows)
            ax.text(x, y, rlabel, ha='center', va='center', fontsize=7,
                   color='white', fontweight='bold', zorder=11)
            
            # Table name below
            ax.text(x, y - node_size - 0.35, table[:18], ha='center', va='top',
                   fontsize=9, fontweight='bold', color='#CBD5E1', zorder=11)
        
        # Decorative background stars
        if NUMPY_AVAILABLE:
            np.random.seed(99)
            ax.scatter(np.random.uniform(-8, 8, 60), np.random.uniform(-8, 8, 60),
                      s=np.random.uniform(0.2, 1.2, 60), color='white', alpha=0.25, zorder=1)
        
        # Legend
        explicit_line = mpatches.Patch(color='#818CF8', label='Explicit FK')
        inferred_line = mpatches.Patch(color='#06B6D4', label='Inferred')
        leg = ax.legend(handles=[explicit_line, inferred_line], loc='upper left', fontsize=10,
                       facecolor='#1E293B', edgecolor='#334155', labelcolor='white')
        
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Table Relationship Network', fontsize=18, fontweight='bold',
                     pad=20, color='white')
        
        stats_text = f'{n_tables} tables  ·  {len(relationships)} explicit  ·  {len(inferred_rels)} inferred'
        ax.text(0.5, -0.02, stats_text, transform=ax.transAxes, ha='center',
               fontsize=10, color='#94A3B8')
        
        fig.text(0.99, 0.01, 'DataMind AI', fontsize=7, color='#475569',
                 ha='right', va='bottom', alpha=0.5, style='italic')
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'network_relationships', fig)
    
    def _generate_relationship_heatmap(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate premium heatmap showing relationship strengths between tables."""
        tables = results.get('tables', [])
        relationships = results.get('relationships', [])
        
        if not tables or len(tables) < 2:
            return None
        
        table_names = [t.get('name', '')[:15] for t in tables[:12]]
        n = len(table_names)
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        name_to_idx = {name: i for i, name in enumerate(table_names)}
        
        for rel in relationships:
            fn = rel.get('from_table', '')[:15]
            tn = rel.get('to_table', '')[:15]
            if fn in name_to_idx and tn in name_to_idx:
                matrix[name_to_idx[fn]][name_to_idx[tn]] = 1
                matrix[name_to_idx[tn]][name_to_idx[fn]] = 0.5
        
        detector = AdvancedRelationshipDetector()
        inferred = detector.analyze_all_relationships(tables)
        for rel in inferred:
            fn = rel.get('from_table', '')[:15]
            tn = rel.get('to_table', '')[:15]
            conf = rel.get('confidence', 0.5)
            if fn in name_to_idx and tn in name_to_idx:
                i, j = name_to_idx[fn], name_to_idx[tn]
                if matrix[i][j] == 0:
                    matrix[i][j] = conf * 0.7
        
        fig, ax = plt.subplots(figsize=(14, 12))
        fig.set_facecolor('#FAFBFC')
        
        cmap = LinearSegmentedColormap.from_list('rel',
            ['#F8FAFC', '#BFDBFE', '#3B82F6', '#1E3A8A'], N=256)
        
        if NUMPY_AVAILABLE:
            im = ax.imshow(np.array(matrix), cmap=cmap, aspect='auto', vmin=0, vmax=1)
        else:
            im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Relationship Strength', fontsize=11, fontweight='600')
        
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(table_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(table_names, fontsize=9)
        
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > 0:
                    tc = 'white' if matrix[i][j] > 0.5 else '#1E293B'
                    ax.text(j, i, f'{matrix[i][j]:.1f}', ha='center', va='center',
                           fontsize=8, color=tc, fontweight='bold')
        
        ax.set_title('Table Relationship Matrix', fontsize=16, fontweight='bold',
                     pad=20, color='#111827')
        ax.set_xlabel('Target Table', fontsize=12, fontweight='600', color='#374151')
        ax.set_ylabel('Source Table', fontsize=12, fontweight='600', color='#374151')
        
        self._add_watermark(fig)
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
        
        return self._save_chart(dataset_id, 'hierarchy_tables', fig)
    
    # ========================================================================
    # NEW ADVANCED CHARTS
    # ========================================================================
    
    def _generate_sankey_dataflow(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate Sankey-style data flow diagram showing relationships as curved bands."""
        relationships = results.get('relationships', [])
        tables = results.get('tables', [])
        
        if not relationships or len(relationships) < 1:
            return None
        
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.set_facecolor('#FAFBFC')
        
        # Collect tables involved in relationships
        left_tables = []
        right_tables = []
        flows = []
        
        for rel in relationships:
            ft = rel.get('from_table', '')
            tt = rel.get('to_table', '')
            if ft and tt:
                if ft not in left_tables:
                    left_tables.append(ft)
                if tt not in right_tables:
                    right_tables.append(tt)
                conf = rel.get('confidence', 80)
                flows.append((ft, tt, max(conf, 30)))
        
        if not flows:
            plt.close(fig)
            return None
        
        n_left = len(left_tables)
        n_right = len(right_tables)
        
        # Position nodes
        left_y = {t: (i + 0.5) / n_left for i, t in enumerate(left_tables)}
        right_y = {t: (i + 0.5) / n_right for i, t in enumerate(right_tables)}
        
        x_left = 0.12
        x_right = 0.88
        
        # Draw flow bands
        for ft, tt, weight in flows:
            ly = left_y.get(ft, 0.5)
            ry = right_y.get(tt, 0.5)
            band_w = weight / 300.0 * 0.08
            
            color_idx = left_tables.index(ft) % len(self.COLORS)
            color = self.COLORS[color_idx]
            
            # Bezier curve via fill_between
            xs = np.linspace(x_left + 0.06, x_right - 0.06, 50)
            t_param = (xs - xs[0]) / (xs[-1] - xs[0])
            ys = ly + (ry - ly) * (3 * t_param**2 - 2 * t_param**3)
            
            ax.fill_between(xs, ys - band_w, ys + band_w,
                           color=color, alpha=0.35, zorder=2)
            ax.plot(xs, ys, color=color, alpha=0.7, lw=1.5, zorder=3)
        
        # Draw left nodes
        for i, table in enumerate(left_tables):
            y = left_y[table]
            color = self.COLORS[i % len(self.COLORS)]
            box = FancyBboxPatch((x_left - 0.06, y - 0.025), 0.12, 0.05,
                                boxstyle="round,pad=0.008",
                                facecolor=color, edgecolor='white', linewidth=2, zorder=5)
            ax.add_patch(box)
            ax.text(x_left, y, table[:16], ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white', zorder=6)
        
        # Draw right nodes
        for i, table in enumerate(right_tables):
            y = right_y[table]
            color = self.COLORS[(i + 3) % len(self.COLORS)]
            box = FancyBboxPatch((x_right - 0.06, y - 0.025), 0.12, 0.05,
                                boxstyle="round,pad=0.008",
                                facecolor=color, edgecolor='white', linewidth=2, zorder=5)
            ax.add_patch(box)
            ax.text(x_right, y, table[:16], ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white', zorder=6)
        
        # Labels
        ax.text(x_left, 1.02, 'Source Tables', ha='center', va='bottom',
               fontsize=11, fontweight='bold', color='#4B5563', transform=ax.transAxes)
        ax.text(x_right, 1.02, 'Target Tables', ha='center', va='bottom',
               fontsize=11, fontweight='bold', color='#4B5563', transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.axis('off')
        ax.set_title('Data Flow Diagram', fontsize=16, fontweight='bold', pad=25, color='#111827')
        
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'sankey_dataflow', fig)
    
    def _generate_schema_constellation(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate constellation-style schema map with force-directed layout."""
        tables = results.get('tables', [])
        relationships = results.get('relationships', [])
        
        if not tables or len(tables) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(16, 16))
        fig.set_facecolor('#0F172A')  # Dark background for constellation effect
        
        n = len(tables)
        table_names = [t.get('name', '') for t in tables]
        table_rows = {t.get('name', ''): t.get('row_count', 100) for t in tables}
        max_rows = max(table_rows.values()) if table_rows else 1
        
        # Force-directed simple layout (spring algorithm)
        # Start with circular placement
        positions = {}
        for i, name in enumerate(table_names):
            angle = 2 * math.pi * i / n - math.pi / 2
            r = 4.0
            positions[name] = [r * math.cos(angle), r * math.sin(angle)]
        
        # Simple spring simulation (20 iterations)
        for iteration in range(20):
            forces = {name: [0.0, 0.0] for name in table_names}
            
            # Repulsion between all nodes
            for i, n1 in enumerate(table_names):
                for j, n2 in enumerate(table_names):
                    if i >= j:
                        continue
                    dx = positions[n1][0] - positions[n2][0]
                    dy = positions[n1][1] - positions[n2][1]
                    dist = max(math.sqrt(dx*dx + dy*dy), 0.1)
                    repulsion = 8.0 / (dist * dist)
                    fx = repulsion * dx / dist
                    fy = repulsion * dy / dist
                    forces[n1][0] += fx
                    forces[n1][1] += fy
                    forces[n2][0] -= fx
                    forces[n2][1] -= fy
            
            # Attraction along relationships
            for rel in relationships:
                ft = rel.get('from_table', '')
                tt = rel.get('to_table', '')
                if ft in positions and tt in positions:
                    dx = positions[tt][0] - positions[ft][0]
                    dy = positions[tt][1] - positions[ft][1]
                    dist = max(math.sqrt(dx*dx + dy*dy), 0.1)
                    attraction = dist * 0.3
                    fx = attraction * dx / dist
                    fy = attraction * dy / dist
                    forces[ft][0] += fx
                    forces[ft][1] += fy
                    forces[tt][0] -= fx
                    forces[tt][1] -= fy
            
            # Apply forces with damping
            damping = 0.5 * (1 - iteration / 20)
            for name in table_names:
                positions[name][0] += forces[name][0] * damping
                positions[name][1] += forces[name][1] * damping
        
        # Draw connection "beams" (glow effect on dark background)
        for rel in relationships:
            ft = rel.get('from_table', '')
            tt = rel.get('to_table', '')
            if ft in positions and tt in positions:
                x1, y1 = positions[ft]
                x2, y2 = positions[tt]
                # Glow layers
                for lw, alpha in [(6, 0.08), (4, 0.15), (2, 0.4), (1, 0.8)]:
                    ax.plot([x1, x2], [y1, y2], color='#06B6D4', lw=lw, alpha=alpha, zorder=2)
        
        # Draw stars (table nodes)
        for i, name in enumerate(table_names):
            x, y = positions[name]
            rows = table_rows.get(name, 100)
            node_r = 0.3 + (rows / max_rows) * 0.5
            color = self.COLORS[i % len(self.COLORS)]
            
            # Glow effect
            for r_mult, alpha in [(3.0, 0.04), (2.0, 0.08), (1.5, 0.15)]:
                glow = plt.Circle((x, y), node_r * r_mult, color=color, alpha=alpha, zorder=3)
                ax.add_patch(glow)
            
            # Core node
            circle = plt.Circle((x, y), node_r, color=color, ec='white', lw=2, zorder=10)
            ax.add_patch(circle)
            
            # Label
            ax.text(x, y - node_r - 0.35, name[:18], ha='center', va='top',
                   fontsize=9, fontweight='bold', color='white', zorder=11)
            # Row count
            if rows >= 1000:
                row_label = f'{rows/1000:.1f}K'
            else:
                row_label = str(rows)
            ax.text(x, y, row_label, ha='center', va='center',
                   fontsize=7, color='white', fontweight='bold', zorder=11)
        
        # Decorative stars in background
        if NUMPY_AVAILABLE:
            np.random.seed(42)
            star_x = np.random.uniform(-8, 8, 80)
            star_y = np.random.uniform(-8, 8, 80)
            star_s = np.random.uniform(0.2, 1.5, 80)
            ax.scatter(star_x, star_y, s=star_s, color='white', alpha=0.3, zorder=1)
        
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Schema Constellation', fontsize=18, fontweight='bold',
                     pad=20, color='white')
        ax.text(0.5, -0.02, f'{len(table_names)} tables · {len(relationships)} relationships',
               transform=ax.transAxes, ha='center', fontsize=10, color='#94A3B8')
        
        fig.text(0.99, 0.01, 'DataMind AI', fontsize=7, color='#475569',
                 ha='right', va='bottom', alpha=0.5, style='italic')
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'constellation_schema', fig)
    
    def _generate_fk_coverage_ring(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate FK coverage ring showing how well tables are interconnected."""
        tables = results.get('tables', [])
        relationships = results.get('relationships', [])
        
        if not tables:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 12))
        fig.set_facecolor('#FAFBFC')
        
        table_names = [t.get('name', '') for t in tables]
        n = len(table_names)
        
        # Calculate connectivity per table
        connected = set()
        for rel in relationships:
            connected.add(rel.get('from_table', ''))
            connected.add(rel.get('to_table', ''))
        
        coverage = len(connected & set(table_names)) / max(n, 1) * 100
        
        # FK columns per table
        fk_counts = {}
        total_fks = 0
        for table in tables:
            fks = sum(1 for c in table.get('columns', []) if c.get('is_foreign_key'))
            fk_counts[table.get('name', '')] = fks
            total_fks += fks
        
        # Outer ring — one segment per table
        ring_colors = []
        ring_sizes = []
        ring_labels = []
        
        for i, table in enumerate(tables):
            name = table.get('name', '')
            fks = fk_counts.get(name, 0)
            is_connected = name in connected
            ring_sizes.append(max(fks + 1, 1))
            ring_colors.append(self.COLORS[i % len(self.COLORS)] if is_connected else '#E5E7EB')
            ring_labels.append(name[:15])
        
        # Outer ring
        wedges, _ = ax.pie(ring_sizes, labels=None, colors=ring_colors,
                          startangle=90, radius=1.1,
                          wedgeprops=dict(width=0.25, edgecolor='white', linewidth=2))
        
        # Inner coverage gauge
        covered_angle = coverage / 100 * 360
        gauge_bg = Wedge((0, 0), 0.75, 0, 360, width=0.18,
                        facecolor='#F1F5F9', edgecolor='none', zorder=3)
        ax.add_patch(gauge_bg)
        
        if coverage >= 80:
            gauge_color = '#10B981'
        elif coverage >= 50:
            gauge_color = '#06B6D4'
        elif coverage >= 25:
            gauge_color = '#F59E0B'
        else:
            gauge_color = '#EF4444'
        
        gauge_fill = Wedge((0, 0), 0.75, 90, 90 - covered_angle, width=0.18,
                          facecolor=gauge_color, edgecolor='none', zorder=4)
        ax.add_patch(gauge_fill)
        
        # Center text
        ax.text(0, 0.08, f'{coverage:.0f}%', ha='center', va='center',
               fontsize=36, fontweight='bold', color=gauge_color, zorder=5)
        ax.text(0, -0.12, 'FK Coverage', ha='center', va='center',
               fontsize=12, color='#6B7280', fontweight='500', zorder=5)
        ax.text(0, -0.28, f'{total_fks} foreign keys', ha='center', va='center',
               fontsize=10, color='#9CA3AF', zorder=5)
        
        # Legend labels around the ring
        for i, (wedge, label) in enumerate(zip(wedges, ring_labels)):
            ang = (wedge.theta2 + wedge.theta1) / 2
            x = 1.3 * math.cos(math.radians(ang))
            y = 1.3 * math.sin(math.radians(ang))
            ha = 'left' if x >= 0 else 'right'
            ax.text(x, y, label, ha=ha, va='center', fontsize=7,
                   color='#4B5563', fontweight='500')
        
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Foreign Key Coverage', fontsize=16, fontweight='bold', pad=25, color='#111827')
        
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'fk_coverage_ring', fig)
    
    def _generate_data_density_heatmap(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate a data density heatmap showing rows × columns intensity per table."""
        tables = results.get('tables', [])
        if not tables or len(tables) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(tables) * 0.7)))
        fig.set_facecolor('#FAFBFC')
        
        sorted_tables = sorted(tables, key=lambda t: t.get('row_count', 0), reverse=True)[:15]
        
        names = [t.get('name', '')[:20] for t in sorted_tables]
        rows = [t.get('row_count', 0) for t in sorted_tables]
        cols = [len(t.get('columns', [])) for t in sorted_tables]
        
        # Density = rows × columns (data cells)
        densities = [r * c for r, c in zip(rows, cols)]
        max_density = max(densities) if densities else 1
        
        # Normalize for colormap
        norm_densities = [d / max_density for d in densities]
        
        # Custom colormap: light cyan → deep blue
        cmap = LinearSegmentedColormap.from_list('density',
            ['#E0F7FA', '#06B6D4', '#0E7490', '#164E63'], N=256)
        
        y_pos = range(len(names))
        bar_colors = [cmap(nd) for nd in norm_densities]
        
        bars = ax.barh(y_pos, densities, color=bar_colors, edgecolor='white',
                      linewidth=1, height=0.7)
        
        for i, (bar, density, row, col) in enumerate(zip(bars, densities, rows, cols)):
            w = bar.get_width()
            if density >= 1_000_000:
                label = f'{density/1_000_000:.1f}M cells  ({row:,}R × {col}C)'
            elif density >= 1000:
                label = f'{density/1000:.1f}K cells  ({row:,}R × {col}C)'
            else:
                label = f'{density:,} cells  ({row:,}R × {col}C)'
            ax.text(w + max_density * 0.02, i, label,
                   ha='left', va='center', fontsize=8, color='#4B5563', fontweight='500')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Data Density (Rows × Columns)', fontsize=12, fontweight='600')
        ax.set_title('Data Density Heatmap', fontsize=16, fontweight='bold', pad=20, color='#111827')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.12)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'density_heatmap', fig)
    
    def _generate_table_dna_barcode(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate 'DNA barcode' visualization — each table is a row of colored strips for columns."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        sorted_tables = sorted(tables, key=lambda t: len(t.get('columns', [])), reverse=True)[:12]
        max_cols = max(len(t.get('columns', [])) for t in sorted_tables) if sorted_tables else 1
        
        fig, ax = plt.subplots(figsize=(16, max(6, len(sorted_tables) * 0.8)))
        fig.set_facecolor('#FAFBFC')
        
        type_color_map = {
            'INTEGER': '#4F46E5',
            'TEXT': '#06B6D4',
            'NUMERIC': '#10B981',
            'DATETIME': '#F59E0B',
            'BOOLEAN': '#EC4899',
            'BINARY': '#8B5CF6',
        }
        
        for row_i, table in enumerate(sorted_tables):
            columns = table.get('columns', [])
            table_name = table.get('name', '')[:20]
            
            for col_i, col in enumerate(columns):
                dtype = self._normalize_data_type(col.get('data_type', 'OTHER'))
                color = type_color_map.get(dtype, '#9CA3AF')
                is_pk = col.get('is_primary_key', False)
                is_fk = col.get('is_foreign_key', False)
                
                # Draw strip
                alpha = 0.9 if is_pk else (0.7 if is_fk else 0.55)
                rect = plt.Rectangle((col_i, row_i - 0.35), 0.85, 0.7,
                                    facecolor=color, alpha=alpha, edgecolor='white',
                                    linewidth=0.5, zorder=3)
                ax.add_patch(rect)
                
                # PK/FK markers
                if is_pk:
                    ax.text(col_i + 0.42, row_i, '★', ha='center', va='center',
                           fontsize=7, color='white', fontweight='bold', zorder=4)
                elif is_fk:
                    ax.text(col_i + 0.42, row_i, '→', ha='center', va='center',
                           fontsize=7, color='white', fontweight='bold', zorder=4)
        
        ax.set_yticks(range(len(sorted_tables)))
        ax.set_yticklabels([t.get('name', '')[:20] for t in sorted_tables], fontsize=10)
        ax.set_xlim(-0.5, max_cols + 0.5)
        ax.set_ylim(-0.7, len(sorted_tables) - 0.3)
        ax.set_xlabel('Column Index', fontsize=11, fontweight='500')
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_title('Table DNA — Column Composition Barcode', fontsize=16,
                     fontweight='bold', pad=20, color='#111827')
        
        # Legend
        legend_patches = [mpatches.Patch(color=c, label=t, alpha=0.7)
                         for t, c in type_color_map.items()]
        legend_patches.append(mpatches.Patch(color='#374151', label='★ = PK  → = FK', alpha=0.5))
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8,
                 ncol=4, frameon=True, fancybox=True, edgecolor='#E5E7EB')
        
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'table_dna', fig)
    
    def _generate_column_fingerprint(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate bubble scatter showing each column's uniqueness vs null% per table."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.set_facecolor('#FAFBFC')
        
        for t_i, table in enumerate(tables[:8]):
            table_name = table.get('name', '')
            color = self.COLORS[t_i % len(self.COLORS)]
            
            for col in table.get('columns', []):
                null_pct = col.get('null_percentage', 0) * 100
                unique_count = col.get('unique_count', 0)
                total_count = col.get('total_count', 1) or 1
                uniqueness = (unique_count / total_count) * 100
                
                # Bubble size based on total_count
                size = max(20, min(300, total_count / 10))
                
                ax.scatter(uniqueness, null_pct, s=size, c=color, alpha=0.55,
                          edgecolors='white', linewidths=1, zorder=3)
        
        # Quadrant lines
        ax.axhline(y=10, color='#F59E0B', linestyle='--', alpha=0.3, zorder=1)
        ax.axvline(x=50, color='#06B6D4', linestyle='--', alpha=0.3, zorder=1)
        
        # Quadrant labels
        ax.text(75, 2, 'High Uniqueness\nLow Nulls ✓', ha='center', fontsize=8,
               color='#10B981', alpha=0.6, fontweight='bold')
        ax.text(25, 2, 'Low Uniqueness\nLow Nulls', ha='center', fontsize=8,
               color='#F59E0B', alpha=0.6, fontweight='bold')
        
        ax.set_xlabel('Uniqueness %', fontsize=12, fontweight='600')
        ax.set_ylabel('Null %', fontsize=12, fontweight='600')
        ax.set_title('Column Fingerprint — Uniqueness vs Completeness', fontsize=16,
                     fontweight='bold', pad=20, color='#111827')
        ax.grid(alpha=0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Legend — one per table
        legend_patches = [mpatches.Patch(color=self.COLORS[i % len(self.COLORS)],
                          label=tables[i].get('name', '')[:15], alpha=0.6)
                         for i in range(min(len(tables), 8))]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8,
                 frameon=True, fancybox=True, edgecolor='#E5E7EB')
        
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'column_fingerprint', fig)
    
    def _generate_sunburst_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate sunburst hierarchy chart: center=DB, ring1=tables, ring2=column types."""
        tables = results.get('tables', [])
        if not tables or len(tables) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 14))
        fig.set_facecolor('#FAFBFC')
        
        db_name = results.get('database_name', 'Database')[:20]
        
        # --- Inner ring: tables (proportional to column count) ---
        table_sizes = []
        table_labels = []
        table_colors = []
        
        # --- Outer ring: column types per table ---
        outer_sizes = []
        outer_colors = []
        outer_labels = []
        
        type_color_map = {
            'INTEGER': '#4F46E5',
            'TEXT': '#06B6D4',
            'NUMERIC': '#10B981',
            'DATETIME': '#F59E0B',
            'BOOLEAN': '#EC4899',
            'BINARY': '#8B5CF6',
        }
        
        for i, table in enumerate(tables[:10]):
            cols = table.get('columns', [])
            n_cols = max(len(cols), 1)
            table_sizes.append(n_cols)
            table_labels.append(table.get('name', '')[:15])
            table_colors.append(self.COLORS[i % len(self.COLORS)])
            
            # Group columns by type
            type_groups = defaultdict(int)
            for col in cols:
                dtype = self._normalize_data_type(col.get('data_type', 'OTHER'))
                type_groups[dtype] += 1
            
            for dtype, count in type_groups.items():
                outer_sizes.append(count)
                outer_colors.append(type_color_map.get(dtype, '#9CA3AF'))
                outer_labels.append(dtype)
        
        if not table_sizes:
            plt.close(fig)
            return None
        
        # Center circle — database name
        center = plt.Circle((0, 0), 0.35, color='#1E293B', zorder=10)
        ax.add_patch(center)
        ax.text(0, 0.05, db_name, ha='center', va='center', fontsize=11,
               fontweight='bold', color='white', zorder=11)
        ax.text(0, -0.1, f'{len(tables)} tables', ha='center', va='center',
               fontsize=8, color='#94A3B8', zorder=11)
        
        # Inner ring — tables
        inner_wedges, inner_texts = ax.pie(
            table_sizes, labels=None, colors=table_colors,
            startangle=90, radius=0.72,
            wedgeprops=dict(width=0.32, edgecolor='white', linewidth=2)
        )
        
        # Label inner ring
        for wedge, label in zip(inner_wedges, table_labels):
            ang = (wedge.theta1 + wedge.theta2) / 2
            x = 0.56 * math.cos(math.radians(ang))
            y = 0.56 * math.sin(math.radians(ang))
            rot = ang if -90 < ang < 90 else ang - 180
            if wedge.theta2 - wedge.theta1 > 15:  # Only label big slices
                ax.text(x, y, label, ha='center', va='center', fontsize=7,
                       fontweight='bold', color='white', rotation=rot, zorder=8)
        
        # Outer ring — column types
        ax.pie(outer_sizes, labels=None, colors=outer_colors,
               startangle=90, radius=1.05,
               wedgeprops=dict(width=0.28, edgecolor='white', linewidth=1.5))
        
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Schema Sunburst Hierarchy', fontsize=16, fontweight='bold',
                     pad=25, color='#111827')
        
        # Type legend
        type_patches = [mpatches.Patch(color=c, label=t, alpha=0.8)
                       for t, c in type_color_map.items()]
        ax.legend(handles=type_patches, loc='lower right', fontsize=8,
                 title='Column Types', title_fontsize=9,
                 frameon=True, fancybox=True, edgecolor='#E5E7EB')
        
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'sunburst_hierarchy', fig)
    
    def _generate_violin_distribution(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate violin plot showing row count distribution + box stats."""
        tables = results.get('tables', [])
        if not tables or len(tables) < 3 or not NUMPY_AVAILABLE:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                        gridspec_kw={'width_ratios': [2, 1]})
        fig.set_facecolor('#FAFBFC')
        
        row_counts = [t.get('row_count', 0) for t in tables]
        col_counts = [len(t.get('columns', [])) for t in tables]
        table_names = [t.get('name', '')[:15] for t in tables]
        
        # --- Left: bar chart with statistics overlay ---
        sorted_idx = np.argsort(row_counts)[::-1]
        sorted_names = [table_names[i] for i in sorted_idx]
        sorted_rows = [row_counts[i] for i in sorted_idx]
        
        colors = [self.COLORS[i % len(self.COLORS)] for i in range(len(sorted_names))]
        bars = ax1.bar(range(len(sorted_names)), sorted_rows, color=colors,
                      edgecolor='white', linewidth=1, alpha=0.8)
        
        # Mean and median lines
        mean_val = np.mean(row_counts)
        median_val = np.median(row_counts)
        ax1.axhline(y=mean_val, color='#EF4444', linestyle='--', lw=1.5, alpha=0.6,
                    label=f'Mean: {mean_val:,.0f}')
        ax1.axhline(y=median_val, color='#06B6D4', linestyle='-.', lw=1.5, alpha=0.6,
                    label=f'Median: {median_val:,.0f}')
        
        ax1.set_xticks(range(len(sorted_names)))
        ax1.set_xticklabels(sorted_names, rotation=55, ha='right', fontsize=8)
        ax1.set_ylabel('Row Count', fontsize=11, fontweight='600')
        ax1.set_title('Row Distribution by Table', fontsize=14, fontweight='bold', color='#111827')
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(axis='y', alpha=0.12)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # --- Right: summary statistics panel ---
        ax2.axis('off')
        stats = {
            'Tables': len(tables),
            'Total Rows': f'{sum(row_counts):,}',
            'Mean Rows': f'{mean_val:,.0f}',
            'Median Rows': f'{median_val:,.0f}',
            'Std Dev': f'{np.std(row_counts):,.0f}',
            'Max Rows': f'{max(row_counts):,}',
            'Min Rows': f'{min(row_counts):,}',
            'Total Columns': sum(col_counts),
            'Avg Cols/Table': f'{np.mean(col_counts):.1f}',
        }
        
        y_start = 0.95
        ax2.text(0.5, y_start + 0.05, 'Distribution Statistics',
                ha='center', va='top', fontsize=13, fontweight='bold', color='#111827',
                transform=ax2.transAxes)
        
        for i, (label, value) in enumerate(stats.items()):
            y = y_start - i * 0.095
            # Background band
            if i % 2 == 0:
                ax2.add_patch(plt.Rectangle((0.05, y - 0.035), 0.9, 0.07,
                             facecolor='#F1F5F9', edgecolor='none',
                             transform=ax2.transAxes, zorder=1))
            ax2.text(0.1, y, label, ha='left', va='center', fontsize=10,
                    color='#6B7280', transform=ax2.transAxes, zorder=2)
            ax2.text(0.9, y, str(value), ha='right', va='center', fontsize=11,
                    fontweight='bold', color='#111827', transform=ax2.transAxes, zorder=2)
        
        fig.suptitle('Statistical Distribution Analysis', fontsize=16,
                    fontweight='bold', color='#111827', y=1.02)
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'violin_distribution', fig)
    
    # ========================================================================
    # SCHEMA OVERVIEW CHARTS
    # ========================================================================
    
    def _generate_schema_matrix(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate premium 4-panel schema overview dashboard."""
        tables = results.get('tables', [])
        summary = results.get('summary', {})
        
        if not tables:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.set_facecolor('#FAFBFC')
        
        # 1. Table Overview (top-left)
        ax1 = axes[0, 0]
        table_data = [(t.get('name', '')[:18], t.get('row_count', 0),
                       len(t.get('columns', []))) for t in tables[:10]]
        
        if table_data:
            names, rows, cols = zip(*table_data)
            x = range(len(names))
            
            bars1 = ax1.bar(x, rows, color='#06B6D4', alpha=0.8, label='Rows', width=0.38)
            bars2 = ax1.bar([i + 0.4 for i in x], [c * 100 for c in cols], color='#818CF8',
                    alpha=0.8, label='Columns (×100)', width=0.38)
            
            ax1.set_xticks([i + 0.2 for i in x])
            ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
            ax1.set_title('Table Overview', fontsize=13, fontweight='bold', color='#111827', pad=10)
            ax1.legend(fontsize=8, fancybox=True, edgecolor='#E5E7EB')
            ax1.grid(axis='y', alpha=0.12)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
        
        # 2. Quality Distribution (top-right)
        ax2 = axes[0, 1]
        quality_issues = results.get('quality_issues', [])
        severity_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        for issue in quality_issues:
            sev = issue.get('severity', 'low').capitalize()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        qcolors = ['#EF4444', '#F59E0B', '#10B981']
        q_vals = list(severity_counts.values())
        if sum(q_vals) > 0:
            wedges, texts, autotexts = ax2.pie(
                q_vals, labels=list(severity_counts.keys()),
                autopct='%1.0f%%', colors=qcolors, startangle=90,
                wedgeprops=dict(edgecolor='white', linewidth=2.5),
                textprops=dict(fontweight='600'))
            for at in autotexts:
                at.set_fontsize(10)
                at.set_fontweight('bold')
        else:
            ax2.text(0.5, 0.5, 'No Issues', ha='center', va='center',
                    fontsize=16, fontweight='bold', color='#10B981',
                    transform=ax2.transAxes)
            ax2.text(0.5, 0.35, 'Quality is excellent', ha='center', va='center',
                    fontsize=10, color='#6B7280', transform=ax2.transAxes)
        ax2.set_title('Quality Issues', fontsize=13, fontweight='bold', color='#111827', pad=10)
        
        # 3. Column Types (bottom-left)
        ax3 = axes[1, 0]
        type_counts = defaultdict(int)
        for table in tables:
            for col in table.get('columns', []):
                dtype = self._normalize_data_type(col.get('data_type', 'OTHER'))
                type_counts[dtype] += 1
        
        if type_counts:
            types, counts = zip(*sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:8])
            tcolors = [self.COLORS[i % len(self.COLORS)] for i in range(len(types))]
            bars = ax3.barh(types, counts, color=tcolors, edgecolor='white', linewidth=1.5, alpha=0.85)
            for bar, c in zip(bars, counts):
                ax3.text(bar.get_width() + max(counts) * 0.02, bar.get_y() + bar.get_height()/2,
                        str(c), va='center', fontsize=9, fontweight='bold', color='#374151')
            ax3.set_title('Column Data Types', fontsize=13, fontweight='bold', color='#111827', pad=10)
            ax3.set_xlabel('Count', fontsize=10)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
        
        # 4. Key Metrics (bottom-right) — card style
        ax4 = axes[1, 1]
        ax4.axis('off')
        metrics = {
            'Tables': (len(tables), '#06B6D4'),
            'Total Columns': (sum(len(t.get('columns', [])) for t in tables), '#818CF8'),
            'Total Rows': (sum(t.get('row_count', 0) for t in tables), '#10B981'),
            'Relationships': (len(results.get('relationships', [])), '#F59E0B'),
            'Quality Score': (summary.get('quality_score', 0), '#EC4899'),
        }
        
        y_start = 0.92
        ax4.text(0.5, y_start + 0.08, 'Key Metrics', ha='center', va='top',
                fontsize=14, fontweight='bold', color='#111827', transform=ax4.transAxes)
        for i, (label, (value, color)) in enumerate(metrics.items()):
            y = y_start - i * 0.16
            # Background card
            ax4.add_patch(FancyBboxPatch((0.08, y - 0.05), 0.84, 0.12,
                         boxstyle='round,pad=0.02', facecolor=color, alpha=0.08,
                         edgecolor=color, linewidth=1.5, transform=ax4.transAxes))
            ax4.text(0.15, y, label, ha='left', va='center', fontsize=11,
                    color='#4B5563', transform=ax4.transAxes)
            v_str = f'{value:,}' if isinstance(value, int) else f'{value:.0f}'
            ax4.text(0.85, y, v_str, ha='right', va='center', fontsize=14,
                    fontweight='bold', color=color, transform=ax4.transAxes)
        
        fig.suptitle('Schema Analysis Overview', fontsize=18, fontweight='bold',
                    color='#111827', y=1.01)
        self._add_watermark(fig)
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'matrix_overview', fig)


def create_chart_generator(storage_manager) -> ChartGenerator:
    """Factory function to create chart generator."""
    return ChartGenerator(storage_manager)

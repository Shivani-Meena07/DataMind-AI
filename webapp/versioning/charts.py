"""
DataMind AI - Chart Generation Module
Generates deterministic, reproducible charts from analysis results

All charts are derived artifacts stored under:
outputs/<dataset_id>/charts/

Naming convention:
- bar_<metric>.png
- pie_<distribution>.png  
- line_<trend>.png
- heatmap_<correlation>.png
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import io

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class ChartGenerator:
    """
    Generates deterministic charts from analysis results.
    
    Features:
    - Consistent styling across all charts
    - Deterministic output (same data = same chart)
    - Multiple chart types supported
    - Automatic chart selection based on data
    """
    
    # Color palette (consistent across charts)
    COLORS = [
        '#4F46E5',  # Indigo
        '#10B981',  # Emerald
        '#F59E0B',  # Amber
        '#EF4444',  # Red
        '#8B5CF6',  # Violet
        '#06B6D4',  # Cyan
        '#F97316',  # Orange
        '#EC4899',  # Pink
        '#84CC16',  # Lime
        '#6366F1',  # Indigo light
    ]
    
    # Chart style settings
    STYLE = {
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#E5E7EB',
        'axes.labelcolor': '#374151',
        'axes.titleweight': 'bold',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.color': '#6B7280',
        'ytick.color': '#6B7280',
        'grid.color': '#F3F4F6',
        'grid.linestyle': '-',
        'grid.alpha': 0.8,
        'font.family': 'sans-serif',
    }
    
    def __init__(self, storage_manager=None):
        """
        Initialize chart generator.
        
        Args:
            storage_manager: StorageManager instance for storing charts
        """
        self.storage = storage_manager
        
        if MATPLOTLIB_AVAILABLE:
            plt.rcParams.update(self.STYLE)
    
    def generate_all_charts(self, dataset_id: str, analysis_results: Dict) -> List[str]:
        """
        Generate all applicable charts for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            analysis_results: Analysis results dictionary
            
        Returns:
            List of generated chart filenames
        """
        if not MATPLOTLIB_AVAILABLE:
            return []
        
        charts = []
        
        # Generate table size distribution (bar chart)
        chart = self._generate_table_size_chart(dataset_id, analysis_results)
        if chart:
            charts.append(chart)
        
        # Generate table type distribution (pie chart)
        chart = self._generate_table_type_chart(dataset_id, analysis_results)
        if chart:
            charts.append(chart)
        
        # Generate quality score gauge
        chart = self._generate_quality_gauge(dataset_id, analysis_results)
        if chart:
            charts.append(chart)
        
        # Generate column type distribution
        chart = self._generate_column_type_chart(dataset_id, analysis_results)
        if chart:
            charts.append(chart)
        
        # Generate relationship diagram
        chart = self._generate_relationship_chart(dataset_id, analysis_results)
        if chart:
            charts.append(chart)
        
        # Generate null distribution chart
        chart = self._generate_null_distribution_chart(dataset_id, analysis_results)
        if chart:
            charts.append(chart)
        
        # Generate data quality issues chart
        chart = self._generate_quality_issues_chart(dataset_id, analysis_results)
        if chart:
            charts.append(chart)
        
        return charts
    
    def _save_chart(self, dataset_id: str, chart_name: str, fig) -> Optional[str]:
        """Save chart to storage."""
        if not self.storage:
            return None
        
        # Convert to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        chart_data = buf.read()
        plt.close(fig)
        
        # Store chart
        self.storage.store_chart(dataset_id, chart_data, chart_name)
        
        return f"{chart_name}.png"
    
    def _generate_table_size_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate bar chart of table row counts."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        # Sort by row count
        sorted_tables = sorted(tables, key=lambda t: t.get('row_count', 0), reverse=True)[:10]
        
        names = [t.get('name', 'Unknown')[:20] for t in sorted_tables]
        counts = [t.get('row_count', 0) for t in sorted_tables]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(names, counts, color=self.COLORS[0], edgecolor='white', linewidth=1)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width + max(counts) * 0.02, bar.get_y() + bar.get_height()/2,
                   f'{count:,}', ha='left', va='center', fontsize=10, color='#374151')
        
        ax.set_xlabel('Number of Rows')
        ax.set_title('Table Size Distribution')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'bar_table_size_distribution', fig)
    
    def _generate_table_type_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate pie chart of table types."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        # Count table types
        type_counts = {}
        for table in tables:
            table_type = table.get('table_type', 'unknown')
            type_counts[table_type] = type_counts.get(table_type, 0) + 1
        
        if not type_counts:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        colors = self.COLORS[:len(labels)]
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.02] * len(sizes)
        )
        
        # Style
        for text in texts:
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        ax.set_title('Table Type Distribution')
        ax.axis('equal')
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'pie_table_type_distribution', fig)
    
    def _generate_quality_gauge(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate quality score gauge chart."""
        summary = results.get('summary', {})
        quality_score = summary.get('quality_score', 0)
        
        fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={'projection': 'polar'})
        
        # Create semi-circle gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones(100)
        
        # Background arc
        ax.plot(theta, r, color='#E5E7EB', linewidth=25, solid_capstyle='round')
        
        # Score arc
        score_theta = np.linspace(0, np.pi * (quality_score / 100), 50)
        score_r = np.ones(50)
        
        # Color based on score
        if quality_score >= 80:
            color = '#10B981'  # Green
        elif quality_score >= 60:
            color = '#F59E0B'  # Amber
        else:
            color = '#EF4444'  # Red
        
        ax.plot(score_theta, score_r, color=color, linewidth=25, solid_capstyle='round')
        
        # Add score text
        ax.text(np.pi/2, 0.3, f'{quality_score}', ha='center', va='center',
               fontsize=48, fontweight='bold', color=color)
        ax.text(np.pi/2, 0.0, 'Quality Score', ha='center', va='center',
               fontsize=14, color='#6B7280')
        
        ax.set_rticks([])
        ax.set_thetagrids([])
        ax.spines['polar'].set_visible(False)
        ax.set_ylim(0, 1.2)
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'gauge_quality_score', fig)
    
    def _generate_column_type_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate bar chart of column data types."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        # Count data types
        type_counts = {}
        for table in tables:
            for col in table.get('columns', []):
                data_type = col.get('data_type', 'UNKNOWN').upper()
                # Normalize types
                if 'INT' in data_type:
                    data_type = 'INTEGER'
                elif 'CHAR' in data_type or 'TEXT' in data_type:
                    data_type = 'TEXT'
                elif 'REAL' in data_type or 'FLOAT' in data_type or 'DOUBLE' in data_type:
                    data_type = 'REAL'
                elif 'DATE' in data_type or 'TIME' in data_type:
                    data_type = 'DATETIME'
                
                type_counts[data_type] = type_counts.get(data_type, 0) + 1
        
        if not type_counts:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors = self.COLORS[:len(types)]
        
        bars = ax.bar(types, counts, color=colors, edgecolor='white', linewidth=1)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max(counts) * 0.02,
                   str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Data Type')
        ax.set_ylabel('Number of Columns')
        ax.set_title('Column Data Type Distribution')
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'bar_column_type_distribution', fig)
    
    def _generate_relationship_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate relationship overview chart."""
        relationships = results.get('relationships', [])
        tables = results.get('tables', [])
        
        if not relationships or not tables:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique tables involved in relationships
        involved_tables = set()
        for rel in relationships:
            involved_tables.add(rel.get('from_table'))
            involved_tables.add(rel.get('to_table'))
        
        # Position tables in a circle
        n_tables = len(involved_tables)
        if n_tables < 2:
            return None
        
        table_list = sorted(list(involved_tables))
        angles = np.linspace(0, 2 * np.pi, n_tables, endpoint=False)
        
        radius = 3
        positions = {}
        for i, table in enumerate(table_list):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            positions[table] = (x, y)
        
        # Draw connections
        for rel in relationships:
            from_pos = positions.get(rel.get('from_table'))
            to_pos = positions.get(rel.get('to_table'))
            
            if from_pos and to_pos:
                color = '#4F46E5' if rel.get('is_explicit', True) else '#9CA3AF'
                linestyle = '-' if rel.get('is_explicit', True) else '--'
                
                ax.annotate('', xy=to_pos, xytext=from_pos,
                           arrowprops=dict(arrowstyle='->', color=color,
                                          linestyle=linestyle, lw=2))
        
        # Draw table nodes
        for table, (x, y) in positions.items():
            circle = plt.Circle((x, y), 0.5, color='#4F46E5', ec='white', lw=2)
            ax.add_patch(circle)
            ax.text(x, y - 0.9, table[:15], ha='center', va='top', fontsize=9,
                   fontweight='bold')
        
        # Legend
        explicit_patch = mpatches.Patch(color='#4F46E5', label='Explicit FK')
        inferred_patch = mpatches.Patch(color='#9CA3AF', label='Inferred')
        ax.legend(handles=[explicit_patch, inferred_patch], loc='upper right')
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Table Relationships')
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'diagram_relationships', fig)
    
    def _generate_null_distribution_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate chart showing null value distribution."""
        tables = results.get('tables', [])
        if not tables:
            return None
        
        # Collect columns with significant null percentages
        null_data = []
        for table in tables:
            for col in table.get('columns', []):
                null_pct = col.get('null_percentage', 0)
                if null_pct > 0:
                    null_data.append({
                        'table': table.get('name', 'Unknown'),
                        'column': col.get('name', 'Unknown'),
                        'null_pct': null_pct
                    })
        
        if not null_data:
            return None
        
        # Sort and take top 15
        null_data.sort(key=lambda x: x['null_pct'], reverse=True)
        null_data = null_data[:15]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = [f"{d['table']}.{d['column']}"[:30] for d in null_data]
        values = [d['null_pct'] for d in null_data]
        
        # Color based on severity
        colors = []
        for v in values:
            if v > 80:
                colors.append('#EF4444')  # Red
            elif v > 50:
                colors.append('#F59E0B')  # Amber
            else:
                colors.append('#10B981')  # Green
        
        bars = ax.barh(labels, values, color=colors, edgecolor='white', linewidth=1)
        
        # Add percentage labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Null Percentage')
        ax.set_title('Columns with Null Values')
        ax.set_xlim(0, 110)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add threshold line
        ax.axvline(x=50, color='#F59E0B', linestyle='--', alpha=0.7, label='50% threshold')
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'bar_null_distribution', fig)
    
    def _generate_quality_issues_chart(self, dataset_id: str, results: Dict) -> Optional[str]:
        """Generate chart of quality issues by severity."""
        quality_issues = results.get('quality_issues', [])
        
        if not quality_issues:
            return None
        
        # Count by severity
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for issue in quality_issues:
            severity = issue.get('severity', 'low')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        severities = ['high', 'medium', 'low']
        counts = [severity_counts[s] for s in severities]
        colors = ['#EF4444', '#F59E0B', '#10B981']
        
        bars = ax.bar(severities, counts, color=colors, edgecolor='white', linewidth=2)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                       str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Severity Level')
        ax.set_ylabel('Number of Issues')
        ax.set_title('Data Quality Issues by Severity')
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Capitalize labels
        ax.set_xticklabels(['High', 'Medium', 'Low'])
        
        plt.tight_layout()
        
        return self._save_chart(dataset_id, 'bar_quality_issues_severity', fig)


def create_chart_generator(storage_manager) -> ChartGenerator:
    """Factory function to create chart generator."""
    return ChartGenerator(storage_manager)

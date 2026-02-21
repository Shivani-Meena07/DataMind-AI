"""
Intelligence Store

Central data structure that holds all analyzed information about
the database schema, relationships, data quality, and inferred semantics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from datetime import datetime
import json


class ColumnType(Enum):
    """Semantic column type classification."""
    IDENTIFIER = "identifier"           # Primary/foreign keys, unique IDs
    METRIC = "metric"                   # Numeric measures (revenue, counts)
    DIMENSION = "dimension"             # Categorical data (status, type)
    TIMESTAMP = "timestamp"             # Date/time columns
    TEXT = "text"                       # Free-form text
    BOOLEAN = "boolean"                 # True/false flags
    GEOGRAPHIC = "geographic"           # Location data
    FINANCIAL = "financial"             # Currency/money
    UNKNOWN = "unknown"


class TableType(Enum):
    """Semantic table type classification."""
    FACT = "fact"                       # Transaction/event tables
    DIMENSION = "dimension"             # Reference/lookup tables
    BRIDGE = "bridge"                   # Many-to-many junction tables
    SNAPSHOT = "snapshot"               # Point-in-time snapshots
    STAGING = "staging"                 # ETL staging tables
    AUDIT = "audit"                     # Audit/logging tables
    CONFIGURATION = "configuration"     # Config/settings tables
    UNKNOWN = "unknown"


class DataQualitySeverity(Enum):
    """Severity levels for data quality issues."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ColumnProfile:
    """Detailed profile of a single column."""
    # Basic metadata
    name: str
    data_type: str
    nullable: bool
    default_value: Optional[str] = None
    
    # Key information
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references_table: Optional[str] = None
    references_column: Optional[str] = None
    
    # Semantic classification
    semantic_type: ColumnType = ColumnType.UNKNOWN
    
    # Statistics
    total_count: int = 0
    null_count: int = 0
    distinct_count: int = 0
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    avg_value: Optional[float] = None
    
    # Data quality metrics
    null_percentage: float = 0.0
    uniqueness_ratio: float = 0.0
    
    # Sample values
    sample_values: List[Any] = field(default_factory=list)
    most_common_values: List[tuple] = field(default_factory=list)  # (value, count)
    
    # LLM-inferred information
    business_description: Optional[str] = None
    business_relevance: Optional[str] = None
    common_usage: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_sparse(self) -> bool:
        """Check if column has high null percentage."""
        return self.null_percentage > 0.5
    
    @property
    def is_likely_identifier(self) -> bool:
        """Check if column is likely an identifier."""
        return self.uniqueness_ratio > 0.95 or self.is_primary_key


@dataclass
class TableProfile:
    """Detailed profile of a single table."""
    # Basic metadata
    name: str
    schema: Optional[str] = None
    
    # Row count
    row_count: int = 0
    
    # Columns
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)
    
    # Keys
    primary_key: Optional[List[str]] = None
    foreign_keys: List[Dict[str, Any]] = field(default_factory=list)
    
    # Indexes
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Constraints
    unique_constraints: List[Dict[str, Any]] = field(default_factory=list)
    check_constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Semantic classification
    table_type: TableType = TableType.UNKNOWN
    
    # LLM-inferred information
    business_entity: Optional[str] = None
    business_description: Optional[str] = None
    business_purpose: Optional[str] = None
    primary_users: Optional[str] = None
    update_frequency: Optional[str] = None
    
    # Data quality summary
    overall_quality_score: float = 0.0
    quality_issues: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def column_count(self) -> int:
        """Get number of columns."""
        return len(self.columns)
    
    @property
    def has_primary_key(self) -> bool:
        """Check if table has a primary key."""
        return self.primary_key is not None and len(self.primary_key) > 0


@dataclass
class Relationship:
    """Represents a relationship between two tables."""
    # Source table
    source_table: str
    source_columns: List[str]
    
    # Target table
    target_table: str
    target_columns: List[str]
    
    # Relationship type
    cardinality: str = "unknown"  # "1:1", "1:N", "N:1", "N:M"
    
    # Relationship direction
    is_identifying: bool = False  # Part of child's primary key
    
    # LLM-inferred information
    business_description: Optional[str] = None
    data_flow_description: Optional[str] = None


@dataclass
class DataQualityIssue:
    """Represents a data quality issue."""
    table: str
    column: Optional[str]
    severity: DataQualitySeverity
    issue_type: str
    description: str
    recommendation: Optional[str] = None
    affected_rows: Optional[int] = None


@dataclass
class BusinessInsight:
    """Business insight derived from the database."""
    category: str  # "KPI", "UseCase", "Analytics", "Warning"
    title: str
    description: str
    related_tables: List[str] = field(default_factory=list)
    related_columns: List[str] = field(default_factory=list)
    sample_query: Optional[str] = None
    priority: str = "medium"  # "low", "medium", "high"


class IntelligenceStore:
    """
    Central repository for all database intelligence.
    
    This is the backbone of the DataMind system, storing all
    analyzed information in a structured, queryable format.
    """
    
    def __init__(self):
        """Initialize the intelligence store."""
        # Database metadata
        self.database_name: Optional[str] = None
        self.database_type: Optional[str] = None
        self.database_version: Optional[str] = None
        self.analysis_timestamp: datetime = datetime.now()
        
        # Tables
        self.tables: Dict[str, TableProfile] = {}
        
        # Relationships
        self.relationships: List[Relationship] = []
        
        # Data quality
        self.quality_issues: List[DataQualityIssue] = []
        self.overall_quality_score: float = 0.0
        
        # Business insights
        self.insights: List[BusinessInsight] = []
        
        # Dependency graph (for topological sorting)
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # LLM-generated summaries
        self.executive_summary: Optional[str] = None
        self.database_overview: Optional[str] = None
        self.warnings_and_caveats: List[str] = []
        self.improvement_suggestions: List[str] = []
    
    def add_table(self, table: TableProfile):
        """Add a table profile to the store."""
        self.tables[table.name] = table
        if table.name not in self.dependency_graph:
            self.dependency_graph[table.name] = set()
    
    def add_relationship(self, relationship: Relationship):
        """Add a relationship to the store."""
        self.relationships.append(relationship)
        
        # Update dependency graph
        if relationship.source_table not in self.dependency_graph:
            self.dependency_graph[relationship.source_table] = set()
        self.dependency_graph[relationship.source_table].add(relationship.target_table)
    
    def add_quality_issue(self, issue: DataQualityIssue):
        """Add a data quality issue."""
        self.quality_issues.append(issue)
    
    def add_insight(self, insight: BusinessInsight):
        """Add a business insight."""
        self.insights.append(insight)
    
    def get_table(self, name: str) -> Optional[TableProfile]:
        """Get a table profile by name."""
        return self.tables.get(name)
    
    def get_relationships_for_table(self, table_name: str) -> List[Relationship]:
        """Get all relationships involving a table."""
        return [
            r for r in self.relationships
            if r.source_table == table_name or r.target_table == table_name
        ]
    
    def get_outgoing_relationships(self, table_name: str) -> List[Relationship]:
        """Get relationships where table is the source (has FK)."""
        return [r for r in self.relationships if r.source_table == table_name]
    
    def get_incoming_relationships(self, table_name: str) -> List[Relationship]:
        """Get relationships where table is the target (is referenced)."""
        return [r for r in self.relationships if r.target_table == table_name]
    
    def get_quality_issues_for_table(self, table_name: str) -> List[DataQualityIssue]:
        """Get quality issues for a specific table."""
        return [i for i in self.quality_issues if i.table == table_name]
    
    def get_core_tables(self) -> List[str]:
        """
        Get core/dimension tables (tables with no outgoing FKs but incoming FKs).
        These are typically the foundation tables of the schema.
        """
        tables_with_incoming = set()
        tables_with_outgoing = set()
        
        for rel in self.relationships:
            tables_with_outgoing.add(rel.source_table)
            tables_with_incoming.add(rel.target_table)
        
        # Core tables have incoming but no outgoing
        return list(tables_with_incoming - tables_with_outgoing)
    
    def get_transaction_tables(self) -> List[str]:
        """
        Get transaction/fact tables (tables with outgoing FKs).
        These typically represent events or transactions.
        """
        tables_with_outgoing = set()
        
        for rel in self.relationships:
            tables_with_outgoing.add(rel.source_table)
        
        return list(tables_with_outgoing)
    
    def get_isolated_tables(self) -> List[str]:
        """Get tables with no relationships."""
        related_tables = set()
        for rel in self.relationships:
            related_tables.add(rel.source_table)
            related_tables.add(rel.target_table)
        
        return [t for t in self.tables.keys() if t not in related_tables]
    
    def get_topological_order(self) -> List[str]:
        """
        Get tables in topological order (dependencies first).
        Useful for understanding data loading order.
        """
        # Simple topological sort using Kahn's algorithm
        in_degree = {t: 0 for t in self.tables}
        
        for rel in self.relationships:
            if rel.target_table in in_degree:
                in_degree[rel.source_table] += 1
        
        queue = [t for t, d in in_degree.items() if d == 0]
        result = []
        
        while queue:
            table = queue.pop(0)
            result.append(table)
            
            for rel in self.get_incoming_relationships(table):
                in_degree[rel.source_table] -= 1
                if in_degree[rel.source_table] == 0:
                    queue.append(rel.source_table)
        
        # Add any remaining tables (cycles or isolated)
        remaining = [t for t in self.tables if t not in result]
        result.extend(remaining)
        
        return result
    
    def calculate_overall_quality_score(self) -> float:
        """Calculate overall database quality score (0-100)."""
        if not self.tables:
            return 0.0
        
        total_score = 0.0
        
        for table in self.tables.values():
            table_score = 100.0
            
            # Deduct for quality issues
            table_issues = self.get_quality_issues_for_table(table.name)
            for issue in table_issues:
                if issue.severity == DataQualitySeverity.CRITICAL:
                    table_score -= 20
                elif issue.severity == DataQualitySeverity.WARNING:
                    table_score -= 10
                else:
                    table_score -= 2
            
            # Deduct for missing primary key
            if not table.has_primary_key:
                table_score -= 15
            
            # Ensure score doesn't go below 0
            table.overall_quality_score = max(0, table_score)
            total_score += table.overall_quality_score
        
        self.overall_quality_score = total_score / len(self.tables)
        return self.overall_quality_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert store to dictionary for serialization."""
        return {
            "database_name": self.database_name,
            "database_type": self.database_type,
            "database_version": self.database_version,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "table_count": len(self.tables),
            "relationship_count": len(self.relationships),
            "overall_quality_score": self.overall_quality_score,
            "tables": list(self.tables.keys()),
            "executive_summary": self.executive_summary,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert store to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_columns = sum(t.column_count for t in self.tables.values())
        total_rows = sum(t.row_count for t in self.tables.values())
        
        return {
            "total_tables": len(self.tables),
            "total_columns": total_columns,
            "total_rows": total_rows,
            "total_relationships": len(self.relationships),
            "total_quality_issues": len(self.quality_issues),
            "critical_issues": len([i for i in self.quality_issues if i.severity == DataQualitySeverity.CRITICAL]),
            "warning_issues": len([i for i in self.quality_issues if i.severity == DataQualitySeverity.WARNING]),
            "core_tables": len(self.get_core_tables()),
            "transaction_tables": len(self.get_transaction_tables()),
            "isolated_tables": len(self.get_isolated_tables()),
            "overall_quality_score": self.overall_quality_score,
        }

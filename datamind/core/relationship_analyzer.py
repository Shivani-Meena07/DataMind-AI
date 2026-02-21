"""
Relationship Analyzer

Analyzes and infers relationships between tables, including
foreign key detection, cardinality analysis, and dependency mapping.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from collections import defaultdict

from sqlalchemy import text

from datamind.core.connection import DatabaseConnectionManager
from datamind.core.intelligence_store import (
    IntelligenceStore,
    TableProfile,
    Relationship,
    TableType,
)
from datamind.config import DataMindConfig

logger = logging.getLogger(__name__)


class RelationshipAnalyzer:
    """
    Analyzes table relationships and data dependencies.
    
    This module handles:
    - Foreign key relationship extraction
    - Cardinality analysis (1:1, 1:N, N:M)
    - Implicit relationship detection
    - Table dependency hierarchy
    - Data flow direction analysis
    - Table type classification (fact/dimension)
    """
    
    def __init__(
        self,
        connection_manager: DatabaseConnectionManager,
        config: DataMindConfig,
    ):
        """
        Initialize the relationship analyzer.
        
        Args:
            connection_manager: Database connection manager
            config: DataMind configuration
        """
        self.conn_manager = connection_manager
        self.config = config
    
    def analyze(self, store: IntelligenceStore) -> IntelligenceStore:
        """
        Analyze all relationships in the database.
        
        Args:
            store: Intelligence store with schema information
            
        Returns:
            Intelligence store with relationship data
        """
        logger.info("Starting relationship analysis...")
        
        # Extract relationships from foreign keys
        self._extract_fk_relationships(store)
        
        # Detect implicit relationships
        self._detect_implicit_relationships(store)
        
        # Analyze cardinality for each relationship
        self._analyze_cardinality(store)
        
        # Classify table types
        self._classify_table_types(store)
        
        # Build dependency hierarchy
        self._build_dependency_hierarchy(store)
        
        logger.info(f"Relationship analysis complete. Found {len(store.relationships)} relationships.")
        return store
    
    def _extract_fk_relationships(self, store: IntelligenceStore):
        """Extract relationships from defined foreign keys."""
        for table_name, table_profile in store.tables.items():
            for fk in table_profile.foreign_keys:
                source_cols = fk.get('constrained_columns', [])
                target_table = fk.get('referred_table')
                target_cols = fk.get('referred_columns', [])
                
                if not source_cols or not target_table or not target_cols:
                    continue
                
                # Don't add relationship if target table not in our scope
                if target_table not in store.tables:
                    logger.debug(f"Skipping FK to external table: {target_table}")
                    continue
                
                relationship = Relationship(
                    source_table=table_name,
                    source_columns=source_cols,
                    target_table=target_table,
                    target_columns=target_cols,
                )
                
                # Check if this is an identifying relationship
                # (FK is part of the primary key)
                if table_profile.primary_key:
                    relationship.is_identifying = any(
                        col in table_profile.primary_key for col in source_cols
                    )
                
                store.add_relationship(relationship)
                logger.debug(f"Found FK relationship: {table_name} -> {target_table}")
    
    def _detect_implicit_relationships(self, store: IntelligenceStore):
        """
        Detect implicit relationships based on naming conventions.
        
        This finds potential relationships where FKs aren't explicitly defined
        but column names suggest relationships (e.g., customer_id -> customers.id)
        """
        # Build a map of potential parent tables and their PK columns
        pk_map: Dict[str, str] = {}  # table_name -> pk_column
        
        for table_name, table_profile in store.tables.items():
            if table_profile.primary_key and len(table_profile.primary_key) == 1:
                pk_map[table_name] = table_profile.primary_key[0]
        
        # Get existing relationships to avoid duplicates
        existing_rels = set()
        for rel in store.relationships:
            key = (rel.source_table, tuple(rel.source_columns), 
                   rel.target_table, tuple(rel.target_columns))
            existing_rels.add(key)
        
        # Check each table for columns that might reference other tables
        for table_name, table_profile in store.tables.items():
            for col_name, col_profile in table_profile.columns.items():
                # Skip if already a foreign key
                if col_profile.is_foreign_key:
                    continue
                
                # Skip primary keys
                if col_profile.is_primary_key:
                    continue
                
                # Look for naming patterns like: {table}_id, {table}id, fk_{table}
                potential_targets = self._find_potential_targets(
                    col_name, pk_map, table_name
                )
                
                for target_table, target_col in potential_targets:
                    # Check if relationship already exists
                    key = (table_name, (col_name,), target_table, (target_col,))
                    if key in existing_rels:
                        continue
                    
                    # Verify the relationship makes sense by checking data types
                    target_table_profile = store.tables.get(target_table)
                    if not target_table_profile:
                        continue
                    
                    target_col_profile = target_table_profile.columns.get(target_col)
                    if not target_col_profile:
                        continue
                    
                    # Basic type compatibility check
                    if not self._types_compatible(
                        col_profile.data_type, 
                        target_col_profile.data_type
                    ):
                        continue
                    
                    # Create implicit relationship
                    relationship = Relationship(
                        source_table=table_name,
                        source_columns=[col_name],
                        target_table=target_table,
                        target_columns=[target_col],
                    )
                    
                    store.add_relationship(relationship)
                    existing_rels.add(key)
                    logger.debug(
                        f"Found implicit relationship: {table_name}.{col_name} -> "
                        f"{target_table}.{target_col}"
                    )
    
    def _find_potential_targets(
        self, 
        col_name: str, 
        pk_map: Dict[str, str],
        current_table: str
    ) -> List[Tuple[str, str]]:
        """Find potential target tables for a column based on naming conventions."""
        targets = []
        col_lower = col_name.lower()
        
        for table_name, pk_col in pk_map.items():
            if table_name == current_table:
                continue
            
            table_lower = table_name.lower()
            
            # Pattern: {table}_id
            if col_lower == f"{table_lower}_id":
                targets.append((table_name, pk_col))
            # Pattern: {table}id (no underscore)
            elif col_lower == f"{table_lower}id":
                targets.append((table_name, pk_col))
            # Pattern: fk_{table}
            elif col_lower == f"fk_{table_lower}":
                targets.append((table_name, pk_col))
            # Pattern: {singular}_id where table is {singular}s
            elif table_lower.endswith('s') and col_lower == f"{table_lower[:-1]}_id":
                targets.append((table_name, pk_col))
            # Pattern: id_{table}
            elif col_lower == f"id_{table_lower}":
                targets.append((table_name, pk_col))
        
        return targets
    
    def _types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two column types are compatible for a relationship."""
        type1_lower = type1.lower()
        type2_lower = type2.lower()
        
        # Integer types
        int_types = {'int', 'integer', 'bigint', 'smallint', 'tinyint', 'serial', 'bigserial'}
        type1_is_int = any(t in type1_lower for t in int_types)
        type2_is_int = any(t in type2_lower for t in int_types)
        if type1_is_int and type2_is_int:
            return True
        
        # String types
        str_types = {'varchar', 'char', 'text', 'string', 'nvarchar', 'uuid'}
        type1_is_str = any(t in type1_lower for t in str_types)
        type2_is_str = any(t in type2_lower for t in str_types)
        if type1_is_str and type2_is_str:
            return True
        
        return False
    
    def _analyze_cardinality(self, store: IntelligenceStore):
        """Analyze cardinality for each relationship."""
        for relationship in store.relationships:
            cardinality = self._compute_cardinality(
                relationship.source_table,
                relationship.source_columns,
                relationship.target_table,
                relationship.target_columns,
                store
            )
            relationship.cardinality = cardinality
    
    def _compute_cardinality(
        self,
        source_table: str,
        source_columns: List[str],
        target_table: str,
        target_columns: List[str],
        store: IntelligenceStore
    ) -> str:
        """
        Compute the cardinality of a relationship.
        
        Returns one of: "1:1", "1:N", "N:1", "N:M"
        """
        if len(source_columns) != 1 or len(target_columns) != 1:
            return "N:M"  # Composite keys are complex
        
        source_col = source_columns[0]
        target_col = target_columns[0]
        
        # Get source column profile
        source_table_profile = store.tables.get(source_table)
        target_table_profile = store.tables.get(target_table)
        
        if not source_table_profile or not target_table_profile:
            return "unknown"
        
        source_col_profile = source_table_profile.columns.get(source_col)
        target_col_profile = target_table_profile.columns.get(target_col)
        
        if not source_col_profile or not target_col_profile:
            return "unknown"
        
        # Check if source column is unique (1:1 from source side)
        source_is_unique = (
            source_col_profile.is_primary_key or 
            source_col_profile.uniqueness_ratio > 0.99
        )
        
        # Target is typically the PK side (unique)
        target_is_unique = (
            target_col_profile.is_primary_key or
            target_col_profile.uniqueness_ratio > 0.99
        )
        
        if source_is_unique and target_is_unique:
            return "1:1"
        elif target_is_unique:
            return "N:1"  # Many source rows reference one target
        elif source_is_unique:
            return "1:N"  # One source row referenced by many (unusual for FK)
        else:
            return "N:M"
    
    def _classify_table_types(self, store: IntelligenceStore):
        """Classify tables as fact, dimension, bridge, etc."""
        # Count incoming and outgoing relationships
        incoming = defaultdict(int)
        outgoing = defaultdict(int)
        
        for rel in store.relationships:
            outgoing[rel.source_table] += 1
            incoming[rel.target_table] += 1
        
        for table_name, table_profile in store.tables.items():
            out_count = outgoing.get(table_name, 0)
            in_count = incoming.get(table_name, 0)
            
            # Bridge/junction table: has multiple outgoing FKs and those FKs are part of PK
            if out_count >= 2 and in_count == 0:
                # Check if PKs are composed of FKs
                fk_columns = set()
                for rel in store.relationships:
                    if rel.source_table == table_name:
                        fk_columns.update(rel.source_columns)
                
                if table_profile.primary_key:
                    pk_set = set(table_profile.primary_key)
                    if pk_set.issubset(fk_columns):
                        table_profile.table_type = TableType.BRIDGE
                        continue
                
                # Otherwise likely a fact table
                table_profile.table_type = TableType.FACT
            
            # Dimension table: only incoming references, no outgoing FKs
            elif in_count > 0 and out_count == 0:
                table_profile.table_type = TableType.DIMENSION
            
            # Fact table: has outgoing FKs (references dimensions)
            elif out_count > 0:
                table_profile.table_type = TableType.FACT
            
            # Isolated table - check by naming convention
            else:
                name_lower = table_name.lower()
                if any(p in name_lower for p in ['log', 'audit', 'history']):
                    table_profile.table_type = TableType.AUDIT
                elif any(p in name_lower for p in ['config', 'setting', 'param']):
                    table_profile.table_type = TableType.CONFIGURATION
                elif any(p in name_lower for p in ['staging', 'temp', 'tmp']):
                    table_profile.table_type = TableType.STAGING
                else:
                    table_profile.table_type = TableType.UNKNOWN
    
    def _build_dependency_hierarchy(self, store: IntelligenceStore):
        """Build the dependency hierarchy for tables."""
        # Already partially done in add_relationship
        # Here we can add additional analysis
        
        # Find root tables (no dependencies)
        root_tables = []
        dependent_tables = set()
        
        for rel in store.relationships:
            dependent_tables.add(rel.source_table)
        
        for table_name in store.tables:
            if table_name not in dependent_tables:
                root_tables.append(table_name)
        
        logger.debug(f"Root tables (no dependencies): {root_tables}")
    
    def get_relationship_summary(self, store: IntelligenceStore) -> Dict[str, Any]:
        """Get a summary of relationship analysis."""
        cardinality_counts = defaultdict(int)
        for rel in store.relationships:
            cardinality_counts[rel.cardinality] += 1
        
        table_type_counts = defaultdict(int)
        for table in store.tables.values():
            table_type_counts[table.table_type.value] += 1
        
        return {
            "total_relationships": len(store.relationships),
            "cardinality_distribution": dict(cardinality_counts),
            "table_type_distribution": dict(table_type_counts),
            "core_tables": store.get_core_tables(),
            "transaction_tables": store.get_transaction_tables(),
            "isolated_tables": store.get_isolated_tables(),
        }
    
    def generate_relationship_text(self, store: IntelligenceStore) -> str:
        """Generate human-readable relationship descriptions."""
        lines = []
        
        for rel in store.relationships:
            source = rel.source_table
            target = rel.target_table
            
            # Generate description based on cardinality
            if rel.cardinality == "N:1":
                desc = f"Many records in '{source}' reference one record in '{target}'"
            elif rel.cardinality == "1:1":
                desc = f"One record in '{source}' corresponds to one record in '{target}'"
            elif rel.cardinality == "1:N":
                desc = f"One record in '{source}' can be referenced by many in '{target}'"
            else:
                desc = f"'{source}' has a many-to-many relationship with '{target}'"
            
            lines.append(f"• {source} → {target} ({rel.cardinality}): {desc}")
        
        return "\n".join(lines)
    
    def detect_circular_dependencies(self, store: IntelligenceStore) -> List[List[str]]:
        """Detect circular dependencies in the schema."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(table: str, path: List[str]):
            visited.add(table)
            rec_stack.add(table)
            path.append(table)
            
            for rel in store.get_outgoing_relationships(table):
                next_table = rel.target_table
                if next_table not in visited:
                    dfs(next_table, path.copy())
                elif next_table in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(next_table)
                    cycles.append(path[cycle_start:] + [next_table])
            
            rec_stack.remove(table)
        
        for table in store.tables:
            if table not in visited:
                dfs(table, [])
        
        return cycles

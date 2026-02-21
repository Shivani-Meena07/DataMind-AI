"""
Schema Scanner

Extracts complete schema metadata from databases including
tables, columns, keys, constraints, and indexes.
"""

from typing import Dict, List, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import text, inspect
from sqlalchemy.engine import Inspector

from datamind.core.connection import DatabaseConnectionManager
from datamind.core.intelligence_store import (
    IntelligenceStore,
    TableProfile,
    ColumnProfile,
    ColumnType,
)
from datamind.config import DataMindConfig, DatabaseType

logger = logging.getLogger(__name__)


class SchemaScanner:
    """
    Extracts comprehensive schema metadata from databases.
    
    This module handles:
    - Table discovery
    - Column metadata extraction
    - Primary key detection
    - Foreign key detection
    - Index information
    - Constraint extraction
    """
    
    def __init__(
        self,
        connection_manager: DatabaseConnectionManager,
        config: DataMindConfig,
    ):
        """
        Initialize the schema scanner.
        
        Args:
            connection_manager: Database connection manager
            config: DataMind configuration
        """
        self.conn_manager = connection_manager
        self.config = config
        self._inspector: Optional[Inspector] = None
    
    @property
    def inspector(self) -> Inspector:
        """Get SQLAlchemy inspector."""
        if self._inspector is None:
            self._inspector = self.conn_manager.get_inspector()
        return self._inspector
    
    def scan(self, store: IntelligenceStore) -> IntelligenceStore:
        """
        Perform complete schema scan and populate intelligence store.
        
        Args:
            store: Intelligence store to populate
            
        Returns:
            Populated intelligence store
        """
        logger.info("Starting schema scan...")
        
        # Get database info
        db_info = self.conn_manager.get_database_info()
        store.database_type = db_info.get("type")
        store.database_version = db_info.get("version")
        
        # Get table names
        tables = self._get_filtered_tables()
        logger.info(f"Found {len(tables)} tables to scan")
        
        # Scan each table
        for table_name in tables:
            try:
                table_profile = self._scan_table(table_name)
                store.add_table(table_profile)
                logger.debug(f"Scanned table: {table_name}")
            except Exception as e:
                logger.error(f"Error scanning table {table_name}: {e}")
        
        logger.info(f"Schema scan complete. Scanned {len(store.tables)} tables.")
        return store
    
    def _get_filtered_tables(self) -> List[str]:
        """Get list of tables after applying filters."""
        schema = self.config.database.schema if self.config.database else None
        all_tables = self.inspector.get_table_names(schema=schema)
        
        # Apply inclusion filter
        if self.config.include_tables:
            all_tables = [t for t in all_tables if t in self.config.include_tables]
        
        # Apply exclusion filter
        if self.config.exclude_tables:
            all_tables = [t for t in all_tables if t not in self.config.exclude_tables]
        
        # Exclude system tables
        if self.config.exclude_system_tables:
            system_prefixes = ('sqlite_', 'pg_', 'sql_', 'sys', 'information_schema')
            all_tables = [
                t for t in all_tables 
                if not t.lower().startswith(system_prefixes)
            ]
        
        return sorted(all_tables)
    
    def _scan_table(self, table_name: str) -> TableProfile:
        """
        Scan a single table and extract all metadata.
        
        Args:
            table_name: Name of the table to scan
            
        Returns:
            TableProfile with complete metadata
        """
        schema = self.config.database.schema if self.config.database else None
        
        # Create table profile
        profile = TableProfile(name=table_name, schema=schema)
        
        # Get row count
        profile.row_count = self._get_row_count(table_name)
        
        # Get columns
        profile.columns = self._get_columns(table_name, schema)
        
        # Get primary key
        profile.primary_key = self._get_primary_key(table_name, schema)
        
        # Update columns with PK info
        if profile.primary_key:
            for pk_col in profile.primary_key:
                if pk_col in profile.columns:
                    profile.columns[pk_col].is_primary_key = True
        
        # Get foreign keys
        profile.foreign_keys = self._get_foreign_keys(table_name, schema)
        
        # Update columns with FK info
        for fk in profile.foreign_keys:
            for col in fk.get('constrained_columns', []):
                if col in profile.columns:
                    profile.columns[col].is_foreign_key = True
                    # Get referenced table/column
                    ref_table = fk.get('referred_table')
                    ref_cols = fk.get('referred_columns', [])
                    if ref_table:
                        profile.columns[col].references_table = ref_table
                    if ref_cols:
                        profile.columns[col].references_column = ref_cols[0]
        
        # Get indexes
        profile.indexes = self._get_indexes(table_name, schema)
        
        # Get unique constraints
        profile.unique_constraints = self._get_unique_constraints(table_name, schema)
        
        # Get check constraints (if supported)
        profile.check_constraints = self._get_check_constraints(table_name, schema)
        
        return profile
    
    def _get_row_count(self, table_name: str) -> int:
        """Get row count for a table."""
        try:
            # Use quoted identifier for safety
            query = f'SELECT COUNT(*) FROM "{table_name}"'
            result = self.conn_manager.execute_query(query)
            if result:
                return list(result[0].values())[0]
        except Exception as e:
            logger.warning(f"Could not get row count for {table_name}: {e}")
        return 0
    
    def _get_columns(
        self, 
        table_name: str, 
        schema: Optional[str]
    ) -> Dict[str, ColumnProfile]:
        """Get column information for a table."""
        columns = {}
        
        try:
            col_info = self.inspector.get_columns(table_name, schema=schema)
            
            for col in col_info:
                name = col['name']
                
                # Determine data type string
                col_type = str(col['type'])
                
                profile = ColumnProfile(
                    name=name,
                    data_type=col_type,
                    nullable=col.get('nullable', True),
                    default_value=str(col.get('default')) if col.get('default') else None,
                )
                
                # Infer semantic type from data type
                profile.semantic_type = self._infer_semantic_type(col_type, name)
                
                columns[name] = profile
                
        except Exception as e:
            logger.error(f"Error getting columns for {table_name}: {e}")
        
        return columns
    
    def _infer_semantic_type(self, data_type: str, column_name: str) -> ColumnType:
        """
        Infer semantic column type from data type and name.
        
        This is a preliminary classification that will be refined
        by the data profiler and LLM inference.
        """
        data_type_lower = data_type.lower()
        name_lower = column_name.lower()
        
        # Timestamp patterns
        if any(t in data_type_lower for t in ['timestamp', 'datetime', 'date', 'time']):
            return ColumnType.TIMESTAMP
        
        # Boolean patterns
        if any(t in data_type_lower for t in ['bool', 'bit']):
            return ColumnType.BOOLEAN
        
        # ID patterns in name
        if any(p in name_lower for p in ['_id', 'id_', '_key', '_code', '_uuid']):
            return ColumnType.IDENTIFIER
        
        if name_lower.endswith('id') or name_lower == 'id':
            return ColumnType.IDENTIFIER
        
        # Metric patterns (numeric with metric-like names)
        if any(t in data_type_lower for t in ['int', 'float', 'decimal', 'numeric', 'double', 'real']):
            metric_patterns = ['amount', 'price', 'cost', 'total', 'count', 'qty', 'quantity', 
                             'revenue', 'sales', 'score', 'rating', 'weight', 'height', 'length',
                             'sum', 'avg', 'value']
            if any(p in name_lower for p in metric_patterns):
                return ColumnType.METRIC
        
        # Status/flag patterns
        if any(p in name_lower for p in ['status', 'state', 'type', 'category', 'flag', 'is_', 'has_']):
            return ColumnType.DIMENSION
        
        # Geographic patterns
        if any(p in name_lower for p in ['lat', 'lng', 'longitude', 'latitude', 'geo', 'zip', 
                                          'postal', 'city', 'state', 'country', 'address']):
            return ColumnType.GEOGRAPHIC
        
        # Financial patterns
        if any(p in name_lower for p in ['price', 'cost', 'amount', 'payment', 'fee', 'tax']):
            if any(t in data_type_lower for t in ['decimal', 'numeric', 'money']):
                return ColumnType.FINANCIAL
        
        # Text patterns
        if any(t in data_type_lower for t in ['text', 'varchar', 'char', 'string', 'clob']):
            name_patterns = ['name', 'title', 'description', 'comment', 'note', 'message']
            if any(p in name_lower for p in name_patterns):
                return ColumnType.TEXT
        
        return ColumnType.UNKNOWN
    
    def _get_primary_key(
        self, 
        table_name: str, 
        schema: Optional[str]
    ) -> Optional[List[str]]:
        """Get primary key columns for a table."""
        try:
            pk_info = self.inspector.get_pk_constraint(table_name, schema=schema)
            if pk_info:
                return pk_info.get('constrained_columns', [])
        except Exception as e:
            logger.warning(f"Could not get primary key for {table_name}: {e}")
        return None
    
    def _get_foreign_keys(
        self, 
        table_name: str, 
        schema: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Get foreign key information for a table."""
        try:
            return self.inspector.get_foreign_keys(table_name, schema=schema)
        except Exception as e:
            logger.warning(f"Could not get foreign keys for {table_name}: {e}")
            return []
    
    def _get_indexes(
        self, 
        table_name: str, 
        schema: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Get index information for a table."""
        try:
            return self.inspector.get_indexes(table_name, schema=schema)
        except Exception as e:
            logger.warning(f"Could not get indexes for {table_name}: {e}")
            return []
    
    def _get_unique_constraints(
        self, 
        table_name: str, 
        schema: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Get unique constraints for a table."""
        try:
            return self.inspector.get_unique_constraints(table_name, schema=schema)
        except Exception as e:
            logger.debug(f"Could not get unique constraints for {table_name}: {e}")
            return []
    
    def _get_check_constraints(
        self, 
        table_name: str, 
        schema: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Get check constraints for a table (if supported)."""
        try:
            return self.inspector.get_check_constraints(table_name, schema=schema)
        except Exception as e:
            logger.debug(f"Could not get check constraints for {table_name}: {e}")
            return []
    
    def get_schema_summary(self, store: IntelligenceStore) -> Dict[str, Any]:
        """
        Get a summary of the scanned schema.
        
        Args:
            store: Populated intelligence store
            
        Returns:
            Schema summary dictionary
        """
        total_columns = sum(len(t.columns) for t in store.tables.values())
        total_rows = sum(t.row_count for t in store.tables.values())
        tables_with_pk = sum(1 for t in store.tables.values() if t.has_primary_key)
        tables_with_fk = sum(1 for t in store.tables.values() if t.foreign_keys)
        
        # Column type distribution
        type_distribution = {}
        for table in store.tables.values():
            for col in table.columns.values():
                col_type = col.semantic_type.value
                type_distribution[col_type] = type_distribution.get(col_type, 0) + 1
        
        return {
            "total_tables": len(store.tables),
            "total_columns": total_columns,
            "total_rows": total_rows,
            "tables_with_primary_key": tables_with_pk,
            "tables_with_foreign_keys": tables_with_fk,
            "column_type_distribution": type_distribution,
            "average_columns_per_table": total_columns / len(store.tables) if store.tables else 0,
            "database_type": store.database_type,
            "database_version": store.database_version,
        }

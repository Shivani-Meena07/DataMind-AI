"""
Data Profiler

Analyzes actual data in database tables to compute statistics,
detect data quality issues, and identify patterns.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import Counter
import re

from sqlalchemy import text

from datamind.core.connection import DatabaseConnectionManager
from datamind.core.intelligence_store import (
    IntelligenceStore,
    TableProfile,
    ColumnProfile,
    ColumnType,
    DataQualityIssue,
    DataQualitySeverity,
)
from datamind.config import DataMindConfig, DatabaseType

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Analyzes actual data in tables to compute statistics and detect issues.
    
    This module handles:
    - NULL value analysis
    - Uniqueness calculation
    - Value distribution analysis
    - Data quality scoring
    - Pattern detection (PII, anomalies)
    - Sample value extraction
    """
    
    def __init__(
        self,
        connection_manager: DatabaseConnectionManager,
        config: DataMindConfig,
    ):
        """
        Initialize the data profiler.
        
        Args:
            connection_manager: Database connection manager
            config: DataMind configuration
        """
        self.conn_manager = connection_manager
        self.config = config
        self.profiling_config = config.profiling
    
    def profile(self, store: IntelligenceStore) -> IntelligenceStore:
        """
        Profile all tables in the intelligence store.
        
        Args:
            store: Intelligence store with schema information
            
        Returns:
            Intelligence store with profiling data
        """
        logger.info("Starting data profiling...")
        
        for table_name, table_profile in store.tables.items():
            try:
                # Skip tiny tables
                if table_profile.row_count < self.profiling_config.min_rows_for_profiling:
                    logger.debug(f"Skipping {table_name} (too few rows)")
                    continue
                
                self._profile_table(table_profile)
                logger.debug(f"Profiled table: {table_name}")
                
                # Detect quality issues
                issues = self._detect_quality_issues(table_profile)
                for issue in issues:
                    store.add_quality_issue(issue)
                    
            except Exception as e:
                logger.error(f"Error profiling table {table_name}: {e}")
        
        # Calculate overall quality score
        store.calculate_overall_quality_score()
        
        logger.info(f"Data profiling complete. Found {len(store.quality_issues)} quality issues.")
        return store
    
    def _profile_table(self, table: TableProfile):
        """Profile all columns in a table."""
        table_name = table.name
        
        for col_name, col_profile in table.columns.items():
            try:
                self._profile_column(table_name, col_profile, table.row_count)
            except Exception as e:
                logger.warning(f"Error profiling column {table_name}.{col_name}: {e}")
    
    def _profile_column(
        self, 
        table_name: str, 
        column: ColumnProfile,
        total_rows: int
    ):
        """
        Profile a single column.
        
        Args:
            table_name: Name of the table
            column: Column profile to update
            total_rows: Total row count of the table
        """
        col_name = column.name
        column.total_count = total_rows
        
        # Get basic statistics in one query
        stats = self._get_column_statistics(table_name, col_name, column.data_type)
        
        if stats:
            column.null_count = stats.get('null_count', 0)
            column.distinct_count = stats.get('distinct_count', 0)
            column.min_value = stats.get('min_value')
            column.max_value = stats.get('max_value')
            column.avg_value = stats.get('avg_value')
            
            # Calculate ratios
            if total_rows > 0:
                column.null_percentage = column.null_count / total_rows
                column.uniqueness_ratio = column.distinct_count / total_rows
        
        # Get sample values
        column.sample_values = self._get_sample_values(table_name, col_name)
        
        # Get most common values
        if self.profiling_config.compute_distributions:
            column.most_common_values = self._get_most_common_values(
                table_name, col_name
            )
        
        # Refine semantic type based on data
        self._refine_semantic_type(column)
        
        # Check for PII
        if self.profiling_config.detect_pii:
            self._detect_pii(column)
    
    def _get_column_statistics(
        self, 
        table_name: str, 
        column_name: str,
        data_type: str
    ) -> Dict[str, Any]:
        """Get basic statistics for a column."""
        # Determine if numeric for avg calculation
        is_numeric = any(t in data_type.lower() for t in [
            'int', 'float', 'decimal', 'numeric', 'double', 'real', 'money'
        ])
        
        # Build query
        avg_clause = f', AVG(CAST("{column_name}" AS FLOAT))' if is_numeric else ''
        min_max_clause = f', MIN("{column_name}"), MAX("{column_name}")'
        
        query = f"""
            SELECT 
                COUNT(*) - COUNT("{column_name}") as null_count,
                COUNT(DISTINCT "{column_name}") as distinct_count
                {min_max_clause}
                {avg_clause}
            FROM "{table_name}"
        """
        
        try:
            # Handle sample size limitation
            if self.profiling_config.sample_size:
                # For sampling, we need to adjust the query based on DB type
                db_type = self.config.database.db_type
                if db_type == DatabaseType.POSTGRESQL:
                    query = f"""
                        SELECT 
                            COUNT(*) - COUNT("{column_name}") as null_count,
                            COUNT(DISTINCT "{column_name}") as distinct_count
                            {min_max_clause}
                            {avg_clause}
                        FROM (
                            SELECT "{column_name}" FROM "{table_name}" 
                            ORDER BY RANDOM() LIMIT {self.profiling_config.sample_size}
                        ) sample
                    """
                elif db_type == DatabaseType.MYSQL:
                    query = f"""
                        SELECT 
                            COUNT(*) - COUNT(`{column_name}`) as null_count,
                            COUNT(DISTINCT `{column_name}`) as distinct_count
                            {min_max_clause.replace('"', '`')}
                            {avg_clause.replace('"', '`')}
                        FROM (
                            SELECT `{column_name}` FROM `{table_name}` 
                            ORDER BY RAND() LIMIT {self.profiling_config.sample_size}
                        ) sample
                    """
                elif db_type == DatabaseType.SQLITE:
                    query = f"""
                        SELECT 
                            COUNT(*) - COUNT("{column_name}") as null_count,
                            COUNT(DISTINCT "{column_name}") as distinct_count
                            {min_max_clause}
                            {avg_clause}
                        FROM (
                            SELECT "{column_name}" FROM "{table_name}" 
                            ORDER BY RANDOM() LIMIT {self.profiling_config.sample_size}
                        )
                    """
            
            result = self.conn_manager.execute_query(query)
            if result:
                row = result[0]
                keys = list(row.keys())
                return {
                    'null_count': row.get(keys[0], 0) or 0,
                    'distinct_count': row.get(keys[1], 0) or 0,
                    'min_value': row.get(keys[2]) if len(keys) > 2 else None,
                    'max_value': row.get(keys[3]) if len(keys) > 3 else None,
                    'avg_value': row.get(keys[4]) if len(keys) > 4 else None,
                }
        except Exception as e:
            logger.warning(f"Error getting statistics for {table_name}.{column_name}: {e}")
        
        return {}
    
    def _get_sample_values(
        self, 
        table_name: str, 
        column_name: str,
        limit: int = 10
    ) -> List[Any]:
        """Get sample non-null values from a column."""
        try:
            query = f"""
                SELECT DISTINCT "{column_name}" 
                FROM "{table_name}" 
                WHERE "{column_name}" IS NOT NULL 
                LIMIT {limit}
            """
            result = self.conn_manager.execute_query(query)
            return [row[column_name] for row in result]
        except Exception as e:
            logger.debug(f"Error getting sample values: {e}")
            return []
    
    def _get_most_common_values(
        self, 
        table_name: str, 
        column_name: str,
        limit: int = 10
    ) -> List[Tuple[Any, int]]:
        """Get most common values and their counts."""
        try:
            query = f"""
                SELECT "{column_name}", COUNT(*) as cnt 
                FROM "{table_name}" 
                WHERE "{column_name}" IS NOT NULL
                GROUP BY "{column_name}" 
                ORDER BY cnt DESC 
                LIMIT {limit}
            """
            result = self.conn_manager.execute_query(query)
            return [(row[column_name], row['cnt']) for row in result]
        except Exception as e:
            logger.debug(f"Error getting most common values: {e}")
            return []
    
    def _refine_semantic_type(self, column: ColumnProfile):
        """Refine semantic type based on actual data patterns."""
        # Check for high uniqueness = identifier
        if column.uniqueness_ratio > self.profiling_config.uniqueness_threshold:
            if column.semantic_type == ColumnType.UNKNOWN:
                column.semantic_type = ColumnType.IDENTIFIER
        
        # Check for low cardinality = dimension
        if column.distinct_count < 50 and column.distinct_count > 0:
            if column.semantic_type == ColumnType.UNKNOWN:
                column.semantic_type = ColumnType.DIMENSION
        
        # Check sample values for patterns
        if column.sample_values:
            self._detect_value_patterns(column)
    
    def _detect_value_patterns(self, column: ColumnProfile):
        """Detect patterns in sample values."""
        samples = [str(v) for v in column.sample_values if v is not None]
        if not samples:
            return
        
        # Email pattern
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if all(re.match(email_pattern, s) for s in samples[:5]):
            column.warnings.append("Contains email addresses (potential PII)")
        
        # Phone pattern
        phone_pattern = r'^[\d\s\-\+\(\)]{7,}$'
        if all(re.match(phone_pattern, s) for s in samples[:5]):
            column.warnings.append("May contain phone numbers (potential PII)")
        
        # UUID pattern
        uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
        if all(re.match(uuid_pattern, s.lower()) for s in samples[:5]):
            column.semantic_type = ColumnType.IDENTIFIER
        
        # Status-like values
        status_keywords = {'active', 'inactive', 'pending', 'completed', 'cancelled', 
                          'approved', 'rejected', 'open', 'closed', 'new', 'processing',
                          'shipped', 'delivered', 'paid', 'unpaid', 'yes', 'no', 'true', 'false'}
        if all(s.lower() in status_keywords for s in samples[:5]):
            column.semantic_type = ColumnType.DIMENSION
    
    def _detect_pii(self, column: ColumnProfile):
        """Detect potential PII columns."""
        name_lower = column.name.lower()
        
        pii_patterns = {
            'email': ['email', 'e_mail', 'e-mail', 'mail'],
            'phone': ['phone', 'mobile', 'cell', 'tel', 'fax'],
            'ssn': ['ssn', 'social_security', 'social_sec'],
            'address': ['address', 'street', 'addr'],
            'name': ['first_name', 'last_name', 'full_name', 'firstname', 'lastname'],
            'dob': ['birth', 'dob', 'birthday'],
            'credit_card': ['card_number', 'credit_card', 'cc_num'],
        }
        
        for pii_type, patterns in pii_patterns.items():
            if any(p in name_lower for p in patterns):
                column.warnings.append(f"Potential PII detected: {pii_type}")
                break
    
    def _detect_quality_issues(self, table: TableProfile) -> List[DataQualityIssue]:
        """Detect data quality issues in a table."""
        issues = []
        
        # Check for missing primary key
        if not table.has_primary_key:
            issues.append(DataQualityIssue(
                table=table.name,
                column=None,
                severity=DataQualitySeverity.WARNING,
                issue_type="missing_primary_key",
                description=f"Table '{table.name}' has no primary key defined",
                recommendation="Consider adding a primary key for data integrity and query performance",
            ))
        
        # Check each column
        for col_name, col in table.columns.items():
            # High null percentage
            if col.null_percentage >= self.profiling_config.null_threshold_critical:
                issues.append(DataQualityIssue(
                    table=table.name,
                    column=col_name,
                    severity=DataQualitySeverity.CRITICAL,
                    issue_type="high_null_rate",
                    description=f"Column '{col_name}' has {col.null_percentage:.1%} NULL values",
                    recommendation="Investigate if column is necessary or add data validation",
                    affected_rows=col.null_count,
                ))
            elif col.null_percentage >= self.profiling_config.null_threshold_warning:
                issues.append(DataQualityIssue(
                    table=table.name,
                    column=col_name,
                    severity=DataQualitySeverity.WARNING,
                    issue_type="moderate_null_rate",
                    description=f"Column '{col_name}' has {col.null_percentage:.1%} NULL values",
                    recommendation="Consider if NULL values are expected or require cleanup",
                    affected_rows=col.null_count,
                ))
            
            # Single value (no variation)
            if col.distinct_count == 1 and table.row_count > 100:
                issues.append(DataQualityIssue(
                    table=table.name,
                    column=col_name,
                    severity=DataQualitySeverity.INFO,
                    issue_type="single_value",
                    description=f"Column '{col_name}' contains only one distinct value",
                    recommendation="Consider if this column is necessary or if it should be removed",
                ))
            
            # Non-nullable FK with nulls
            if col.is_foreign_key and col.null_count > 0 and not col.nullable:
                issues.append(DataQualityIssue(
                    table=table.name,
                    column=col_name,
                    severity=DataQualitySeverity.CRITICAL,
                    issue_type="fk_nulls",
                    description=f"Foreign key column '{col_name}' has NULL values but is marked NOT NULL",
                    recommendation="Investigate potential data integrity issues",
                    affected_rows=col.null_count,
                ))
        
        return issues
    
    def detect_orphan_records(self, store: IntelligenceStore) -> List[DataQualityIssue]:
        """
        Detect orphan records (FK values with no parent).
        
        This is a more expensive operation that checks referential integrity.
        """
        issues = []
        
        for rel in store.relationships:
            source_table = rel.source_table
            target_table = rel.target_table
            source_cols = rel.source_columns
            target_cols = rel.target_columns
            
            if len(source_cols) != 1 or len(target_cols) != 1:
                continue  # Skip composite keys for simplicity
            
            source_col = source_cols[0]
            target_col = target_cols[0]
            
            try:
                # Check for orphan records
                query = f"""
                    SELECT COUNT(*) as orphan_count
                    FROM "{source_table}" s
                    LEFT JOIN "{target_table}" t ON s."{source_col}" = t."{target_col}"
                    WHERE s."{source_col}" IS NOT NULL AND t."{target_col}" IS NULL
                """
                result = self.conn_manager.execute_query(query)
                
                if result and result[0]['orphan_count'] > 0:
                    orphan_count = result[0]['orphan_count']
                    issues.append(DataQualityIssue(
                        table=source_table,
                        column=source_col,
                        severity=DataQualitySeverity.CRITICAL,
                        issue_type="orphan_records",
                        description=f"{orphan_count} records in '{source_table}.{source_col}' reference non-existent values in '{target_table}.{target_col}'",
                        recommendation="Review foreign key constraint enforcement or clean up orphan records",
                        affected_rows=orphan_count,
                    ))
            except Exception as e:
                logger.debug(f"Error checking orphans for {source_table}.{source_col}: {e}")
        
        return issues
    
    def get_profiling_summary(self, store: IntelligenceStore) -> Dict[str, Any]:
        """Get a summary of profiling results."""
        total_columns = 0
        nullable_columns = 0
        columns_with_nulls = 0
        high_null_columns = 0
        
        for table in store.tables.values():
            for col in table.columns.values():
                total_columns += 1
                if col.nullable:
                    nullable_columns += 1
                if col.null_count > 0:
                    columns_with_nulls += 1
                if col.null_percentage > 0.5:
                    high_null_columns += 1
        
        return {
            "total_columns_profiled": total_columns,
            "nullable_columns": nullable_columns,
            "columns_with_nulls": columns_with_nulls,
            "high_null_columns": high_null_columns,
            "quality_issues": len(store.quality_issues),
            "critical_issues": len([i for i in store.quality_issues 
                                   if i.severity == DataQualitySeverity.CRITICAL]),
            "overall_quality_score": store.overall_quality_score,
        }

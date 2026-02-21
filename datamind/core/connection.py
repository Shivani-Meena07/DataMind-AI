"""
Database Connection Manager

Handles database connections for multiple database types with
connection pooling, automatic reconnection, and health checking.
"""

from typing import Optional, Any, List, Dict, Generator
from contextlib import contextmanager
import logging

from sqlalchemy import create_engine, text, inspect, MetaData
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

from datamind.config import DatabaseConfig, DatabaseType

logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors."""
    pass


class DatabaseConnectionManager:
    """
    Manages database connections with support for multiple database types.
    
    Features:
    - Connection pooling for performance
    - Automatic reconnection on failure
    - Health checking
    - Context manager support for safe connection handling
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize the connection manager.
        
        Args:
            config: Database configuration object
        """
        self.config = config
        self._engine: Optional[Engine] = None
        self._metadata: Optional[MetaData] = None
        
    @property
    def engine(self) -> Engine:
        """Get or create the database engine."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine
    
    @property
    def metadata(self) -> MetaData:
        """Get reflected metadata."""
        if self._metadata is None:
            self._metadata = MetaData()
            self._metadata.reflect(bind=self.engine)
        return self._metadata
    
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine with appropriate settings."""
        connection_string = self.config.get_connection_string()
        
        # Engine configuration based on database type
        engine_kwargs = {
            "pool_pre_ping": True,  # Enable connection health checks
            "pool_recycle": 3600,   # Recycle connections after 1 hour
        }
        
        # SQLite doesn't support connection pooling the same way
        if self.config.db_type != DatabaseType.SQLITE:
            engine_kwargs.update({
                "poolclass": QueuePool,
                "pool_size": 5,
                "max_overflow": 10,
            })
        
        try:
            engine = create_engine(connection_string, **engine_kwargs)
            logger.info(f"Created database engine for {self.config.db_type.value}")
            return engine
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to create engine: {e}")
    
    @contextmanager
    def get_connection(self) -> Generator[Connection, None, None]:
        """
        Get a database connection as a context manager.
        
        Yields:
            Database connection
            
        Example:
            with manager.get_connection() as conn:
                result = conn.execute(query)
        """
        connection = None
        try:
            connection = self.engine.connect()
            yield connection
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            raise DatabaseConnectionError(f"Connection error: {e}")
        finally:
            if connection:
                connection.close()
    
    def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_inspector(self):
        """Get SQLAlchemy inspector for metadata operations."""
        return inspect(self.engine)
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Execute a query and return results as list of dictionaries.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            List of row dictionaries
        """
        with self.get_connection() as conn:
            result = conn.execute(text(query), params or {})
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
    
    def execute_query_chunked(
        self, 
        query: str, 
        chunk_size: int = 1000,
        params: Optional[Dict] = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Execute a query and yield results in chunks.
        
        Useful for processing large result sets without loading
        everything into memory.
        
        Args:
            query: SQL query string
            chunk_size: Number of rows per chunk
            params: Optional query parameters
            
        Yields:
            Chunks of row dictionaries
        """
        with self.get_connection() as conn:
            result = conn.execute(text(query), params or {})
            columns = result.keys()
            
            while True:
                rows = result.fetchmany(chunk_size)
                if not rows:
                    break
                yield [dict(zip(columns, row)) for row in rows]
    
    def get_table_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of table names in the database."""
        inspector = self.get_inspector()
        return inspector.get_table_names(schema=schema or self.config.schema)
    
    def get_view_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of view names in the database."""
        inspector = self.get_inspector()
        return inspector.get_view_names(schema=schema or self.config.schema)
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database server information."""
        info = {
            "type": self.config.db_type.value,
            "dialect": str(self.engine.dialect.name),
        }
        
        try:
            with self.get_connection() as conn:
                if self.config.db_type == DatabaseType.POSTGRESQL:
                    result = conn.execute(text("SELECT version()"))
                    info["version"] = result.scalar()
                elif self.config.db_type == DatabaseType.MYSQL:
                    result = conn.execute(text("SELECT version()"))
                    info["version"] = result.scalar()
                elif self.config.db_type == DatabaseType.SQLITE:
                    result = conn.execute(text("SELECT sqlite_version()"))
                    info["version"] = result.scalar()
        except Exception as e:
            logger.warning(f"Could not get database version: {e}")
            info["version"] = "Unknown"
        
        return info
    
    def close(self):
        """Close the database connection and dispose of the engine."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._metadata = None
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

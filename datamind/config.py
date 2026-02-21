"""
Configuration management for DataMind AI.

Handles all configuration options including database connections,
LLM settings, and output preferences.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import os
from pathlib import Path
import yaml


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    SQLSERVER = "sqlserver"
    ORACLE = "oracle"


class OutputFormat(Enum):
    """Supported output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    db_type: DatabaseType
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    db_path: Optional[str] = None  # For SQLite
    schema: Optional[str] = None  # For PostgreSQL schema filtering
    ssl_mode: Optional[str] = None
    connection_timeout: int = 30
    
    def get_connection_string(self) -> str:
        """Generate SQLAlchemy connection string."""
        if self.db_type == DatabaseType.SQLITE:
            return f"sqlite:///{self.db_path}"
        elif self.db_type == DatabaseType.POSTGRESQL:
            auth = f"{self.username}:{self.password}@" if self.username else ""
            return f"postgresql+psycopg2://{auth}{self.host}:{self.port}/{self.database}"
        elif self.db_type == DatabaseType.MYSQL:
            auth = f"{self.username}:{self.password}@" if self.username else ""
            return f"mysql+pymysql://{auth}{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


@dataclass
class LLMConfig:
    """LLM configuration for inference engine."""
    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    api_key: Optional[str] = None
    temperature: float = 0.3  # Lower for more deterministic outputs
    max_tokens: int = 4096
    timeout: int = 60
    retry_attempts: int = 3
    batch_size: int = 5  # Number of entities to process in one LLM call
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")


@dataclass
class ProfilingConfig:
    """Data profiling configuration."""
    sample_size: int = 10000  # Max rows to sample per table
    null_threshold_warning: float = 0.1  # 10% nulls triggers warning
    null_threshold_critical: float = 0.5  # 50% nulls triggers critical
    uniqueness_threshold: float = 0.95  # 95% unique = likely identifier
    high_cardinality_threshold: int = 1000  # For categorical detection
    min_rows_for_profiling: int = 10  # Skip profiling for tiny tables
    detect_pii: bool = True  # Flag potential PII columns
    compute_distributions: bool = True  # Compute value distributions


@dataclass
class OutputConfig:
    """Output configuration."""
    format: OutputFormat = OutputFormat.MARKDOWN
    output_dir: str = "./output"
    include_sample_queries: bool = True
    include_er_diagram: bool = True
    include_data_quality: bool = True
    include_business_glossary: bool = True
    max_sample_values: int = 10  # Sample values to show per column
    language: str = "en"  # Output language


@dataclass
class DataMindConfig:
    """Main configuration container."""
    database: DatabaseConfig = None
    llm: LLMConfig = field(default_factory=LLMConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Processing options
    parallel_workers: int = 4
    verbose: bool = False
    cache_enabled: bool = True
    cache_dir: str = "./.datamind_cache"
    
    # Table filtering
    include_tables: Optional[List[str]] = None
    exclude_tables: Optional[List[str]] = None
    exclude_system_tables: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "DataMindConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "DataMindConfig":
        """Create config from dictionary."""
        db_data = data.get("database", {})
        db_config = DatabaseConfig(
            db_type=DatabaseType(db_data.get("type", "sqlite")),
            host=db_data.get("host"),
            port=db_data.get("port"),
            database=db_data.get("database"),
            username=db_data.get("username"),
            password=db_data.get("password"),
            db_path=db_data.get("path"),
            schema=db_data.get("schema"),
        )
        
        llm_data = data.get("llm", {})
        llm_config = LLMConfig(
            provider=llm_data.get("provider", "openai"),
            model=llm_data.get("model", "gpt-4-turbo-preview"),
            api_key=llm_data.get("api_key"),
            temperature=llm_data.get("temperature", 0.3),
        )
        
        return cls(
            database=db_config,
            llm=llm_config,
            verbose=data.get("verbose", False),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "database": {
                "type": self.database.db_type.value if self.database else None,
                "host": self.database.host if self.database else None,
                "port": self.database.port if self.database else None,
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
            },
            "profiling": {
                "sample_size": self.profiling.sample_size,
                "null_threshold_warning": self.profiling.null_threshold_warning,
            },
            "output": {
                "format": self.output.format.value,
                "output_dir": self.output.output_dir,
            },
        }


def create_default_config(
    db_type: str,
    db_path: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    output_dir: str = "./output",
) -> DataMindConfig:
    """Factory function to create a default configuration."""
    
    db_config = DatabaseConfig(
        db_type=DatabaseType(db_type),
        db_path=db_path,
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
    )
    
    return DataMindConfig(
        database=db_config,
        llm=LLMConfig(),
        profiling=ProfilingConfig(),
        output=OutputConfig(output_dir=output_dir),
    )

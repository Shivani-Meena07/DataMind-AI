"""
DataMind AI - Main Entry Point

Command-line interface and orchestration for the database documentation agent.
"""

import sys
import logging
from typing import Optional
from pathlib import Path
import time

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from datamind.config import (
    DataMindConfig, 
    DatabaseConfig, 
    DatabaseType, 
    OutputFormat,
    create_default_config,
)
from datamind.core.connection import DatabaseConnectionManager
from datamind.core.schema_scanner import SchemaScanner
from datamind.core.data_profiler import DataProfiler
from datamind.core.relationship_analyzer import RelationshipAnalyzer
from datamind.core.intelligence_store import IntelligenceStore
from datamind.inference.llm_engine import LLMInferenceEngine
from datamind.generators.doc_generator import DocumentationGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


class DataMindAgent:
    """
    Main orchestrator for the DataMind database documentation agent.
    
    This class coordinates all modules to:
    1. Connect to the database
    2. Scan the schema
    3. Profile the data
    4. Analyze relationships
    5. Run LLM inference
    6. Generate documentation
    """
    
    def __init__(self, config: DataMindConfig):
        """
        Initialize the DataMind agent.
        
        Args:
            config: Complete configuration object
        """
        self.config = config
        self.store = IntelligenceStore()
        
        # Initialize components
        self.conn_manager: Optional[DatabaseConnectionManager] = None
        self.schema_scanner: Optional[SchemaScanner] = None
        self.data_profiler: Optional[DataProfiler] = None
        self.relationship_analyzer: Optional[RelationshipAnalyzer] = None
        self.llm_engine: Optional[LLMInferenceEngine] = None
        self.doc_generator: Optional[DocumentationGenerator] = None
    
    def run(self, skip_llm: bool = False) -> str:
        """
        Execute the complete documentation pipeline.
        
        Args:
            skip_llm: Skip LLM inference (faster, less detailed)
            
        Returns:
            Path to generated documentation
        """
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            # Phase 1: Connect to database
            task1 = progress.add_task("Connecting to database...", total=1)
            self._connect()
            progress.update(task1, completed=1)
            
            # Phase 2: Scan schema
            task2 = progress.add_task("Scanning database schema...", total=1)
            self._scan_schema()
            progress.update(task2, completed=1)
            
            # Phase 3: Profile data
            task3 = progress.add_task("Profiling data quality...", total=1)
            self._profile_data()
            progress.update(task3, completed=1)
            
            # Phase 4: Analyze relationships
            task4 = progress.add_task("Analyzing relationships...", total=1)
            self._analyze_relationships()
            progress.update(task4, completed=1)
            
            # Phase 5: LLM inference (optional)
            if not skip_llm:
                task5 = progress.add_task("Running AI inference...", total=1)
                self._run_inference()
                progress.update(task5, completed=1)
            
            # Phase 6: Generate documentation
            task6 = progress.add_task("Generating documentation...", total=1)
            doc_path = self._generate_documentation()
            progress.update(task6, completed=1)
        
        # Print summary
        elapsed = time.time() - start_time
        self._print_summary(elapsed, doc_path)
        
        # Cleanup
        self._cleanup()
        
        return doc_path
    
    def _connect(self):
        """Establish database connection."""
        self.conn_manager = DatabaseConnectionManager(self.config.database)
        
        if not self.conn_manager.test_connection():
            raise ConnectionError("Failed to connect to database")
        
        db_info = self.conn_manager.get_database_info()
        self.store.database_type = db_info.get("type")
        self.store.database_version = db_info.get("version")
        self.store.database_name = self.config.database.database or self.config.database.db_path
        
        logger.info(f"Connected to {self.store.database_type} database")
    
    def _scan_schema(self):
        """Scan database schema."""
        self.schema_scanner = SchemaScanner(self.conn_manager, self.config)
        self.store = self.schema_scanner.scan(self.store)
        
        logger.info(f"Scanned {len(self.store.tables)} tables")
    
    def _profile_data(self):
        """Profile data quality."""
        self.data_profiler = DataProfiler(self.conn_manager, self.config)
        self.store = self.data_profiler.profile(self.store)
        
        # Also check for orphan records
        orphan_issues = self.data_profiler.detect_orphan_records(self.store)
        for issue in orphan_issues:
            self.store.add_quality_issue(issue)
        
        logger.info(f"Found {len(self.store.quality_issues)} data quality issues")
    
    def _analyze_relationships(self):
        """Analyze table relationships."""
        self.relationship_analyzer = RelationshipAnalyzer(self.conn_manager, self.config)
        self.store = self.relationship_analyzer.analyze(self.store)
        
        logger.info(f"Found {len(self.store.relationships)} relationships")
    
    def _run_inference(self):
        """Run LLM inference."""
        self.llm_engine = LLMInferenceEngine(self.config)
        self.store = self.llm_engine.infer(self.store)
        
        logger.info("LLM inference complete")
    
    def _generate_documentation(self) -> str:
        """Generate and save documentation."""
        self.doc_generator = DocumentationGenerator(self.config)
        document = self.doc_generator.generate(self.store)
        
        # Determine filename
        db_name = self.store.database_name or "database"
        if "/" in db_name or "\\" in db_name:
            db_name = Path(db_name).stem
        
        filename = f"{db_name}_user_manual"
        doc_path = self.doc_generator.save(document, filename)
        
        return doc_path
    
    def _print_summary(self, elapsed: float, doc_path: str):
        """Print execution summary."""
        console.print()
        
        # Create summary table
        table = Table(title="📊 Analysis Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        stats = self.store.get_statistics()
        table.add_row("Tables Analyzed", str(stats["total_tables"]))
        table.add_row("Total Columns", str(stats["total_columns"]))
        table.add_row("Total Records", f"{stats['total_rows']:,}")
        table.add_row("Relationships Found", str(stats["total_relationships"]))
        table.add_row("Quality Issues", str(stats["total_quality_issues"]))
        table.add_row("Quality Score", f"{stats['overall_quality_score']:.1f}/100")
        table.add_row("Processing Time", f"{elapsed:.1f}s")
        
        console.print(table)
        console.print()
        
        # Print output location
        console.print(Panel(
            f"📘 Documentation saved to:\n[bold green]{doc_path}[/bold green]",
            title="Output",
            border_style="green"
        ))
    
    def _cleanup(self):
        """Cleanup resources."""
        if self.conn_manager:
            self.conn_manager.close()


# CLI Commands
@click.group()
@click.version_option(version="1.0.0", prog_name="DataMind AI")
def cli():
    """DataMind AI - Intelligent Database Documentation Agent"""
    pass


@cli.command()
@click.option('--db-type', '-t', type=click.Choice(['sqlite', 'postgresql', 'mysql']),
              required=True, help='Database type')
@click.option('--db-path', '-p', type=click.Path(), help='Path to SQLite database')
@click.option('--host', '-h', help='Database host')
@click.option('--port', type=int, help='Database port')
@click.option('--database', '-d', help='Database name')
@click.option('--user', '-u', help='Database username')
@click.option('--password', help='Database password')
@click.option('--output', '-o', default='./output', help='Output directory')
@click.option('--skip-llm', is_flag=True, help='Skip LLM inference')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analyze(db_type, db_path, host, port, database, user, password, output, skip_llm, verbose):
    """
    Analyze a database and generate documentation.
    
    Examples:
    
        datamind analyze -t sqlite -p ./data/olist.db
        
        datamind analyze -t postgresql -h localhost -P 5432 -d mydb -u admin -p secret
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(Panel(
        "[bold blue]DataMind AI[/bold blue]\n"
        "[dim]Intelligent Database Documentation Agent[/dim]",
        border_style="blue"
    ))
    
    try:
        # Create configuration
        config = create_default_config(
            db_type=db_type,
            db_path=db_path,
            host=host,
            port=port,
            database=database,
            username=user,
            password=password,
            output_dir=output,
        )
        
        # Run agent
        agent = DataMindAgent(config)
        doc_path = agent.run(skip_llm=skip_llm)
        
        console.print("\n[bold green]✓ Documentation generated successfully![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {e}[/bold red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--db-type', '-t', type=click.Choice(['sqlite', 'postgresql', 'mysql']),
              required=True, help='Database type')
@click.option('--db-path', '-p', type=click.Path(), help='Path to SQLite database')
@click.option('--host', '-h', help='Database host')
@click.option('--port', type=int, help='Database port')
@click.option('--database', '-d', help='Database name')
@click.option('--user', '-u', help='Database username')
@click.option('--password', help='Database password')
def test_connection(db_type, db_path, host, port, database, user, password):
    """Test database connection."""
    try:
        db_config = DatabaseConfig(
            db_type=DatabaseType(db_type),
            db_path=db_path,
            host=host,
            port=port,
            database=database,
            username=user,
            password=password,
        )
        
        conn_manager = DatabaseConnectionManager(db_config)
        
        if conn_manager.test_connection():
            console.print("[bold green]✓ Connection successful![/bold green]")
            
            db_info = conn_manager.get_database_info()
            console.print(f"  Database Type: {db_info.get('type')}")
            console.print(f"  Version: {db_info.get('version')}")
            
            tables = conn_manager.get_table_names()
            console.print(f"  Tables Found: {len(tables)}")
        else:
            console.print("[bold red]✗ Connection failed![/bold red]")
            
        conn_manager.close()
        
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")


@cli.command()
def version():
    """Show version information."""
    console.print(Panel(
        "[bold]DataMind AI[/bold] v1.0.0\n\n"
        "An AI system that replaces tribal knowledge in databases.\n\n"
        "Components:\n"
        "  • Schema Scanner\n"
        "  • Data Profiler\n"
        "  • Relationship Analyzer\n"
        "  • LLM Inference Engine\n"
        "  • Documentation Generator",
        title="About",
        border_style="blue"
    ))


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()

"""
DataMind AI Demo Runner

This script demonstrates the full capabilities of DataMind AI
using the Olist E-Commerce dataset.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from datamind.demo.setup_olist import setup_olist_database
from datamind.config import create_default_config
from datamind.main import DataMindAgent

console = Console()


def run_demo(skip_llm: bool = True):
    """
    Run the complete DataMind AI demonstration.
    
    Args:
        skip_llm: Skip LLM inference (set to False if you have an API key)
    """
    console.print(Panel(
        "[bold blue]🧠 DataMind AI Demo[/bold blue]\n\n"
        "[dim]Intelligent Database Documentation Agent[/dim]\n\n"
        "This demo will:\n"
        "1. Create a sample Olist E-Commerce database\n"
        "2. Analyze the schema and data\n"
        "3. Generate a comprehensive user manual",
        title="Welcome",
        border_style="blue"
    ))
    
    console.print()
    
    # Step 1: Create the demo database
    console.print("[bold cyan]Step 1: Creating Olist E-Commerce Database[/bold cyan]")
    console.print("-" * 50)
    
    db_path = "./data/olist.db"
    setup_olist_database(db_path)
    
    console.print()
    
    # Step 2: Run DataMind Agent
    console.print("[bold cyan]Step 2: Running DataMind AI Analysis[/bold cyan]")
    console.print("-" * 50)
    
    # Create configuration
    config = create_default_config(
        db_type="sqlite",
        db_path=db_path,
        output_dir="./output",
    )
    
    # Check for API key
    if os.environ.get("OPENAI_API_KEY") and not skip_llm:
        console.print("[green]OpenAI API key detected - LLM inference enabled[/green]")
        use_llm = True
    else:
        console.print("[yellow]No OpenAI API key - running without LLM inference[/yellow]")
        console.print("[dim]Set OPENAI_API_KEY environment variable for full AI features[/dim]")
        use_llm = False
    
    console.print()
    
    # Run the agent
    agent = DataMindAgent(config)
    doc_path = agent.run(skip_llm=not use_llm or skip_llm)
    
    console.print()
    console.print("[bold green]✓ Demo completed successfully![/bold green]")
    console.print()
    console.print(f"[dim]Open the generated documentation at:[/dim]")
    console.print(f"[bold]{doc_path}[/bold]")
    
    return doc_path


def main():
    """Main entry point for demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DataMind AI Demo")
    parser.add_argument(
        "--with-llm", 
        action="store_true",
        help="Enable LLM inference (requires OPENAI_API_KEY)"
    )
    
    args = parser.parse_args()
    
    run_demo(skip_llm=not args.with_llm)


if __name__ == "__main__":
    main()

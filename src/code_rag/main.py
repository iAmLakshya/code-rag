"""Main entry point for Code RAG."""

import argparse
import asyncio
import sys
from pathlib import Path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Code RAG - AI-powered code search and analysis"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # TUI command (default)
    tui_parser = subparsers.add_parser("tui", help="Launch the TUI interface")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a repository")
    index_parser.add_argument("path", help="Path to the repository")
    index_parser.add_argument(
        "--name", "-n", help="Project name (defaults to directory name)"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the indexed codebase")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--limit", "-l", type=int, default=10, help="Max results")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the codebase")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", "-l", type=int, default=10, help="Max results")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show indexing status")

    args = parser.parse_args()

    if args.command is None or args.command == "tui":
        run_tui()
    elif args.command == "index":
        asyncio.run(run_index(args.path, args.name))
    elif args.command == "query":
        asyncio.run(run_query(args.question, args.limit))
    elif args.command == "search":
        asyncio.run(run_search(args.query, args.limit))
    elif args.command == "status":
        asyncio.run(run_status())
    else:
        parser.print_help()


def run_tui():
    """Run the TUI interface."""
    from code_rag.tui.app import run_app

    run_app()


async def run_index(path: str, name: str | None = None):
    """Run indexing from CLI."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    from code_rag.pipeline.orchestrator import PipelineOrchestrator
    from code_rag.pipeline.progress import PipelineProgress

    console = Console()
    path = Path(path).resolve()

    if not path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        sys.exit(1)

    console.print(f"[blue]Indexing repository: {path}[/blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Indexing...", total=100)

        def on_progress(p: PipelineProgress):
            progress.update(task, completed=p.overall_percentage)
            if p.current_stage:
                progress.update(
                    task,
                    description=f"{p.current_stage.value.replace('_', ' ').title()}...",
                )

        orchestrator = PipelineOrchestrator(
            repo_path=path,
            project_name=name,
            progress_callback=on_progress,
        )

        try:
            result = await orchestrator.run()
            progress.update(task, completed=100)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    console.print()
    console.print("[green]Indexing complete![/green]")
    console.print(f"  Files indexed: {result['files_indexed']}")
    console.print(f"  Entities found: {result['entities_found']}")
    console.print(f"  Graph nodes: {result['graph_nodes']}")
    console.print(f"  Chunks embedded: {result['chunks_embedded']}")
    console.print(f"  Time elapsed: {result['elapsed_seconds']:.1f}s")


async def run_query(question: str, limit: int = 10):
    """Run a query from CLI."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    from code_rag.query.engine import QueryEngine

    console = Console()

    console.print(f"[blue]Query:[/blue] {question}")
    console.print()

    async with QueryEngine() as engine:
        try:
            result = await engine.query(question, limit=limit)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[yellow]Make sure Docker containers are running.[/yellow]")
            sys.exit(1)

    # Print answer
    console.print(Panel(Markdown(result.answer), title="Answer", border_style="green"))
    console.print()

    # Print sources
    if result.sources:
        console.print("[blue]Sources:[/blue]")
        for i, source in enumerate(result.sources[:5], 1):
            console.print(
                f"  {i}. {source.file_path}:{source.start_line} "
                f"[dim]({source.entity_name})[/dim]"
            )


async def run_search(query: str, limit: int = 10):
    """Run a search from CLI."""
    from rich.console import Console
    from rich.table import Table

    from code_rag.query.engine import QueryEngine

    console = Console()

    async with QueryEngine() as engine:
        try:
            results = await engine.search(query, limit=limit)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Search Results: {query}")
    table.add_column("Score", style="cyan", width=8)
    table.add_column("Type", style="magenta", width=10)
    table.add_column("Name", style="green")
    table.add_column("File", style="dim")
    table.add_column("Lines", style="dim", width=10)

    for result in results:
        table.add_row(
            f"{result.score:.2f}",
            result.entity_type,
            result.entity_name,
            result.file_path,
            f"{result.start_line or '?'}-{result.end_line or '?'}",
        )

    console.print(table)


async def run_status():
    """Show indexing status from CLI."""
    from rich.console import Console
    from rich.table import Table

    from code_rag.query.engine import QueryEngine

    console = Console()

    async with QueryEngine() as engine:
        try:
            stats = await engine.get_statistics()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[yellow]Make sure Docker containers are running.[/yellow]")
            sys.exit(1)

    table = Table(title="Codebase Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")

    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


if __name__ == "__main__":
    main()

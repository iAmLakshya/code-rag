import argparse
import asyncio
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Code RAG - AI-powered code search and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  code-rag index ./my-project              Index a repository
  code-rag index ./my-project --force      Force re-index (regenerate summaries)
  code-rag projects list                   List all indexed projects
  code-rag projects delete my-project      Delete a project
  code-rag query "How does auth work?"     Query the codebase
  code-rag search "password validation"    Search for code
  code-rag status                          Show database statistics
  code-rag settings                        Show current configuration
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    index_parser = subparsers.add_parser("index", help="Index a repository")
    index_parser.add_argument("path", help="Path to the repository")
    index_parser.add_argument(
        "--name", "-n", help="Project name (defaults to directory name)"
    )
    index_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force re-index all files (bypass incremental check)"
    )

    projects_parser = subparsers.add_parser("projects", help="Manage indexed projects")
    projects_subparsers = projects_parser.add_subparsers(dest="projects_command")

    projects_subparsers.add_parser("list", help="List all projects")

    projects_show_parser = projects_subparsers.add_parser("show", help="Show project details")
    projects_show_parser.add_argument("name", help="Project name")

    projects_delete_parser = projects_subparsers.add_parser("delete", help="Delete a project")
    projects_delete_parser.add_argument("name", help="Project name")
    projects_delete_parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )

    query_parser = subparsers.add_parser("query", help="Query the indexed codebase")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--project", "-p", help="Project name to query")
    query_parser.add_argument("--limit", "-l", type=int, default=15, help="Max results")
    query_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed execution stats and reasoning"
    )

    search_parser = subparsers.add_parser("search", help="Search the codebase")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--project", "-p", help="Project name to search")
    search_parser.add_argument("--limit", "-l", type=int, default=15, help="Max results")

    subparsers.add_parser("status", help="Show indexing status and statistics")

    subparsers.add_parser("settings", help="Show current configuration")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)
    elif args.command == "index":
        asyncio.run(run_index(args.path, args.name, args.force))
    elif args.command == "projects":
        if args.projects_command == "list" or args.projects_command is None:
            asyncio.run(run_projects_list())
        elif args.projects_command == "show":
            asyncio.run(run_projects_show(args.name))
        elif args.projects_command == "delete":
            asyncio.run(run_projects_delete(args.name, args.yes))
    elif args.command == "query":
        asyncio.run(run_query(args.question, args.limit, args.verbose, args.project))
    elif args.command == "search":
        asyncio.run(run_search(args.query, args.limit, args.project))
    elif args.command == "status":
        asyncio.run(run_status())
    elif args.command == "settings":
        run_settings()
    else:
        parser.print_help()


async def run_index(path: str, name: str | None = None, force: bool = False):
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from code_rag.pipeline.orchestrator import PipelineOrchestrator
    from code_rag.pipeline.progress import PipelineProgress

    console = Console()
    path = Path(path).resolve()

    if not path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        sys.exit(1)

    if not path.is_dir():
        console.print(f"[red]Error: Path is not a directory: {path}[/red]")
        sys.exit(1)

    if force:
        mode_str = "[bold magenta]Force indexing[/bold magenta]"
    else:
        mode_str = "[bold blue]Indexing[/bold blue]"
    console.print(f"{mode_str} repository: [cyan]{path}[/cyan]")
    if force:
        console.print("[dim]All files will be re-processed (summaries regenerated)[/dim]")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Starting...", total=100)

        def on_progress(p: PipelineProgress):
            progress.update(task, completed=p.overall_percentage)
            if p.current_stage:
                stage_name = p.current_stage.value.replace("_", " ").title()
                stage_progress = p.stages.get(p.current_stage)
                if stage_progress and stage_progress.total > 0:
                    detail = f"({stage_progress.completed}/{stage_progress.total})"
                else:
                    detail = ""
                progress.update(task, description=f"{stage_name} {detail}")

        orchestrator = PipelineOrchestrator(
            repo_path=path,
            project_name=name,
            progress_callback=on_progress,
            force=force,
        )

        try:
            result = await orchestrator.run()
            progress.update(task, completed=100, description="Complete")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            sys.exit(1)

    console.print()
    console.print("[green]Indexing complete![/green]")
    console.print(f"  [cyan]Files indexed:[/cyan]    {result['files_indexed']}")
    console.print(f"  [cyan]Entities found:[/cyan]   {result['entities_found']}")
    console.print(f"  [cyan]Graph nodes:[/cyan]      {result['graph_nodes']}")
    console.print(f"  [cyan]Summaries:[/cyan]        {result['summaries']}")
    console.print(f"  [cyan]Chunks embedded:[/cyan]  {result['chunks_embedded']}")
    console.print(f"  [cyan]Time elapsed:[/cyan]     {result['elapsed_seconds']:.1f}s")


async def run_projects_list():
    from rich.console import Console
    from rich.table import Table

    from code_rag.projects.manager import ProjectManager

    console = Console()

    try:
        async with ProjectManager() as manager:
            projects = await manager.list_projects()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure Docker containers are running.[/yellow]")
        sys.exit(1)

    if not projects:
        console.print("[yellow]No projects found.[/yellow]")
        console.print("[dim]Use 'code-rag index <path>' to index a repository.[/dim]")
        return

    table = Table(title="Indexed Projects")
    table.add_column("Name", style="cyan")
    table.add_column("Files", style="green", justify="right")
    table.add_column("Entities", style="green", justify="right")
    table.add_column("Chunks", style="green", justify="right")
    table.add_column("Last Indexed", style="dim")

    for project in projects:
        last_indexed = ""
        if project.last_indexed_at:
            last_indexed = project.last_indexed_at.strftime("%Y-%m-%d %H:%M")

        table.add_row(
            project.name,
            str(project.total_files),
            str(project.total_entities),
            str(project.total_chunks),
            last_indexed,
        )

    console.print(table)


async def run_projects_show(name: str):
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from code_rag.projects.manager import ProjectManager

    console = Console()

    try:
        async with ProjectManager() as manager:
            stats = await manager.get_project_stats(name)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure Docker containers are running.[/yellow]")
        sys.exit(1)

    if not stats:
        console.print(f"[red]Project '{name}' not found.[/red]")
        sys.exit(1)

    created = stats["created_at"].strftime("%Y-%m-%d %H:%M") if stats["created_at"] else "N/A"
    last_indexed = (
        stats["last_indexed_at"].strftime("%Y-%m-%d %H:%M")
        if stats["last_indexed_at"]
        else "N/A"
    )

    info = (
        f"[cyan]Files:[/cyan] {stats['total_files']}\n"
        f"[cyan]Entities:[/cyan] {stats['total_entities']}\n"
        f"[cyan]Chunks:[/cyan] {stats['total_chunks']}\n"
        f"[cyan]Created:[/cyan] {created}\n"
        f"[cyan]Last Indexed:[/cyan] {last_indexed}"
    )
    console.print(Panel(info, title=f"Project: {name}", border_style="cyan"))

    if stats["indexes"]:
        table = Table(title="Indexes")
        table.add_column("Path", style="dim")
        table.add_column("Files", justify="right")
        table.add_column("Entities", justify="right")

        for idx in stats["indexes"]:
            table.add_row(
                idx.get("path", "N/A"),
                str(idx.get("file_count", 0)),
                str(idx.get("entity_count", 0)),
            )

        console.print(table)


async def run_projects_delete(name: str, skip_confirm: bool = False):
    from rich.console import Console

    from code_rag.projects.manager import ProjectManager

    console = Console()

    if not skip_confirm:
        console.print(f"[yellow]Delete project '{name}'?[/yellow]")
        console.print("[dim]This will remove all graph nodes, vectors, and indexes.[/dim]")
        response = input("Type 'yes' to confirm: ")
        if response.lower() != "yes":
            console.print("[dim]Cancelled.[/dim]")
            return

    try:
        async with ProjectManager() as manager:
            deleted = await manager.delete_project(name)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    if deleted:
        console.print(f"[green]Deleted project '{name}'.[/green]")
    else:
        console.print(f"[red]Project '{name}' not found.[/red]")
        sys.exit(1)


async def run_query(
    question: str, limit: int = 15, verbose: bool = False, project: str | None = None
):
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    from code_rag.query import QueryEngine

    console = Console()

    console.print(f"[blue]Query:[/blue] {question}")
    if project:
        console.print(f"[dim]Project: {project}[/dim]")
    console.print()

    async with QueryEngine() as engine:
        try:
            result = await engine.query(question, limit=limit, project_name=project)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[yellow]Make sure Docker containers are running.[/yellow]")
            sys.exit(1)

    if verbose:
        console.print(
            Panel(
                f"[cyan]Intent:[/cyan] {result.query_plan.primary_intent.value}\n"
                f"[cyan]Entities:[/cyan] {', '.join(e.name for e in result.query_plan.entities)}\n"
                f"[cyan]Multi-hop:[/cyan] {result.query_plan.requires_multi_hop} "
                f"(max {result.query_plan.max_hops} hops)\n"
                f"[cyan]Graph entities:[/cyan] {result.context.total_entities_found}\n"
                f"[cyan]Execution time:[/cyan] {sum(result.execution_stats.values())}ms",
                title="Query Analysis",
                border_style="cyan",
            )
        )
        console.print()

        if result.context.reasoning_notes:
            console.print("[cyan]Reasoning:[/cyan]")
            for note in result.context.reasoning_notes:
                console.print(f"  - {note}")
            console.print()

    console.print(Panel(Markdown(result.answer), title="Answer", border_style="green"))
    console.print()

    if result.sources:
        console.print("[blue]Sources:[/blue]")
        for i, source in enumerate(result.sources[:5], 1):
            score_info = f"[score: {source.final_score:.2f}]" if verbose else ""
            rel_info = (
                f" [{source.relationship_path}]"
                if source.relationship_path and verbose
                else ""
            )
            console.print(
                f"  {i}. {source.file_path}:{source.start_line or '?'} "
                f"[dim]({source.entity_name}){rel_info} {score_info}[/dim]"
            )


async def run_search(query: str, limit: int = 15, project: str | None = None):
    from rich.console import Console
    from rich.table import Table

    from code_rag.query import QueryEngine

    console = Console()

    if project:
        console.print(f"[dim]Project: {project}[/dim]")

    async with QueryEngine() as engine:
        try:
            results = await engine.search(query, limit=limit, project_name=project)
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
            f"{result.final_score:.2f}",
            result.entity_type,
            result.entity_name,
            result.file_path,
            f"{result.start_line or '?'}-{result.end_line or '?'}",
        )

    console.print(table)


async def run_status():
    from rich.console import Console
    from rich.table import Table

    from code_rag.projects.manager import ProjectManager
    from code_rag.query import QueryEngine

    console = Console()

    # Get database stats
    try:
        async with QueryEngine() as engine:
            stats = await engine.get_statistics()
    except Exception as e:
        console.print(f"[red]Error connecting to databases: {e}[/red]")
        console.print("[yellow]Make sure Docker containers are running:[/yellow]")
        console.print("  docker-compose up -d")
        sys.exit(1)

    table = Table(title="Database Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)
    console.print()

    try:
        async with ProjectManager() as manager:
            projects = await manager.list_projects()
            console.print(f"[cyan]Total projects:[/cyan] {len(projects)}")
            if projects:
                console.print("[dim]Use 'code-rag projects list' for details.[/dim]")
    except Exception:
        pass


def run_settings():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from code_rag.config import get_settings

    console = Console()
    settings = get_settings()

    db_table = Table(title="Database Configuration", show_header=False)
    db_table.add_column("Setting", style="cyan")
    db_table.add_column("Value", style="green")

    db_table.add_row("Memgraph Host", settings.database.memgraph_host)
    db_table.add_row("Memgraph Port", str(settings.database.memgraph_port))
    db_table.add_row("Qdrant Host", settings.database.qdrant_host)
    db_table.add_row("Qdrant Port", str(settings.database.qdrant_port))

    console.print(db_table)
    console.print()

    ai_table = Table(title="AI Configuration", show_header=False)
    ai_table.add_column("Setting", style="cyan")
    ai_table.add_column("Value", style="green")

    ai_table.add_row("LLM Provider", settings.ai.llm_provider)
    ai_table.add_row("LLM Model", settings.ai.llm_model)
    ai_table.add_row("Embedding Provider", settings.ai.embedding_provider)
    ai_table.add_row("Embedding Model", settings.ai.embedding_model)
    ai_table.add_row("Embedding Dimensions", str(settings.ai.embedding_dimensions))
    ai_table.add_row("Temperature", str(settings.ai.llm_temperature))

    openai_key = settings.ai.openai_api_key.get_secret_value()
    openai_status = "[green]set[/green]" if openai_key else "[red]not set[/red]"
    ai_table.add_row("OpenAI API Key", openai_status)

    anthropic_key = settings.ai.anthropic_api_key.get_secret_value()
    if anthropic_key:
        ai_table.add_row("Anthropic API Key", "[green]set[/green]")

    google_key = settings.ai.google_api_key.get_secret_value()
    if google_key:
        ai_table.add_row("Google API Key", "[green]set[/green]")

    console.print(ai_table)
    console.print()

    idx_table = Table(title="Indexing Configuration", show_header=False)
    idx_table.add_column("Setting", style="cyan")
    idx_table.add_column("Value", style="green")

    idx_table.add_row("Batch Size", str(settings.indexing.batch_size))
    idx_table.add_row("Max Concurrent Requests", str(settings.indexing.max_concurrent_requests))
    idx_table.add_row("Chunk Max Tokens", str(settings.indexing.chunk_max_tokens))
    idx_table.add_row("Chunk Overlap Tokens", str(settings.indexing.chunk_overlap_tokens))

    console.print(idx_table)
    console.print()

    console.print(
        Panel(
            f"[cyan]Supported Extensions:[/cyan] {', '.join(settings.files.supported_extensions)}\n"
            f"[cyan]Ignore Patterns:[/cyan] {', '.join(settings.files.ignore_patterns[:5])}...",
            title="File Configuration",
            border_style="dim",
        )
    )


if __name__ == "__main__":
    main()

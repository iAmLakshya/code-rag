"""MCP tool definitions for Code RAG.

These tools can be registered with an MCP server to expose Code RAG
functionality to Claude Code and other MCP-compatible clients.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standard result format for MCP tools."""
    success: bool
    data: Any = None
    message: str = ""
    error: str | None = None


@dataclass
class CodeSnippet:
    """Result from code retrieval tool."""
    qualified_name: str
    source_code: str = ""
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    docstring: str | None = None
    found: bool = True
    error_message: str | None = None


@dataclass
class SearchResult:
    """Result from semantic search tool."""
    qualified_name: str
    entity_type: str
    file_path: str
    score: float
    summary: str | None = None


@dataclass
class GraphQueryResult:
    """Result from graph query tool."""
    query_used: str
    results: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""


# =============================================================================
# Tool Factory Functions
# =============================================================================

def create_index_tool(
    orchestrator_factory: callable,
) -> dict[str, Any]:
    """Create the index_repository tool.

    Args:
        orchestrator_factory: Factory function to create PipelineOrchestrator.

    Returns:
        Tool definition dict for MCP registration.
    """
    async def index_repository(
        repo_path: str,
        project_name: str | None = None,
    ) -> ToolResult:
        """Index a codebase into the knowledge graph and vector store.

        This tool scans a repository, parses the code, builds a knowledge graph
        of code relationships, generates AI summaries, and creates embeddings
        for semantic search.

        Args:
            repo_path: Path to the repository to index.
            project_name: Optional name for the project. Defaults to directory name.

        Returns:
            ToolResult with indexing statistics.
        """
        logger.info(f"[Tool:Index] Indexing repository: {repo_path}")

        try:
            path = Path(repo_path).resolve()
            if not path.exists():
                return ToolResult(
                    success=False,
                    error=f"Repository path does not exist: {repo_path}",
                )

            name = project_name or path.name
            orchestrator = orchestrator_factory(path, name)

            stats = await orchestrator.run()

            return ToolResult(
                success=True,
                data=stats,
                message=f"Successfully indexed {stats.get('files_indexed', 0)} files "
                        f"with {stats.get('entities_found', 0)} entities.",
            )

        except Exception as e:
            logger.error(f"[Tool:Index] Error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                error=str(e),
            )

    return {
        "name": "index_repository",
        "description": (
            "Index a codebase into the knowledge graph and vector store. "
            "Parses code, builds relationships, generates summaries, and creates embeddings."
        ),
        "function": index_repository,
        "parameters": {
            "repo_path": {
                "type": "string",
                "description": "Path to the repository to index",
                "required": True,
            },
            "project_name": {
                "type": "string",
                "description": "Optional project name (defaults to directory name)",
                "required": False,
            },
        },
    }


def create_query_tool(
    query_engine_factory: callable,
) -> dict[str, Any]:
    """Create the query_code_graph tool.

    Args:
        query_engine_factory: Factory function to create QueryEngine.

    Returns:
        Tool definition dict for MCP registration.
    """
    async def query_code_graph(
        question: str,
        limit: int = 10,
    ) -> ToolResult:
        """Query the codebase using natural language.

        Ask questions about the codebase structure, functionality, relationships,
        or implementation details. The tool uses hybrid search (graph + vector)
        and generates an AI response based on relevant code.

        Args:
            question: Natural language question about the codebase.
            limit: Maximum number of search results to use (default: 10).

        Returns:
            ToolResult with the answer and sources.

        Examples:
            - "How does authentication work?"
            - "What functions call the User class?"
            - "Explain the payment processing flow"
            - "Where is error handling implemented?"
        """
        logger.info(f"[Tool:Query] Question: '{question}'")

        try:
            engine = query_engine_factory()
            result = await engine.query(question, limit=limit)

            sources = [
                {
                    "file": s.file_path,
                    "entity": s.entity_name,
                    "line": s.start_line,
                }
                for s in result.sources[:5]
            ]

            return ToolResult(
                success=True,
                data={
                    "answer": result.answer,
                    "sources": sources,
                    "query_type": result.query_analysis.query_type.value if result.query_analysis else None,
                },
                message=f"Found {len(result.sources)} relevant code sections.",
            )

        except Exception as e:
            logger.error(f"[Tool:Query] Error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                error=str(e),
            )

    return {
        "name": "query_code_graph",
        "description": (
            "Query the codebase using natural language. Ask about structure, "
            "functionality, relationships, or implementation details."
        ),
        "function": query_code_graph,
        "parameters": {
            "question": {
                "type": "string",
                "description": "Natural language question about the codebase",
                "required": True,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum search results to use (default: 10)",
                "required": False,
            },
        },
    }


def create_code_retrieval_tool(
    project_root: Path,
    graph_client_factory: callable,
) -> dict[str, Any]:
    """Create the get_code_snippet tool.

    Args:
        project_root: Root path of the indexed project.
        graph_client_factory: Factory function to create MemgraphClient.

    Returns:
        Tool definition dict for MCP registration.
    """
    async def get_code_snippet(
        qualified_name: str,
    ) -> CodeSnippet:
        """Retrieve source code for a specific function, class, or method.

        Use the fully qualified name to retrieve the actual source code
        from the codebase. This is useful after finding entities through
        search or query tools.

        Args:
            qualified_name: Full qualified name (e.g., "myproject.models.User.save").

        Returns:
            CodeSnippet with source code and location info.

        Examples:
            - "myproject.models.User"
            - "myproject.api.endpoints.create_user"
            - "myproject.utils.helpers.format_date"
        """
        logger.info(f"[Tool:GetCode] Retrieving: {qualified_name}")

        try:
            client = graph_client_factory()
            await client.connect()

            try:
                query = """
                    MATCH (n) WHERE n.qualified_name = $qn
                    OPTIONAL MATCH (m:File)-[*]-(n)
                    RETURN n.name AS name, n.start_line AS start, n.end_line AS end,
                           m.path AS path, n.docstring AS docstring
                    LIMIT 1
                """
                results = await client.execute(query, {"qn": qualified_name})

                if not results:
                    return CodeSnippet(
                        qualified_name=qualified_name,
                        found=False,
                        error_message="Entity not found in graph.",
                    )

                result = results[0]
                file_path = result.get("path")
                start_line = result.get("start")
                end_line = result.get("end")

                if not all([file_path, start_line, end_line]):
                    return CodeSnippet(
                        qualified_name=qualified_name,
                        file_path=file_path or "",
                        found=False,
                        error_message="Graph entry missing location data.",
                    )

                full_path = (project_root / file_path).resolve()
                if not full_path.is_relative_to(project_root.resolve()):
                    return CodeSnippet(
                        qualified_name=qualified_name,
                        file_path=file_path,
                        found=False,
                        error_message="Path traversal detected - access denied.",
                    )

                if not full_path.exists():
                    return CodeSnippet(
                        qualified_name=qualified_name,
                        file_path=file_path,
                        found=False,
                        error_message=f"Source file not found: {file_path}",
                    )

                try:
                    with full_path.open("r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                except (OSError, PermissionError) as e:
                    return CodeSnippet(
                        qualified_name=qualified_name,
                        file_path=file_path,
                        found=False,
                        error_message=f"Failed to read file: {e}",
                    )

                source_code = "".join(lines[start_line - 1:end_line])

                return CodeSnippet(
                    qualified_name=qualified_name,
                    source_code=source_code,
                    file_path=file_path,
                    line_start=start_line,
                    line_end=end_line,
                    docstring=result.get("docstring"),
                    found=True,
                )

            finally:
                await client.close()

        except Exception as e:
            logger.error(f"[Tool:GetCode] Error: {e}", exc_info=True)
            return CodeSnippet(
                qualified_name=qualified_name,
                found=False,
                error_message=str(e),
            )

    return {
        "name": "get_code_snippet",
        "description": (
            "Retrieve source code for a specific function, class, or method "
            "by its fully qualified name."
        ),
        "function": get_code_snippet,
        "parameters": {
            "qualified_name": {
                "type": "string",
                "description": "Full qualified name (e.g., 'myproject.models.User')",
                "required": True,
            },
        },
    }


def create_semantic_search_tool(
    vector_searcher_factory: callable,
) -> dict[str, Any]:
    """Create the semantic_search tool.

    Args:
        vector_searcher_factory: Factory function to create VectorSearcher.

    Returns:
        Tool definition dict for MCP registration.
    """
    async def semantic_search(
        query: str,
        limit: int = 5,
        entity_type: str | None = None,
    ) -> ToolResult:
        """Search for code by functionality or intent using natural language.

        Use this tool to find code that performs specific functionality
        based on intent rather than exact names. Perfect for exploratory
        questions about what code exists.

        Args:
            query: Natural language description of desired functionality.
            limit: Maximum number of results (default: 5).
            entity_type: Filter by type: "function", "class", "method" (optional).

        Returns:
            ToolResult with matching code entities.

        Examples:
            - "Find error handling functions"
            - "Authentication related code"
            - "Database query implementations"
            - "File I/O operations"
        """
        logger.info(f"[Tool:SemanticSearch] Query: '{query}'")

        try:
            searcher = vector_searcher_factory()
            results = await searcher.search_code(
                query=query,
                limit=limit,
                entity_type=entity_type,
            )

            search_results = [
                SearchResult(
                    qualified_name=r.entity_name,
                    entity_type=r.entity_type,
                    file_path=r.file_path,
                    score=r.score,
                    summary=r.summary,
                )
                for r in results
            ]

            return ToolResult(
                success=True,
                data=[vars(r) for r in search_results],
                message=f"Found {len(search_results)} matches for '{query}'.",
            )

        except Exception as e:
            logger.error(f"[Tool:SemanticSearch] Error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                error=str(e),
            )

    return {
        "name": "semantic_search",
        "description": (
            "Search for code by functionality or intent using natural language. "
            "Find code based on what it does, not its name."
        ),
        "function": semantic_search,
        "parameters": {
            "query": {
                "type": "string",
                "description": "Natural language description of functionality",
                "required": True,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default: 5)",
                "required": False,
            },
            "entity_type": {
                "type": "string",
                "description": "Filter by type: function, class, method",
                "required": False,
            },
        },
    }

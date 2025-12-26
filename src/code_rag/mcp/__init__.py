"""Model Context Protocol (MCP) server for Claude Code integration.

This module provides an MCP server that exposes Code RAG functionality
as tools for use with Claude Code and other MCP-compatible clients.

Usage with Claude Code:
    claude mcp add --transport stdio code-rag \\
      --env TARGET_REPO_PATH=/path/to/project \\
      -- uv run code-rag mcp-server

Available Tools:
    - index_repository: Index a codebase into the knowledge graph
    - query_code_graph: Query the graph using natural language
    - get_code_snippet: Retrieve code by qualified name
    - semantic_search: Search code by intent/functionality
"""

from code_rag.mcp.server import MCPServer
from code_rag.mcp.tools import (
    create_index_tool,
    create_query_tool,
    create_code_retrieval_tool,
    create_semantic_search_tool,
)

__all__ = [
    "MCPServer",
    "create_index_tool",
    "create_query_tool",
    "create_code_retrieval_tool",
    "create_semantic_search_tool",
]

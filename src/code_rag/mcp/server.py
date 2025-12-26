"""MCP Server implementation for Code RAG.

This module provides an MCP (Model Context Protocol) server that exposes
Code RAG functionality as tools for Claude Code integration.

Usage:
    # Start the MCP server
    python -m code_rag.mcp.server

    # Or via CLI
    code-rag mcp-server

Claude Code Integration:
    claude mcp add --transport stdio code-rag \\
      --env TARGET_REPO_PATH=/path/to/project \\
      -- uv run code-rag mcp-server
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from code_rag.config import get_settings
from code_rag.mcp.tools import (
    create_index_tool,
    create_query_tool,
    create_code_retrieval_tool,
    create_semantic_search_tool,
)

logger = logging.getLogger(__name__)


class MCPServer:
    """Model Context Protocol server for Code RAG.

    This server implements a subset of the MCP protocol to expose
    Code RAG functionality as tools for Claude Code.

    The server communicates via stdio using JSON-RPC 2.0.
    """

    def __init__(
        self,
        repo_path: Path | None = None,
        project_name: str | None = None,
    ):
        """Initialize the MCP server.

        Args:
            repo_path: Path to the repository to work with.
            project_name: Name of the project.
        """
        self.repo_path = repo_path or Path(os.environ.get("TARGET_REPO_PATH", ".")).resolve()
        self.project_name = project_name or self.repo_path.name
        self.tools: dict[str, dict[str, Any]] = {}
        self._running = False

        # Register tools
        self._register_tools()

    def _register_tools(self) -> None:
        # Index tool
        index_tool = create_index_tool(self._create_orchestrator)
        self.tools[index_tool["name"]] = index_tool

        # Query tool
        query_tool = create_query_tool(self._create_query_engine)
        self.tools[query_tool["name"]] = query_tool

        # Code retrieval tool
        code_tool = create_code_retrieval_tool(
            self.repo_path,
            self._create_graph_client,
        )
        self.tools[code_tool["name"]] = code_tool

        # Semantic search tool
        search_tool = create_semantic_search_tool(self._create_vector_searcher)
        self.tools[search_tool["name"]] = search_tool

        logger.info(f"Registered {len(self.tools)} tools: {list(self.tools.keys())}")

    def _create_orchestrator(self, repo_path: Path, project_name: str):
        from code_rag.pipeline.orchestrator import PipelineOrchestrator
        return PipelineOrchestrator(repo_path=repo_path, project_name=project_name)

    def _create_query_engine(self):
        from code_rag.query.engine import QueryEngine
        return QueryEngine()

    def _create_graph_client(self):
        from code_rag.graph.client import MemgraphClient
        return MemgraphClient()

    def _create_vector_searcher(self):
        from code_rag.embeddings.indexer import VectorSearcher
        return VectorSearcher()

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a JSON-RPC request.

        Args:
            request: JSON-RPC request dict.

        Returns:
            JSON-RPC response dict.
        """
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        try:
            if method == "initialize":
                return self._handle_initialize(request_id, params)

            elif method == "tools/list":
                return self._handle_list_tools(request_id)

            elif method == "tools/call":
                return await self._handle_call_tool(request_id, params)

            elif method == "shutdown":
                self._running = False
                return {"jsonrpc": "2.0", "id": request_id, "result": None}

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": str(e),
                },
            }

    def _handle_initialize(
        self,
        request_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "code-rag",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "tools": {},
                },
            },
        }

    def _handle_list_tools(self, request_id: Any) -> dict[str, Any]:
        tools_list = []

        for tool in self.tools.values():
            tools_list.append({
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        name: {
                            "type": param.get("type", "string"),
                            "description": param.get("description", ""),
                        }
                        for name, param in tool.get("parameters", {}).items()
                    },
                    "required": [
                        name
                        for name, param in tool.get("parameters", {}).items()
                        if param.get("required", False)
                    ],
                },
            })

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": tools_list},
        }

    async def _handle_call_tool(
        self,
        request_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tools:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": f"Unknown tool: {tool_name}",
                },
            }

        tool = self.tools[tool_name]
        func = tool["function"]

        try:
            # Call the tool function
            if asyncio.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)

            # Convert dataclass to dict if needed
            if hasattr(result, "__dict__"):
                result_data = vars(result)
            else:
                result_data = result

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result_data, indent=2, default=str),
                        }
                    ],
                },
            }

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"error": str(e)}),
                        }
                    ],
                    "isError": True,
                },
            }

    async def run_stdio(self) -> None:
        """Run the server using stdio transport.

        Reads JSON-RPC requests from stdin and writes responses to stdout.
        """
        self._running = True
        logger.info("Starting MCP server on stdio")

        # Use asyncio for non-blocking stdin reading
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(
            lambda: protocol, sys.stdin
        )

        while self._running:
            try:
                # Read a line (JSON-RPC request)
                line = await reader.readline()
                if not line:
                    break

                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                # Parse and handle request
                request = json.loads(line_str)
                response = await self.handle_request(request)

                # Write response
                response_str = json.dumps(response) + "\n"
                sys.stdout.write(response_str)
                sys.stdout.flush()

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
            except Exception as e:
                logger.error(f"Error in stdio loop: {e}", exc_info=True)

        logger.info("MCP server stopped")


async def main():
    """Main entry point for the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # Log to stderr to not interfere with stdio transport
    )

    server = MCPServer()
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())

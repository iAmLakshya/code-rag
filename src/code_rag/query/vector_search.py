"""Vector-based semantic search using Qdrant."""

import logging
from dataclasses import dataclass

from code_rag.core.errors import EmbeddingError, QueryError, VectorStoreError
from code_rag.embeddings.client import CollectionName, QdrantManager
from code_rag.embeddings.embedder import OpenAIEmbedder

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_LIMIT = 10
EXCLUDE_FILE_BUFFER = 5


@dataclass
class CodeSearchResult:
    """Result from code search."""

    score: float
    file_path: str
    entity_type: str
    entity_name: str
    content: str
    language: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    graph_node_id: str | None = None


@dataclass
class SummarySearchResult:
    """Result from summary search."""

    score: float
    file_path: str
    entity_type: str
    entity_name: str
    summary: str
    graph_node_id: str | None = None


class VectorSearcher:
    """Performs semantic search on code embeddings."""

    def __init__(
        self,
        qdrant: QdrantManager,
        embedder: OpenAIEmbedder,
    ):
        """Initialize vector searcher.

        Args:
            qdrant: Qdrant manager instance.
            embedder: OpenAI embedder instance.
        """
        self.qdrant = qdrant
        self.embedder = embedder

    async def search_code(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        language: str | None = None,
        entity_type: str | None = None,
    ) -> list[dict]:
        """Search for similar code.

        Args:
            query: Natural language search query.
            limit: Maximum results.
            language: Filter by programming language.
            entity_type: Filter by entity type.

        Returns:
            List of matching code chunks with scores.

        Raises:
            QueryError: If search fails or input is invalid.
        """
        if not query or not query.strip():
            raise QueryError("Search query cannot be empty")

        try:
            logger.debug(f"Searching code: query='{query}', limit={limit}, language={language}")
            query_embedding = await self.embedder.embed(query)

            filters = {}
            if language:
                filters["language"] = language
            if entity_type:
                filters["entity_type"] = entity_type

            results = await self.qdrant.search(
                collection=CollectionName.CODE_CHUNKS.value,
                query_vector=query_embedding,
                limit=limit,
                filters=filters if filters else None,
            )

            logger.debug(f"Found {len(results)} code results")
            return [
                self._transform_code_result(r)
                for r in results
            ]

        except EmbeddingError as e:
            logger.error(f"Embedding error: {e}")
            raise QueryError("Failed to embed search query", cause=e)
        except VectorStoreError as e:
            logger.error(f"Vector store error: {e}")
            raise QueryError("Failed to search code", cause=e)

    async def search_summaries(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[dict]:
        """Search for entities by summary.

        Args:
            query: Natural language search query.
            limit: Maximum results.

        Returns:
            List of matching summaries with scores.

        Raises:
            QueryError: If search fails or input is invalid.
        """
        if not query or not query.strip():
            raise QueryError("Search query cannot be empty")

        try:
            logger.debug(f"Searching summaries: query='{query}', limit={limit}")
            query_embedding = await self.embedder.embed(query)

            results = await self.qdrant.search(
                collection=CollectionName.SUMMARIES.value,
                query_vector=query_embedding,
                limit=limit,
            )

            logger.debug(f"Found {len(results)} summary results")
            return [
                self._transform_summary_result(r)
                for r in results
            ]

        except EmbeddingError as e:
            logger.error(f"Embedding error: {e}")
            raise QueryError("Failed to embed search query", cause=e)
        except VectorStoreError as e:
            logger.error(f"Vector store error: {e}")
            raise QueryError("Failed to search summaries", cause=e)

    async def find_similar_code(
        self,
        code_snippet: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        exclude_file: str | None = None,
    ) -> list[dict]:
        """Find code similar to a given snippet.

        Args:
            code_snippet: Code to find similar matches for.
            limit: Maximum results.
            exclude_file: Optional file path to exclude.

        Returns:
            List of similar code chunks.

        Raises:
            QueryError: If search fails or input is invalid.
        """
        if not code_snippet or not code_snippet.strip():
            raise QueryError("Code snippet cannot be empty")

        try:
            logger.debug(f"Finding similar code: limit={limit}, exclude={exclude_file}")
            query_embedding = await self.embedder.embed(code_snippet)

            search_limit = limit + EXCLUDE_FILE_BUFFER if exclude_file else limit
            results = await self.qdrant.search(
                collection=CollectionName.CODE_CHUNKS.value,
                query_vector=query_embedding,
                limit=search_limit,
            )

            matches = []
            for result in results:
                if exclude_file and result["payload"].get("file_path") == exclude_file:
                    continue

                matches.append(self._transform_similar_code_result(result))

                if len(matches) >= limit:
                    break

            logger.debug(f"Found {len(matches)} similar code results")
            return matches

        except EmbeddingError as e:
            logger.error(f"Embedding error: {e}")
            raise QueryError("Failed to embed code snippet", cause=e)
        except VectorStoreError as e:
            logger.error(f"Vector store error: {e}")
            raise QueryError("Failed to find similar code", cause=e)

    def _transform_code_result(self, result: dict) -> dict:
        """Transform vector search result to code result dictionary.

        Args:
            result: Raw search result.

        Returns:
            Transformed result dictionary.
        """
        payload = result["payload"]
        return {
            "score": result["score"],
            "file_path": payload.get("file_path"),
            "entity_type": payload.get("entity_type"),
            "entity_name": payload.get("entity_name"),
            "language": payload.get("language"),
            "content": payload.get("content"),
            "start_line": payload.get("start_line"),
            "end_line": payload.get("end_line"),
            "graph_node_id": payload.get("graph_node_id"),
        }

    def _transform_summary_result(self, result: dict) -> dict:
        """Transform vector search result to summary result dictionary.

        Args:
            result: Raw search result.

        Returns:
            Transformed result dictionary.
        """
        payload = result["payload"]
        return {
            "score": result["score"],
            "file_path": payload.get("file_path"),
            "entity_type": payload.get("entity_type"),
            "entity_name": payload.get("entity_name"),
            "summary": payload.get("summary"),
            "graph_node_id": payload.get("graph_node_id"),
        }

    def _transform_similar_code_result(self, result: dict) -> dict:
        """Transform vector search result to similar code result dictionary.

        Args:
            result: Raw search result.

        Returns:
            Transformed result dictionary.
        """
        payload = result["payload"]
        return {
            "score": result["score"],
            "file_path": payload.get("file_path"),
            "entity_type": payload.get("entity_type"),
            "entity_name": payload.get("entity_name"),
            "content": payload.get("content"),
            "start_line": payload.get("start_line"),
            "end_line": payload.get("end_line"),
        }

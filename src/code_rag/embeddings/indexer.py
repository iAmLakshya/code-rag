"""Vector indexer and searcher for managing code embeddings in Qdrant."""

import logging
import uuid
from dataclasses import dataclass
from typing import Callable

from code_rag.core.errors import IndexingError
from code_rag.embeddings.chunker import CodeChunker
from code_rag.embeddings.client import CollectionName, QdrantManager
from code_rag.embeddings.embedder import OpenAIEmbedder
from code_rag.parsing.models import ParsedFile

logger = logging.getLogger(__name__)


@dataclass
class CodeSearchResult:
    """Search result for code chunks."""

    score: float
    file_path: str
    entity_type: str
    entity_name: str
    content: str
    start_line: int
    end_line: int


@dataclass
class SummarySearchResult:
    """Search result for summaries."""

    score: float
    file_path: str
    entity_type: str
    entity_name: str
    summary: str


class VectorIndexer:
    """Indexes code chunks into Qdrant vector database."""

    def __init__(
        self,
        qdrant: QdrantManager,
        embedder: OpenAIEmbedder,
        chunker: CodeChunker | None = None,
    ):
        """Initialize the vector indexer.

        Args:
            qdrant: Qdrant manager instance.
            embedder: OpenAI embedder instance.
            chunker: Code chunker instance.
        """
        self.qdrant = qdrant
        self.embedder = embedder
        self.chunker = chunker or CodeChunker()

    async def index_file(
        self,
        parsed_file: ParsedFile,
        progress_callback: Callable[[int, int], None] | None = None,
        force: bool = False,
        project_name: str | None = None,
    ) -> int:
        """Index a single parsed file.

        Args:
            parsed_file: Parsed file to index.
            progress_callback: Optional progress callback.
            force: Force re-indexing even if file hasn't changed.
            project_name: Project name for filtering.

        Returns:
            Number of chunks indexed (0 if skipped).
        """
        try:
            file_path = str(parsed_file.file_info.path)
            content_hash = parsed_file.file_info.content_hash

            if not force and not await self._needs_indexing(
                file_path, content_hash
            ):
                logger.debug(f"Skipping unchanged file: {file_path}")
                return 0

            await self.qdrant.delete(
                CollectionName.CODE_CHUNKS.value,
                {"file_path": file_path},
            )

            chunks = self.chunker.chunk_file(parsed_file, project_name=project_name)
            if not chunks:
                logger.debug(f"No chunks generated for file: {file_path}")
                return 0

            texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedder.embed_with_progress(
                texts,
                progress_callback=progress_callback,
            )

            ids = [str(uuid.uuid4()) for _ in chunks]
            payloads = [chunk.to_payload() for chunk in chunks]

            await self.qdrant.upsert(
                collection=CollectionName.CODE_CHUNKS.value,
                ids=ids,
                vectors=embeddings,
                payloads=payloads,
            )

            logger.info(f"Indexed {len(chunks)} chunks from {file_path}")
            return len(chunks)
        except Exception as e:
            raise IndexingError(
                f"Failed to index file {parsed_file.file_info.path}",
                stage="file_indexing",
                cause=e,
            )

    async def index_files(
        self,
        parsed_files: list[ParsedFile],
        progress_callback: Callable[[int, int], None] | None = None,
        project_name: str | None = None,
    ) -> int:
        """Index multiple parsed files.

        Args:
            parsed_files: List of parsed files to index.
            progress_callback: Optional progress callback.
            project_name: Project name for filtering.

        Returns:
            Total number of chunks indexed.
        """
        total_chunks = 0

        for i, parsed_file in enumerate(parsed_files):
            try:
                chunks = await self.index_file(
                    parsed_file, project_name=project_name
                )
                total_chunks += chunks

                if progress_callback:
                    progress_callback(i + 1, len(parsed_files))
            except IndexingError as e:
                logger.error(f"Failed to index file: {e}")
                continue

        logger.info(f"Indexed total of {total_chunks} chunks from {len(parsed_files)} files")
        return total_chunks

    async def index_summary(
        self,
        file_path: str,
        entity_type: str,
        entity_name: str,
        summary: str,
        graph_node_id: str | None = None,
    ) -> None:
        """Index an AI-generated summary.

        Args:
            file_path: Source file path.
            entity_type: Type of entity (file, class, function).
            entity_name: Name of the entity.
            summary: AI-generated summary text.
            graph_node_id: Optional graph node ID for linking.
        """
        try:
            embedding = await self.embedder.embed(summary)

            payload = {
                "file_path": file_path,
                "entity_type": entity_type,
                "entity_name": entity_name,
                "summary": summary,
                "graph_node_id": graph_node_id,
            }

            await self.qdrant.upsert(
                collection=CollectionName.SUMMARIES.value,
                ids=[str(uuid.uuid4())],
                vectors=[embedding],
                payloads=[payload],
            )

            logger.info(f"Indexed summary for {entity_name} in {file_path}")
        except Exception as e:
            raise IndexingError(
                f"Failed to index summary for {entity_name}",
                stage="summary_indexing",
                cause=e,
            )

    async def _needs_indexing(
        self,
        file_path: str,
        content_hash: str,
    ) -> bool:
        """Check if a file needs re-indexing.

        Args:
            file_path: File path to check.
            content_hash: Current content hash.

        Returns:
            True if file needs indexing.
        """
        return await self.qdrant.file_needs_update(
            CollectionName.CODE_CHUNKS.value,
            file_path,
            content_hash,
        )


class VectorSearcher:
    """Searches for code and summaries in Qdrant vector database."""

    def __init__(
        self,
        qdrant: QdrantManager,
        embedder: OpenAIEmbedder,
    ):
        """Initialize the vector searcher.

        Args:
            qdrant: Qdrant manager instance.
            embedder: OpenAI embedder instance.
        """
        self.qdrant = qdrant
        self.embedder = embedder

    async def search_code(
        self,
        query: str,
        limit: int = 10,
        language: str | None = None,
        entity_type: str | None = None,
        project_name: str | None = None,
    ) -> list[CodeSearchResult]:
        """Search for similar code chunks.

        Args:
            query: Search query text.
            limit: Maximum results to return.
            language: Filter by programming language.
            entity_type: Filter by entity type.
            project_name: Filter by project name.

        Returns:
            List of matching code results.
        """
        try:
            query_embedding = await self.embedder.embed(query)

            filters = {}
            if language:
                filters["language"] = language
            if entity_type:
                filters["entity_type"] = entity_type
            if project_name:
                filters["project_name"] = project_name

            results = await self.qdrant.search(
                collection=CollectionName.CODE_CHUNKS.value,
                query_vector=query_embedding,
                limit=limit,
                filters=filters if filters else None,
            )

            return self._format_code_results(results)
        except Exception as e:
            logger.error(f"Code search failed: {e}")
            raise IndexingError(
                f"Failed to search code for query: {query}",
                stage="code_search",
                cause=e,
            )

    async def search_summaries(
        self,
        query: str,
        limit: int = 10,
        entity_type: str | None = None,
    ) -> list[SummarySearchResult]:
        """Search for similar summaries.

        Args:
            query: Search query text.
            limit: Maximum results to return.
            entity_type: Filter by entity type.

        Returns:
            List of matching summary results.
        """
        try:
            query_embedding = await self.embedder.embed(query)

            filters = {}
            if entity_type:
                filters["entity_type"] = entity_type

            results = await self.qdrant.search(
                collection=CollectionName.SUMMARIES.value,
                query_vector=query_embedding,
                limit=limit,
                filters=filters if filters else None,
            )

            return self._format_summary_results(results)
        except Exception as e:
            logger.error(f"Summary search failed: {e}")
            raise IndexingError(
                f"Failed to search summaries for query: {query}",
                stage="summary_search",
                cause=e,
            )

    def _format_code_results(
        self, results: list[dict]
    ) -> list[CodeSearchResult]:
        """Format code search results.

        Args:
            results: Raw search results from Qdrant.

        Returns:
            Formatted code search results.
        """
        return [
            CodeSearchResult(
                score=result["score"],
                file_path=result["payload"].get("file_path", ""),
                entity_type=result["payload"].get("entity_type", ""),
                entity_name=result["payload"].get("entity_name", ""),
                content=result["payload"].get("content", ""),
                start_line=result["payload"].get("start_line", 0),
                end_line=result["payload"].get("end_line", 0),
            )
            for result in results
        ]

    def _format_summary_results(
        self, results: list[dict]
    ) -> list[SummarySearchResult]:
        """Format summary search results.

        Args:
            results: Raw search results from Qdrant.

        Returns:
            Formatted summary search results.
        """
        return [
            SummarySearchResult(
                score=result["score"],
                file_path=result["payload"].get("file_path", ""),
                entity_type=result["payload"].get("entity_type", ""),
                entity_name=result["payload"].get("entity_name", ""),
                summary=result["payload"].get("summary", ""),
            )
            for result in results
        ]

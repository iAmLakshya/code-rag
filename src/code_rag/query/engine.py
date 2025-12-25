"""Hybrid query engine combining graph and vector search."""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from code_rag.config import get_settings
from code_rag.core.errors import QueryError
from code_rag.core.types import QueryType
from code_rag.embeddings.client import CollectionName, QdrantManager
from code_rag.embeddings.embedder import OpenAIEmbedder
from code_rag.graph.client import MemgraphClient
from code_rag.query.graph_search import GraphSearcher
from code_rag.query.reranker import ResultReranker, SearchResult
from code_rag.query.responder import ResponseGenerator
from code_rag.query.vector_search import VectorSearcher

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_LIMIT = 10
MAX_ENTITY_QUERIES = 3


@dataclass
class QueryAnalysis:
    """Analysis of a user query."""

    query_type: QueryType
    entities: list[str]
    intent: str
    filters: dict[str, str]


@dataclass
class QueryResult:
    """Result of a query."""

    answer: str
    sources: list[SearchResult]
    query_analysis: QueryAnalysis


class QueryAnalyzer:
    """Analyzes queries to determine type and extract entities."""

    CAMEL_CASE_PATTERN = r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b'
    SNAKE_CASE_PATTERN = r'\b([a-z]+(?:_[a-z]+)+)\b'
    BACKTICK_PATTERN = r'`([^`]+)`'

    STRUCTURAL_KEYWORDS = {
        "find_callers": ["what calls", "who calls", "callers of"],
        "find_callees": ["calls what", "what does.*call"],
        "find_hierarchy": ["extends", "inherits", "subclass"],
    }

    NAVIGATIONAL_KEYWORDS = ["show me", "find the", "where is"]
    EXPLANATORY_KEYWORDS = ["how does", "explain", "what does.*do"]

    def analyze(self, question: str) -> QueryAnalysis:
        """Analyze a query to determine type and extract entities.

        Args:
            question: User question.

        Returns:
            Query analysis.
        """
        if not question or not question.strip():
            raise QueryError("Question cannot be empty")

        logger.debug(f"Analyzing query: {question}")

        question_lower = question.lower()
        query_type, intent = self._detect_query_type(question_lower)
        entities = self._extract_entities(question)

        logger.debug(f"Query type: {query_type}, intent: {intent}, entities: {entities}")

        return QueryAnalysis(
            query_type=query_type,
            entities=entities,
            intent=intent,
            filters={},
        )

    def _detect_query_type(self, question_lower: str) -> tuple[QueryType, str]:
        """Detect query type and intent from question.

        Args:
            question_lower: Lowercased question.

        Returns:
            Tuple of (query_type, intent).
        """
        for intent, keywords in self.STRUCTURAL_KEYWORDS.items():
            if any(kw in question_lower for kw in keywords):
                return QueryType.STRUCTURAL, intent

        if any(kw in question_lower for kw in self.NAVIGATIONAL_KEYWORDS):
            return QueryType.NAVIGATIONAL, "locate"

        if any(kw in question_lower for kw in self.EXPLANATORY_KEYWORDS):
            return QueryType.EXPLANATORY, "explain"

        return QueryType.SEMANTIC, "search"

    def _extract_entities(self, question: str) -> list[str]:
        """Extract entity names from question.

        Args:
            question: User question.

        Returns:
            List of extracted entity names.
        """
        entities = []

        entities.extend(re.findall(self.CAMEL_CASE_PATTERN, question))
        entities.extend(re.findall(self.SNAKE_CASE_PATTERN, question))
        entities.extend(re.findall(self.BACKTICK_PATTERN, question))

        return list(set(entities))


class QueryEngine:
    """Hybrid search engine combining graph and vector search."""

    def __init__(
        self,
        memgraph: MemgraphClient | None = None,
        qdrant: QdrantManager | None = None,
        embedder: OpenAIEmbedder | None = None,
        graph_searcher: GraphSearcher | None = None,
        vector_searcher: VectorSearcher | None = None,
        responder: ResponseGenerator | None = None,
        reranker: ResultReranker | None = None,
        analyzer: QueryAnalyzer | None = None,
    ):
        """Initialize query engine.

        Args:
            memgraph: Memgraph client. Created if not provided.
            qdrant: Qdrant manager. Created if not provided.
            embedder: OpenAI embedder. Created if not provided.
            graph_searcher: Graph searcher. Created if not provided.
            vector_searcher: Vector searcher. Created if not provided.
            responder: Response generator. Created if not provided.
            reranker: Result reranker. Created if not provided.
            analyzer: Query analyzer. Created if not provided.
        """
        self._memgraph = memgraph
        self._qdrant = qdrant
        self._embedder = embedder
        self._graph_searcher = graph_searcher
        self._vector_searcher = vector_searcher
        self._responder = responder
        self._reranker = reranker or ResultReranker()
        self._analyzer = analyzer or QueryAnalyzer()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        logger.info("Initializing query engine")

        if self._memgraph is None:
            self._memgraph = MemgraphClient()
            await self._memgraph.connect()

        if self._qdrant is None:
            self._qdrant = QdrantManager()
            await self._qdrant.connect()

        if self._embedder is None:
            self._embedder = OpenAIEmbedder()

        if self._graph_searcher is None:
            self._graph_searcher = GraphSearcher(self._memgraph)

        if self._vector_searcher is None:
            self._vector_searcher = VectorSearcher(self._qdrant, self._embedder)

        if self._responder is None:
            self._responder = ResponseGenerator()

        self._initialized = True
        logger.info("Query engine initialized")

    async def close(self) -> None:
        """Close all connections."""
        logger.info("Closing query engine")

        if self._memgraph:
            await self._memgraph.close()
        if self._qdrant:
            await self._qdrant.close()

        self._initialized = False

    async def query(
        self,
        question: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        language: str | None = None,
    ) -> QueryResult:
        """Execute a query and generate a response.

        Args:
            question: Natural language question.
            limit: Maximum search results.
            language: Optional language filter.

        Returns:
            Query result with answer and sources.

        Raises:
            QueryError: If query execution fails.
        """
        await self.initialize()

        try:
            logger.info(f"Executing query: {question}")

            analysis = self._analyzer.analyze(question)

            graph_results, vector_results = await self._execute_hybrid_search(
                question,
                analysis,
                limit,
                language,
            )

            fused_results = self._fuse_and_deduplicate(graph_results, vector_results)

            answer = await self._responder.generate_response(question, fused_results)

            logger.info(f"Query completed, returned {len(fused_results[:limit])} sources")

            return QueryResult(
                answer=answer,
                sources=fused_results[:limit],
                query_analysis=analysis,
            )

        except QueryError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during query: {e}")
            raise QueryError("Query execution failed", cause=e)

    async def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        language: str | None = None,
        entity_type: str | None = None,
    ) -> list[SearchResult]:
        """Execute a search without generating a response.

        Args:
            query: Search query.
            limit: Maximum results.
            language: Optional language filter.
            entity_type: Optional entity type filter.

        Returns:
            List of search results.

        Raises:
            QueryError: If search fails.
        """
        await self.initialize()

        try:
            logger.info(f"Executing search: {query}")

            analysis = self._analyzer.analyze(query)

            graph_results, vector_results = await self._execute_hybrid_search(
                query,
                analysis,
                limit,
                language,
            )

            fused = self._fuse_and_deduplicate(graph_results, vector_results)

            logger.info(f"Search completed, found {len(fused[:limit])} results")
            return fused[:limit]

        except QueryError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise QueryError("Search execution failed", cause=e)

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the indexed codebase.

        Returns:
            Statistics dictionary.

        Raises:
            QueryError: If statistics retrieval fails.
        """
        await self.initialize()

        try:
            logger.debug("Getting statistics")

            graph_stats = await self._graph_searcher.get_statistics()

            try:
                code_info = await self._qdrant.get_collection_info(
                    CollectionName.CODE_CHUNKS.value
                )
                vector_count = code_info.points_count
            except Exception as e:
                logger.warning(f"Failed to get vector count: {e}")
                vector_count = 0

            return {
                **graph_stats,
                "vector_count": vector_count,
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise QueryError("Failed to get statistics", cause=e)

    async def _execute_hybrid_search(
        self,
        query: str,
        analysis: QueryAnalysis,
        limit: int,
        language: str | None,
    ) -> tuple[list[dict], list[dict]]:
        """Execute both graph and vector searches.

        Args:
            query: Search query.
            analysis: Query analysis.
            limit: Maximum results.
            language: Optional language filter.

        Returns:
            Tuple of (graph_results, vector_results).
        """
        graph_results, vector_results = await asyncio.gather(
            self._execute_graph_search(analysis, limit),
            self._execute_vector_search(query, analysis, limit, language),
        )

        return graph_results, vector_results

    def _fuse_and_deduplicate(
        self,
        graph_results: list[dict],
        vector_results: list[dict],
    ) -> list[SearchResult]:
        """Fuse and deduplicate search results.

        Args:
            graph_results: Results from graph search.
            vector_results: Results from vector search.

        Returns:
            Fused and deduplicated results.
        """
        fused_results = self._reranker.fuse_results(graph_results, vector_results)
        return self._reranker.deduplicate(fused_results)

    async def _execute_graph_search(
        self,
        analysis: QueryAnalysis,
        limit: int,
    ) -> list[dict]:
        """Execute graph-based search.

        Args:
            analysis: Query analysis.
            limit: Maximum results.

        Returns:
            Graph search results.
        """
        results = []

        if analysis.query_type == QueryType.STRUCTURAL:
            results.extend(await self._execute_structural_search(analysis))
        elif analysis.query_type == QueryType.NAVIGATIONAL:
            results.extend(await self._execute_navigational_search(analysis))

        results.extend(await self._search_entities_by_name(analysis.entities))

        return results[:limit]

    async def _execute_vector_search(
        self,
        query: str,
        analysis: QueryAnalysis,
        limit: int,
        language: str | None,
    ) -> list[dict]:
        """Execute vector-based search.

        Args:
            query: Search query.
            analysis: Query analysis.
            limit: Maximum results.
            language: Optional language filter.

        Returns:
            Vector search results.
        """
        code_results = await self._vector_searcher.search_code(
            query=query,
            limit=limit,
            language=language,
        )

        if analysis.query_type in (QueryType.EXPLANATORY, QueryType.SEMANTIC):
            summary_results = await self._vector_searcher.search_summaries(
                query=query,
                limit=limit // 2,
            )
            code_results.extend(summary_results)

        return code_results[:limit]

    async def _execute_structural_search(
        self,
        analysis: QueryAnalysis,
    ) -> list[dict]:
        """Execute structural queries for entities.

        Args:
            analysis: Query analysis.

        Returns:
            Structural search results.
        """
        results = []

        for entity in analysis.entities:
            try:
                if analysis.intent == "find_callers":
                    entity_results = await self._graph_searcher.find_callers(entity)
                elif analysis.intent == "find_callees":
                    entity_results = await self._graph_searcher.find_callees(entity)
                elif analysis.intent == "find_hierarchy":
                    entity_results = await self._graph_searcher.find_class_hierarchy(entity)
                else:
                    continue

                results.extend(entity_results)

            except QueryError as e:
                logger.warning(f"Failed structural search for {entity}: {e}")

        return results

    async def _execute_navigational_search(
        self,
        analysis: QueryAnalysis,
    ) -> list[dict]:
        """Execute navigational queries for entities.

        Args:
            analysis: Query analysis.

        Returns:
            Navigational search results.
        """
        results = []

        for entity in analysis.entities:
            try:
                entity_results = await self._graph_searcher.find_entity_by_name(entity)
                results.extend(entity_results)
            except QueryError as e:
                logger.warning(f"Failed navigational search for {entity}: {e}")

        return results

    async def _search_entities_by_name(
        self,
        entities: list[str],
    ) -> list[dict]:
        """Search for entities by name.

        Args:
            entities: Entity names to search for.

        Returns:
            Entity search results.
        """
        results = []

        for entity in entities[:MAX_ENTITY_QUERIES]:
            try:
                entity_results = await self._graph_searcher.find_entity_by_name(entity)
                results.extend(entity_results)
            except QueryError as e:
                logger.warning(f"Failed entity name search for {entity}: {e}")

        return results

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

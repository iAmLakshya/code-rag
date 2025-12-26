"""Context enrichment for building rich, informative query responses."""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from code_rag.core.errors import QueryError
from code_rag.embeddings.client import CollectionName, QdrantManager
from code_rag.graph.client import MemgraphClient
from code_rag.query.graph_reasoning import GraphContext, GraphNode, GraphPath
from code_rag.query.query_planner import QueryIntent, QueryPlan

logger = logging.getLogger(__name__)

MAX_CODE_SNIPPET_LENGTH = 3000
MAX_CONTEXT_ENTITIES = 20
MAX_RELATED_CODE_SNIPPETS = 5


@dataclass
class CodeSnippet:
    """A code snippet with metadata."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    entity_name: str
    entity_type: str
    language: str | None = None
    relevance_score: float = 0.0


@dataclass
class EntityContext:
    """Rich context for a single entity."""

    entity: GraphNode
    code_snippet: CodeSnippet | None = None
    implementation_summary: str | None = None
    caller_summaries: list[str] = field(default_factory=list)
    callee_summaries: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)


@dataclass
class EnrichedContext:
    """Fully enriched context for query response generation."""

    query: str
    intent: QueryIntent
    primary_contexts: list[EntityContext]
    call_chain_explanations: list[str]
    hierarchy_explanations: list[str]
    file_summaries: dict[str, str]
    dependency_map: dict[str, list[str]]
    code_snippets: list[CodeSnippet]
    graph_summary: str
    total_entities_found: int
    reasoning_notes: list[str]


class ContextBuilder:
    """Builds rich context from graph and vector search results."""

    def __init__(
        self,
        memgraph: MemgraphClient,
        qdrant: QdrantManager,
    ):
        """Initialize the context builder.

        Args:
            memgraph: Memgraph client instance.
            qdrant: Qdrant manager instance.
        """
        self.memgraph = memgraph
        self.qdrant = qdrant

    async def build_enriched_context(
        self,
        plan: QueryPlan,
        graph_context: GraphContext,
        vector_results: list[dict[str, Any]],
    ) -> EnrichedContext:
        """Build enriched context from query results.

        Args:
            plan: The query plan.
            graph_context: Context from graph reasoning.
            vector_results: Results from vector search.

        Returns:
            EnrichedContext with all gathered information.

        Raises:
            QueryError: If context building fails.
        """
        logger.debug(f"Building enriched context for intent: {plan.primary_intent}")

        try:
            enriched = EnrichedContext(
                query=plan.original_query,
                intent=plan.primary_intent,
                primary_contexts=[],
                call_chain_explanations=[],
                hierarchy_explanations=[],
                file_summaries={},
                dependency_map={},
                code_snippets=[],
                graph_summary="",
                total_entities_found=0,
                reasoning_notes=[],
            )

            # Build context for primary entities
            primary_tasks = [
                self._build_entity_context(entity, graph_context)
                for entity in graph_context.primary_entities[:MAX_CONTEXT_ENTITIES]
            ]

            if primary_tasks:
                primary_results = await asyncio.gather(*primary_tasks, return_exceptions=True)
                for result in primary_results:
                    if isinstance(result, EntityContext):
                        enriched.primary_contexts.append(result)

            # Extract code snippets from vector results
            enriched.code_snippets = self._extract_code_snippets(vector_results)

            # Build call chain explanations
            if graph_context.call_chains:
                enriched.call_chain_explanations = self._build_call_chain_explanations(
                    graph_context.call_chains
                )

            # Build hierarchy explanations
            if graph_context.parent_classes or graph_context.child_classes:
                enriched.hierarchy_explanations = self._build_hierarchy_explanations(
                    graph_context
                )

            # Build file summaries for relevant files
            relevant_files = self._collect_relevant_files(graph_context, vector_results)
            enriched.file_summaries = await self._build_file_summaries(relevant_files)

            # Build graph summary
            enriched.graph_summary = self._build_graph_summary(graph_context)

            # Count total entities
            enriched.total_entities_found = (
                len(graph_context.primary_entities)
                + len(graph_context.callers)
                + len(graph_context.callees)
                + len(graph_context.methods)
            )

            # Add reasoning notes based on what was found
            enriched.reasoning_notes = self._generate_reasoning_notes(plan, graph_context, enriched)

            logger.debug(
                f"Context built: {len(enriched.primary_contexts)} primary contexts, "
                f"{len(enriched.code_snippets)} code snippets"
            )

            return enriched

        except Exception as e:
            logger.error(f"Error building enriched context: {e}")
            raise QueryError(f"Failed to build context: {e}", cause=e)

    async def _build_entity_context(
        self,
        entity: GraphNode,
        graph_context: GraphContext,
    ) -> EntityContext:
        """Build rich context for a single entity.

        Args:
            entity: The entity to build context for.
            graph_context: Full graph context.

        Returns:
            EntityContext with all relevant information.
        """
        context = EntityContext(entity=entity)

        # Try to fetch code snippet from vector store
        try:
            code_results = await self.qdrant.search(
                collection=CollectionName.CODE_CHUNKS.value,
                query_vector=None,  # We'll use filter-only search
                limit=1,
                filters={
                    "entity_name": entity.name,
                    "file_path": entity.file_path,
                },
            )

            if code_results:
                payload = code_results[0].get("payload", {})
                context.code_snippet = CodeSnippet(
                    content=payload.get("content", "")[:MAX_CODE_SNIPPET_LENGTH],
                    file_path=entity.file_path,
                    start_line=entity.start_line or 0,
                    end_line=entity.end_line or 0,
                    entity_name=entity.name,
                    entity_type=entity.node_type,
                    language=payload.get("language"),
                    relevance_score=code_results[0].get("score", 0),
                )
        except Exception as e:
            logger.debug(f"Could not fetch code snippet: {e}")

        if entity.summary:
            context.implementation_summary = entity.summary
        elif entity.docstring:
            context.implementation_summary = entity.docstring[:500]

        context.caller_summaries = [
            f"{c.name} ({c.node_type}) in {c.file_path}:{c.start_line or '?'}"
            for c in graph_context.callers[:5]
            if c.name and c.file_path
        ]

        context.callee_summaries = [
            f"{c.name} ({c.node_type}) - {c.summary or 'No summary'}"
            for c in graph_context.callees[:5]
            if c.file_path
        ]

        related = set()
        if entity.node_type == "Class":
            for method in graph_context.methods:
                if method.parent_class == entity.qualified_name:
                    related.add(f"Method: {method.name}")
        for sibling in graph_context.file_context:
            if sibling.file_path == entity.file_path and sibling.name != entity.name:
                related.add(f"{sibling.node_type}: {sibling.name}")

        context.related_entities = list(related)[:10]

        return context

    def _extract_code_snippets(self, vector_results: list[dict[str, Any]]) -> list[CodeSnippet]:
        """Extract code snippets from vector search results.

        Args:
            vector_results: Vector search results.

        Returns:
            List of CodeSnippet objects.
        """
        snippets = []

        for result in vector_results[:MAX_RELATED_CODE_SNIPPETS]:
            content = result.get("content", "")
            if content:
                snippets.append(
                    CodeSnippet(
                        content=content[:MAX_CODE_SNIPPET_LENGTH],
                        file_path=result.get("file_path", ""),
                        start_line=result.get("start_line", 0),
                        end_line=result.get("end_line", 0),
                        entity_name=result.get("entity_name", ""),
                        entity_type=result.get("entity_type", ""),
                        language=result.get("language"),
                        relevance_score=result.get("score", 0),
                    )
                )

        return snippets

    def _build_call_chain_explanations(self, call_chains: list[GraphPath]) -> list[str]:
        """Build human-readable explanations of call chains.

        Args:
            call_chains: List of call chain paths.

        Returns:
            List of explanation strings.
        """
        explanations = []

        for chain in call_chains[:5]:
            if len(chain.nodes) >= 2:
                path_desc = " → ".join(
                    f"{node.name} ({node.node_type})"
                    for node in chain.nodes
                )
                explanation = f"Call chain ({chain.total_length} hops): {path_desc}"
                explanations.append(explanation)

        return explanations

    def _build_hierarchy_explanations(self, graph_context: GraphContext) -> list[str]:
        """Build human-readable explanations of inheritance hierarchies.

        Args:
            graph_context: Graph context with hierarchy info.

        Returns:
            List of explanation strings.
        """
        explanations = []

        if graph_context.parent_classes:
            ancestors = " → ".join(
                p.name for p in graph_context.parent_classes[:5]
            )
            explanations.append(f"Inherits from: {ancestors}")

        if graph_context.child_classes:
            children = ", ".join(
                c.name for c in graph_context.child_classes[:5]
            )
            count = len(graph_context.child_classes)
            if count > 5:
                children += f" (and {count - 5} more)"
            explanations.append(f"Extended by: {children}")

        return explanations

    def _collect_relevant_files(
        self,
        graph_context: GraphContext,
        vector_results: list[dict[str, Any]],
    ) -> set[str]:
        """Collect all relevant file paths.

        Args:
            graph_context: Graph context.
            vector_results: Vector search results.

        Returns:
            Set of file paths.
        """
        files = set()

        for entity in graph_context.primary_entities:
            if entity.file_path:
                files.add(entity.file_path)

        for entity in graph_context.callers[:5]:
            if entity.file_path:
                files.add(entity.file_path)

        for entity in graph_context.callees[:5]:
            if entity.file_path:
                files.add(entity.file_path)

        for result in vector_results[:5]:
            if result.get("file_path"):
                files.add(result["file_path"])

        return files

    async def _build_file_summaries(self, file_paths: set[str]) -> dict[str, str]:
        """Build summaries for relevant files.

        Args:
            file_paths: Set of file paths.

        Returns:
            Dictionary mapping file path to summary.
        """
        summaries = {}

        for file_path in list(file_paths)[:10]:
            try:
                # Query the graph for file info
                result = await self.memgraph.execute(
                    """
                    MATCH (f:File {path: $path})
                    OPTIONAL MATCH (f)-[:DEFINES]->(e)
                    RETURN f.summary as summary,
                           f.language as language,
                           count(e) as entity_count
                    """,
                    {"path": file_path},
                )

                if result:
                    r = result[0]
                    file_name = Path(file_path).name
                    entity_count = r.get("entity_count", 0)
                    language = r.get("language", "unknown")
                    summary = r.get("summary", "")

                    if summary:
                        summaries[file_path] = f"{file_name} ({language}, {entity_count} entities): {summary}"
                    else:
                        summaries[file_path] = f"{file_name} ({language}, {entity_count} entities)"

            except Exception as e:
                logger.debug(f"Could not get file summary for {file_path}: {e}")

        return summaries

    def _build_graph_summary(self, graph_context: GraphContext) -> str:
        """Build a summary of the graph context.

        Args:
            graph_context: Graph context.

        Returns:
            Summary string.
        """
        parts = []

        if graph_context.primary_entities:
            entity_names = ", ".join(e.name for e in graph_context.primary_entities[:3])
            parts.append(f"Found {len(graph_context.primary_entities)} primary entities: {entity_names}")

        if graph_context.callers:
            parts.append(f"{len(graph_context.callers)} callers found")

        if graph_context.callees:
            parts.append(f"{len(graph_context.callees)} callees found")

        if graph_context.methods:
            parts.append(f"{len(graph_context.methods)} methods found")

        if graph_context.call_chains:
            parts.append(f"{len(graph_context.call_chains)} call chains identified")

        if graph_context.parent_classes or graph_context.child_classes:
            total = len(graph_context.parent_classes) + len(graph_context.child_classes)
            parts.append(f"{total} classes in inheritance hierarchy")

        return ". ".join(parts) if parts else "No significant graph relationships found."

    def _generate_reasoning_notes(
        self,
        plan: QueryPlan,
        graph_context: GraphContext,
        enriched: EnrichedContext,
    ) -> list[str]:
        """Generate reasoning notes about the search process.

        Args:
            plan: Query plan.
            graph_context: Graph context.
            enriched: Enriched context being built.

        Returns:
            List of reasoning notes.
        """
        notes = []

        if plan.reasoning:
            notes.append(f"Query analysis: {plan.reasoning}")

        if plan.requires_multi_hop:
            notes.append(f"Used multi-hop graph traversal (up to {plan.max_hops} hops)")

        if not graph_context.primary_entities:
            notes.append("No exact entity matches found; using semantic search results")
        elif len(graph_context.primary_entities) > 1:
            notes.append(f"Found {len(graph_context.primary_entities)} entities matching the query")

        if graph_context.call_chains:
            notes.append(f"Identified {len(graph_context.call_chains)} call paths between entities")

        if graph_context.callers and graph_context.callees:
            notes.append(
                f"Entity has {len(graph_context.callers)} callers and calls {len(graph_context.callees)} other functions"
            )

        if "implementation_details" in plan.context_requirements:
            if enriched.code_snippets:
                notes.append(f"Retrieved {len(enriched.code_snippets)} relevant code snippets")
            else:
                notes.append("Implementation details requested but no code snippets found")

        return notes


def format_context_for_llm(enriched: EnrichedContext) -> str:
    """Format enriched context for LLM consumption.

    Args:
        enriched: EnrichedContext object.

    Returns:
        Formatted context string.
    """
    sections = []

    sections.append(f"## Query Context\n")
    sections.append(f"**Intent**: {enriched.intent.value}\n")
    sections.append(f"**Summary**: {enriched.graph_summary}\n")

    if enriched.primary_contexts:
        sections.append("\n## Primary Entities\n")

        for i, ctx in enumerate(enriched.primary_contexts, 1):
            entity = ctx.entity
            sections.append(f"### {i}. {entity.name} ({entity.node_type})\n")
            sections.append(f"**File**: `{entity.file_path}`")

            if entity.start_line:
                sections.append(f" (lines {entity.start_line}-{entity.end_line})")
            sections.append("\n")

            if entity.signature:
                sections.append(f"**Signature**: `{entity.signature}`\n")

            if ctx.implementation_summary:
                sections.append(f"**Summary**: {ctx.implementation_summary}\n")

            if ctx.code_snippet:
                lang = ctx.code_snippet.language or ""
                sections.append(f"\n```{lang}\n{ctx.code_snippet.content}\n```\n")

            if ctx.caller_summaries:
                sections.append(f"\n**Called by**: {', '.join(ctx.caller_summaries[:3])}\n")

            if ctx.callee_summaries:
                sections.append(f"**Calls**: {', '.join(ctx.callee_summaries[:3])}\n")

            if ctx.related_entities:
                sections.append(f"**Related**: {', '.join(ctx.related_entities[:5])}\n")

    if enriched.call_chain_explanations:
        sections.append("\n## Call Chains\n")
        for explanation in enriched.call_chain_explanations:
            sections.append(f"- {explanation}\n")

    caller_info = []
    for ctx in enriched.primary_contexts:
        if ctx.caller_summaries:
            caller_info.extend(ctx.caller_summaries)
    if caller_info:
        sections.append("\n## Callers (Call Sites)\n")
        for caller in caller_info[:10]:
            sections.append(f"- {caller}\n")

    if enriched.hierarchy_explanations:
        sections.append("\n## Inheritance Hierarchy\n")
        for explanation in enriched.hierarchy_explanations:
            sections.append(f"- {explanation}\n")

    if enriched.code_snippets:
        sections.append("\n## Related Code\n")
        for snippet in enriched.code_snippets[:3]:
            sections.append(f"### {snippet.entity_name} ({snippet.entity_type})\n")
            sections.append(f"**File**: `{snippet.file_path}` (lines {snippet.start_line}-{snippet.end_line})\n")
            lang = snippet.language or ""
            sections.append(f"```{lang}\n{snippet.content}\n```\n")

    if enriched.file_summaries:
        sections.append("\n## Relevant Files\n")
        for file_path, summary in list(enriched.file_summaries.items())[:5]:
            sections.append(f"- {summary}\n")

    if enriched.reasoning_notes:
        sections.append("\n## Analysis Notes\n")
        for note in enriched.reasoning_notes:
            sections.append(f"- {note}\n")

    return "".join(sections)

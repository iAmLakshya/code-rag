"""Advanced multi-hop graph reasoning for code understanding."""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from code_rag.core.errors import GraphError, QueryError
from code_rag.graph.client import MemgraphClient
from code_rag.query.query_planner import ExtractedEntity, QueryIntent, QueryPlan

logger = logging.getLogger(__name__)

# Limits for graph traversals
MAX_TRAVERSAL_DEPTH = 5
MAX_RESULTS_PER_QUERY = 50
MAX_PATH_LENGTH = 10
MAX_RELATED_ENTITIES = 30


class TraversalDirection(Enum):
    """Direction for graph traversal."""

    INCOMING = "incoming"
    OUTGOING = "outgoing"
    BOTH = "both"


@dataclass
class GraphNode:
    """A node from the graph with full context."""

    node_type: str  # Class, Function, Method, File, Import
    name: str
    qualified_name: str
    file_path: str
    signature: str | None = None
    docstring: str | None = None
    summary: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    is_async: bool = False
    parent_class: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphPath:
    """A path through the graph representing a relationship chain."""

    nodes: list[GraphNode]
    relationships: list[str]  # Relationship types between consecutive nodes
    total_length: int
    path_type: str  # call_chain, inheritance, dependency, etc.


@dataclass
class GraphContext:
    """Rich context gathered from graph traversal."""

    primary_entities: list[GraphNode]
    callers: list[GraphNode]
    callees: list[GraphNode]
    parent_classes: list[GraphNode]
    child_classes: list[GraphNode]
    methods: list[GraphNode]  # Methods if entity is a class
    containing_class: GraphNode | None  # Class if entity is a method
    file_context: list[GraphNode]  # Other entities in the same file
    dependencies: list[GraphNode]  # Imported entities
    dependents: list[GraphNode]  # Entities that import this
    call_chains: list[GraphPath]  # Paths through call graph
    inheritance_chains: list[GraphPath]  # Paths through inheritance


class MultiHopGraphQueries:
    """Advanced Cypher queries for multi-hop graph reasoning."""

    # Find transitive callers (entities that call entities that call target)
    # Uses exact match first, then fuzzy CONTAINS match
    FIND_TRANSITIVE_CALLERS = """
    MATCH path = (caller)-[:CALLS*1..{max_hops}]->(target)
    WHERE target.name = $name
       OR target.qualified_name = $name
       OR target.name CONTAINS $name
       OR toLower(target.name) CONTAINS toLower($name)
    WITH path, caller, target, length(path) as depth
    WHERE caller <> target
    RETURN DISTINCT
        labels(caller)[0] as node_type,
        caller.name as name,
        caller.qualified_name as qualified_name,
        caller.file_path as file_path,
        caller.signature as signature,
        caller.docstring as docstring,
        caller.summary as summary,
        caller.start_line as start_line,
        caller.end_line as end_line,
        caller.is_async as is_async,
        depth,
        target.name as target_name
    ORDER BY depth
    LIMIT $limit
    """

    # Find transitive callees (entities called by entities called by source)
    # Uses exact match first, then fuzzy CONTAINS match
    FIND_TRANSITIVE_CALLEES = """
    MATCH path = (source)-[:CALLS*1..{max_hops}]->(callee)
    WHERE source.name = $name
       OR source.qualified_name = $name
       OR source.name CONTAINS $name
       OR toLower(source.name) CONTAINS toLower($name)
    WITH path, source, callee, length(path) as depth
    WHERE callee <> source
    RETURN DISTINCT
        labels(callee)[0] as node_type,
        callee.name as name,
        callee.qualified_name as qualified_name,
        callee.file_path as file_path,
        callee.signature as signature,
        callee.docstring as docstring,
        callee.summary as summary,
        callee.start_line as start_line,
        callee.end_line as end_line,
        callee.is_async as is_async,
        depth,
        source.name as source_name
    ORDER BY depth
    LIMIT $limit
    """

    # Find call chain between two entities
    # Note: Uses .format(max_hops=X) so {{ escapes to { for Cypher map literals
    FIND_CALL_CHAIN = """
    MATCH path = shortestPath((source)-[:CALLS*1..{max_hops}]->(target))
    WHERE (source.name = $source_name OR source.qualified_name = $source_name)
    AND (target.name = $target_name OR target.qualified_name = $target_name)
    RETURN
        [node in nodes(path) | {{
            node_type: labels(node)[0],
            name: node.name,
            qualified_name: node.qualified_name,
            file_path: node.file_path,
            signature: node.signature,
            summary: node.summary
        }}] as path_nodes,
        length(path) as path_length
    LIMIT 5
    """

    # Find all paths between two entities (not just shortest)
    # Note: Uses .format(max_hops=X) so {{ escapes to { for Cypher map literals
    FIND_ALL_PATHS = """
    MATCH path = (source)-[:CALLS*1..{max_hops}]->(target)
    WHERE (source.name = $source_name OR source.qualified_name = $source_name)
    AND (target.name = $target_name OR target.qualified_name = $target_name)
    WITH path, length(path) as path_length
    ORDER BY path_length
    LIMIT 10
    RETURN
        [node in nodes(path) | {{
            node_type: labels(node)[0],
            name: node.name,
            qualified_name: node.qualified_name,
            file_path: node.file_path,
            signature: node.signature,
            summary: node.summary
        }}] as path_nodes,
        path_length
    """

    # Find full inheritance hierarchy (both ancestors and descendants)
    # Note: NOT formatted, uses single { for Cypher map literals
    FIND_FULL_HIERARCHY = """
    MATCH (target:Class)
    WHERE target.name = $name OR target.qualified_name = $name
    OPTIONAL MATCH ancestor_path = (target)-[:EXTENDS*1..5]->(ancestor:Class)
    OPTIONAL MATCH descendant_path = (descendant:Class)-[:EXTENDS*1..5]->(target)
    WITH target,
         collect(DISTINCT {
             node: ancestor,
             depth: length(ancestor_path),
             direction: 'ancestor'
         }) as ancestors,
         collect(DISTINCT {
             node: descendant,
             depth: length(descendant_path),
             direction: 'descendant'
         }) as descendants
    RETURN
        {
            node_type: 'Class',
            name: target.name,
            qualified_name: target.qualified_name,
            file_path: target.file_path,
            signature: target.signature,
            summary: target.summary
        } as target_node,
        [a in ancestors WHERE a.node IS NOT NULL | {
            node_type: 'Class',
            name: a.node.name,
            qualified_name: a.node.qualified_name,
            file_path: a.node.file_path,
            depth: a.depth,
            direction: a.direction
        }] + [d in descendants WHERE d.node IS NOT NULL | {
            node_type: 'Class',
            name: d.node.name,
            qualified_name: d.node.qualified_name,
            file_path: d.node.file_path,
            depth: d.depth,
            direction: d.direction
        }] as hierarchy_nodes
    """

    # Find entity with all its methods (for classes)
    # Note: NOT formatted, uses single { for Cypher map literals
    FIND_CLASS_WITH_METHODS = """
    MATCH (c:Class)
    WHERE c.name = $name OR c.qualified_name = $name
    OPTIONAL MATCH (c)-[:DEFINES_METHOD]->(m:Method)
    RETURN
        {
            node_type: 'Class',
            name: c.name,
            qualified_name: c.qualified_name,
            file_path: c.file_path,
            signature: c.signature,
            docstring: c.docstring,
            summary: c.summary,
            start_line: c.start_line,
            end_line: c.end_line
        } as class_node,
        collect({
            node_type: 'Method',
            name: m.name,
            qualified_name: m.qualified_name,
            file_path: m.file_path,
            signature: m.signature,
            docstring: m.docstring,
            summary: m.summary,
            start_line: m.start_line,
            end_line: m.end_line,
            is_async: m.is_async,
            is_static: m.is_static,
            is_classmethod: m.is_classmethod
        }) as methods
    """

    # Find all entities in a file with their relationships
    # Note: NOT formatted, uses single { for Cypher map literals
    FIND_FILE_CONTEXT = """
    MATCH (f:File {path: $file_path})-[:DEFINES]->(entity)
    OPTIONAL MATCH (entity)-[:CALLS]->(callee)
    OPTIONAL MATCH (caller)-[:CALLS]->(entity)
    OPTIONAL MATCH (entity)-[:EXTENDS]->(parent:Class)
    OPTIONAL MATCH (child:Class)-[:EXTENDS]->(entity)
    RETURN
        {
            node_type: labels(entity)[0],
            name: entity.name,
            qualified_name: entity.qualified_name,
            file_path: entity.file_path,
            signature: entity.signature,
            docstring: entity.docstring,
            summary: entity.summary,
            start_line: entity.start_line,
            end_line: entity.end_line
        } as entity_node,
        count(DISTINCT callee) as callee_count,
        count(DISTINCT caller) as caller_count,
        collect(DISTINCT parent.qualified_name)[0] as parent_class,
        count(DISTINCT child) as child_count
    ORDER BY entity.start_line
    """

    # Find implementation details - entity with all related context
    # Note: NOT formatted, uses single { for Cypher map literals
    FIND_IMPLEMENTATION_CONTEXT = """
    MATCH (entity)
    WHERE entity.name = $name OR entity.qualified_name = $name

    // Get direct callers
    OPTIONAL MATCH (caller)-[:CALLS]->(entity)
    WITH entity, collect(DISTINCT {
        node_type: labels(caller)[0],
        name: caller.name,
        qualified_name: caller.qualified_name,
        file_path: caller.file_path,
        summary: caller.summary
    })[0..10] as callers

    // Get direct callees
    OPTIONAL MATCH (entity)-[:CALLS]->(callee)
    WITH entity, callers, collect(DISTINCT {
        node_type: labels(callee)[0],
        name: callee.name,
        qualified_name: callee.qualified_name,
        file_path: callee.file_path,
        summary: callee.summary
    })[0..10] as callees

    // Get containing file's other entities
    OPTIONAL MATCH (f:File {path: entity.file_path})-[:DEFINES]->(sibling)
    WHERE sibling <> entity
    WITH entity, callers, callees, collect(DISTINCT {
        node_type: labels(sibling)[0],
        name: sibling.name,
        qualified_name: sibling.qualified_name,
        summary: sibling.summary,
        start_line: sibling.start_line
    })[0..10] as siblings

    RETURN
        {
            node_type: labels(entity)[0],
            name: entity.name,
            qualified_name: entity.qualified_name,
            file_path: entity.file_path,
            signature: entity.signature,
            docstring: entity.docstring,
            summary: entity.summary,
            start_line: entity.start_line,
            end_line: entity.end_line,
            is_async: entity.is_async,
            parent_class: entity.parent_class
        } as entity_node,
        callers,
        callees,
        siblings
    """

    # Find usage patterns - where and how an entity is used
    # Note: NOT formatted, uses single { for Cypher map literals
    FIND_USAGE_PATTERNS = """
    MATCH (user)-[:CALLS]->(target)
    WHERE target.name = $name OR target.qualified_name = $name
    WITH user, target
    OPTIONAL MATCH (user)-[:CALLS]->(other_callee)
    WHERE other_callee <> target
    WITH user, target, collect(DISTINCT other_callee.name)[0..5] as context_calls
    RETURN
        {
            node_type: labels(user)[0],
            name: user.name,
            qualified_name: user.qualified_name,
            file_path: user.file_path,
            signature: user.signature,
            summary: user.summary,
            start_line: user.start_line,
            end_line: user.end_line
        } as user_node,
        context_calls
    ORDER BY user.file_path, user.start_line
    LIMIT $limit
    """

    # Find module/file dependencies
    # Note: NOT formatted, uses single { for Cypher map literals
    FIND_MODULE_DEPENDENCIES = """
    MATCH (f:File {path: $file_path})-[:IMPORTS]->(i:Import)
    OPTIONAL MATCH (imported_entity)
    WHERE imported_entity.name = i.name OR imported_entity.qualified_name CONTAINS i.name
    RETURN
        i.name as import_name,
        i.source as import_source,
        i.is_external as is_external,
        {
            node_type: labels(imported_entity)[0],
            name: imported_entity.name,
            qualified_name: imported_entity.qualified_name,
            file_path: imported_entity.file_path,
            summary: imported_entity.summary
        } as resolved_entity
    """

    # Find entities by fuzzy name match with context
    # Searches for partial matches and ranks by relevance
    FIND_ENTITY_FUZZY = """
    MATCH (n)
    WHERE n.name CONTAINS $name
       OR n.qualified_name CONTAINS $name
       OR toLower(n.name) CONTAINS toLower($name)
       OR toLower(n.qualified_name) CONTAINS toLower($name)
    WITH n,
         CASE
             WHEN n.name = $name THEN 0
             WHEN n.qualified_name = $name THEN 1
             WHEN toLower(n.name) = toLower($name) THEN 2
             WHEN n.name STARTS WITH $name THEN 3
             WHEN n.name ENDS WITH $name THEN 4
             WHEN n.name CONTAINS $name THEN 5
             ELSE 6
         END as match_score
    ORDER BY match_score, n.name
    LIMIT $limit
    RETURN
        labels(n)[0] as node_type,
        n.name as name,
        n.qualified_name as qualified_name,
        n.file_path as file_path,
        n.signature as signature,
        n.docstring as docstring,
        n.summary as summary,
        n.start_line as start_line,
        n.end_line as end_line,
        n.is_async as is_async,
        n.parent_class as parent_class,
        match_score
    """

    # Get graph centrality metrics for importance ranking
    GET_ENTITY_CENTRALITY = """
    MATCH (n)
    WHERE n.name = $name OR n.qualified_name = $name
    OPTIONAL MATCH (caller)-[:CALLS]->(n)
    WITH n, count(DISTINCT caller) as in_degree
    OPTIONAL MATCH (n)-[:CALLS]->(callee)
    WITH n, in_degree, count(DISTINCT callee) as out_degree
    OPTIONAL MATCH (n)-[r]-()
    RETURN
        n.name as name,
        n.qualified_name as qualified_name,
        in_degree,
        out_degree,
        in_degree + out_degree as total_degree,
        count(DISTINCT r) as relationship_count
    """


class GraphReasoningEngine:
    """Advanced graph reasoning engine for multi-hop queries."""

    def __init__(self, client: MemgraphClient):
        """Initialize the graph reasoning engine.

        Args:
            client: Memgraph client instance.
        """
        self.client = client

    async def execute_query_plan(self, plan: QueryPlan) -> GraphContext:
        """Execute a query plan and gather comprehensive graph context.

        Args:
            plan: Query plan from QueryPlanner.

        Returns:
            GraphContext with all relevant entities and relationships.

        Raises:
            QueryError: If execution fails.
        """
        logger.debug(f"Executing query plan: intent={plan.primary_intent}")

        context = GraphContext(
            primary_entities=[],
            callers=[],
            callees=[],
            parent_classes=[],
            child_classes=[],
            methods=[],
            containing_class=None,
            file_context=[],
            dependencies=[],
            dependents=[],
            call_chains=[],
            inheritance_chains=[],
        )

        try:
            # First, resolve primary entities
            for entity in plan.entities:
                if entity.is_primary or not context.primary_entities:
                    nodes = await self._find_entity(entity.name, entity.entity_type)
                    context.primary_entities.extend(nodes)

            if not context.primary_entities:
                # Try fuzzy search if exact match fails
                for entity in plan.entities:
                    nodes = await self._find_entity_fuzzy(entity.name)
                    context.primary_entities.extend(nodes)

            # Based on intent, gather relevant context
            if plan.primary_intent in (QueryIntent.FIND_CALLERS, QueryIntent.FIND_USAGES):
                await self._gather_caller_context(context, plan)

            elif plan.primary_intent == QueryIntent.FIND_CALLEES:
                await self._gather_callee_context(context, plan)

            elif plan.primary_intent == QueryIntent.FIND_CALL_CHAIN:
                await self._gather_call_chain_context(context, plan)

            elif plan.primary_intent in (QueryIntent.FIND_HIERARCHY, QueryIntent.FIND_IMPLEMENTATIONS):
                await self._gather_hierarchy_context(context, plan)

            elif plan.primary_intent == QueryIntent.EXPLAIN_IMPLEMENTATION:
                await self._gather_implementation_context(context, plan)

            elif plan.primary_intent in (QueryIntent.FIND_DEPENDENCIES, QueryIntent.FIND_DEPENDENTS):
                await self._gather_dependency_context(context, plan)

            else:
                # Default: gather comprehensive context
                await self._gather_comprehensive_context(context, plan)

            logger.debug(
                f"Context gathered: {len(context.primary_entities)} primary, "
                f"{len(context.callers)} callers, {len(context.callees)} callees"
            )

            return context

        except Exception as e:
            logger.error(f"Error executing query plan: {e}")
            raise QueryError(f"Graph reasoning failed: {e}", cause=e)

    async def find_transitive_callers(
        self,
        entity_name: str,
        max_hops: int = 3,
        limit: int = MAX_RESULTS_PER_QUERY,
    ) -> list[GraphNode]:
        """Find all entities that transitively call the target.

        Args:
            entity_name: Name of the target entity.
            max_hops: Maximum call chain depth.
            limit: Maximum results.

        Returns:
            List of caller nodes with depth information.
        """
        max_hops = min(max_hops, MAX_TRAVERSAL_DEPTH)
        query = MultiHopGraphQueries.FIND_TRANSITIVE_CALLERS.format(max_hops=max_hops)

        try:
            results = await self.client.execute(
                query,
                {"name": entity_name, "limit": limit},
            )
            return [self._result_to_node(r) for r in results]
        except GraphError as e:
            logger.warning(f"Error finding transitive callers: {e}")
            return []

    async def find_transitive_callees(
        self,
        entity_name: str,
        max_hops: int = 3,
        limit: int = MAX_RESULTS_PER_QUERY,
    ) -> list[GraphNode]:
        """Find all entities transitively called by the source.

        Args:
            entity_name: Name of the source entity.
            max_hops: Maximum call chain depth.
            limit: Maximum results.

        Returns:
            List of callee nodes with depth information.
        """
        max_hops = min(max_hops, MAX_TRAVERSAL_DEPTH)
        query = MultiHopGraphQueries.FIND_TRANSITIVE_CALLEES.format(max_hops=max_hops)

        try:
            results = await self.client.execute(
                query,
                {"name": entity_name, "limit": limit},
            )
            return [self._result_to_node(r) for r in results]
        except GraphError as e:
            logger.warning(f"Error finding transitive callees: {e}")
            return []

    async def find_call_chain(
        self,
        source_name: str,
        target_name: str,
        max_hops: int = 5,
    ) -> list[GraphPath]:
        """Find call chains between two entities.

        Args:
            source_name: Name of the source entity.
            target_name: Name of the target entity.
            max_hops: Maximum chain length.

        Returns:
            List of paths connecting the entities.
        """
        max_hops = min(max_hops, MAX_PATH_LENGTH)
        query = MultiHopGraphQueries.FIND_ALL_PATHS.format(max_hops=max_hops)

        try:
            results = await self.client.execute(
                query,
                {"source_name": source_name, "target_name": target_name},
            )

            paths = []
            for r in results:
                path_nodes = r.get("path_nodes", [])
                if path_nodes:
                    nodes = [self._dict_to_node(n) for n in path_nodes]
                    paths.append(
                        GraphPath(
                            nodes=nodes,
                            relationships=["CALLS"] * (len(nodes) - 1),
                            total_length=r.get("path_length", len(nodes) - 1),
                            path_type="call_chain",
                        )
                    )

            return paths
        except GraphError as e:
            logger.warning(f"Error finding call chain: {e}")
            return []

    async def find_full_hierarchy(self, class_name: str) -> tuple[GraphNode | None, list[GraphNode], list[GraphNode]]:
        """Find the full inheritance hierarchy for a class.

        Args:
            class_name: Name of the class.

        Returns:
            Tuple of (target_node, ancestors, descendants).
        """
        try:
            results = await self.client.execute(
                MultiHopGraphQueries.FIND_FULL_HIERARCHY,
                {"name": class_name},
            )

            if not results:
                return None, [], []

            r = results[0]
            target_node = self._dict_to_node(r.get("target_node", {}))

            ancestors = []
            descendants = []

            for h in r.get("hierarchy_nodes", []):
                node = self._dict_to_node(h)
                if h.get("direction") == "ancestor":
                    ancestors.append(node)
                else:
                    descendants.append(node)

            return target_node, ancestors, descendants
        except GraphError as e:
            logger.warning(f"Error finding hierarchy: {e}")
            return None, [], []

    async def find_implementation_context(self, entity_name: str) -> dict[str, Any]:
        """Find comprehensive implementation context for an entity.

        Args:
            entity_name: Name of the entity.

        Returns:
            Dictionary with entity details, callers, callees, and siblings.
        """
        try:
            results = await self.client.execute(
                MultiHopGraphQueries.FIND_IMPLEMENTATION_CONTEXT,
                {"name": entity_name},
            )

            if not results:
                return {}

            r = results[0]
            return {
                "entity": self._dict_to_node(r.get("entity_node", {})),
                "callers": [self._dict_to_node(c) for c in r.get("callers", []) if c.get("name")],
                "callees": [self._dict_to_node(c) for c in r.get("callees", []) if c.get("name")],
                "siblings": [self._dict_to_node(s) for s in r.get("siblings", []) if s.get("name")],
            }
        except GraphError as e:
            logger.warning(f"Error finding implementation context: {e}")
            return {}

    async def find_class_with_methods(self, class_name: str) -> tuple[GraphNode | None, list[GraphNode]]:
        """Find a class and all its methods.

        Args:
            class_name: Name of the class.

        Returns:
            Tuple of (class_node, list of method nodes).
        """
        try:
            results = await self.client.execute(
                MultiHopGraphQueries.FIND_CLASS_WITH_METHODS,
                {"name": class_name},
            )

            if not results:
                return None, []

            r = results[0]
            class_node = self._dict_to_node(r.get("class_node", {}))
            methods = [self._dict_to_node(m) for m in r.get("methods", []) if m.get("name")]

            return class_node, methods
        except GraphError as e:
            logger.warning(f"Error finding class with methods: {e}")
            return None, []

    async def find_file_context(self, file_path: str) -> list[dict[str, Any]]:
        """Find all entities in a file with their relationship counts.

        Args:
            file_path: Path to the file.

        Returns:
            List of entity dictionaries with relationship counts.
        """
        try:
            results = await self.client.execute(
                MultiHopGraphQueries.FIND_FILE_CONTEXT,
                {"file_path": file_path},
            )

            entities = []
            for r in results:
                entity = self._dict_to_node(r.get("entity_node", {}))
                entities.append(
                    {
                        "entity": entity,
                        "callee_count": r.get("callee_count", 0),
                        "caller_count": r.get("caller_count", 0),
                        "parent_class": r.get("parent_class"),
                        "child_count": r.get("child_count", 0),
                    }
                )

            return entities
        except GraphError as e:
            logger.warning(f"Error finding file context: {e}")
            return []

    async def get_entity_centrality(self, entity_name: str) -> dict[str, int]:
        """Get centrality metrics for an entity.

        Args:
            entity_name: Name of the entity.

        Returns:
            Dictionary with centrality metrics.
        """
        try:
            results = await self.client.execute(
                MultiHopGraphQueries.GET_ENTITY_CENTRALITY,
                {"name": entity_name},
            )

            if not results:
                return {"in_degree": 0, "out_degree": 0, "total_degree": 0}

            r = results[0]
            return {
                "in_degree": r.get("in_degree", 0),
                "out_degree": r.get("out_degree", 0),
                "total_degree": r.get("total_degree", 0),
                "relationship_count": r.get("relationship_count", 0),
            }
        except GraphError as e:
            logger.warning(f"Error getting entity centrality: {e}")
            return {"in_degree": 0, "out_degree": 0, "total_degree": 0}

    # Private helper methods

    async def _find_entity(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> list[GraphNode]:
        """Find entities by exact name match."""
        if entity_type:
            type_label = entity_type.capitalize()
            query = f"""
            MATCH (n:{type_label})
            WHERE n.name = $name OR n.qualified_name = $name
            RETURN
                labels(n)[0] as node_type,
                n.name as name,
                n.qualified_name as qualified_name,
                n.file_path as file_path,
                n.signature as signature,
                n.docstring as docstring,
                n.summary as summary,
                n.start_line as start_line,
                n.end_line as end_line,
                n.is_async as is_async,
                n.parent_class as parent_class
            """
        else:
            query = """
            MATCH (n)
            WHERE n.name = $name OR n.qualified_name = $name
            RETURN
                labels(n)[0] as node_type,
                n.name as name,
                n.qualified_name as qualified_name,
                n.file_path as file_path,
                n.signature as signature,
                n.docstring as docstring,
                n.summary as summary,
                n.start_line as start_line,
                n.end_line as end_line,
                n.is_async as is_async,
                n.parent_class as parent_class
            """

        try:
            results = await self.client.execute(query, {"name": name})
            return [self._result_to_node(r) for r in results]
        except GraphError:
            return []

    async def _find_entity_fuzzy(self, name: str, limit: int = 10) -> list[GraphNode]:
        """Find entities by fuzzy name match."""
        try:
            results = await self.client.execute(
                MultiHopGraphQueries.FIND_ENTITY_FUZZY,
                {"name": name, "limit": limit},
            )
            return [self._result_to_node(r) for r in results]
        except GraphError:
            return []

    async def _gather_caller_context(self, context: GraphContext, plan: QueryPlan) -> None:
        """Gather caller-focused context."""
        max_hops = plan.max_hops if plan.requires_multi_hop else 1

        # If we have primary entities, search from them
        if context.primary_entities:
            for entity in context.primary_entities:
                callers = await self.find_transitive_callers(
                    entity.qualified_name or entity.name,
                    max_hops=max_hops,
                )
                context.callers.extend(callers)
        else:
            # Fall back to searching by entity names from the plan
            for entity in plan.entities:
                callers = await self.find_transitive_callers(
                    entity.name,
                    max_hops=max_hops,
                )
                context.callers.extend(callers)

    async def _gather_callee_context(self, context: GraphContext, plan: QueryPlan) -> None:
        """Gather callee-focused context."""
        max_hops = plan.max_hops if plan.requires_multi_hop else 1

        # If we have primary entities, search from them
        if context.primary_entities:
            for entity in context.primary_entities:
                callees = await self.find_transitive_callees(
                    entity.qualified_name or entity.name,
                    max_hops=max_hops,
                )
                context.callees.extend(callees)
        else:
            # Fall back to searching by entity names from the plan
            for entity in plan.entities:
                callees = await self.find_transitive_callees(
                    entity.name,
                    max_hops=max_hops,
                )
                context.callees.extend(callees)

    async def _gather_call_chain_context(self, context: GraphContext, plan: QueryPlan) -> None:
        """Gather call chain context between entities."""
        if len(plan.entities) >= 2:
            source = plan.entities[0].name
            target = plan.entities[1].name
            chains = await self.find_call_chain(source, target, plan.max_hops)
            context.call_chains.extend(chains)

    async def _gather_hierarchy_context(self, context: GraphContext, plan: QueryPlan) -> None:
        """Gather inheritance hierarchy context.

        For classes: finds parent classes and child classes (inheritance hierarchy).
        For methods/functions: finds callers and callees (usage context).
        """
        # Process resolved primary entities
        for entity in context.primary_entities:
            if entity.node_type == "Class":
                _, ancestors, descendants = await self.find_full_hierarchy(
                    entity.qualified_name or entity.name
                )
                context.parent_classes.extend(ancestors)
                context.child_classes.extend(descendants)
            else:
                # For methods and functions, gather caller/callee context
                callers = await self.find_transitive_callers(
                    entity.qualified_name or entity.name,
                    max_hops=plan.max_hops if plan.requires_multi_hop else 1,
                )
                context.callers.extend(callers)

        # Also search for entities from plan that weren't in primary_entities
        # This handles compound queries like "hierarchy of X and callers of Y"
        resolved_names = {e.name.lower() for e in context.primary_entities}
        resolved_names.update(e.qualified_name.lower() for e in context.primary_entities if e.qualified_name)

        for plan_entity in plan.entities:
            if plan_entity.name.lower() not in resolved_names:
                # Try to find and get callers for this entity
                callers = await self.find_transitive_callers(
                    plan_entity.name,
                    max_hops=plan.max_hops if plan.requires_multi_hop else 1,
                )
                context.callers.extend(callers)

    async def _gather_implementation_context(self, context: GraphContext, plan: QueryPlan) -> None:
        """Gather comprehensive implementation context."""
        for entity in context.primary_entities:
            impl_context = await self.find_implementation_context(
                entity.qualified_name or entity.name
            )

            if impl_context:
                context.callers.extend(impl_context.get("callers", []))
                context.callees.extend(impl_context.get("callees", []))
                context.file_context.extend(impl_context.get("siblings", []))

            # If it's a class, get methods too
            if entity.node_type == "Class":
                _, methods = await self.find_class_with_methods(
                    entity.qualified_name or entity.name
                )
                context.methods.extend(methods)

    async def _gather_dependency_context(self, context: GraphContext, plan: QueryPlan) -> None:
        """Gather dependency context."""
        for entity in context.primary_entities:
            if entity.file_path:
                file_context = await self.find_file_context(entity.file_path)
                for fc in file_context:
                    if fc.get("entity"):
                        context.file_context.append(fc["entity"])

    async def _gather_comprehensive_context(self, context: GraphContext, plan: QueryPlan) -> None:
        """Gather comprehensive context for general queries."""
        # Run multiple context gathering operations in parallel
        tasks = []

        for entity in context.primary_entities[:3]:  # Limit to top 3 entities
            entity_name = entity.qualified_name or entity.name

            tasks.append(self.find_transitive_callers(entity_name, max_hops=2, limit=10))
            tasks.append(self.find_transitive_callees(entity_name, max_hops=2, limit=10))

            if entity.node_type == "Class":
                tasks.append(self.find_class_with_methods(entity_name))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            idx = 0
            for entity in context.primary_entities[:3]:
                # Callers result
                if idx < len(results) and isinstance(results[idx], list):
                    context.callers.extend(results[idx])
                idx += 1

                # Callees result
                if idx < len(results) and isinstance(results[idx], list):
                    context.callees.extend(results[idx])
                idx += 1

                # Class methods result
                if entity.node_type == "Class":
                    if idx < len(results) and isinstance(results[idx], tuple):
                        _, methods = results[idx]
                        context.methods.extend(methods)
                    idx += 1

    def _result_to_node(self, result: dict[str, Any]) -> GraphNode:
        """Convert a query result to a GraphNode."""
        return GraphNode(
            node_type=result.get("node_type", "Unknown"),
            name=result.get("name", ""),
            qualified_name=result.get("qualified_name", ""),
            file_path=result.get("file_path", ""),
            signature=result.get("signature"),
            docstring=result.get("docstring"),
            summary=result.get("summary"),
            start_line=result.get("start_line"),
            end_line=result.get("end_line"),
            is_async=result.get("is_async", False),
            parent_class=result.get("parent_class"),
            metadata={"depth": result.get("depth")} if "depth" in result else {},
        )

    def _dict_to_node(self, d: dict[str, Any]) -> GraphNode:
        """Convert a dictionary to a GraphNode."""
        return GraphNode(
            node_type=d.get("node_type", "Unknown"),
            name=d.get("name", ""),
            qualified_name=d.get("qualified_name", ""),
            file_path=d.get("file_path", ""),
            signature=d.get("signature"),
            docstring=d.get("docstring"),
            summary=d.get("summary"),
            start_line=d.get("start_line"),
            end_line=d.get("end_line"),
            is_async=d.get("is_async", False),
            parent_class=d.get("parent_class"),
        )

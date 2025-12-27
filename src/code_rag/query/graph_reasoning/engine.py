import asyncio
import logging
from typing import Any

from code_rag.core.errors import GraphError, QueryError
from code_rag.graph.client import MemgraphClient
from code_rag.query.graph_reasoning.models import (
    MAX_PATH_LENGTH,
    MAX_RESULTS_PER_QUERY,
    MAX_TRAVERSAL_DEPTH,
    GraphContext,
    GraphNode,
    GraphPath,
)
from code_rag.query.graph_reasoning.queries import MultiHopGraphQueries
from code_rag.query.query_planner import QueryIntent, QueryPlan

logger = logging.getLogger(__name__)


class GraphReasoningEngine:
    def __init__(self, client: MemgraphClient):
        self.client = client

    async def execute_query_plan(self, plan: QueryPlan) -> GraphContext:
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
            for entity in plan.entities:
                if entity.is_primary or not context.primary_entities:
                    nodes = await self._find_entity(entity.name, entity.entity_type)
                    context.primary_entities.extend(nodes)

            if not context.primary_entities:
                for entity in plan.entities:
                    nodes = await self._find_entity_fuzzy(entity.name)
                    context.primary_entities.extend(nodes)

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

    async def _find_entity(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> list[GraphNode]:
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
        try:
            results = await self.client.execute(
                MultiHopGraphQueries.FIND_ENTITY_FUZZY,
                {"name": name, "limit": limit},
            )
            return [self._result_to_node(r) for r in results]
        except GraphError:
            return []

    async def _gather_caller_context(self, context: GraphContext, plan: QueryPlan) -> None:
        max_hops = plan.max_hops if plan.requires_multi_hop else 1

        if context.primary_entities:
            for entity in context.primary_entities:
                callers = await self.find_transitive_callers(
                    entity.qualified_name or entity.name,
                    max_hops=max_hops,
                )
                context.callers.extend(callers)
        else:
            for entity in plan.entities:
                callers = await self.find_transitive_callers(
                    entity.name,
                    max_hops=max_hops,
                )
                context.callers.extend(callers)

    async def _gather_callee_context(self, context: GraphContext, plan: QueryPlan) -> None:
        max_hops = plan.max_hops if plan.requires_multi_hop else 1

        if context.primary_entities:
            for entity in context.primary_entities:
                callees = await self.find_transitive_callees(
                    entity.qualified_name or entity.name,
                    max_hops=max_hops,
                )
                context.callees.extend(callees)
        else:
            for entity in plan.entities:
                callees = await self.find_transitive_callees(
                    entity.name,
                    max_hops=max_hops,
                )
                context.callees.extend(callees)

    async def _gather_call_chain_context(self, context: GraphContext, plan: QueryPlan) -> None:
        if len(plan.entities) >= 2:
            source = plan.entities[0].name
            target = plan.entities[1].name
            chains = await self.find_call_chain(source, target, plan.max_hops)
            context.call_chains.extend(chains)

    async def _gather_hierarchy_context(self, context: GraphContext, plan: QueryPlan) -> None:
        for entity in context.primary_entities:
            if entity.node_type == "Class":
                _, ancestors, descendants = await self.find_full_hierarchy(
                    entity.qualified_name or entity.name
                )
                context.parent_classes.extend(ancestors)
                context.child_classes.extend(descendants)
            else:
                callers = await self.find_transitive_callers(
                    entity.qualified_name or entity.name,
                    max_hops=plan.max_hops if plan.requires_multi_hop else 1,
                )
                context.callers.extend(callers)

        resolved_names = {e.name.lower() for e in context.primary_entities}
        resolved_names.update(e.qualified_name.lower() for e in context.primary_entities if e.qualified_name)

        for plan_entity in plan.entities:
            if plan_entity.name.lower() not in resolved_names:
                callers = await self.find_transitive_callers(
                    plan_entity.name,
                    max_hops=plan.max_hops if plan.requires_multi_hop else 1,
                )
                context.callers.extend(callers)

    async def _gather_implementation_context(self, context: GraphContext, plan: QueryPlan) -> None:
        for entity in context.primary_entities:
            impl_context = await self.find_implementation_context(
                entity.qualified_name or entity.name
            )

            if impl_context:
                context.callers.extend(impl_context.get("callers", []))
                context.callees.extend(impl_context.get("callees", []))
                context.file_context.extend(impl_context.get("siblings", []))

            if entity.node_type == "Class":
                _, methods = await self.find_class_with_methods(
                    entity.qualified_name or entity.name
                )
                context.methods.extend(methods)

    async def _gather_dependency_context(self, context: GraphContext, plan: QueryPlan) -> None:
        for entity in context.primary_entities:
            if entity.file_path:
                file_context = await self.find_file_context(entity.file_path)
                for fc in file_context:
                    if fc.get("entity"):
                        context.file_context.append(fc["entity"])

    async def _gather_comprehensive_context(self, context: GraphContext, plan: QueryPlan) -> None:
        tasks = []

        for entity in context.primary_entities[:3]:
            entity_name = entity.qualified_name or entity.name

            tasks.append(self.find_transitive_callers(entity_name, max_hops=2, limit=10))
            tasks.append(self.find_transitive_callees(entity_name, max_hops=2, limit=10))

            if entity.node_type == "Class":
                tasks.append(self.find_class_with_methods(entity_name))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            idx = 0
            for entity in context.primary_entities[:3]:
                if idx < len(results) and isinstance(results[idx], list):
                    context.callers.extend(results[idx])
                idx += 1

                if idx < len(results) and isinstance(results[idx], list):
                    context.callees.extend(results[idx])
                idx += 1

                if entity.node_type == "Class":
                    if idx < len(results) and isinstance(results[idx], tuple):
                        _, methods = results[idx]
                        context.methods.extend(methods)
                    idx += 1

    def _result_to_node(self, result: dict[str, Any]) -> GraphNode:
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

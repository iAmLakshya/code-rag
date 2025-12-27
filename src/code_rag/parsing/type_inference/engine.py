from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from code_rag.core.cache import ASTCache, FunctionRegistry

if TYPE_CHECKING:
    from tree_sitter import Node

logger = logging.getLogger(__name__)

_RE_METHOD_CHAIN = re.compile(r"\)\.[^)]*$")
_RE_FINAL_METHOD = re.compile(r"\.([^.()]+)$")


def safe_decode_text(node: Node) -> str | None:
    if node.text:
        return node.text.decode("utf-8")
    return None


class TypeInferenceEngine:
    """Infers types for local variables and method returns."""

    def __init__(
        self,
        function_registry: FunctionRegistry | None = None,
        import_mapping: dict[str, dict[str, str]] | None = None,
        ast_cache: ASTCache | None = None,
        module_qn_to_file_path: dict[str, Path] | None = None,
        simple_name_lookup: dict[str, set[str]] | None = None,
    ):
        self.function_registry = function_registry or FunctionRegistry()
        self.import_mapping = import_mapping or {}
        self.ast_cache = ast_cache or ASTCache()
        self.module_qn_to_file_path = module_qn_to_file_path or {}
        self.simple_name_lookup = simple_name_lookup or {}

        self._method_return_type_cache: dict[str, str | None] = {}
        self._type_inference_in_progress: set[str] = set()

    def build_local_variable_type_map(
        self,
        caller_node: Node,
        module_qn: str,
        language: str,
    ) -> dict[str, str]:
        local_var_types: dict[str, str] = {}

        if language != "python":
            return local_var_types

        try:
            self._infer_parameter_types(caller_node, local_var_types, module_qn)
            self._traverse_single_pass(caller_node, local_var_types, module_qn)
        except Exception as e:
            logger.debug(f"Failed to build local variable type map: {e}")

        return local_var_types

    def _infer_parameter_types(
        self,
        caller_node: Node,
        local_var_types: dict[str, str],
        module_qn: str,
    ) -> None:
        params_node = caller_node.child_by_field_name("parameters")
        if not params_node:
            return

        for param in params_node.children:
            if param.type == "identifier":
                param_name = safe_decode_text(param)
                if param_name and param_name not in ("self", "cls"):
                    inferred_type = self._infer_type_from_parameter_name(
                        param_name, module_qn
                    )
                    if inferred_type:
                        local_var_types[param_name] = inferred_type
            elif param.type == "typed_parameter":
                param_name_node = param.child_by_field_name("name")
                param_type_node = param.child_by_field_name("type")
                if param_name_node and param_type_node:
                    param_name = safe_decode_text(param_name_node)
                    param_type = safe_decode_text(param_type_node)
                    if param_name and param_type:
                        local_var_types[param_name] = param_type

    def _infer_type_from_parameter_name(
        self,
        param_name: str,
        module_qn: str,
    ) -> str | None:
        available_class_names = []

        for qn, entity_type in self.function_registry.all_entries().items():
            if entity_type == "Class" and qn.startswith(module_qn + "."):
                remaining = qn[len(module_qn) + 1:]
                if "." not in remaining:
                    available_class_names.append(remaining)

        if module_qn in self.import_mapping:
            for local_name, imported_qn in self.import_mapping[module_qn].items():
                if self.function_registry.get(imported_qn) == "Class":
                    available_class_names.append(local_name)

        param_lower = param_name.lower()
        best_match = None
        highest_score = 0

        for class_name in available_class_names:
            class_lower = class_name.lower()
            score = 0

            if param_lower == class_lower:
                score = 100
            elif class_lower.endswith(param_lower) or param_lower.endswith(class_lower):
                score = 90
            elif class_lower in param_lower:
                score = int(80 * (len(class_lower) / len(param_lower)))

            if score > highest_score:
                highest_score = score
                best_match = class_name

        return best_match if highest_score > 50 else None

    def _traverse_single_pass(
        self,
        node: Node,
        local_var_types: dict[str, str],
        module_qn: str,
    ) -> None:
        """Single-pass AST traversal combining multiple inference operations."""
        assignments: list[Node] = []
        comprehensions: list[Node] = []
        for_statements: list[Node] = []

        stack: list[Node] = [node]
        while stack:
            current = stack.pop()
            node_type = current.type

            if node_type == "assignment":
                assignments.append(current)
            elif node_type == "list_comprehension":
                comprehensions.append(current)
            elif node_type == "for_statement":
                for_statements.append(current)

            stack.extend(reversed(current.children))

        for assignment in assignments:
            self._process_assignment_simple(assignment, local_var_types, module_qn)

        for assignment in assignments:
            self._process_assignment_complex(assignment, local_var_types, module_qn)

        for comp in comprehensions:
            self._analyze_comprehension(comp, local_var_types, module_qn)

        for for_stmt in for_statements:
            self._analyze_for_loop(for_stmt, local_var_types, module_qn)

        self._infer_instance_variable_types_from_assignments(
            assignments, local_var_types, module_qn
        )

    def _process_assignment_simple(
        self,
        assignment_node: Node,
        local_var_types: dict[str, str],
        module_qn: str,
    ) -> None:
        left_node = assignment_node.child_by_field_name("left")
        right_node = assignment_node.child_by_field_name("right")

        if not left_node or not right_node:
            return

        var_name = self._extract_variable_name(left_node)
        if not var_name:
            return

        inferred_type = self._infer_type_from_expression_simple(right_node, module_qn)
        if inferred_type:
            local_var_types[var_name] = inferred_type

    def _process_assignment_complex(
        self,
        assignment_node: Node,
        local_var_types: dict[str, str],
        module_qn: str,
    ) -> None:
        left_node = assignment_node.child_by_field_name("left")
        right_node = assignment_node.child_by_field_name("right")

        if not left_node or not right_node:
            return

        var_name = self._extract_variable_name(left_node)
        if not var_name or var_name in local_var_types:
            return

        inferred_type = self._infer_type_from_expression_complex(
            right_node, module_qn, local_var_types
        )
        if inferred_type:
            local_var_types[var_name] = inferred_type

    def _infer_type_from_expression_simple(
        self,
        node: Node,
        module_qn: str,
    ) -> str | None:
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node and func_node.type == "identifier":
                class_name = safe_decode_text(func_node)
                if class_name and class_name[0].isupper():
                    return class_name

        elif node.type == "list_comprehension":
            body_node = node.child_by_field_name("body")
            if body_node:
                return self._infer_type_from_expression_simple(body_node, module_qn)

        return None

    def _infer_type_from_expression_complex(
        self,
        node: Node,
        module_qn: str,
        local_var_types: dict[str, str],
    ) -> str | None:
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node and func_node.type == "attribute":
                method_call_text = safe_decode_text(func_node)
                if method_call_text:
                    return self._infer_method_call_return_type(
                        method_call_text, module_qn, local_var_types
                    )

        return None

    def _analyze_comprehension(
        self,
        comp_node: Node,
        local_var_types: dict[str, str],
        module_qn: str,
    ) -> None:
        for child in comp_node.children:
            if child.type == "for_in_clause":
                left_node = child.child_by_field_name("left")
                right_node = child.child_by_field_name("right")
                if left_node and right_node:
                    self._infer_loop_var_from_iterable(
                        left_node, right_node, local_var_types, module_qn
                    )

    def _analyze_for_loop(
        self,
        for_node: Node,
        local_var_types: dict[str, str],
        module_qn: str,
    ) -> None:
        left_node = for_node.child_by_field_name("left")
        right_node = for_node.child_by_field_name("right")

        if left_node and right_node:
            self._infer_loop_var_from_iterable(
                left_node, right_node, local_var_types, module_qn
            )

    def _infer_loop_var_from_iterable(
        self,
        left_node: Node,
        right_node: Node,
        local_var_types: dict[str, str],
        module_qn: str,
    ) -> None:
        loop_var = self._extract_variable_name(left_node)
        if not loop_var:
            return

        element_type = self._infer_iterable_element_type(
            right_node, local_var_types, module_qn
        )
        if element_type:
            local_var_types[loop_var] = element_type

    def _infer_iterable_element_type(
        self,
        iterable_node: Node,
        local_var_types: dict[str, str],
        module_qn: str,
    ) -> str | None:
        if iterable_node.type == "list":
            for child in iterable_node.children:
                if child.type == "call":
                    func_node = child.child_by_field_name("function")
                    if func_node and func_node.type == "identifier":
                        class_name = safe_decode_text(func_node)
                        if class_name and class_name[0].isupper():
                            return class_name

        elif iterable_node.type == "identifier":
            var_name = safe_decode_text(iterable_node)
            if var_name and var_name in local_var_types:
                var_type = local_var_types[var_name]
                if var_type and var_type != "list":
                    return var_type

        return None

    def _infer_instance_variable_types_from_assignments(
        self,
        assignments: list[Node],
        local_var_types: dict[str, str],
        module_qn: str,
    ) -> None:
        for assignment in assignments:
            left_node = assignment.child_by_field_name("left")
            right_node = assignment.child_by_field_name("right")

            if left_node and right_node and left_node.type == "attribute":
                left_text = safe_decode_text(left_node)
                if left_text and left_text.startswith("self."):
                    assigned_type = self._infer_type_from_expression_simple(
                        right_node, module_qn
                    )
                    if assigned_type:
                        local_var_types[left_text] = assigned_type

    def _infer_method_call_return_type(
        self,
        method_call: str,
        module_qn: str,
        local_var_types: dict[str, str] | None = None,
    ) -> str | None:
        cache_key = f"{module_qn}:{method_call}"

        if cache_key in self._type_inference_in_progress:
            logger.debug(f"Recursion guard: skipping {method_call}")
            return None

        self._type_inference_in_progress.add(cache_key)
        try:
            if "." in method_call and self._is_method_chain(method_call):
                return self._infer_chained_call_return_type(
                    method_call, module_qn, local_var_types
                )

            return self._infer_simple_method_return_type(
                method_call, module_qn, local_var_types
            )
        finally:
            self._type_inference_in_progress.discard(cache_key)

    def _is_method_chain(self, call_name: str) -> bool:
        if "(" in call_name and ")" in call_name:
            return bool(_RE_METHOD_CHAIN.search(call_name))
        return False

    def _infer_chained_call_return_type(
        self,
        call_name: str,
        module_qn: str,
        local_var_types: dict[str, str] | None = None,
    ) -> str | None:
        match = _RE_FINAL_METHOD.search(call_name)
        if not match:
            return None

        final_method = match.group(1)
        object_expr = call_name[:match.start()]

        object_type = self._infer_object_type_for_chained_call(
            object_expr, module_qn, local_var_types
        )

        if object_type:
            method_qn = f"{object_type}.{final_method}"
            return self._get_method_return_type_from_registry(method_qn)

        return None

    def _infer_object_type_for_chained_call(
        self,
        object_expr: str,
        module_qn: str,
        local_var_types: dict[str, str] | None = None,
    ) -> str | None:
        if "(" not in object_expr and local_var_types and object_expr in local_var_types:
            return local_var_types[object_expr]

        if "(" in object_expr and ")" in object_expr:
            return self._infer_method_call_return_type(
                object_expr, module_qn, local_var_types
            )

        return None

    def _infer_simple_method_return_type(
        self,
        method_call: str,
        module_qn: str,
        local_var_types: dict[str, str] | None = None,
    ) -> str | None:
        if "." not in method_call:
            return None

        parts = method_call.split(".")
        if len(parts) < 2:
            return None

        class_name = parts[0]
        method_name = parts[-1].split("(")[0] if "(" in parts[-1] else parts[-1]

        if local_var_types and class_name in local_var_types:
            var_type = local_var_types[class_name]
            method_qn = f"{var_type}.{method_name}"
            return self._get_method_return_type_from_registry(method_qn)

        resolved_class = self._resolve_class_name(class_name, module_qn)
        if resolved_class:
            method_qn = f"{resolved_class}.{method_name}"
            return self._get_method_return_type_from_registry(method_qn)

        return None

    def _get_method_return_type_from_registry(self, method_qn: str) -> str | None:
        if method_qn in self._method_return_type_cache:
            return self._method_return_type_cache[method_qn]

        self._method_return_type_cache[method_qn] = None
        return None

    def _resolve_class_name(self, class_name: str, module_qn: str) -> str | None:
        local_qn = f"{module_qn}.{class_name}"
        if local_qn in self.function_registry:
            return local_qn

        if module_qn in self.import_mapping:
            if class_name in self.import_mapping[module_qn]:
                return self.import_mapping[module_qn][class_name]

        if class_name in self.simple_name_lookup:
            matches = self.simple_name_lookup[class_name]
            if len(matches) == 1:
                return next(iter(matches))

        return None

    def _extract_variable_name(self, node: Node) -> str | None:
        if node.type == "identifier":
            return safe_decode_text(node)
        return None

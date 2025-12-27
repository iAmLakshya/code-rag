from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from code_rag.parsing.call_resolution.builtins import (
    JS_BUILTIN_PATTERNS,
    JS_BUILTIN_TYPES,
    PYTHON_BUILTINS,
)

if TYPE_CHECKING:
    from tree_sitter import Node

    from code_rag.core.cache import FunctionRegistry
    from code_rag.parsing.import_processor import ImportProcessor
    from code_rag.parsing.type_inference.engine import TypeInferenceEngine

_RE_METHOD_CHAIN = re.compile(r"\)\.")
_RE_FINAL_METHOD = re.compile(r"\.([^.()]+)$")

logger = logging.getLogger(__name__)


def safe_decode_text(node: Node) -> str | None:
    if node.text:
        return node.text.decode("utf-8")
    return None


class CallProcessor:
    def __init__(
        self,
        function_registry: FunctionRegistry,
        import_processor: ImportProcessor,
        type_inference: TypeInferenceEngine,
        class_inheritance: dict[str, list[str]],
        project_name: str,
        repo_path: Path,
    ):
        self.function_registry = function_registry
        self.import_processor = import_processor
        self.type_inference = type_inference
        self.class_inheritance = class_inheritance
        self.project_name = project_name
        self.repo_path = repo_path

    def resolve_call(
        self,
        call_name: str,
        module_qn: str,
        local_var_types: dict[str, str] | None = None,
        class_context: str | None = None,
        language: str = "python",
    ) -> tuple[str, str] | None:
        if not call_name:
            return None

        if call_name == "super" or call_name.startswith("super.") or call_name.startswith("super()"):
            return self._resolve_super_call(call_name, module_qn, class_context)

        if "." in call_name and self._is_method_chain(call_name):
            return self._resolve_chained_call(call_name, module_qn, local_var_types)

        import_result = self._resolve_via_imports(
            call_name, module_qn, local_var_types, class_context
        )
        if import_result:
            return import_result

        same_module_result = self._resolve_same_module_call(call_name, module_qn)
        if same_module_result:
            return same_module_result

        builtin_result = self._resolve_builtin_call(call_name, language)
        if builtin_result:
            return builtin_result

        return self._resolve_by_simple_name(call_name, module_qn)

    def _resolve_super_call(
        self,
        call_name: str,
        module_qn: str,
        class_context: str | None,
    ) -> tuple[str, str] | None:
        if not class_context:
            logger.debug(f"No class context for super() call: {call_name}")
            return None

        if call_name == "super" or call_name == "super()":
            method_name = "__init__"
        elif call_name.startswith("super()."):
            method_name = call_name.split(".", 1)[1].split("(")[0]
        elif call_name.startswith("super."):
            method_name = call_name.split(".", 1)[1].split("(")[0]
        else:
            return None

        if class_context not in self.class_inheritance:
            logger.debug(f"No inheritance info for {class_context}")
            return None

        result = self._resolve_inherited_method(class_context, method_name)
        if result:
            logger.debug(f"Resolved super() call: {call_name} -> {result[1]}")
            return result

        logger.debug(f"Could not resolve super() call: {call_name}")
        return None

    def _resolve_inherited_method(
        self,
        class_qn: str,
        method_name: str,
    ) -> tuple[str, str] | None:
        if class_qn not in self.class_inheritance:
            return None

        visited = set()
        queue = list(self.class_inheritance.get(class_qn, []))

        while queue:
            parent_qn = queue.pop(0)
            if parent_qn in visited:
                continue
            visited.add(parent_qn)

            method_qn = f"{parent_qn}.{method_name}"
            entity_type = self.function_registry.get(method_qn)
            if entity_type:
                return (entity_type, method_qn)

            if parent_qn in self.class_inheritance:
                for grandparent_qn in self.class_inheritance[parent_qn]:
                    if grandparent_qn not in visited:
                        queue.append(grandparent_qn)

        return None

    def _is_method_chain(self, call_name: str) -> bool:
        if "(" in call_name and ")" in call_name:
            return bool(_RE_METHOD_CHAIN.search(call_name))
        return False

    def _resolve_chained_call(
        self,
        call_name: str,
        module_qn: str,
        local_var_types: dict[str, str] | None,
    ) -> tuple[str, str] | None:
        match = _RE_FINAL_METHOD.search(call_name)
        if not match:
            return None

        final_method = match.group(1)
        object_expr = call_name[: match.start()]

        object_type = self._infer_expression_type(object_expr, module_qn, local_var_types)
        if not object_type:
            return None

        resolved_class = self._resolve_class_name(object_type, module_qn)
        if not resolved_class:
            resolved_class = object_type

        method_qn = f"{resolved_class}.{final_method}"
        entity_type = self.function_registry.get(method_qn)
        if entity_type:
            logger.debug(f"Resolved chained call: {call_name} -> {method_qn}")
            return (entity_type, method_qn)

        inherited = self._resolve_inherited_method(resolved_class, final_method)
        if inherited:
            logger.debug(f"Resolved chained inherited call: {call_name} -> {inherited[1]}")
            return inherited

        return None

    def _infer_expression_type(
        self,
        expr: str,
        module_qn: str,
        local_var_types: dict[str, str] | None,
    ) -> str | None:
        if local_var_types and expr in local_var_types:
            return local_var_types[expr]

        if "(" in expr:
            if self.type_inference:
                return self.type_inference._infer_method_call_return_type(
                    expr, module_qn, local_var_types
                )

        return None

    def _resolve_via_imports(
        self,
        call_name: str,
        module_qn: str,
        local_var_types: dict[str, str] | None,
        class_context: str | None,
    ) -> tuple[str, str] | None:
        import_map = self.import_processor.get_import_mapping(module_qn)
        if not import_map:
            return None

        if call_name in import_map:
            imported_qn = import_map[call_name]
            entity_type = self.function_registry.get(imported_qn)
            if entity_type:
                logger.debug(f"Direct import resolved: {call_name} -> {imported_qn}")
                return (entity_type, imported_qn)

        if "." in call_name:
            parts = call_name.split(".")
            if len(parts) >= 2:
                object_name = parts[0]
                method_name = ".".join(parts[1:]).split("(")[0]

                if local_var_types and object_name in local_var_types:
                    var_type = local_var_types[object_name]
                    class_qn = self._resolve_type_to_class(var_type, module_qn, import_map)
                    if class_qn:
                        return self._try_resolve_method(class_qn, method_name)

                if object_name in import_map:
                    imported_qn = import_map[object_name]
                    method_qn = f"{imported_qn}.{method_name}"
                    entity_type = self.function_registry.get(method_qn)
                    if entity_type:
                        logger.debug(f"Import method resolved: {call_name} -> {method_qn}")
                        return (entity_type, method_qn)

        for local_name, imported_qn in import_map.items():
            if local_name.startswith("*"):
                wildcard_qn = f"{imported_qn}.{call_name}"
                entity_type = self.function_registry.get(wildcard_qn)
                if entity_type:
                    logger.debug(f"Wildcard import resolved: {call_name} -> {wildcard_qn}")
                    return (entity_type, wildcard_qn)

        return None

    def _resolve_type_to_class(
        self,
        type_name: str,
        module_qn: str,
        import_map: dict[str, str],
    ) -> str | None:
        if "." in type_name:
            return type_name

        if type_name in import_map:
            return import_map[type_name]

        local_qn = f"{module_qn}.{type_name}"
        if self.function_registry.get(local_qn) == "Class":
            return local_qn

        matches = self.function_registry.find_by_simple_name(type_name)
        for match in matches:
            if self.function_registry.get(match) == "Class":
                return match

        return None

    def _try_resolve_method(
        self,
        class_qn: str,
        method_name: str,
    ) -> tuple[str, str] | None:
        method_qn = f"{class_qn}.{method_name}"
        entity_type = self.function_registry.get(method_qn)
        if entity_type:
            return (entity_type, method_qn)

        return self._resolve_inherited_method(class_qn, method_name)

    def _resolve_same_module_call(
        self,
        call_name: str,
        module_qn: str,
    ) -> tuple[str, str] | None:
        simple_name = call_name.split(".")[0].split("(")[0]

        local_qn = f"{module_qn}.{simple_name}"
        entity_type = self.function_registry.get(local_qn)
        if entity_type:
            logger.debug(f"Same-module resolved: {call_name} -> {local_qn}")
            return (entity_type, local_qn)

        return None

    def _resolve_builtin_call(
        self,
        call_name: str,
        language: str,
    ) -> tuple[str, str] | None:
        simple_name = call_name.split("(")[0]

        if language == "python":
            if simple_name in PYTHON_BUILTINS:
                return ("Function", f"builtins.{simple_name}")

        elif language in ("javascript", "typescript", "jsx", "tsx"):
            if call_name in JS_BUILTIN_PATTERNS:
                return ("Function", f"builtin.{call_name}")
            if simple_name in JS_BUILTIN_TYPES:
                return ("Class", f"builtin.{simple_name}")

        return None

    def _resolve_by_simple_name(
        self,
        call_name: str,
        module_qn: str,
    ) -> tuple[str, str] | None:
        simple_name = call_name.split(".")[-1].split("(")[0]

        matches = self.function_registry.find_by_simple_name(simple_name)
        if not matches:
            return None

        matches.sort(key=lambda qn: self._calculate_distance(qn, module_qn))
        best_qn = matches[0]

        entity_type = self.function_registry.get(best_qn)
        if entity_type:
            logger.debug(f"Fallback resolved: {call_name} -> {best_qn}")
            return (entity_type, best_qn)

        return None

    def _calculate_distance(self, candidate_qn: str, caller_module_qn: str) -> int:
        caller_parts = caller_module_qn.split(".")
        candidate_parts = candidate_qn.split(".")

        common_prefix = 0
        for i in range(min(len(caller_parts), len(candidate_parts))):
            if caller_parts[i] == candidate_parts[i]:
                common_prefix += 1
            else:
                break

        distance = (len(caller_parts) - common_prefix) + (len(candidate_parts) - common_prefix)

        if candidate_qn.startswith(caller_module_qn + "."):
            distance -= 2

        return distance

    def _resolve_class_name(self, class_name: str, module_qn: str) -> str | None:
        local_qn = f"{module_qn}.{class_name}"
        if self.function_registry.get(local_qn) == "Class":
            return local_qn

        import_map = self.import_processor.get_import_mapping(module_qn)
        if import_map and class_name in import_map:
            return import_map[class_name]

        matches = self.function_registry.find_by_simple_name(class_name)
        for match in matches:
            if self.function_registry.get(match) == "Class":
                return match

        return None

    def extract_calls_from_node(
        self,
        node: Node,
        source: str,
        language: str,
    ) -> list[str]:
        calls = set()

        stack = [node]
        while stack:
            current = stack.pop()

            if self._is_call_node(current, language):
                call_name = self._get_call_name(current, source, language)
                if call_name:
                    calls.add(call_name)

            stack.extend(reversed(current.children))

        return list(calls)

    def _is_call_node(self, node: Node, language: str) -> bool:
        if language == "python":
            return node.type == "call"
        elif language in ("javascript", "typescript", "jsx", "tsx"):
            return node.type == "call_expression"
        elif language == "java":
            return node.type == "method_invocation"
        else:
            return node.type in ("call", "call_expression")

    def _get_call_name(self, node: Node, source: str, language: str) -> str | None:
        func_node = None

        if language == "python":
            if node.children:
                func_node = node.children[0]
        else:
            func_node = node.child_by_field_name("function")

        if func_node:
            text = safe_decode_text(func_node)
            if text:
                return text

        return None

"""Enhanced call processor for resolving function and method calls.

This module provides sophisticated call resolution including:
- Import-based resolution
- Inheritance chain resolution (method calls on parent classes)
- Super call handling (super(), super().method())
- Chained method call resolution (a.b().c())
- Builtin type handling (JavaScript/Python)
- Wildcard import resolution
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tree_sitter import Node

    from code_rag.core.cache import FunctionRegistry
    from code_rag.parsing.import_processor import ImportProcessor
    from code_rag.parsing.type_inference.engine import TypeInferenceEngine

# Pre-compiled regex patterns
_RE_METHOD_CHAIN = re.compile(r"\)\.")
_RE_FINAL_METHOD = re.compile(r"\.([^.()]+)$")

logger = logging.getLogger(__name__)


# Built-in types and patterns for different languages
_PYTHON_BUILTINS = {
    "print",
    "len",
    "range",
    "int",
    "str",
    "float",
    "bool",
    "list",
    "dict",
    "set",
    "tuple",
    "open",
    "type",
    "isinstance",
    "hasattr",
    "getattr",
    "setattr",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sorted",
    "reversed",
    "any",
    "all",
    "sum",
    "min",
    "max",
    "abs",
    "round",
    "input",
    "super",
    "classmethod",
    "staticmethod",
    "property",
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
}

_JS_BUILTIN_TYPES = {
    "Array",
    "Object",
    "String",
    "Number",
    "Date",
    "RegExp",
    "Function",
    "Map",
    "Set",
    "Promise",
    "Error",
    "Boolean",
}

_JS_BUILTIN_PATTERNS = {
    "Object.create",
    "Object.keys",
    "Object.values",
    "Object.entries",
    "Object.assign",
    "Array.from",
    "Array.isArray",
    "parseInt",
    "parseFloat",
    "console.log",
    "console.error",
    "console.warn",
    "JSON.parse",
    "JSON.stringify",
    "Math.random",
    "Math.floor",
    "Math.ceil",
    "Math.round",
    "Date.now",
}


def safe_decode_text(node: "Node") -> str | None:
    """Safely decode text from a tree-sitter node."""
    if node.text:
        return node.text.decode("utf-8")
    return None


class CallProcessor:
    """Handles processing and resolution of function and method calls.

    This class provides sophisticated call resolution that handles:
    - Simple function calls: foo()
    - Method calls: obj.method()
    - Chained calls: obj.a().b().c()
    - Super calls: super().__init__()
    - Imported functions: from module import func; func()
    - Inheritance-based resolution: calls to parent class methods

    Attributes:
        function_registry: Registry of known functions/classes.
        import_processor: Processor for import mappings.
        type_inference: Engine for inferring variable types.
        class_inheritance: Map of class -> parent classes.
        project_name: Name of the current project.
        repo_path: Root path of the repository.
    """

    def __init__(
        self,
        function_registry: "FunctionRegistry",
        import_processor: "ImportProcessor",
        type_inference: "TypeInferenceEngine",
        class_inheritance: dict[str, list[str]],
        project_name: str,
        repo_path: Path,
    ):
        """Initialize the call processor.

        Args:
            function_registry: Registry of known functions/classes.
            import_processor: Processor for import mappings.
            type_inference: Engine for inferring variable types.
            class_inheritance: Map of class qualified name -> list of parent qualified names.
            project_name: Name of the current project.
            repo_path: Root path of the repository.
        """
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
        """Resolve a function/method call to its qualified name and type.

        Args:
            call_name: The call expression (e.g., "obj.method", "func", "super().method").
            module_qn: The qualified name of the current module.
            local_var_types: Map of local variable names to their inferred types.
            class_context: The qualified name of the enclosing class (if in a method).
            language: The programming language.

        Returns:
            Tuple of (entity_type, qualified_name) or None if unresolved.
        """
        if not call_name:
            return None

        # Handle super calls
        if call_name == "super" or call_name.startswith("super.") or call_name.startswith("super()"):
            return self._resolve_super_call(call_name, module_qn, class_context)

        # Handle chained method calls
        if "." in call_name and self._is_method_chain(call_name):
            return self._resolve_chained_call(call_name, module_qn, local_var_types)

        # Handle import-resolved calls
        import_result = self._resolve_via_imports(
            call_name, module_qn, local_var_types, class_context
        )
        if import_result:
            return import_result

        # Handle same-module calls
        same_module_result = self._resolve_same_module_call(call_name, module_qn)
        if same_module_result:
            return same_module_result

        # Handle builtin calls
        builtin_result = self._resolve_builtin_call(call_name, language)
        if builtin_result:
            return builtin_result

        # Fallback: try fuzzy matching by simple name
        return self._resolve_by_simple_name(call_name, module_qn)

    def _resolve_super_call(
        self,
        call_name: str,
        module_qn: str,
        class_context: str | None,
    ) -> tuple[str, str] | None:
        """Resolve super() calls to parent class methods.

        Handles patterns like:
        - super()  -> calls parent __init__
        - super().__init__() -> explicit parent __init__
        - super().method() -> parent method

        Args:
            call_name: The super call expression.
            module_qn: Current module qualified name.
            class_context: Enclosing class qualified name.

        Returns:
            Tuple of (entity_type, qualified_name) or None.
        """
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
        """Resolve a method by searching the inheritance chain.

        Uses BFS to search parent classes in MRO order.

        Args:
            class_qn: The class to start searching from.
            method_name: The method name to find.

        Returns:
            Tuple of (entity_type, qualified_name) or None.
        """
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
        """Check if this is a chained method call (not just obj.method).

        Returns True for patterns like:
        - obj.method().other() - chained calls
        - obj().method() - call result method

        Args:
            call_name: The call expression.

        Returns:
            True if this is a method chain.
        """
        if "(" in call_name and ")" in call_name:
            return bool(_RE_METHOD_CHAIN.search(call_name))
        return False

    def _resolve_chained_call(
        self,
        call_name: str,
        module_qn: str,
        local_var_types: dict[str, str] | None,
    ) -> tuple[str, str] | None:
        """Resolve chained method calls like obj.method().other().

        Works by:
        1. Extracting the final method name
        2. Inferring the return type of the preceding expression
        3. Looking up the method on that type

        Args:
            call_name: The chained call expression.
            module_qn: Current module qualified name.
            local_var_types: Local variable type map.

        Returns:
            Tuple of (entity_type, qualified_name) or None.
        """
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
        """Infer the type of an expression.

        Args:
            expr: The expression to analyze.
            module_qn: Current module qualified name.
            local_var_types: Local variable type map.

        Returns:
            Inferred type name or None.
        """
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
        """Resolve a call using import mappings.

        Args:
            call_name: The call expression.
            module_qn: Current module qualified name.
            local_var_types: Local variable type map.
            class_context: Enclosing class qualified name.

        Returns:
            Tuple of (entity_type, qualified_name) or None.
        """
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
        """Resolve a type name to its fully qualified class name.

        Args:
            type_name: The type name (simple or qualified).
            module_qn: Current module qualified name.
            import_map: Import mapping for the module.

        Returns:
            Fully qualified class name or None.
        """
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
        """Try to resolve a method on a class, including inheritance.

        Args:
            class_qn: Class qualified name.
            method_name: Method name to find.

        Returns:
            Tuple of (entity_type, qualified_name) or None.
        """
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
        """Resolve a call to something in the same module.

        Args:
            call_name: The call expression (simple name).
            module_qn: Current module qualified name.

        Returns:
            Tuple of (entity_type, qualified_name) or None.
        """
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
        """Resolve built-in function calls.

        Args:
            call_name: The call expression.
            language: Programming language.

        Returns:
            Tuple of (entity_type, qualified_name) or None.
        """
        simple_name = call_name.split("(")[0]

        if language == "python":
            if simple_name in _PYTHON_BUILTINS:
                return ("Function", f"builtins.{simple_name}")

        elif language in ("javascript", "typescript", "jsx", "tsx"):
            if call_name in _JS_BUILTIN_PATTERNS:
                return ("Function", f"builtin.{call_name}")
            if simple_name in _JS_BUILTIN_TYPES:
                return ("Class", f"builtin.{simple_name}")

        return None

    def _resolve_by_simple_name(
        self,
        call_name: str,
        module_qn: str,
    ) -> tuple[str, str] | None:
        """Fallback resolution using simple name matching.

        Finds functions with matching simple names and ranks by distance.

        Args:
            call_name: The call expression.
            module_qn: Current module qualified name.

        Returns:
            Tuple of (entity_type, qualified_name) or None.
        """
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
        """Calculate distance between two qualified names.

        Lower distance means more likely to be the correct resolution.

        Args:
            candidate_qn: Candidate qualified name.
            caller_module_qn: Caller's module qualified name.

        Returns:
            Distance score (lower is better).
        """
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
        """Convert a simple class name to its qualified name.

        Args:
            class_name: Simple class name.
            module_qn: Current module qualified name.

        Returns:
            Qualified class name or None.
        """
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
        node: "Node",
        source: str,
        language: str,
    ) -> list[str]:
        """Extract all function/method calls from an AST node.

        Args:
            node: Tree-sitter node to analyze.
            source: Source code string.
            language: Programming language.

        Returns:
            List of call expressions.
        """
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

    def _is_call_node(self, node: "Node", language: str) -> bool:
        """Check if a node is a function call."""
        if language == "python":
            return node.type == "call"
        elif language in ("javascript", "typescript", "jsx", "tsx"):
            return node.type == "call_expression"
        elif language == "java":
            return node.type == "method_invocation"
        else:
            return node.type in ("call", "call_expression")

    def _get_call_name(self, node: "Node", source: str, language: str) -> str | None:
        """Extract the call target name from a call node."""
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

"""Python-specific type inference logic.

Handles Python-specific patterns for inferring variable types:
- Type annotations
- Constructor calls (User())
- Assignment from method returns
- Parameter types
- Instance attributes (self.x)
- Loop variables
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from code_rag.parsing.type_inference.models import (
    InferredType,
    TypeInferenceContext,
    TypeSource,
    VariableTypeMap,
)

if TYPE_CHECKING:
    from tree_sitter import Node

logger = logging.getLogger(__name__)


class PythonTypeInference:
    """Python-specific type inference engine.

    Analyzes Python AST nodes to infer variable types within a function scope.
    """

    def __init__(
        self,
        function_registry: Any = None,
        import_mapping: dict[str, dict[str, str]] | None = None,
    ):
        """Initialize Python type inference.

        Args:
            function_registry: Registry of known functions/classes.
            import_mapping: Map of module_qn -> {local_name: imported_qn}.
        """
        self.function_registry = function_registry
        self.import_mapping = import_mapping or {}

        # Cache for method return type analysis
        self._return_type_cache: dict[str, str | None] = {}
        self._in_progress: set[str] = set()  # Recursion guard

    def infer_local_types(
        self,
        function_node: Node,
        context: TypeInferenceContext,
    ) -> VariableTypeMap:
        """Infer types for all local variables in a function.

        Args:
            function_node: Tree-sitter node for the function.
            context: Type inference context.

        Returns:
            Map of variable names to inferred types.
        """
        type_map = VariableTypeMap()

        try:
            self._infer_parameter_types(function_node, type_map, context)

            assignments = self._collect_assignments(function_node)

            for assignment in assignments:
                self._process_simple_assignment(assignment, type_map, context)

            for assignment in assignments:
                self._process_complex_assignment(assignment, type_map, context)

            self._infer_loop_variable_types(function_node, type_map, context)

            if context.class_name:
                self._infer_instance_attrs_from_init(
                    function_node, type_map, context
                )

        except Exception as e:
            logger.debug(f"Error inferring types: {e}")

        return type_map

    def _infer_parameter_types(
        self,
        function_node: Node,
        type_map: VariableTypeMap,
        context: TypeInferenceContext,
    ) -> None:
        """Infer types from function parameters.

        Handles:
        - Type annotations: def foo(user: User)
        - Self/cls parameters
        - Naming conventions: def foo(user_service) -> UserService
        """
        params_node = function_node.child_by_field_name("parameters")
        if not params_node:
            return

        for param in params_node.children:
            if param.type == "identifier":
                param_name = self._get_node_text(param)
                if param_name:
                    if param_name in ("self", "cls"):
                        continue

                    inferred = self._infer_type_from_name(param_name, context)
                    if inferred:
                        type_map.set_type(param_name, inferred)

            elif param.type == "typed_parameter":
                name_node = param.child_by_field_name("name")
                type_node = param.child_by_field_name("type")

                if name_node and type_node:
                    param_name = self._get_node_text(name_node)
                    type_name = self._get_node_text(type_node)

                    if param_name and type_name:
                        resolved_qn = self._resolve_type_name(type_name, context)
                        type_map.set_type(
                            param_name,
                            InferredType(
                                type_name=type_name,
                                qualified_name=resolved_qn,
                                source=TypeSource.ANNOTATION,
                            ),
                        )

            elif param.type == "default_parameter":
                name_node = param.child_by_field_name("name")
                value_node = param.child_by_field_name("value")

                if name_node and value_node:
                    param_name = self._get_node_text(name_node)
                    if param_name:
                        inferred = self._infer_type_from_expression(
                            value_node, context
                        )
                        if inferred:
                            type_map.set_type(param_name, inferred)

    def _collect_assignments(self, node: Node) -> list[Node]:
        """Collect all assignment nodes in a function body.

        Args:
            node: Root node to search.

        Returns:
            List of assignment nodes.
        """
        assignments = []
        stack = [node]

        while stack:
            current = stack.pop()
            if current.type == "assignment":
                assignments.append(current)
            if current.type not in ("function_definition", "class_definition"):
                stack.extend(reversed(current.children))

        return assignments

    def _process_simple_assignment(
        self,
        assignment: Node,
        type_map: VariableTypeMap,
        context: TypeInferenceContext,
    ) -> None:
        """Process simple assignments (constructors, literals).

        Args:
            assignment: Assignment AST node.
            type_map: Type map to update.
            context: Inference context.
        """
        left = assignment.child_by_field_name("left")
        right = assignment.child_by_field_name("right")

        if not left or not right:
            return

        var_name = self._extract_variable_name(left)
        if not var_name:
            return

        if var_name in type_map:
            return

        inferred = self._infer_simple_type(right, context)
        if inferred:
            type_map.set_type(var_name, inferred)

    def _process_complex_assignment(
        self,
        assignment: Node,
        type_map: VariableTypeMap,
        context: TypeInferenceContext,
    ) -> None:
        """Process complex assignments (method returns).

        Args:
            assignment: Assignment AST node.
            type_map: Type map to update.
            context: Inference context.
        """
        left = assignment.child_by_field_name("left")
        right = assignment.child_by_field_name("right")

        if not left or not right:
            return

        var_name = self._extract_variable_name(left)
        if not var_name:
            return

        if var_name in type_map:
            return

        if right.type == "call":
            inferred = self._infer_method_return_type(right, type_map, context)
            if inferred:
                type_map.set_type(var_name, inferred)

    def _infer_simple_type(
        self,
        expr_node: Node,
        context: TypeInferenceContext,
    ) -> InferredType | None:
        """Infer type from simple expressions.

        Handles:
        - Constructor calls: User()
        - Literals: [], {}, "", etc.
        """
        if expr_node.type == "call":
            func_node = expr_node.child_by_field_name("function")
            if func_node and func_node.type == "identifier":
                func_name = self._get_node_text(func_node)
                if func_name and func_name[0].isupper():
                    resolved_qn = self._resolve_type_name(func_name, context)
                    return InferredType(
                        type_name=func_name,
                        qualified_name=resolved_qn,
                        source=TypeSource.CONSTRUCTOR,
                    )

        elif expr_node.type == "list":
            return InferredType(type_name="list", source=TypeSource.INFERRED)

        elif expr_node.type == "dictionary":
            return InferredType(type_name="dict", source=TypeSource.INFERRED)

        elif expr_node.type == "string":
            return InferredType(type_name="str", source=TypeSource.INFERRED)

        elif expr_node.type in ("integer", "float"):
            type_name = "int" if expr_node.type == "integer" else "float"
            return InferredType(type_name=type_name, source=TypeSource.INFERRED)

        return None

    def _infer_method_return_type(
        self,
        call_node: Node,
        type_map: VariableTypeMap,
        context: TypeInferenceContext,
    ) -> InferredType | None:
        """Infer type from a method call's return type.

        Args:
            call_node: Call expression AST node.
            type_map: Current type map for resolving receiver.
            context: Inference context.

        Returns:
            Inferred return type or None.
        """
        func_node = call_node.child_by_field_name("function")
        if not func_node:
            return None

        if func_node.type == "identifier":
            func_name = self._get_node_text(func_node)
            if func_name and func_name[0].isupper():
                resolved_qn = self._resolve_type_name(func_name, context)
                return InferredType(
                    type_name=func_name,
                    qualified_name=resolved_qn,
                    source=TypeSource.CONSTRUCTOR,
                )
            return None

        if func_node.type == "attribute":
            return self._infer_attribute_call_type(func_node, type_map, context)

        return None

    def _infer_attribute_call_type(
        self,
        attr_node: Node,
        type_map: VariableTypeMap,
        context: TypeInferenceContext,
    ) -> InferredType | None:
        """Infer type from attribute method call.

        Args:
            attr_node: Attribute AST node (receiver.method).
            type_map: Current type map.
            context: Inference context.

        Returns:
            Inferred return type or None.
        """
        object_node = attr_node.child_by_field_name("object")
        attr_name_node = attr_node.child_by_field_name("attribute")

        if not object_node or not attr_name_node:
            return None

        method_name = self._get_node_text(attr_name_node)
        if not method_name:
            return None

        receiver_type = self._get_receiver_type(object_node, type_map, context)
        if not receiver_type:
            return None

        method_qn = f"{receiver_type}.{method_name}"

        if method_qn in self._in_progress:
            return None

        if method_qn in self._return_type_cache:
            cached = self._return_type_cache[method_qn]
            if cached:
                return InferredType(
                    type_name=cached.split(".")[-1],
                    qualified_name=cached,
                    source=TypeSource.METHOD_RETURN,
                )
            return None

        return None

    def _get_receiver_type(
        self,
        object_node: Node,
        type_map: VariableTypeMap,
        context: TypeInferenceContext,
    ) -> str | None:
        """Get the type of a method call receiver.

        Args:
            object_node: AST node for the receiver.
            type_map: Current type map.
            context: Inference context.

        Returns:
            Qualified type name or None.
        """
        if object_node.type == "identifier":
            var_name = self._get_node_text(object_node)
            if var_name:
                if var_name in type_map:
                    inferred = type_map[var_name]
                    return inferred.qualified_name or inferred.type_name

                if var_name == "self" and context.class_qn:
                    return context.class_qn

                if context.module_qn in self.import_mapping:
                    imports = self.import_mapping[context.module_qn]
                    if var_name in imports:
                        return imports[var_name]

        elif object_node.type == "attribute":
            return self._resolve_attribute_type(object_node, type_map, context)

        return None

    def _resolve_attribute_type(
        self,
        attr_node: Node,
        type_map: VariableTypeMap,
        context: TypeInferenceContext,
    ) -> str | None:
        """Resolve the type of an attribute access.

        Args:
            attr_node: Attribute AST node.
            type_map: Current type map.
            context: Inference context.

        Returns:
            Qualified type name or None.
        """
        object_node = attr_node.child_by_field_name("object")
        attr_name_node = attr_node.child_by_field_name("attribute")

        if not object_node or not attr_name_node:
            return None

        attr_name = self._get_node_text(attr_name_node)
        if not attr_name:
            return None

        if object_node.type == "identifier":
            obj_name = self._get_node_text(object_node)
            if obj_name == "self":
                attr_type = type_map.get_instance_attr(attr_name)
                if attr_type:
                    return attr_type.qualified_name or attr_type.type_name

        return None

    def _infer_loop_variable_types(
        self,
        function_node: Node,
        type_map: VariableTypeMap,
        context: TypeInferenceContext,
    ) -> None:
        """Infer types for loop variables.

        Args:
            function_node: Function AST node.
            type_map: Type map to update.
            context: Inference context.
        """
        stack = [function_node]

        while stack:
            current = stack.pop()

            if current.type == "for_statement":
                left = current.child_by_field_name("left")
                right = current.child_by_field_name("right")

                if left and right:
                    var_name = self._extract_variable_name(left)
                    if var_name and var_name not in type_map:
                        elem_type = self._infer_iterable_element_type(
                            right, type_map, context
                        )
                        if elem_type:
                            type_map.set_type(var_name, elem_type)

            if current.type not in ("function_definition", "class_definition"):
                stack.extend(reversed(current.children))

    def _infer_iterable_element_type(
        self,
        iterable_node: Node,
        type_map: VariableTypeMap,
        context: TypeInferenceContext,
    ) -> InferredType | None:
        """Infer the element type of an iterable.

        Args:
            iterable_node: AST node for the iterable.
            type_map: Current type map.
            context: Inference context.

        Returns:
            Inferred element type or None.
        """
        if iterable_node.type == "list":
            for child in iterable_node.children:
                if child.type == "call":
                    elem_type = self._infer_simple_type(child, context)
                    if elem_type:
                        elem_type.source = TypeSource.LOOP_VARIABLE
                        return elem_type

        return None

    def _infer_instance_attrs_from_init(
        self,
        function_node: Node,
        type_map: VariableTypeMap,
        context: TypeInferenceContext,
    ) -> None:
        """Infer instance attribute types from __init__ method.

        Args:
            function_node: Current function AST node.
            type_map: Type map to update.
            context: Inference context.
        """
        class_node = self._find_containing_class(function_node)
        if not class_node:
            return

        init_node = self._find_init_method(class_node)
        if not init_node:
            return

        self._analyze_self_assignments(init_node, type_map, context)

    def _analyze_self_assignments(
        self,
        node: Node,
        type_map: VariableTypeMap,
        context: TypeInferenceContext,
    ) -> None:
        """Analyze self.x = ... assignments.

        Args:
            node: AST node to search.
            type_map: Type map to update.
            context: Inference context.
        """
        stack = [node]

        while stack:
            current = stack.pop()

            if current.type == "assignment":
                left = current.child_by_field_name("left")
                right = current.child_by_field_name("right")

                if left and right and left.type == "attribute":
                    left_text = self._get_node_text(left)
                    if left_text and left_text.startswith("self."):
                        attr_name = left_text[5:]  # Remove "self."
                        inferred = self._infer_type_from_expression(right, context)
                        if inferred:
                            type_map.set_instance_attr(attr_name, inferred)

            if current.type not in ("function_definition", "class_definition"):
                stack.extend(reversed(current.children))

    def _infer_type_from_expression(
        self,
        expr_node: Node,
        context: TypeInferenceContext,
    ) -> InferredType | None:
        """Infer type from any expression.

        Args:
            expr_node: Expression AST node.
            context: Inference context.

        Returns:
            Inferred type or None.
        """
        # Try simple type first
        simple = self._infer_simple_type(expr_node, context)
        if simple:
            return simple

        # Future enhancement: Handle more complex expressions
        # e.g., list comprehensions, conditional expressions, etc.
        return None

    def _find_containing_class(self, node: Node) -> Node | None:
        """Find the class containing a node.

        Args:
            node: AST node.

        Returns:
            Class definition node or None.
        """
        current = node.parent
        while current:
            if current.type == "class_definition":
                return current
            current = current.parent
        return None

    def _find_init_method(self, class_node: Node) -> Node | None:
        """Find __init__ method in a class.

        Args:
            class_node: Class definition node.

        Returns:
            __init__ function node or None.
        """
        body = class_node.child_by_field_name("body")
        if not body:
            return None

        for child in body.children:
            if child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = self._get_node_text(name_node)
                    if name == "__init__":
                        return child

        return None

    def _extract_variable_name(self, node: Node) -> str | None:
        """Extract variable name from assignment LHS.

        Args:
            node: Left-hand side AST node.

        Returns:
            Variable name or None.
        """
        if node.type == "identifier":
            return self._get_node_text(node)
        return None

    def _infer_type_from_name(
        self,
        name: str,
        context: TypeInferenceContext,
    ) -> InferredType | None:
        """Infer type from variable naming convention.

        Examples:
        - user_service -> UserService
        - user -> User

        Args:
            name: Variable name.
            context: Inference context.

        Returns:
            Inferred type or None.
        """
        # Convert snake_case to PascalCase
        if "_" in name:
            parts = name.split("_")
            class_name = "".join(p.capitalize() for p in parts)
        else:
            class_name = name.capitalize()

        # Check if this class exists
        resolved = self._resolve_type_name(class_name, context)
        if resolved:
            return InferredType(
                type_name=class_name,
                qualified_name=resolved,
                source=TypeSource.INFERRED,
                confidence=0.5,  # Lower confidence for name-based inference
            )

        return None

    def _resolve_type_name(
        self,
        type_name: str,
        context: TypeInferenceContext,
    ) -> str | None:
        """Resolve a type name to its qualified name.

        Args:
            type_name: Simple type name.
            context: Inference context.

        Returns:
            Qualified name or None.
        """
        # Check imports
        if context.module_qn in self.import_mapping:
            imports = self.import_mapping[context.module_qn]
            if type_name in imports:
                return imports[type_name]

        # Check local module
        local_qn = f"{context.module_qn}.{type_name}"
        if self.function_registry and local_qn in self.function_registry:
            return local_qn

        # Check function registry by simple name
        if self.function_registry:
            if hasattr(self.function_registry, 'find_by_simple_name'):
                matches = self.function_registry.find_by_simple_name(type_name)
                if len(matches) == 1:
                    return matches[0]

        return None

    def _get_node_text(self, node: Node) -> str | None:
        """Get text content of a node.

        Args:
            node: AST node.

        Returns:
            Node text or None.
        """
        if node.text:
            return node.text.decode("utf-8")
        return None

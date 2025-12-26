"""Import processor for managing import mappings and resolution.

This module provides comprehensive import processing including:
- Standard imports (import foo)
- Aliased imports (import foo as bar)
- From imports (from foo import bar)
- Relative imports (from . import bar)
- Wildcard imports (from foo import *)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tree_sitter import Node

    from code_rag.core.cache import FunctionRegistry

logger = logging.getLogger(__name__)


def safe_decode_text(node: "Node") -> str | None:
    """Safely decode text from a tree-sitter node."""
    if node.text:
        return node.text.decode("utf-8")
    return None


class ImportProcessor:
    """Handles parsing and management of import statements.

    This class maintains a mapping of local names to fully qualified names
    for each module, enabling accurate resolution of function/class calls.

    Attributes:
        import_mapping: Dict mapping module_qn -> {local_name: qualified_name}.
        function_registry: Registry of known functions/classes.
        project_name: Name of the current project.
        repo_path: Root path of the repository.
    """

    def __init__(
        self,
        function_registry: "FunctionRegistry",
        project_name: str,
        repo_path: Path,
    ):
        """Initialize the import processor.

        Args:
            function_registry: Registry of known functions/classes.
            project_name: Name of the current project.
            repo_path: Root path of the repository.
        """
        self.function_registry = function_registry
        self.project_name = project_name
        self.repo_path = repo_path
        self.import_mapping: dict[str, dict[str, str]] = {}

    def get_import_mapping(self, module_qn: str) -> dict[str, str]:
        """Get the import mapping for a module.

        Args:
            module_qn: Module qualified name.

        Returns:
            Dict mapping local names to qualified names.
        """
        return self.import_mapping.get(module_qn, {})

    def clear_module(self, module_qn: str) -> None:
        """Clear import mappings for a module.

        Args:
            module_qn: Module qualified name.
        """
        if module_qn in self.import_mapping:
            del self.import_mapping[module_qn]

    def parse_imports(
        self,
        root_node: "Node",
        module_qn: str,
        language: str,
    ) -> None:
        """Parse import statements from an AST and build import mapping.

        Args:
            root_node: Root AST node.
            module_qn: Module qualified name.
            language: Programming language.
        """
        self.import_mapping[module_qn] = {}

        if language == "python":
            self._parse_python_imports(root_node, module_qn)
        elif language in ("javascript", "typescript", "jsx", "tsx"):
            self._parse_js_ts_imports(root_node, module_qn)
        elif language == "java":
            self._parse_java_imports(root_node, module_qn)
        else:
            logger.debug(f"Import parsing not implemented for {language}")

        logger.debug(f"Parsed {len(self.import_mapping[module_qn])} imports in {module_qn}")

    def _parse_python_imports(self, root_node: "Node", module_qn: str) -> None:
        """Parse Python import statements."""
        for node in self._walk_tree(root_node, {"import_statement", "import_from_statement"}):
            if node.type == "import_statement":
                self._handle_python_import_statement(node, module_qn)
            elif node.type == "import_from_statement":
                self._handle_python_import_from_statement(node, module_qn)

    def _handle_python_import_statement(self, node: "Node", module_qn: str) -> None:
        """Handle 'import module' statements."""
        for child in node.children:
            if child.type == "dotted_name":
                module_name = safe_decode_text(child)
                if module_name:
                    local_name = module_name.split(".")[0]
                    full_name = self._resolve_python_module(module_name)
                    self.import_mapping[module_qn][local_name] = full_name
                    logger.debug(f"Import: {local_name} -> {full_name}")

            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node and alias_node:
                    module_name = safe_decode_text(name_node)
                    alias = safe_decode_text(alias_node)
                    if module_name and alias:
                        full_name = self._resolve_python_module(module_name)
                        self.import_mapping[module_qn][alias] = full_name
                        logger.debug(f"Aliased import: {alias} -> {full_name}")

    def _handle_python_import_from_statement(self, node: "Node", module_qn: str) -> None:
        """Handle 'from module import name' statements."""
        module_name_node = node.child_by_field_name("module_name")
        if module_name_node is None:
            for child in node.children:
                if child.type == "dotted_name":
                    module_name_node = child
                    break
                elif child.type == "relative_import":
                    module_name_node = child
                    break

        if not module_name_node:
            return

        if module_name_node.type == "relative_import":
            base_module = self._resolve_relative_import(module_name_node, module_qn)
        else:
            module_text = safe_decode_text(module_name_node)
            base_module = self._resolve_python_module(module_text) if module_text else ""

        if not base_module:
            return

        is_wildcard = any(child.type == "wildcard_import" for child in node.children)

        if is_wildcard:
            wildcard_key = f"*{base_module}"
            self.import_mapping[module_qn][wildcard_key] = base_module
            logger.debug(f"Wildcard import: * -> {base_module}")
            return

        for child in node.children:
            if child.type == "dotted_name" and child != module_name_node:
                name = safe_decode_text(child)
                if name:
                    full_name = f"{base_module}.{name}"
                    self.import_mapping[module_qn][name] = full_name
                    logger.debug(f"From import: {name} -> {full_name}")

            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    for subchild in child.children:
                        if subchild.type in ("identifier", "dotted_name"):
                            name_node = subchild
                            break

                alias_node = child.child_by_field_name("alias")
                if alias_node is None:
                    found_as = False
                    for subchild in child.children:
                        if subchild.type == "as":
                            found_as = True
                        elif found_as and subchild.type == "identifier":
                            alias_node = subchild
                            break

                if name_node:
                    name = safe_decode_text(name_node)
                    alias = safe_decode_text(alias_node) if alias_node else name
                    if name and alias:
                        full_name = f"{base_module}.{name}"
                        self.import_mapping[module_qn][alias] = full_name
                        logger.debug(f"From aliased import: {alias} -> {full_name}")

    def _resolve_relative_import(self, relative_node: "Node", module_qn: str) -> str:
        """Resolve relative imports like '.module' or '..parent.module'.

        Args:
            relative_node: The relative_import AST node.
            module_qn: Current module qualified name.

        Returns:
            Resolved module qualified name.
        """
        module_parts = module_qn.split(".")[1:]
        dots = 0
        module_name = ""

        text = safe_decode_text(relative_node)
        if text:
            while dots < len(text) and text[dots] == ".":
                dots += 1
            module_name = text[dots:]

        if dots > 0:
            target_parts = module_parts[: -(dots)]
        else:
            target_parts = module_parts[:]

        if module_name:
            target_parts.extend(module_name.split("."))

        return f"{self.project_name}.{'.'.join(target_parts)}" if target_parts else self.project_name

    def _resolve_python_module(self, module_name: str) -> str:
        """Resolve a Python module name to its qualified name.

        Checks if the module is local (in repo) or external.

        Args:
            module_name: Module name from import statement.

        Returns:
            Qualified module name.
        """
        if not module_name:
            return module_name

        top_level = module_name.split(".")[0]

        if (self.repo_path / top_level).is_dir() or (self.repo_path / f"{top_level}.py").is_file():
            return f"{self.project_name}.{module_name}"

        return module_name

    def _parse_js_ts_imports(self, root_node: "Node", module_qn: str) -> None:
        """Parse JavaScript/TypeScript import statements."""
        for node in self._walk_tree(root_node, {"import_statement", "lexical_declaration"}):
            if node.type == "import_statement":
                self._handle_js_import_statement(node, module_qn)
            elif node.type == "lexical_declaration":
                self._handle_js_require(node, module_qn)

    def _handle_js_import_statement(self, node: "Node", module_qn: str) -> None:
        """Handle JavaScript import statements."""
        source_module = None
        for child in node.children:
            if child.type == "string":
                source_text = safe_decode_text(child)
                if source_text:
                    source_module = self._resolve_js_module_path(
                        source_text.strip("'\""), module_qn
                    )
                break

        if not source_module:
            return

        for child in node.children:
            if child.type == "import_clause":
                self._parse_js_import_clause(child, source_module, module_qn)

    def _parse_js_import_clause(
        self,
        clause_node: "Node",
        source_module: str,
        module_qn: str,
    ) -> None:
        """Parse JavaScript import clause."""
        for child in clause_node.children:
            if child.type == "identifier":
                name = safe_decode_text(child)
                if name:
                    self.import_mapping[module_qn][name] = f"{source_module}.default"
                    logger.debug(f"JS default import: {name} -> {source_module}.default")

            elif child.type == "named_imports":
                for subchild in child.children:
                    if subchild.type == "import_specifier":
                        name_node = subchild.child_by_field_name("name")
                        alias_node = subchild.child_by_field_name("alias")
                        if name_node:
                            name = safe_decode_text(name_node)
                            local = safe_decode_text(alias_node) if alias_node else name
                            if name and local:
                                self.import_mapping[module_qn][local] = f"{source_module}.{name}"
                                logger.debug(f"JS named import: {local} -> {source_module}.{name}")

            elif child.type == "namespace_import":
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = safe_decode_text(subchild)
                        if name:
                            self.import_mapping[module_qn][name] = source_module
                            logger.debug(f"JS namespace import: {name} -> {source_module}")
                        break

    def _handle_js_require(self, node: "Node", module_qn: str) -> None:
        """Handle CommonJS require() statements."""
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                value_node = child.child_by_field_name("value")

                if name_node and value_node:
                    if name_node.type == "identifier" and value_node.type == "call_expression":
                        func_node = value_node.child_by_field_name("function")
                        args_node = value_node.child_by_field_name("arguments")

                        if func_node and safe_decode_text(func_node) == "require" and args_node:
                            for arg in args_node.children:
                                if arg.type == "string":
                                    var_name = safe_decode_text(name_node)
                                    module_path = safe_decode_text(arg)
                                    if var_name and module_path:
                                        module_path = module_path.strip("'\"")
                                        resolved = self._resolve_js_module_path(module_path, module_qn)
                                        self.import_mapping[module_qn][var_name] = resolved
                                        logger.debug(f"JS require: {var_name} -> {resolved}")
                                    break

    def _resolve_js_module_path(self, import_path: str, module_qn: str) -> str:
        """Resolve JavaScript module path to qualified name."""
        if not import_path.startswith("."):
            return import_path.replace("/", ".")

        current_parts = module_qn.split(".")[:-1]
        import_parts = import_path.split("/")

        for part in import_parts:
            if part == ".":
                continue
            elif part == "..":
                if current_parts:
                    current_parts.pop()
            elif part:
                current_parts.append(part)

        return ".".join(current_parts)

    def _parse_java_imports(self, root_node: "Node", module_qn: str) -> None:
        """Parse Java import statements."""
        for node in self._walk_tree(root_node, {"import_declaration"}):
            is_static = False
            is_wildcard = False
            imported_path = None

            for child in node.children:
                if child.type == "static":
                    is_static = True
                elif child.type == "scoped_identifier":
                    imported_path = safe_decode_text(child)
                elif child.type == "asterisk":
                    is_wildcard = True

            if imported_path:
                if is_wildcard:
                    # Wildcard import
                    wildcard_key = f"*{imported_path}"
                    self.import_mapping[module_qn][wildcard_key] = imported_path
                    logger.debug(f"Java wildcard import: * -> {imported_path}")
                else:
                    # Regular import
                    parts = imported_path.split(".")
                    local_name = parts[-1]
                    self.import_mapping[module_qn][local_name] = imported_path
                    logger.debug(f"Java import: {local_name} -> {imported_path}")

    def _walk_tree(self, node: "Node", target_types: set[str]) -> list["Node"]:
        """Walk tree and collect nodes of specified types."""
        results = []
        stack = [node]

        while stack:
            current = stack.pop()
            if current.type in target_types:
                results.append(current)
            stack.extend(reversed(current.children))

        return results

    def get_imported_names(self, module_qn: str) -> list[str]:
        """Get all imported names for a module.

        Args:
            module_qn: Module qualified name.

        Returns:
            List of imported local names.
        """
        mapping = self.import_mapping.get(module_qn, {})
        return [name for name in mapping.keys() if not name.startswith("*")]

    def get_wildcard_modules(self, module_qn: str) -> list[str]:
        """Get all wildcard imported modules.

        Args:
            module_qn: Module qualified name.

        Returns:
            List of wildcard imported module qualified names.
        """
        mapping = self.import_mapping.get(module_qn, {})
        return [qn for name, qn in mapping.items() if name.startswith("*")]

    def resolve_name(self, name: str, module_qn: str) -> str | None:
        """Resolve a local name to its qualified name via imports.

        Args:
            name: Local name to resolve.
            module_qn: Current module qualified name.

        Returns:
            Qualified name or None if not imported.
        """
        mapping = self.import_mapping.get(module_qn, {})

        if name in mapping:
            return mapping[name]

        for wildcard_key, wildcard_module in mapping.items():
            if wildcard_key.startswith("*"):
                potential_qn = f"{wildcard_module}.{name}"
                if self.function_registry.get(potential_qn):
                    return potential_qn

        return None

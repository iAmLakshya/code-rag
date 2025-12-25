"""TypeScript-specific code extractor."""

from code_rag.core.types import EntityType
from code_rag.parsing.extractors.javascript import JavaScriptExtractor
from code_rag.parsing.models import CodeEntity, ImportInfo


class TypeScriptExtractor(JavaScriptExtractor):
    """Extractor for TypeScript/TSX source code.

    Extends JavaScriptExtractor since TypeScript is a superset of JavaScript.
    """

    def extract_imports(self, root_node, source: str) -> list[ImportInfo]:
        imports = super().extract_imports(root_node, source)
        seen = {(imp.name, imp.source) for imp in imports}

        for node in self._walk_tree(root_node, {"import_statement"}):
            is_type_import = self._has_keyword(node, source, "type")

            if is_type_import:
                source_node = self._find_child_by_type(node, "string")
                source_value = ""
                if source_node:
                    source_value = self._get_node_text(source_node, source).strip("'\"")

                import_clause = self._find_child_by_type(node, "import_clause")
                if import_clause:
                    named_imports = self._find_child_by_type(import_clause, "named_imports")
                    if named_imports:
                        for child in named_imports.children:
                            if child.type == "import_specifier":
                                name_node = child.children[0] if child.children else None
                                if name_node:
                                    name = self._get_node_text(name_node, source)
                                    key = (name, source_value)
                                    if key not in seen:
                                        seen.add(key)
                                        imports.append(
                                            ImportInfo(
                                                name=name,
                                                source=source_value,
                                                is_external=not source_value.startswith("."),
                                                line_number=self._get_node_line(node),
                                            )
                                        )

        return imports

    def _extract_typescript_entities(self, node, source: str) -> list[CodeEntity]:
        entities = []
        if node.type == "interface_declaration":
            entity = self._extract_interface(node, source)
            if entity:
                entities.append(entity)
        elif node.type == "type_alias_declaration":
            entity = self._extract_type_alias(node, source)
            if entity:
                entities.append(entity)
        return entities

    def extract_entities(self, root_node, source: str) -> list[CodeEntity]:
        entities = super().extract_entities(root_node, source)

        for node in root_node.children:
            if node.type in ("interface_declaration", "type_alias_declaration"):
                entities.extend(self._extract_typescript_entities(node, source))

            elif node.type == "export_statement":
                for child in node.children:
                    if child.type in ("interface_declaration", "type_alias_declaration"):
                        entities.extend(self._extract_typescript_entities(child, source))

        return entities

    def _extract_interface(self, node, source: str) -> CodeEntity | None:
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return None
        name = self._get_node_text(name_node, source)

        base_classes = []
        extends_clause = self._find_child_by_type(node, "extends_type_clause")
        if extends_clause:
            for child in extends_clause.children:
                if child.type == "type_identifier":
                    base_classes.append(self._get_node_text(child, source))

        docstring = self._extract_jsdoc(node, source)

        return CodeEntity(
            type=EntityType.INTERFACE,
            name=name,
            qualified_name=name,
            signature=f"interface {name}",
            docstring=docstring,
            code=self._get_node_text(node, source),
            start_line=self._get_node_line(node),
            end_line=self._get_node_end_line(node),
            base_classes=base_classes,
        )

    def _extract_type_alias(self, node, source: str) -> CodeEntity | None:
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return None
        name = self._get_node_text(name_node, source)

        docstring = self._extract_jsdoc(node, source)

        return CodeEntity(
            type=EntityType.TYPE_ALIAS,
            name=name,
            qualified_name=name,
            signature=f"type {name}",
            docstring=docstring,
            code=self._get_node_text(node, source),
            start_line=self._get_node_line(node),
            end_line=self._get_node_end_line(node),
        )

"""JavaScript-specific code extractor."""

from code_rag.core.types import EntityType
from code_rag.parsing.extractors.base import BaseExtractor
from code_rag.parsing.models import CodeEntity, ImportInfo


class JavaScriptExtractor(BaseExtractor):
    """Extractor for JavaScript/JSX source code."""

    def _has_keyword(self, node, source: str, keyword: str) -> bool:
        return any(self._get_node_text(child, source) == keyword for child in node.children)

    def extract_imports(self, root_node, source: str) -> list[ImportInfo]:
        """Extract JavaScript import statements."""
        imports = []

        for node in self._walk_tree(root_node, {"import_statement"}):
            source_node = self._find_child_by_type(node, "string")
            source_value = ""
            if source_node:
                source_value = self._get_node_text(source_node, source).strip("'\"")

            is_external = not source_value.startswith(".")

            # Handle default import
            import_clause = self._find_child_by_type(node, "import_clause")
            if import_clause:
                for child in import_clause.children:
                    if child.type == "identifier":
                        imports.append(
                            ImportInfo(
                                name=self._get_node_text(child, source),
                                source=source_value,
                                is_external=is_external,
                                line_number=self._get_node_line(node),
                            )
                        )

                named_imports = self._find_child_by_type(import_clause, "named_imports")
                if named_imports:
                    for child in named_imports.children:
                        if child.type == "import_specifier":
                            name_node = child.children[0] if child.children else None
                            alias_node = child.children[-1] if len(child.children) > 1 else None

                            if name_node:
                                name = self._get_node_text(name_node, source)
                                alias = None
                                if alias_node and alias_node != name_node:
                                    alias = self._get_node_text(alias_node, source)

                                imports.append(
                                    ImportInfo(
                                        name=name,
                                        alias=alias,
                                        source=source_value,
                                        is_external=is_external,
                                        line_number=self._get_node_line(node),
                                    )
                                )

                namespace_import = self._find_child_by_type(
                    import_clause, "namespace_import"
                )
                if namespace_import:
                    id_node = self._find_child_by_type(namespace_import, "identifier")
                    if id_node:
                        imports.append(
                            ImportInfo(
                                name="*",
                                alias=self._get_node_text(id_node, source),
                                source=source_value,
                                is_external=is_external,
                                line_number=self._get_node_line(node),
                            )
                        )

        for node in self._walk_tree(root_node, {"call_expression"}):
            func_node = node.children[0] if node.children else None
            if func_node and self._get_node_text(func_node, source) == "require":
                args_node = self._find_child_by_type(node, "arguments")
                if args_node:
                    string_node = self._find_child_by_type(args_node, "string")
                    if string_node:
                        source_value = self._get_node_text(string_node, source).strip(
                            "'\"")
                        imports.append(
                            ImportInfo(
                                name=source_value,
                                source=source_value,
                                is_external=not source_value.startswith("."),
                                line_number=self._get_node_line(node),
                            )
                        )

        return imports

    def _extract_lexical_declaration_entities(self, node, source: str) -> list[CodeEntity]:
        entities = []
        for decl in self._find_child_by_type(node, "variable_declarator", find_all=True):
            value_node = None
            for child in decl.children:
                if child.type in ("arrow_function", "function"):
                    value_node = child
                    break

            if value_node:
                name_node = self._find_child_by_type(decl, "identifier")
                if name_node:
                    entity = self._extract_arrow_function(
                        node,
                        source,
                        self._get_node_text(name_node, source),
                        value_node,
                    )
                    if entity:
                        entities.append(entity)
        return entities

    def extract_entities(self, root_node, source: str) -> list[CodeEntity]:
        entities = []

        for node in root_node.children:
            if node.type == "function_declaration":
                entity = self._extract_function(node, source)
                if entity:
                    entities.append(entity)

            elif node.type == "class_declaration":
                entity = self._extract_class(node, source)
                if entity:
                    entities.append(entity)

            elif node.type == "lexical_declaration":
                entities.extend(self._extract_lexical_declaration_entities(node, source))

            elif node.type == "export_statement":
                for child in node.children:
                    if child.type == "function_declaration":
                        entity = self._extract_function(child, source)
                        if entity:
                            entities.append(entity)
                    elif child.type == "class_declaration":
                        entity = self._extract_class(child, source)
                        if entity:
                            entities.append(entity)
                    elif child.type == "lexical_declaration":
                        entities.extend(self._extract_lexical_declaration_entities(child, source))

        return entities

    def _extract_function(self, node, source: str) -> CodeEntity | None:
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None
        name = self._get_node_text(name_node, source)

        params_node = self._find_child_by_type(node, "formal_parameters")
        params = self._get_node_text(params_node, source) if params_node else "()"

        is_async = self._is_async_node(node, source)
        docstring = self._extract_jsdoc(node, source)
        calls = self._extract_calls(node, source)

        return CodeEntity(
            type=EntityType.FUNCTION,
            name=name,
            qualified_name=name,
            signature=f"function {name}{params}",
            docstring=docstring,
            code=self._get_node_text(node, source),
            start_line=self._get_node_line(node),
            end_line=self._get_node_end_line(node),
            is_async=is_async,
            calls=calls,
        )

    def _extract_arrow_function(self, node, source: str, name: str, arrow_node) -> CodeEntity:
        params_node = self._find_child_by_type(arrow_node, "formal_parameters")
        if params_node:
            params = self._get_node_text(params_node, source)
        else:
            param_node = arrow_node.children[0] if arrow_node.children else None
            if param_node and param_node.type == "identifier":
                params = f"({self._get_node_text(param_node, source)})"
            else:
                params = "()"

        is_async = self._is_async_node(arrow_node, source)
        docstring = self._extract_jsdoc(node, source)
        calls = self._extract_calls(arrow_node, source)

        return CodeEntity(
            type=EntityType.FUNCTION,
            name=name,
            qualified_name=name,
            signature=f"const {name} = {params} =>",
            docstring=docstring,
            code=self._get_node_text(node, source),
            start_line=self._get_node_line(node),
            end_line=self._get_node_end_line(node),
            is_async=is_async,
            calls=calls,
        )

    def _extract_class(self, node, source: str) -> CodeEntity | None:
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None
        name = self._get_node_text(name_node, source)

        base_classes = []
        heritage = self._find_child_by_type(node, "class_heritage")
        if heritage:
            for child in heritage.children:
                if child.type == "identifier":
                    base_classes.append(self._get_node_text(child, source))

        body_node = self._find_child_by_type(node, "class_body")
        docstring = self._extract_jsdoc(node, source)

        methods = []
        if body_node:
            for child in body_node.children:
                if child.type == "method_definition":
                    method = self._extract_method(child, source, name)
                    if method:
                        methods.append(method)

        return CodeEntity(
            type=EntityType.CLASS,
            name=name,
            qualified_name=name,
            signature=f"class {name}",
            docstring=docstring,
            code=self._get_node_text(node, source),
            start_line=self._get_node_line(node),
            end_line=self._get_node_end_line(node),
            base_classes=base_classes,
            children=methods,
        )

    def _extract_method(self, node, source: str, class_name: str) -> CodeEntity | None:
        name_node = self._find_child_by_type(node, "property_identifier")
        if not name_node:
            return None
        name = self._get_node_text(name_node, source)

        params_node = self._find_child_by_type(node, "formal_parameters")
        params = self._get_node_text(params_node, source) if params_node else "()"

        is_async = self._is_async_node(node, source)
        is_static = self._has_keyword(node, source, "static")
        docstring = self._extract_jsdoc(node, source)
        calls = self._extract_calls(node, source)

        return CodeEntity(
            type=EntityType.METHOD,
            name=name,
            qualified_name=f"{class_name}.{name}",
            signature=f"{name}{params}",
            docstring=docstring,
            code=self._get_node_text(node, source),
            start_line=self._get_node_line(node),
            end_line=self._get_node_end_line(node),
            is_async=is_async,
            is_static=is_static,
            parent_class=class_name,
            calls=calls,
        )

    def _extract_jsdoc(self, node, source: str) -> str | None:
        """Extract JSDoc comment if present before the node."""
        start_line = self._get_node_line(node)
        lines = source.split("\n")

        if start_line > 1:
            prev_line = lines[start_line - 2].strip()
            if prev_line.endswith("*/"):
                doc_lines = []
                for i in range(start_line - 2, -1, -1):
                    line = lines[i].strip()
                    doc_lines.insert(0, line)
                    if line.startswith("/**"):
                        break

                if doc_lines:
                    doc = "\n".join(doc_lines)
                    doc = doc.replace("/**", "").replace("*/", "")
                    doc = "\n".join(
                        line.lstrip("* ").rstrip()
                        for line in doc.split("\n")
                        if line.strip()
                    )
                    return doc.strip() if doc.strip() else None

        return None

    def _extract_calls(self, node, source: str) -> list[str]:
        calls = set()

        for call_node in self._walk_tree(node, {"call_expression"}):
            func_node = call_node.children[0] if call_node.children else None
            if func_node and func_node.type in ("identifier", "member_expression"):
                calls.add(self._get_node_text(func_node, source))

        return list(calls)

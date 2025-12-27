from code_rag.core.types import EntityType
from code_rag.parsing.extractors.base import BaseExtractor
from code_rag.parsing.models import CodeEntity, ImportInfo


class PythonExtractor(BaseExtractor):
    def extract_imports(self, root_node, source: str) -> list[ImportInfo]:
        imports = []

        for node in self._walk_tree(root_node, {"import_statement", "import_from_statement"}):
            if node.type == "import_statement":
                for name_node in self._find_child_by_type(node, "dotted_name", find_all=True):
                    name = self._get_node_text(name_node, source)
                    alias = None

                    parent = name_node.parent
                    if parent and parent.type == "aliased_import":
                        alias_node = self._find_child_by_type(parent, "identifier")
                        if alias_node:
                            alias = self._get_node_text(alias_node, source)

                    imports.append(
                        ImportInfo(
                            name=name,
                            alias=alias,
                            source=None,
                            is_external=not name.startswith("."),
                            line_number=self._get_node_line(node),
                        )
                    )

            elif node.type == "import_from_statement":
                module_node = self._find_child_by_type(node, "dotted_name")
                if module_node is None:
                    module_node = self._find_child_by_type(node, "relative_import")

                module_name = self._get_node_text(module_node, source) if module_node else ""

                for child in node.children:
                    if child.type == "dotted_name" and child != module_node:
                        name = self._get_node_text(child, source)
                        imports.append(
                            ImportInfo(
                                name=name,
                                source=module_name,
                                is_external=not module_name.startswith("."),
                                line_number=self._get_node_line(node),
                            )
                        )
                    elif child.type == "aliased_import":
                        name_node = self._find_child_by_type(child, "dotted_name")
                        alias_node = self._find_child_by_type(child, "identifier")
                        if name_node:
                            imports.append(
                                ImportInfo(
                                    name=self._get_node_text(name_node, source),
                                    alias=self._get_node_text(alias_node, source)
                                    if alias_node
                                    else None,
                                    source=module_name,
                                    is_external=not module_name.startswith("."),
                                    line_number=self._get_node_line(node),
                                )
                            )

        return imports

    def extract_entities(self, root_node, source: str) -> list[CodeEntity]:
        entities = []

        for node in root_node.children:
            if node.type == "function_definition":
                entity = self._extract_function(node, source)
                if entity:
                    entities.append(entity)

            elif node.type == "class_definition":
                entity = self._extract_class(node, source)
                if entity:
                    entities.append(entity)

            elif node.type == "decorated_definition":
                definition = None
                for child in node.children:
                    if child.type in ("function_definition", "class_definition"):
                        definition = child
                        break

                if definition:
                    if definition.type == "function_definition":
                        entity = self._extract_function(node, source, definition)
                    else:
                        entity = self._extract_class(node, source, definition)
                    if entity:
                        entities.append(entity)

        return entities

    def _extract_callable_base(self, node, source: str, func_node=None, entity_type: EntityType = EntityType.FUNCTION, parent_class: str | None = None):
        if func_node is None:
            func_node = node

        name_node = self._find_child_by_type(func_node, "identifier")
        if not name_node:
            return None
        name = self._get_node_text(name_node, source)

        params_node = self._find_child_by_type(func_node, "parameters")
        params = self._get_node_text(params_node, source) if params_node else "()"

        is_async = self._is_async_node(func_node, source)
        decorators = self._extract_decorators(node, source)

        body_node = self._find_child_by_type(func_node, "block")
        docstring = self._extract_docstring(body_node, source) if body_node else None
        calls = self._extract_calls(func_node, source)

        qualified_name = f"{parent_class}.{name}" if parent_class else name

        return CodeEntity(
            type=entity_type,
            name=name,
            qualified_name=qualified_name,
            signature=f"def {name}{params}",
            docstring=docstring,
            code=self._get_node_text(node, source),
            start_line=self._get_node_line(node),
            end_line=self._get_node_end_line(node),
            is_async=is_async,
            is_static="@staticmethod" in decorators if entity_type == EntityType.METHOD else False,
            is_classmethod="@classmethod" in decorators if entity_type == EntityType.METHOD else False,
            decorators=decorators,
            parent_class=parent_class,
            calls=calls,
        )

    def _extract_function(self, node, source: str, func_node=None) -> CodeEntity | None:
        return self._extract_callable_base(node, source, func_node)

    def _extract_class(self, node, source: str, class_node=None) -> CodeEntity | None:
        if class_node is None:
            class_node = node

        name_node = self._find_child_by_type(class_node, "identifier")
        if not name_node:
            return None
        name = self._get_node_text(name_node, source)

        base_classes = []
        args_node = self._find_child_by_type(class_node, "argument_list")
        if args_node:
            for child in args_node.children:
                if child.type in ("identifier", "attribute"):
                    base_classes.append(self._get_node_text(child, source))

        decorators = self._extract_decorators(node, source)

        body_node = self._find_child_by_type(class_node, "block")
        docstring = self._extract_docstring(body_node, source) if body_node else None

        methods = []
        if body_node:
            for child in body_node.children:
                method = None
                if child.type == "function_definition":
                    method = self._extract_method(child, source, name)
                elif child.type == "decorated_definition":
                    for subchild in child.children:
                        if subchild.type == "function_definition":
                            method = self._extract_method(child, source, name, subchild)
                            break
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
            decorators=decorators,
            base_classes=base_classes,
            children=methods,
        )

    def _extract_method(self, node, source: str, class_name: str, method_node=None) -> CodeEntity:
        return self._extract_callable_base(node, source, method_node, EntityType.METHOD, class_name)

    def _extract_docstring(self, body_node, source: str) -> str | None:
        if body_node is None:
            return None

        for child in body_node.children:
            if child.type == "string":
                return self._clean_string_literal(self._get_node_text(child, source))
            elif child.type == "expression_statement":
                string_node = self._find_child_by_type(child, "string")
                if string_node:
                    return self._clean_string_literal(self._get_node_text(string_node, source))
                break
            elif child.type not in ("comment", "pass_statement"):
                break

        return None

    def _extract_calls(self, node, source: str) -> list[str]:
        calls = set()

        for call_node in self._walk_tree(node, {"call"}):
            func_node = call_node.children[0] if call_node.children else None
            if func_node and func_node.type in ("identifier", "attribute"):
                calls.add(self._get_node_text(func_node, source))

        return list(calls)

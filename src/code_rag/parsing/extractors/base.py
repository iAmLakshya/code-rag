from abc import ABC, abstractmethod

from code_rag.parsing.models import CodeEntity, ImportInfo


class BaseExtractor(ABC):
    @abstractmethod
    def extract_imports(self, root_node, source: str) -> list[ImportInfo]: ...

    @abstractmethod
    def extract_entities(self, root_node, source: str) -> list[CodeEntity]: ...

    def _get_node_text(self, node, source: str) -> str:
        return source[node.start_byte:node.end_byte]

    def _get_node_line(self, node) -> int:
        return node.start_point[0] + 1

    def _get_node_end_line(self, node) -> int:
        return node.end_point[0] + 1

    def _find_child_by_type(self, node, node_type: str, find_all: bool = False):
        if find_all:
            return [child for child in node.children if child.type == node_type]
        for child in node.children:
            if child.type == node_type:
                return child
        return None if not find_all else []

    def _is_async_node(self, node, source: str) -> bool:
        return any(
            child.type == "async" or self._get_node_text(child, source) == "async"
            for child in node.children
        )

    def _extract_decorators(self, node, source: str) -> list[str]:
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                decorator_text = self._get_node_text(child, source).strip()
                decorators.append(decorator_text)
        return decorators

    def _clean_string_literal(self, text: str) -> str:
        text = text.strip()
        if text.startswith('"""') or text.startswith("'''"):
            return text[3:-3].strip()
        elif text.startswith('"') or text.startswith("'"):
            return text[1:-1].strip()
        return text

    def _extract_docstring(self, node, source: str) -> str | None:
        return None

    def _walk_tree(self, node, node_types: set[str]):
        if node.type in node_types:
            yield node
        for child in node.children:
            yield from self._walk_tree(child, node_types)

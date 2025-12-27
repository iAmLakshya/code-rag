from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from code_rag.core.cache import FunctionRegistry
    from code_rag.parsing.import_processor import ImportProcessor

logger = logging.getLogger(__name__)


class InheritanceTracker:
    """Tracks class inheritance for MRO calculation and method resolution."""

    def __init__(
        self,
        function_registry: FunctionRegistry,
        import_processor: ImportProcessor,
    ):
        self.function_registry = function_registry
        self.import_processor = import_processor
        self.class_inheritance: dict[str, list[str]] = {}

    def register_class(
        self,
        class_qn: str,
        base_classes: list[str],
        module_qn: str,
    ) -> None:
        resolved_parents = []

        for base_name in base_classes:
            if not base_name or base_name == "object":
                continue

            resolved = self._resolve_class_name(base_name, module_qn)
            if resolved:
                resolved_parents.append(resolved)
                logger.debug(f"Resolved parent: {base_name} -> {resolved}")
            else:
                resolved_parents.append(base_name)
                logger.debug(f"Unresolved parent: {base_name}")

        self.class_inheritance[class_qn] = resolved_parents
        logger.debug(f"Registered {class_qn} with parents: {resolved_parents}")

    def unregister_class(self, class_qn: str) -> None:
        if class_qn in self.class_inheritance:
            del self.class_inheritance[class_qn]

    def get_parents(self, class_qn: str) -> list[str]:
        return self.class_inheritance.get(class_qn, [])

    def get_mro(self, class_qn: str) -> list[str]:
        """Get Method Resolution Order using simplified C3 linearization."""
        mro = [class_qn]
        visited = {class_qn}

        queue = self.get_parents(class_qn)[:]

        while queue:
            parent = queue.pop(0)
            if parent in visited:
                continue

            visited.add(parent)
            mro.append(parent)

            grandparents = self.get_parents(parent)
            for gp in grandparents:
                if gp not in visited:
                    queue.append(gp)

        return mro

    def get_all_methods(self, class_qn: str) -> list[str]:
        methods = []
        mro = self.get_mro(class_qn)

        for ancestor_qn in mro:
            prefix = f"{ancestor_qn}."
            for entry_qn, entry_type in self.function_registry.all_entries().items():
                if entry_type == "Method" and entry_qn.startswith(prefix):
                    method_name = entry_qn[len(prefix) :]
                    if "." not in method_name:
                        methods.append(entry_qn)

        return methods

    def find_method(self, class_qn: str, method_name: str) -> str | None:
        """Find method by searching MRO for first defining class."""
        mro = self.get_mro(class_qn)

        for ancestor_qn in mro:
            method_qn = f"{ancestor_qn}.{method_name}"
            if self.function_registry.get(method_qn):
                return method_qn

        return None

    def is_subclass(self, child_qn: str, parent_qn: str) -> bool:
        mro = self.get_mro(child_qn)
        return parent_qn in mro

    def get_subclasses(self, class_qn: str) -> list[str]:
        subclasses = []

        for child_qn, parents in self.class_inheritance.items():
            if class_qn in parents:
                subclasses.append(child_qn)
                subclasses.extend(self.get_subclasses(child_qn))

        return subclasses

    def clear(self) -> None:
        self.class_inheritance.clear()

    def clear_by_prefix(self, prefix: str) -> int:
        to_remove = [qn for qn in self.class_inheritance if qn.startswith(prefix)]
        for qn in to_remove:
            del self.class_inheritance[qn]
        return len(to_remove)

    def _resolve_class_name(self, class_name: str, module_qn: str) -> str | None:
        if "." in class_name:
            if self.function_registry.get(class_name) == "Class":
                return class_name
            return class_name

        import_result = self.import_processor.resolve_name(class_name, module_qn)
        if import_result:
            return import_result

        local_qn = f"{module_qn}.{class_name}"
        if self.function_registry.get(local_qn) == "Class":
            return local_qn

        matches = self.function_registry.find_by_simple_name(class_name)
        for match in matches:
            if self.function_registry.get(match) == "Class":
                return match

        return None

    def get_stats(self) -> dict:
        total_classes = len(self.class_inheritance)
        total_parents = sum(len(parents) for parents in self.class_inheritance.values())
        classes_with_parents = sum(1 for parents in self.class_inheritance.values() if parents)

        return {
            "total_classes": total_classes,
            "classes_with_parents": classes_with_parents,
            "total_parent_relationships": total_parents,
            "avg_parents_per_class": total_parents / total_classes if total_classes > 0 else 0,
        }

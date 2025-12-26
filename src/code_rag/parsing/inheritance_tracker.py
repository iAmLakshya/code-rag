"""Class inheritance tracking for method resolution.

This module provides tracking of class inheritance relationships,
enabling proper resolution of inherited method calls and super() calls.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from code_rag.core.cache import FunctionRegistry
    from code_rag.parsing.import_processor import ImportProcessor

logger = logging.getLogger(__name__)


class InheritanceTracker:
    """Tracks class inheritance relationships for method resolution.

    This class maintains a mapping of classes to their parent classes,
    supporting features like:
    - Method Resolution Order (MRO) calculation
    - Inherited method lookup
    - Super call resolution

    Attributes:
        class_inheritance: Dict mapping class_qn -> list of parent class_qns.
        function_registry: Registry of known functions/classes.
        import_processor: Processor for import mappings.
    """

    def __init__(
        self,
        function_registry: "FunctionRegistry",
        import_processor: "ImportProcessor",
    ):
        """Initialize the inheritance tracker.

        Args:
            function_registry: Registry of known functions/classes.
            import_processor: Processor for import mappings.
        """
        self.function_registry = function_registry
        self.import_processor = import_processor
        self.class_inheritance: dict[str, list[str]] = {}

    def register_class(
        self,
        class_qn: str,
        base_classes: list[str],
        module_qn: str,
    ) -> None:
        """Register a class and its parent classes.

        Args:
            class_qn: Qualified name of the class.
            base_classes: List of base class names (may be simple names).
            module_qn: Module containing the class.
        """
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
        """Remove a class from the tracker.

        Args:
            class_qn: Qualified name of the class.
        """
        if class_qn in self.class_inheritance:
            del self.class_inheritance[class_qn]

    def get_parents(self, class_qn: str) -> list[str]:
        """Get direct parent classes of a class.

        Args:
            class_qn: Qualified name of the class.

        Returns:
            List of parent class qualified names.
        """
        return self.class_inheritance.get(class_qn, [])

    def get_mro(self, class_qn: str) -> list[str]:
        """Get Method Resolution Order for a class.

        Uses C3 linearization algorithm (simplified version).

        Args:
            class_qn: Qualified name of the class.

        Returns:
            List of classes in MRO order.
        """
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
        """Get all methods available on a class (including inherited).

        Args:
            class_qn: Qualified name of the class.

        Returns:
            List of method qualified names.
        """
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
        """Find a method on a class or its ancestors.

        Searches the MRO to find the first class that defines the method.

        Args:
            class_qn: Qualified name of the class.
            method_name: Name of the method to find.

        Returns:
            Qualified name of the method or None.
        """
        mro = self.get_mro(class_qn)

        for ancestor_qn in mro:
            method_qn = f"{ancestor_qn}.{method_name}"
            if self.function_registry.get(method_qn):
                return method_qn

        return None

    def is_subclass(self, child_qn: str, parent_qn: str) -> bool:
        """Check if one class is a subclass of another.

        Args:
            child_qn: Potential child class.
            parent_qn: Potential parent class.

        Returns:
            True if child is a subclass of parent.
        """
        mro = self.get_mro(child_qn)
        return parent_qn in mro

    def get_subclasses(self, class_qn: str) -> list[str]:
        """Get all known subclasses of a class.

        Args:
            class_qn: Qualified name of the class.

        Returns:
            List of subclass qualified names.
        """
        subclasses = []

        for child_qn, parents in self.class_inheritance.items():
            if class_qn in parents:
                subclasses.append(child_qn)
                subclasses.extend(self.get_subclasses(child_qn))

        return subclasses

    def clear(self) -> None:
        """Clear all inheritance tracking data."""
        self.class_inheritance.clear()

    def clear_by_prefix(self, prefix: str) -> int:
        """Clear all classes with a given prefix.

        Args:
            prefix: Prefix to match (e.g., module name).

        Returns:
            Number of classes removed.
        """
        to_remove = [qn for qn in self.class_inheritance if qn.startswith(prefix)]
        for qn in to_remove:
            del self.class_inheritance[qn]
        return len(to_remove)

    def _resolve_class_name(self, class_name: str, module_qn: str) -> str | None:
        """Resolve a class name to its qualified name.

        Args:
            class_name: Simple or qualified class name.
            module_qn: Current module qualified name.

        Returns:
            Qualified class name or None.
        """
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
        """Get statistics about tracked inheritance.

        Returns:
            Dictionary with statistics.
        """
        total_classes = len(self.class_inheritance)
        total_parents = sum(len(parents) for parents in self.class_inheritance.values())
        classes_with_parents = sum(1 for parents in self.class_inheritance.values() if parents)

        return {
            "total_classes": total_classes,
            "classes_with_parents": classes_with_parents,
            "total_parent_relationships": total_parents,
            "avg_parents_per_class": total_parents / total_classes if total_classes > 0 else 0,
        }

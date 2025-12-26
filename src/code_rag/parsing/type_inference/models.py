"""Data models for type inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node


class TypeSource(Enum):
    """Source of type inference."""
    ANNOTATION = "annotation"       # From type annotation
    CONSTRUCTOR = "constructor"     # From constructor call (e.g., User())
    ASSIGNMENT = "assignment"       # From assignment RHS
    PARAMETER = "parameter"         # From function parameter
    RETURN_TYPE = "return_type"     # From function return type
    METHOD_RETURN = "method_return" # Inferred from method's return statements
    LOOP_VARIABLE = "loop_variable" # From for loop iteration
    ATTRIBUTE = "attribute"         # From instance attribute (self.x)
    IMPORT = "import"               # From imported name
    INFERRED = "inferred"           # General inference


@dataclass
class InferredType:
    """Represents an inferred type for a variable or expression.

    Attributes:
        type_name: Simple type name (e.g., "User").
        qualified_name: Fully qualified name (e.g., "myproject.models.User").
        source: How the type was inferred.
        confidence: Confidence level (0.0 to 1.0).
        generic_args: Generic type arguments (e.g., ["str"] for List[str]).
    """
    type_name: str
    qualified_name: str | None = None
    source: TypeSource = TypeSource.INFERRED
    confidence: float = 1.0
    generic_args: list[str] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        """Check if type is fully resolved to a qualified name."""
        return self.qualified_name is not None

    def __str__(self) -> str:
        if self.qualified_name:
            return self.qualified_name
        return self.type_name


@dataclass
class VariableTypeMap:
    """Maps variable names to their inferred types within a scope.

    This is used to track local variable types within a function or method.
    """
    _types: dict[str, InferredType] = field(default_factory=dict)
    _instance_attrs: dict[str, InferredType] = field(default_factory=dict)

    def set_type(self, name: str, inferred_type: InferredType) -> None:
        """Set the type for a variable.

        Args:
            name: Variable name.
            inferred_type: Inferred type information.
        """
        self._types[name] = inferred_type

    def get_type(self, name: str) -> InferredType | None:
        """Get the type for a variable.

        Args:
            name: Variable name.

        Returns:
            Inferred type or None if not found.
        """
        return self._types.get(name)

    def set_instance_attr(self, name: str, inferred_type: InferredType) -> None:
        """Set the type for an instance attribute (self.x).

        Args:
            name: Attribute name (without 'self.').
            inferred_type: Inferred type information.
        """
        self._instance_attrs[name] = inferred_type

    def get_instance_attr(self, name: str) -> InferredType | None:
        """Get the type for an instance attribute.

        Args:
            name: Attribute name (without 'self.').

        Returns:
            Inferred type or None if not found.
        """
        return self._instance_attrs.get(name)

    def __contains__(self, name: str) -> bool:
        """Check if a variable has a type."""
        return name in self._types

    def __getitem__(self, name: str) -> InferredType:
        """Get type for a variable."""
        return self._types[name]

    def items(self):
        """Return all (name, type) pairs."""
        return self._types.items()

    def all_types(self) -> dict[str, str]:
        """Get all types as a simple dict (for backward compatibility).

        Returns:
            Dict mapping variable names to type names.
        """
        return {name: t.type_name for name, t in self._types.items()}


@dataclass
class TypeInferenceContext:
    """Context for type inference within a scope.

    Attributes:
        module_qn: Module qualified name (e.g., "myproject.models").
        file_path: Path to the source file.
        language: Programming language.
        class_name: Containing class name (if in a method).
        function_name: Containing function/method name.
        local_types: Map of local variable types.
    """
    module_qn: str
    file_path: Path | None = None
    language: str = "python"
    class_name: str | None = None
    function_name: str | None = None
    local_types: VariableTypeMap = field(default_factory=VariableTypeMap)

    @property
    def class_qn(self) -> str | None:
        """Get the containing class qualified name."""
        if self.class_name:
            return f"{self.module_qn}.{self.class_name}"
        return None

    @property
    def function_qn(self) -> str | None:
        """Get the containing function qualified name."""
        if self.function_name:
            if self.class_name:
                return f"{self.module_qn}.{self.class_name}.{self.function_name}"
            return f"{self.module_qn}.{self.function_name}"
        return None


@dataclass
class MethodCallInfo:
    """Information about a method call to resolve.

    Attributes:
        receiver: The object receiving the call (e.g., "user" or "self.repo").
        method_name: The method being called (e.g., "get_profile").
        arguments: Arguments passed to the method.
        is_chained: Whether this is part of a call chain.
        full_text: Full text of the call expression.
    """
    receiver: str
    method_name: str
    arguments: list[str] = field(default_factory=list)
    is_chained: bool = False
    full_text: str = ""

    @classmethod
    def from_text(cls, text: str) -> "MethodCallInfo | None":
        """Parse a method call from text.

        Args:
            text: Method call text (e.g., "user.get_profile()").

        Returns:
            MethodCallInfo or None if not a valid method call.
        """
        # Remove arguments for parsing
        if "(" in text:
            text_no_args = text[:text.index("(")]
        else:
            text_no_args = text

        parts = text_no_args.split(".")
        if len(parts) < 2:
            return None

        receiver = ".".join(parts[:-1])
        method_name = parts[-1]

        return cls(
            receiver=receiver,
            method_name=method_name,
            full_text=text,
            is_chained="()." in text,
        )

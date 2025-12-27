from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TypeSource(Enum):
    ANNOTATION = "annotation"
    CONSTRUCTOR = "constructor"
    ASSIGNMENT = "assignment"
    PARAMETER = "parameter"
    RETURN_TYPE = "return_type"
    METHOD_RETURN = "method_return"
    LOOP_VARIABLE = "loop_variable"
    ATTRIBUTE = "attribute"
    IMPORT = "import"
    INFERRED = "inferred"


@dataclass
class InferredType:
    type_name: str
    qualified_name: str | None = None
    source: TypeSource = TypeSource.INFERRED
    confidence: float = 1.0
    generic_args: list[str] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return self.qualified_name is not None

    def __str__(self) -> str:
        return self.qualified_name or self.type_name


@dataclass
class VariableTypeMap:
    """Maps variable names to their inferred types within a scope."""

    _types: dict[str, InferredType] = field(default_factory=dict)
    _instance_attrs: dict[str, InferredType] = field(default_factory=dict)

    def set_type(self, name: str, inferred_type: InferredType) -> None:
        self._types[name] = inferred_type

    def get_type(self, name: str) -> InferredType | None:
        return self._types.get(name)

    def set_instance_attr(self, name: str, inferred_type: InferredType) -> None:
        self._instance_attrs[name] = inferred_type

    def get_instance_attr(self, name: str) -> InferredType | None:
        return self._instance_attrs.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._types

    def __getitem__(self, name: str) -> InferredType:
        return self._types[name]

    def items(self):
        return self._types.items()

    def all_types(self) -> dict[str, str]:
        return {name: t.type_name for name, t in self._types.items()}


@dataclass
class TypeInferenceContext:
    module_qn: str
    file_path: Path | None = None
    language: str = "python"
    class_name: str | None = None
    function_name: str | None = None
    local_types: VariableTypeMap = field(default_factory=VariableTypeMap)

    @property
    def class_qn(self) -> str | None:
        if self.class_name:
            return f"{self.module_qn}.{self.class_name}"
        return None

    @property
    def function_qn(self) -> str | None:
        if self.function_name:
            if self.class_name:
                return f"{self.module_qn}.{self.class_name}.{self.function_name}"
            return f"{self.module_qn}.{self.function_name}"
        return None


@dataclass
class MethodCallInfo:
    receiver: str
    method_name: str
    arguments: list[str] = field(default_factory=list)
    is_chained: bool = False
    full_text: str = ""

    @classmethod
    def from_text(cls, text: str) -> MethodCallInfo | None:
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

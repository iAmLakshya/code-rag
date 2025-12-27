"""Type inference engine for resolving method calls and variable types.

This module provides type inference capabilities to accurately resolve
method calls like `user.get_profile().name` to their fully qualified names.

Key components:
- TypeInferenceEngine: Main engine coordinating type inference
- PythonTypeInference: Python-specific type resolution
- VariableTypeMap: Tracks inferred types for local variables

Usage:
    from code_rag.parsing.type_inference import TypeInferenceEngine

    engine = TypeInferenceEngine(function_registry, import_processor)
    local_types = engine.infer_local_types(function_node, module_qn, "python")

    # Resolve a method call
    resolved_qn = engine.resolve_method_call(
        "user.get_profile()",
        module_qn,
        local_types
    )
"""

from code_rag.parsing.type_inference.engine import TypeInferenceEngine
from code_rag.parsing.type_inference.models import (
    InferredType,
    TypeInferenceContext,
    VariableTypeMap,
)
from code_rag.parsing.type_inference.python_inference import PythonTypeInference

__all__ = [
    "TypeInferenceEngine",
    "PythonTypeInference",
    "InferredType",
    "VariableTypeMap",
    "TypeInferenceContext",
]

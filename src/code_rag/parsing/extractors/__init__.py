"""Language-specific code extractors."""

from code_rag.parsing.extractors.base import BaseExtractor
from code_rag.parsing.extractors.javascript import JavaScriptExtractor
from code_rag.parsing.extractors.python import PythonExtractor
from code_rag.parsing.extractors.typescript import TypeScriptExtractor

__all__ = [
    "BaseExtractor",
    "PythonExtractor",
    "JavaScriptExtractor",
    "TypeScriptExtractor",
]

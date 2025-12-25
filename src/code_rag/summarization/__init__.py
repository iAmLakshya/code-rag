"""Summarization module for AI-generated code summaries."""

from code_rag.summarization.prompts import (
    CLASS_CODE_MAX_CHARS,
    FILE_CODE_MAX_CHARS,
    FUNCTION_CODE_MAX_CHARS,
    SummaryPrompts,
)
from code_rag.summarization.summarizer import (
    SYSTEM_MESSAGE,
    CodeSummarizer,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_MAX_WAIT,
    DEFAULT_RETRY_MIN_WAIT,
    DEFAULT_RETRY_MULTIPLIER,
    DEFAULT_TEMPERATURE,
)

__all__ = [
    "CLASS_CODE_MAX_CHARS",
    "CodeSummarizer",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_RETRY_ATTEMPTS",
    "DEFAULT_RETRY_MAX_WAIT",
    "DEFAULT_RETRY_MIN_WAIT",
    "DEFAULT_RETRY_MULTIPLIER",
    "DEFAULT_TEMPERATURE",
    "FILE_CODE_MAX_CHARS",
    "FUNCTION_CODE_MAX_CHARS",
    "SummaryPrompts",
    "SYSTEM_MESSAGE",
]

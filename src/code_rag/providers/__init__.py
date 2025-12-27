"""LLM and Embedding provider implementations.

This module provides a unified interface for multiple AI providers:
- OpenAI (default)
- Ollama (local models)
- Anthropic (Claude)
- Google (Gemini)
- UniXcoder (code-specific embeddings, requires torch/transformers)

Usage:
    from code_rag.providers import get_llm_provider, get_embedding_provider

    llm = get_llm_provider()  # Uses settings to determine provider
    response = await llm.complete(messages=[...])

    embedder = get_embedding_provider()
    vector = await embedder.embed("text")

    # Use UniXcoder for code-specific embeddings (768-dim vs 1536-dim for OpenAI)
    code_embedder = get_embedding_provider(provider="unixcoder")
    code_vector = await code_embedder.embed("def hello(): pass")
"""

from code_rag.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderConfig,
)
from code_rag.providers.factory import (
    get_embedding_provider,
    get_llm_provider,
)

__all__ = [
    "get_llm_provider",
    "get_embedding_provider",
    "BaseLLMProvider",
    "BaseEmbeddingProvider",
    "ProviderConfig",
]

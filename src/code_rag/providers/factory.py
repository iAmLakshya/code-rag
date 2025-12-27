"""Provider factory for creating LLM and embedding providers."""

import logging

from code_rag.config import get_settings
from code_rag.core.errors import ConfigurationError
from code_rag.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderConfig,
)

logger = logging.getLogger(__name__)


def get_llm_provider(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseLLMProvider:
    """Get an LLM provider instance based on configuration.

    The provider is determined by (in order of precedence):
    1. Explicit provider parameter
    2. LLM_PROVIDER environment variable
    3. Default to "openai"

    Args:
        provider: Provider name (openai, ollama, anthropic, google).
        model: Model identifier.
        api_key: API key (optional for Ollama).
        base_url: Custom base URL.
        temperature: Default temperature.
        max_tokens: Default max tokens.

    Returns:
        Configured LLM provider instance.

    Raises:
        ConfigurationError: If provider is unknown or misconfigured.
    """
    settings = get_settings()

    provider_name = (provider or settings.ai.llm_provider).lower()

    config = ProviderConfig(
        provider=provider_name,
        model=model or _get_default_model(provider_name, "llm", settings),
        api_key=api_key or _get_api_key(provider_name, settings),
        base_url=base_url or _get_base_url(provider_name, settings),
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens or 1000,
    )

    return _create_llm_provider(config)


def get_embedding_provider(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseEmbeddingProvider:
    """Get an embedding provider instance based on configuration.

    The provider is determined by (in order of precedence):
    1. Explicit provider parameter
    2. EMBEDDING_PROVIDER environment variable
    3. Default to "openai"

    Args:
        provider: Provider name (openai, ollama, google).
        model: Model identifier.
        api_key: API key (optional for Ollama).
        base_url: Custom base URL.

    Returns:
        Configured embedding provider instance.

    Raises:
        ConfigurationError: If provider is unknown or misconfigured.
    """
    settings = get_settings()

    provider_name = (provider or settings.ai.embedding_provider).lower()

    config = ProviderConfig(
        provider=provider_name,
        model=model or _get_default_model(provider_name, "embedding", settings),
        api_key=api_key or _get_api_key(provider_name, settings),
        base_url=base_url or _get_base_url(provider_name, settings),
    )

    return _create_embedding_provider(config)


def _get_default_model(provider_name: str, model_type: str, settings) -> str:
    defaults = {
        "openai": {
            "llm": settings.llm_model,
            "embedding": settings.embedding_model,
        },
        "ollama": {
            "llm": "llama3.2",
            "embedding": "nomic-embed-text",
        },
        "anthropic": {
            "llm": "claude-sonnet-4-20250514",
            "embedding": None,  # Anthropic doesn't have embeddings
        },
        "google": {
            "llm": "gemini-1.5-flash",
            "embedding": "text-embedding-004",
        },
        "unixcoder": {
            "llm": None,  # UniXcoder is embedding-only
            "embedding": "microsoft/unixcoder-base",
        },
    }

    provider_defaults = defaults.get(provider_name, {})
    return provider_defaults.get(model_type) or settings.llm_model


def _get_api_key(provider_name: str, settings) -> str | None:
    """Get API key for a provider.

    Args:
        provider_name: Provider name.
        settings: Application settings.

    Returns:
        API key or None for local providers.
    """
    if provider_name == "openai":
        return settings.openai_api_key
    elif provider_name == "anthropic":
        return settings.ai.anthropic_api_key
    elif provider_name == "google":
        return settings.ai.google_api_key
    elif provider_name == "ollama":
        return None  # Ollama doesn't require API key
    else:
        return settings.openai_api_key


def _get_base_url(provider_name: str, settings) -> str | None:
    """Get base URL for a provider.

    Args:
        provider_name: Provider name.
        settings: Application settings.

    Returns:
        Base URL or None for default.
    """
    if provider_name == "ollama":
        return settings.ai.ollama_base_url
    return None


def _create_llm_provider(config: ProviderConfig) -> BaseLLMProvider:
    """Create an LLM provider instance.

    Args:
        config: Provider configuration.

    Returns:
        LLM provider instance.

    Raises:
        ConfigurationError: If provider is unknown.
    """
    provider_name = config.provider.lower()

    if provider_name == "openai":
        from code_rag.providers.openai_provider import OpenAILLMProvider
        return OpenAILLMProvider(config)

    elif provider_name == "ollama":
        from code_rag.providers.ollama_provider import OllamaLLMProvider
        return OllamaLLMProvider(config)

    elif provider_name == "anthropic":
        from code_rag.providers.anthropic_provider import AnthropicLLMProvider
        return AnthropicLLMProvider(config)

    elif provider_name == "google":
        from code_rag.providers.google_provider import GoogleLLMProvider
        return GoogleLLMProvider(config)

    else:
        raise ConfigurationError(
            f"Unknown LLM provider: {provider_name}. "
            f"Supported providers: openai, ollama, anthropic, google"
        )


def _create_embedding_provider(config: ProviderConfig) -> BaseEmbeddingProvider:
    """Create an embedding provider instance.

    Args:
        config: Provider configuration.

    Returns:
        Embedding provider instance.

    Raises:
        ConfigurationError: If provider is unknown or doesn't support embeddings.
    """
    provider_name = config.provider.lower()

    if provider_name == "openai":
        from code_rag.providers.openai_provider import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(config)

    elif provider_name == "ollama":
        from code_rag.providers.ollama_provider import OllamaEmbeddingProvider
        return OllamaEmbeddingProvider(config)

    elif provider_name == "google":
        from code_rag.providers.google_provider import GoogleEmbeddingProvider
        return GoogleEmbeddingProvider(config)

    elif provider_name == "unixcoder":
        from code_rag.providers.unixcoder_provider import UniXcoderEmbeddingProvider
        return UniXcoderEmbeddingProvider(config)

    elif provider_name == "anthropic":
        raise ConfigurationError(
            "Anthropic does not provide embedding models. "
            "Use 'openai', 'ollama', 'google', or 'unixcoder' for embeddings."
        )

    else:
        raise ConfigurationError(
            f"Unknown embedding provider: {provider_name}. "
            f"Supported providers: openai, ollama, google, unixcoder"
        )

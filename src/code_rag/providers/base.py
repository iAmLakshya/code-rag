"""Base classes and protocols for LLM and embedding providers."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

RETRY_MAX_ATTEMPTS = 5
RETRY_MULTIPLIER = 1
RETRY_MIN_WAIT = 1
RETRY_MAX_WAIT = 60


@dataclass
class ProviderConfig:
    """Configuration for an AI provider.

    Attributes:
        provider: Provider name (openai, ollama, anthropic, google).
        model: Model identifier.
        api_key: API key (optional for local providers like Ollama).
        base_url: Custom base URL (e.g., Ollama endpoint).
        temperature: Default temperature for completions.
        max_tokens: Default max tokens for completions.
        extra: Additional provider-specific configuration.
    """
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1000
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env_prefix(cls, prefix: str) -> "ProviderConfig":
        """Create config from environment variables with a prefix.

        Args:
            prefix: Environment variable prefix (e.g., "LLM" for LLM_PROVIDER).

        Returns:
            ProviderConfig instance.
        """
        import os

        provider = os.getenv(f"{prefix}_PROVIDER", "openai").lower()
        model = os.getenv(f"{prefix}_MODEL", "gpt-4o")
        api_key = os.getenv(f"{prefix}_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv(f"{prefix}_BASE_URL")

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: ProviderConfig):
        """Initialize the provider.

        Args:
            config: Provider configuration.
        """
        self.config = config
        self._semaphore = asyncio.Semaphore(5)  # Default concurrency limit

    @abstractmethod
    async def _complete_impl(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Implementation-specific completion logic.

        Args:
            messages: Chat messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """
        ...

    @retry(
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=RETRY_MULTIPLIER,
            min=RETRY_MIN_WAIT,
            max=RETRY_MAX_WAIT,
        ),
    )
    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a completion from messages.

        Args:
            messages: Chat messages with 'role' and 'content' keys.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated completion text.
        """
        async with self._semaphore:
            return await self._complete_impl(
                messages=messages,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature if temperature is not None else self.config.temperature,
            )

    def set_concurrency(self, max_concurrent: int) -> None:
        """Set the maximum concurrent requests.

        Args:
            max_concurrent: Maximum concurrent API calls.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: ProviderConfig):
        """Initialize the provider.

        Args:
            config: Provider configuration.
        """
        self.config = config
        self._semaphore = asyncio.Semaphore(5)  # Default concurrency limit

    @abstractmethod
    async def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        """Implementation-specific embedding logic.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    @retry(
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=RETRY_MULTIPLIER,
            min=RETRY_MIN_WAIT,
            max=RETRY_MAX_WAIT,
        ),
    )
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        embeddings = await self._embed_batch_internal([text])
        return embeddings[0]

    async def _embed_batch_internal(self, texts: list[str]) -> list[list[float]]:
        """Internal batch embedding with semaphore.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors.
        """
        async with self._semaphore:
            return await self._embed_impl(texts)

    async def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Embed multiple texts in batches.

        Args:
            texts: Texts to embed.
            batch_size: Number of texts per batch.

        Returns:
            List of embedding vectors.
        """
        all_embeddings = []
        texts_list = list(texts)

        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i:i + batch_size]
            embeddings = await self._embed_batch_internal(batch)
            all_embeddings.extend(embeddings)

        logger.debug(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings

    def set_concurrency(self, max_concurrent: int) -> None:
        """Set the maximum concurrent requests.

        Args:
            max_concurrent: Maximum concurrent API calls.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)

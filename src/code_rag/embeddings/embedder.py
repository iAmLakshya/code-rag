"""Embedding generation with batching and rate limiting.

Supports multiple providers:
- OpenAI (default)
- Ollama (local)
- Google

Usage:
    # Use default provider from settings
    embedder = Embedder()
    vector = await embedder.embed("text")

    # Use specific provider
    embedder = Embedder(provider="ollama", model="nomic-embed-text")
"""

import logging
from typing import Callable, Sequence

from code_rag.config import get_settings
from code_rag.providers import get_embedding_provider
from code_rag.providers.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings using configurable providers with rate limiting.

    This class wraps embedding providers and adds:
    - Progress reporting for batch operations
    - Consistent interface across providers
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_concurrent: int | None = None,
    ):
        """Initialize the embedder.

        Args:
            provider: Provider name (openai, ollama, google). Defaults to settings.
            model: Embedding model name. Defaults to settings/provider default.
            api_key: API key. Defaults to settings.
            base_url: Custom base URL (for Ollama). Defaults to settings.
            max_concurrent: Max concurrent API calls. Defaults to settings.
        """
        settings = get_settings()
        self.max_concurrent = max_concurrent or settings.max_concurrent_requests

        # Get the embedding provider
        self._provider: BaseEmbeddingProvider = get_embedding_provider(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self._provider.set_concurrency(self.max_concurrent)

        logger.info(
            f"Initialized Embedder with {self._provider.config.provider}/"
            f"{self._provider.config.model}"
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        return await self._provider.embed(text)

    async def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts in batches.

        Args:
            texts: Texts to embed.
            batch_size: Number of texts per batch.

        Returns:
            List of embedding vectors.
        """
        return await self._provider.embed_batch(texts, batch_size)

    async def embed_with_progress(
        self,
        texts: Sequence[str],
        batch_size: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Generate embeddings with progress reporting.

        Args:
            texts: Texts to embed.
            batch_size: Number of texts per batch.
            progress_callback: Callback function(current, total).

        Returns:
            List of embedding vectors.
        """
        all_embeddings = []
        texts_list = list(texts)
        total = len(texts_list)

        for i in range(0, total, batch_size):
            batch = texts_list[i:i + batch_size]
            embeddings = await self._provider.embed_batch(batch, batch_size=len(batch))
            all_embeddings.extend(embeddings)

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        logger.info(
            f"Generated {len(all_embeddings)} embeddings across "
            f"{(total + batch_size - 1) // batch_size} batches"
        )
        return all_embeddings


# Backward compatibility alias
OpenAIEmbedder = Embedder

"""OpenAI embedding generation with batching and rate limiting."""

import asyncio
import logging
from typing import Callable, Sequence

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from code_rag.config import get_settings
from code_rag.core.errors import EmbeddingError

logger = logging.getLogger(__name__)

RETRY_MAX_ATTEMPTS = 5
RETRY_MULTIPLIER = 1
RETRY_MIN_WAIT = 1
RETRY_MAX_WAIT = 60


class OpenAIEmbedder:
    """Generates embeddings using OpenAI API with rate limiting."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_concurrent: int | None = None,
    ):
        """Initialize the embedder.

        Args:
            api_key: OpenAI API key. Defaults to settings.
            model: Embedding model name. Defaults to settings.
            max_concurrent: Max concurrent API calls. Defaults to settings.
        """
        settings = get_settings()
        self.model = model or settings.embedding_model
        self.max_concurrent = max_concurrent or settings.max_concurrent_requests
        self._client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    @retry(
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=RETRY_MULTIPLIER, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT
        ),
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with retry logic.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        try:
            async with self._semaphore:
                response = await self._client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                logger.debug(f"Generated embeddings for {len(texts)} texts")
                return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingError(f"Failed to embed batch of {len(texts)} texts", cause=e)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        embeddings = await self._embed_batch([text])
        return embeddings[0]

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
        return await self._process_batches(texts, batch_size, None)

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
        return await self._process_batches(texts, batch_size, progress_callback)

    async def _process_batches(
        self,
        texts: Sequence[str],
        batch_size: int,
        progress_callback: Callable[[int, int], None] | None,
    ) -> list[list[float]]:
        """Process texts in batches with optional progress reporting.

        Args:
            texts: Texts to embed.
            batch_size: Number of texts per batch.
            progress_callback: Optional callback function(current, total).

        Returns:
            List of embedding vectors.
        """
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = list(texts[i : i + batch_size])
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        logger.info(f"Generated {len(all_embeddings)} embeddings across {(total + batch_size - 1) // batch_size} batches")
        return all_embeddings

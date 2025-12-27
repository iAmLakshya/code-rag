import logging
from collections.abc import Callable, Sequence

from code_rag.config import get_settings
from code_rag.providers import get_embedding_provider
from code_rag.providers.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings using configurable providers (OpenAI, Ollama, Google)."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_concurrent: int | None = None,
    ):
        settings = get_settings()
        self.max_concurrent = max_concurrent or settings.max_concurrent_requests

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
        return await self._provider.embed(text)

    async def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        return await self._provider.embed_batch(texts, batch_size)

    async def embed_with_progress(
        self,
        texts: Sequence[str],
        batch_size: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        all_embeddings = []
        texts_list = list(texts)
        total = len(texts_list)

        for i in range(0, total, batch_size):
            batch = texts_list[i : i + batch_size]
            embeddings = await self._provider.embed_batch(batch, batch_size=len(batch))
            all_embeddings.extend(embeddings)

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        logger.info(
            f"Generated {len(all_embeddings)} embeddings across "
            f"{(total + batch_size - 1) // batch_size} batches"
        )
        return all_embeddings


OpenAIEmbedder = Embedder

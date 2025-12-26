"""OpenAI provider implementation."""

import logging

from openai import AsyncOpenAI

from code_rag.core.errors import EmbeddingError, SummarizationError
from code_rag.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderConfig,
)

logger = logging.getLogger(__name__)


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI LLM provider using GPT models."""

    def __init__(self, config: ProviderConfig):
        """Initialize OpenAI LLM provider.

        Args:
            config: Provider configuration with api_key.
        """
        super().__init__(config)
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def _complete_impl(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate completion using OpenAI API.

        Args:
            messages: Chat messages.
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text.

        Raises:
            SummarizationError: If API call fails.
        """
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise SummarizationError(f"OpenAI completion failed: {e}", cause=e)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider using text-embedding models."""

    def __init__(self, config: ProviderConfig):
        """Initialize OpenAI embedding provider.

        Args:
            config: Provider configuration with api_key.
        """
        super().__init__(config)
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If API call fails.
        """
        try:
            response = await self._client.embeddings.create(
                model=self.config.model,
                input=texts,
            )
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise EmbeddingError(f"OpenAI embedding failed: {e}", cause=e)

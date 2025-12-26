"""Ollama provider implementation.

Ollama provides local LLM inference with an OpenAI-compatible API.
Default endpoint: http://localhost:11434/v1

Setup:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull llama3.2
    3. Set environment variables:
       - LLM_PROVIDER=ollama
       - LLM_MODEL=llama3.2
       - OLLAMA_BASE_URL=http://localhost:11434/v1  (optional)
"""

import logging

from openai import AsyncOpenAI

from code_rag.core.errors import EmbeddingError, SummarizationError
from code_rag.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_LLM_MODEL = "llama3.2"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


class OllamaLLMProvider(BaseLLMProvider):
    """Ollama LLM provider for local model inference.

    Uses Ollama's OpenAI-compatible API endpoint.

    Recommended models:
        - llama3.2 (8B, good balance)
        - codellama (7B, code-focused)
        - mistral (7B, fast)
        - deepseek-coder (6.7B, code-focused)
    """

    def __init__(self, config: ProviderConfig):
        """Initialize Ollama LLM provider.

        Args:
            config: Provider configuration. base_url defaults to localhost.
        """
        super().__init__(config)

        # Use default Ollama URL if not specified
        base_url = config.base_url or DEFAULT_OLLAMA_BASE_URL

        # Ollama doesn't require API key, use placeholder
        self._client = AsyncOpenAI(
            api_key="ollama",  # Placeholder, not used
            base_url=base_url,
        )

        logger.info(f"Initialized Ollama LLM provider: {base_url} with model {config.model}")

    async def _complete_impl(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate completion using Ollama API.

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
            logger.error(f"Ollama completion failed: {e}")
            raise SummarizationError(f"Ollama completion failed: {e}", cause=e)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider for local embedding generation.

    Recommended models:
        - nomic-embed-text (768 dims, good quality)
        - mxbai-embed-large (1024 dims, high quality)
        - all-minilm (384 dims, fast)
    """

    def __init__(self, config: ProviderConfig):
        """Initialize Ollama embedding provider.

        Args:
            config: Provider configuration. base_url defaults to localhost.
        """
        super().__init__(config)

        # Use default Ollama URL if not specified
        base_url = config.base_url or DEFAULT_OLLAMA_BASE_URL

        # Ollama doesn't require API key, use placeholder
        self._client = AsyncOpenAI(
            api_key="ollama",  # Placeholder, not used
            base_url=base_url,
        )

        logger.info(f"Initialized Ollama embedding provider: {base_url} with model {config.model}")

    async def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Ollama API.

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
            logger.debug(f"Generated embeddings for {len(texts)} texts via Ollama")
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            raise EmbeddingError(f"Ollama embedding failed: {e}", cause=e)

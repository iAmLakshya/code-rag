"""Google Gemini provider implementation.

Setup:
    1. Get API key from https://makersuite.google.com/app/apikey
    2. Set environment variables:
       - LLM_PROVIDER=google
       - LLM_MODEL=gemini-1.5-flash
       - GOOGLE_API_KEY=...
"""

import logging
from typing import TYPE_CHECKING

from code_rag.core.errors import EmbeddingError, SummarizationError
from code_rag.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderConfig,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import google.generativeai as genai


def _configure_google(api_key: str | None) -> None:
    """Configure Google AI SDK.

    Args:
        api_key: Google API key.

    Raises:
        ImportError: If google-generativeai package is not installed.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
    except ImportError:
        raise ImportError(
            "Google provider requires the 'google-generativeai' package. "
            "Install with: pip install google-generativeai"
        )


class GoogleLLMProvider(BaseLLMProvider):
    """Google Gemini LLM provider.

    Recommended models:
        - gemini-1.5-flash (fast, good for most tasks)
        - gemini-1.5-pro (more capable)
        - gemini-2.0-flash-exp (experimental, very fast)
    """

    def __init__(self, config: ProviderConfig):
        """Initialize Google LLM provider.

        Args:
            config: Provider configuration with api_key.

        Raises:
            ImportError: If google-generativeai package is not installed.
        """
        super().__init__(config)
        _configure_google(config.api_key)

        import google.generativeai as genai
        self._model = genai.GenerativeModel(config.model)
        logger.info(f"Initialized Google LLM provider with model {config.model}")

    async def _complete_impl(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate completion using Google API.

        Args:
            messages: Chat messages (OpenAI format, will be converted).
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text.

        Raises:
            SummarizationError: If API call fails.
        """
        try:
            import google.generativeai as genai

            # Convert OpenAI message format to Google format
            # Google uses a different chat format
            history = []
            current_message = ""

            for msg in messages:
                if msg["role"] == "system":
                    # Prepend system message to first user message
                    current_message = f"Instructions: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    current_message += msg["content"]
                elif msg["role"] == "assistant":
                    if current_message:
                        history.append({"role": "user", "parts": [current_message]})
                        current_message = ""
                    history.append({"role": "model", "parts": [msg["content"]]})

            # Start chat with history
            chat = self._model.start_chat(history=history)

            # Configure generation
            generation_config = genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )

            # Send the final user message
            response = await chat.send_message_async(
                current_message,
                generation_config=generation_config,
            )

            return response.text.strip() if response.text else ""

        except Exception as e:
            logger.error(f"Google completion failed: {e}")
            raise SummarizationError(f"Google completion failed: {e}", cause=e)


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """Google embedding provider using text-embedding models.

    Recommended models:
        - text-embedding-004 (latest, 768 dims) - default
    """

    def __init__(self, config: ProviderConfig):
        """Initialize Google embedding provider.

        Args:
            config: Provider configuration with api_key.

        Raises:
            ImportError: If google-generativeai package is not installed.
        """
        super().__init__(config)
        _configure_google(config.api_key)
        logger.info(f"Initialized Google embedding provider with model {config.model}")

    async def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Google API.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If API call fails.
        """
        try:
            import google.generativeai as genai

            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=f"models/{self.config.model}",
                    content=text,
                    task_type="retrieval_document",
                )
                embeddings.append(result["embedding"])

            logger.debug(f"Generated embeddings for {len(texts)} texts via Google")
            return embeddings

        except Exception as e:
            logger.error(f"Google embedding failed: {e}")
            raise EmbeddingError(f"Google embedding failed: {e}", cause=e)

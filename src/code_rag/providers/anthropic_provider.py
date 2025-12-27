"""Anthropic Claude provider implementation.

Setup:
    1. Get API key from https://console.anthropic.com
    2. Set environment variables:
       - LLM_PROVIDER=anthropic
       - LLM_MODEL=claude-sonnet-4-20250514
       - ANTHROPIC_API_KEY=sk-ant-...
"""

import logging
from typing import TYPE_CHECKING

from code_rag.core.errors import SummarizationError
from code_rag.providers.base import BaseLLMProvider, ProviderConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


def _get_anthropic_client(api_key: str | None) -> "AsyncAnthropic":
    """Get Anthropic client, raising helpful error if not installed.

    Args:
        api_key: Anthropic API key.

    Returns:
        AsyncAnthropic client instance.

    Raises:
        ImportError: If anthropic package is not installed.
    """
    try:
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=api_key)
    except ImportError:
        raise ImportError(
            "Anthropic provider requires the 'anthropic' package. "
            "Install with: pip install anthropic"
        )


class AnthropicLLMProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider.

    Recommended models:
        - claude-sonnet-4-20250514 (balanced)
        - claude-opus-4-20250514 (most capable)
        - claude-3-5-haiku-20241022 (fast)
    """

    def __init__(self, config: ProviderConfig):
        """Initialize Anthropic LLM provider.

        Args:
            config: Provider configuration with api_key.

        Raises:
            ImportError: If anthropic package is not installed.
        """
        super().__init__(config)
        self._client = _get_anthropic_client(config.api_key)
        logger.info(f"Initialized Anthropic LLM provider with model {config.model}")

    async def _complete_impl(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate completion using Anthropic API.

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
            system_message = ""
            anthropic_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })

            response = await self._client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message if system_message else None,
                messages=anthropic_messages,
            )

            content = response.content[0].text if response.content else ""
            return content.strip()

        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
            raise SummarizationError(f"Anthropic completion failed: {e}", cause=e)

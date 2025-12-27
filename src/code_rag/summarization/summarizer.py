"""AI-powered code summarization.

Supports multiple LLM providers:
- OpenAI (default)
- Ollama (local)
- Anthropic (Claude)
- Google (Gemini)
"""

import asyncio
import logging
from collections.abc import Callable

from code_rag.config import get_settings
from code_rag.core.errors import SummarizationError
from code_rag.core.protocols import LLMProvider
from code_rag.core.types import EntityType
from code_rag.parsing.models import CodeEntity, ParsedFile
from code_rag.providers import get_llm_provider
from code_rag.providers.base import BaseLLMProvider
from code_rag.summarization.prompts import SummaryPrompts

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.7

SYSTEM_MESSAGE = "You are a code analysis assistant. Provide concise, accurate summaries of code."


class CodeSummarizer:
    """Generates AI summaries for code entities using configurable LLM providers."""

    def __init__(
        self,
        llm_provider: LLMProvider | BaseLLMProvider | None = None,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_concurrent: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
    ):
        """Initialize the summarizer.

        Args:
            llm_provider: LLM provider instance. If None, creates from settings.
            provider: Provider name (openai, ollama, anthropic, google).
            model: LLM model name. Defaults to settings.
            api_key: API key. Defaults to settings.
            base_url: Custom base URL (for Ollama).
            max_concurrent: Max concurrent API calls. Defaults to settings.
            max_tokens: Maximum tokens for summaries.
            temperature: LLM temperature. Defaults to settings.
        """
        settings = get_settings()
        self.max_tokens = max_tokens
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.max_concurrent = max_concurrent or settings.max_concurrent_requests

        if llm_provider is not None:
            self._llm_provider = llm_provider
        else:
            self._llm_provider = get_llm_provider(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        if hasattr(self._llm_provider, 'set_concurrency'):
            self._llm_provider.set_concurrency(self.max_concurrent)

        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._summarization_strategies = {
            EntityType.CLASS: self._summarize_class,
            EntityType.FUNCTION: self._summarize_function,
            EntityType.METHOD: self._summarize_function,
        }

        logger.info(
            f"Initialized CodeSummarizer with "
            f"{getattr(self._llm_provider, 'config', {})}"
        )

    async def _complete(self, system_message: str, user_message: str) -> str:
        """Generate completion using LLM provider.

        Args:
            system_message: System prompt.
            user_message: User prompt.

        Returns:
            Generated completion text.

        Raises:
            SummarizationError: If completion fails after retries.
        """
        try:
            async with self._semaphore:
                return await self._llm_provider.complete(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise SummarizationError("Failed to generate summary", cause=e)

    async def summarize_file(self, parsed_file: ParsedFile) -> str:
        """Generate a summary for a file.

        Args:
            parsed_file: Parsed file to summarize.

        Returns:
            Generated summary.

        Raises:
            SummarizationError: If summarization fails.
        """
        logger.debug(f"Summarizing file: {parsed_file.file_info.relative_path}")
        prompt = SummaryPrompts.get_file_prompt(
            file_path=parsed_file.file_info.relative_path,
            language=parsed_file.file_info.language.value,
            content=parsed_file.content,
        )
        return await self._complete(SYSTEM_MESSAGE, prompt)

    async def _summarize_function(
        self,
        entity: CodeEntity,
        file_path: str,
        language: str,
    ) -> str:
        """Generate a summary for a function or method.

        Args:
            entity: Function/method entity to summarize.
            file_path: Source file path.
            language: Programming language.

        Returns:
            Generated summary.

        Raises:
            SummarizationError: If summarization fails.
        """
        logger.debug(f"Summarizing function: {entity.qualified_name}")
        prompt = SummaryPrompts.get_function_prompt(
            name=entity.name,
            file_path=file_path,
            signature=entity.signature or "",
            code=entity.code,
            language=language,
            docstring=entity.docstring,
        )
        return await self._complete(SYSTEM_MESSAGE, prompt)

    async def _summarize_class(
        self,
        entity: CodeEntity,
        file_path: str,
        language: str,
    ) -> str:
        """Generate a summary for a class.

        Args:
            entity: Class entity to summarize.
            file_path: Source file path.
            language: Programming language.

        Returns:
            Generated summary.

        Raises:
            SummarizationError: If summarization fails.
        """
        logger.debug(f"Summarizing class: {entity.qualified_name}")
        prompt = SummaryPrompts.get_class_prompt(
            name=entity.name,
            file_path=file_path,
            code=entity.code,
            language=language,
            docstring=entity.docstring,
        )
        return await self._complete(SYSTEM_MESSAGE, prompt)

    async def summarize_entity(
        self,
        entity: CodeEntity,
        file_path: str,
        language: str,
    ) -> str:
        """Generate a summary for any code entity.

        Args:
            entity: Entity to summarize.
            file_path: Source file path.
            language: Programming language.

        Returns:
            Generated summary.

        Raises:
            SummarizationError: If entity type not supported.
        """
        strategy = self._summarization_strategies.get(entity.type)
        if strategy:
            return await strategy(entity, file_path, language)

        logger.warning(f"No summarization strategy for entity type: {entity.type}")
        return ""

    async def summarize_parsed_file(
        self,
        parsed_file: ParsedFile,
        include_entities: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, str]:
        """Generate summaries for a file and its entities.

        Args:
            parsed_file: Parsed file to summarize.
            include_entities: Whether to summarize individual entities.
            progress_callback: Optional progress callback.

        Returns:
            Dictionary mapping entity names to summaries.
        """
        logger.info(f"Summarizing parsed file: {parsed_file.file_info.relative_path}")
        summaries = {}
        file_path = parsed_file.file_info.relative_path
        language = parsed_file.file_info.language.value

        summaries["__file__"] = await self.summarize_file(parsed_file)

        if include_entities:
            entities = parsed_file.all_entities
            total = len(entities) + 1

            for i, entity in enumerate(entities):
                if entity.type in self._summarization_strategies:
                    try:
                        summary = await self.summarize_entity(entity, file_path, language)
                        summaries[entity.qualified_name] = summary
                    except SummarizationError as e:
                        logger.error(f"Failed to summarize {entity.qualified_name}: {e}")
                        summaries[entity.qualified_name] = ""
                    except Exception as e:
                        logger.error(f"Unexpected error summarizing {entity.qualified_name}: {e}")
                        summaries[entity.qualified_name] = ""

                if progress_callback:
                    progress_callback(i + 2, total)

        return summaries

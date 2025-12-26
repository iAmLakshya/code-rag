"""LLM response generation for queries.

Supports multiple LLM providers:
- OpenAI (default)
- Ollama (local)
- Anthropic (Claude)
- Google (Gemini)
"""

import logging

from code_rag.config import get_settings
from code_rag.core.errors import QueryError
from code_rag.providers import get_llm_provider
from code_rag.providers.base import BaseLLMProvider
from code_rag.query.reranker import SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful code assistant. You help developers understand codebases by answering questions about code structure, functionality, and relationships.

When answering questions:
1. Be concise but comprehensive
2. Reference specific files, functions, and line numbers when relevant
3. Explain code relationships and dependencies
4. Provide code snippets when helpful
5. If you're not sure about something, say so

Format your responses using markdown for readability."""

QUERY_PROMPT_TEMPLATE = """Based on the following search results from the codebase, answer the user's question.

User Question: {question}

Search Results:
{context}

Provide a helpful, accurate answer based on the search results. If the results don't fully answer the question, say what you can determine and what's unclear."""

EXPLANATION_WITH_QUESTION_TEMPLATE = """Explain this {language} code, specifically answering: {question}

```{language}
{code}
```"""

EXPLANATION_TEMPLATE = """Explain what this {language} code does:

```{language}
{code}
```

Provide a clear, concise explanation suitable for a developer."""

MAX_RESPONSE_TOKENS = 1500
MAX_EXPLANATION_TOKENS = 1000
MAX_CONTEXT_RESULTS = 10
MAX_CONTENT_LENGTH = 2000


class ResponseGenerator:
    """Generates natural language responses using configurable LLM providers."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize response generator.

        Args:
            provider: Provider name (openai, ollama, anthropic, google).
            model: LLM model name. Defaults to settings.
            api_key: API key. Defaults to settings.
            base_url: Custom base URL (for Ollama).
        """
        settings = get_settings()
        self.temperature = settings.llm_temperature

        self._llm_provider: BaseLLMProvider = get_llm_provider(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=self.temperature,
        )

        logger.info(
            f"Initialized ResponseGenerator with "
            f"{self._llm_provider.config.provider}/{self._llm_provider.config.model}"
        )

    async def generate_response(
        self,
        question: str,
        results: list[SearchResult],
        max_context_results: int = MAX_CONTEXT_RESULTS,
    ) -> str:
        """Generate a response to a question using search results.

        Args:
            question: User's question.
            results: Search results to use as context.
            max_context_results: Maximum results to include in context.

        Returns:
            Generated response.

        Raises:
            QueryError: If response generation fails.
        """
        try:
            logger.debug(f"Generating response for question: {question}")
            context = self._build_context(results[:max_context_results])

            answer = await self._llm_provider.complete(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": QUERY_PROMPT_TEMPLATE.format(
                            question=question,
                            context=context,
                        ),
                    },
                ],
                max_tokens=MAX_RESPONSE_TOKENS,
            )

            logger.debug(f"Generated response length: {len(answer)}")
            return answer

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise QueryError("Failed to generate response", cause=e)

    async def generate_explanation(
        self,
        code: str,
        language: str,
        question: str | None = None,
    ) -> str:
        """Generate an explanation for a code snippet.

        Args:
            code: Code to explain.
            language: Programming language.
            question: Optional specific question about the code.

        Returns:
            Explanation text.

        Raises:
            QueryError: If explanation generation fails.
        """
        try:
            logger.debug(f"Generating explanation for {language} code")

            if question:
                prompt = EXPLANATION_WITH_QUESTION_TEMPLATE.format(
                    language=language,
                    question=question,
                    code=code,
                )
            else:
                prompt = EXPLANATION_TEMPLATE.format(
                    language=language,
                    code=code,
                )

            answer = await self._llm_provider.complete(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_EXPLANATION_TOKENS,
            )

            logger.debug(f"Generated explanation length: {len(answer)}")
            return answer

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            raise QueryError("Failed to generate explanation", cause=e)

    def _build_context(self, results: list[SearchResult]) -> str:
        """Build context string from search results.

        Args:
            results: Search results.

        Returns:
            Formatted context string.
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            parts = [f"### Result {i}"]
            parts.append(f"**File**: {result.file_path}")
            parts.append(f"**Entity**: {result.entity_name} ({result.entity_type})")

            if result.start_line:
                parts.append(f"**Lines**: {result.start_line}-{result.end_line}")

            if result.summary:
                parts.append(f"**Summary**: {result.summary}")

            if result.content:
                content = result.content[:MAX_CONTENT_LENGTH]
                if len(result.content) > MAX_CONTENT_LENGTH:
                    content += "\n... (truncated)"
                parts.append(f"**Code**:\n```\n{content}\n```")

            context_parts.append("\n".join(parts))

        return "\n\n".join(context_parts)

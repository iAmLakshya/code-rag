"""LLM response generation for queries."""

import logging

from openai import AsyncOpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential

from code_rag.config import get_settings
from code_rag.core.errors import QueryError
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
    """Generates natural language responses using LLM."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """Initialize response generator.

        Args:
            api_key: OpenAI API key. Defaults to settings.
            model: LLM model name. Defaults to settings.
        """
        settings = get_settings()
        self.model = model or settings.llm_model
        self.temperature = settings.llm_temperature
        self._client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
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

            answer = await self._call_llm(
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

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise QueryError("Failed to generate response", cause=e)
        except Exception as e:
            logger.error(f"Unexpected error generating response: {e}")
            raise QueryError("Unexpected error during response generation", cause=e)

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

            answer = await self._call_llm(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_EXPLANATION_TOKENS,
            )

            logger.debug(f"Generated explanation length: {len(answer)}")
            return answer

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise QueryError("Failed to generate explanation", cause=e)
        except Exception as e:
            logger.error(f"Unexpected error generating explanation: {e}")
            raise QueryError("Unexpected error during explanation generation", cause=e)

    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> str:
        """Call LLM with error handling.

        Args:
            messages: Chat messages.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text.

        Raises:
            OpenAIError: If API call fails.
        """
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()

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

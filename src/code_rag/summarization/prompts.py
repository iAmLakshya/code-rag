"""Prompt templates for AI summarization."""

FILE_CODE_MAX_CHARS = 8000
FUNCTION_CODE_MAX_CHARS = 4000
CLASS_CODE_MAX_CHARS = 6000


class SummaryPrompts:
    """Collection of prompts for generating code summaries."""

    FILE_SUMMARY = """Analyze this source code file and provide a concise summary.

File: {file_path}
Language: {language}

Code:
```{language}
{content}
```

Provide a summary that includes:
1. The main purpose of this file (1-2 sentences)
2. Key classes/functions defined and what they do
3. Important dependencies or integrations
4. Any notable patterns or design decisions

Keep the summary concise (3-5 sentences) and focus on what a developer would need to know to understand this file's role in the codebase."""

    FUNCTION_SUMMARY = """Analyze this function and provide a concise summary.

Function: {name}
File: {file_path}
Signature: {signature}

Code:
```{language}
{code}
```

{docstring_section}

Provide a summary that includes:
1. What the function does (1 sentence)
2. Input parameters and their purpose
3. Return value and its meaning
4. Any side effects or important behavior

Keep the summary to 2-3 sentences, focusing on practical usage."""

    CLASS_SUMMARY = """Analyze this class and provide a concise summary.

Class: {name}
File: {file_path}

Code:
```{language}
{code}
```

{docstring_section}

Provide a summary that includes:
1. The class's purpose and responsibility (1-2 sentences)
2. Key methods and their roles
3. How this class is typically used
4. Important attributes or state it manages

Keep the summary to 3-4 sentences, focusing on the class's role in the system."""

    CODEBASE_OVERVIEW = """Based on the following file summaries, provide an overview of this codebase.

File summaries:
{summaries}

Provide an overview that includes:
1. The main purpose of this codebase
2. Key architectural patterns used
3. Main components and how they interact
4. Entry points and important flows

Keep the overview concise but comprehensive."""

    @staticmethod
    def _build_docstring_section(docstring: str | None) -> str:
        if docstring:
            return f"Existing docstring:\n{docstring}"
        return ""

    @classmethod
    def get_file_prompt(
        cls,
        file_path: str,
        language: str,
        content: str,
    ) -> str:
        return cls.FILE_SUMMARY.format(
            file_path=file_path,
            language=language,
            content=content[:FILE_CODE_MAX_CHARS],
        )

    @classmethod
    def get_function_prompt(
        cls,
        name: str,
        file_path: str,
        signature: str,
        code: str,
        language: str,
        docstring: str | None = None,
    ) -> str:
        return cls.FUNCTION_SUMMARY.format(
            name=name,
            file_path=file_path,
            signature=signature,
            code=code[:FUNCTION_CODE_MAX_CHARS],
            language=language,
            docstring_section=cls._build_docstring_section(docstring),
        )

    @classmethod
    def get_class_prompt(
        cls,
        name: str,
        file_path: str,
        code: str,
        language: str,
        docstring: str | None = None,
    ) -> str:
        return cls.CLASS_SUMMARY.format(
            name=name,
            file_path=file_path,
            code=code[:CLASS_CODE_MAX_CHARS],
            language=language,
            docstring_section=cls._build_docstring_section(docstring),
        )

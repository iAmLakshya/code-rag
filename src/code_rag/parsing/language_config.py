"""Language configuration system for multi-language code parsing.

This module provides a configuration-driven approach to supporting multiple
programming languages. Each language has a LanguageConfig that defines:
- File extensions
- Tree-sitter node types for various constructs
- Tree-sitter queries for extraction
- Language-specific parsing rules

Adding a new language:
1. Create a LanguageConfig with appropriate node types
2. Add a Tree-sitter query (optional, for complex extractions)
3. Register in LANGUAGE_CONFIGS
4. (Optionally) Create a custom extractor for complex cases
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from code_rag.core.types import Language


@dataclass
class LanguageConfig:
    """Configuration for parsing a programming language.

    Attributes:
        name: Language identifier (matches Language enum).
        display_name: Human-readable name.
        file_extensions: List of file extensions (with dots).
        function_node_types: Tree-sitter node types for functions.
        class_node_types: Tree-sitter node types for classes.
        method_node_types: Tree-sitter node types for methods.
        call_node_types: Tree-sitter node types for function calls.
        import_node_types: Tree-sitter node types for imports.
        module_node_types: Tree-sitter node types for modules/packages.
        comment_node_types: Tree-sitter node types for comments.
        string_node_types: Tree-sitter node types for strings.
        function_query: Tree-sitter query for extracting functions.
        class_query: Tree-sitter query for extracting classes.
        import_query: Tree-sitter query for extracting imports.
        call_query: Tree-sitter query for extracting calls.
        package_indicators: Files that indicate a package/module root.
        ignore_patterns: Patterns to ignore within this language.
    """

    name: str
    display_name: str
    file_extensions: list[str]

    function_node_types: list[str] = field(default_factory=list)
    class_node_types: list[str] = field(default_factory=list)
    method_node_types: list[str] = field(default_factory=list)
    call_node_types: list[str] = field(default_factory=list)
    import_node_types: list[str] = field(default_factory=list)
    module_node_types: list[str] = field(default_factory=list)
    comment_node_types: list[str] = field(default_factory=list)
    string_node_types: list[str] = field(default_factory=list)

    function_query: str | None = None
    class_query: str | None = None
    import_query: str | None = None
    call_query: str | None = None

    package_indicators: list[str] = field(default_factory=list)
    ignore_patterns: list[str] = field(default_factory=list)

    def matches_extension(self, extension: str) -> bool:
        """Check if extension matches this language.

        Args:
            extension: File extension (with or without dot).

        Returns:
            True if extension matches.
        """
        ext = extension if extension.startswith(".") else f".{extension}"
        return ext.lower() in [e.lower() for e in self.file_extensions]


@dataclass
class FQNConfig:
    """Configuration for Fully Qualified Name resolution.

    Used for resolving method calls like user.get_profile() to their
    fully qualified names.
    """

    scope_node_types: set[str]
    function_node_types: set[str]
    class_node_types: set[str]
    get_name: Callable | None = None
    file_to_module_parts: Callable[[Path, Path], list[str]] | None = None


# =============================================================================
# Language Configurations
# =============================================================================

PYTHON_CONFIG = LanguageConfig(
    name="python",
    display_name="Python",
    file_extensions=[".py"],
    function_node_types=["function_definition"],
    class_node_types=["class_definition"],
    method_node_types=["function_definition"],
    call_node_types=["call"],
    import_node_types=["import_statement", "import_from_statement"],
    module_node_types=["module"],
    comment_node_types=["comment"],
    string_node_types=["string", "concatenated_string"],
    function_query="(function_definition name: (identifier) @function)",
    class_query="(class_definition name: (identifier) @class)",
    import_query="""
        (import_statement name: (dotted_name) @import)
        (import_from_statement module_name: (dotted_name) @module)
    """,
    package_indicators=["__init__.py", "pyproject.toml", "setup.py"],
)

JAVASCRIPT_CONFIG = LanguageConfig(
    name="javascript",
    display_name="JavaScript",
    file_extensions=[".js", ".mjs", ".cjs"],
    function_node_types=[
        "function_declaration",
        "function_expression",
        "arrow_function",
        "generator_function_declaration",
    ],
    class_node_types=["class_declaration", "class"],
    method_node_types=["method_definition", "function_expression", "arrow_function"],
    call_node_types=["call_expression"],
    import_node_types=["import_statement", "import_specifier"],
    module_node_types=["program"],
    comment_node_types=["comment"],
    string_node_types=["string", "template_string"],
    function_query="""
        (function_declaration name: (identifier) @function)
        (variable_declarator name: (identifier) @function value: (arrow_function))
    """,
    class_query="(class_declaration name: (identifier) @class)",
    package_indicators=["package.json"],
)

JSX_CONFIG = LanguageConfig(
    name="jsx",
    display_name="JavaScript JSX",
    file_extensions=[".jsx"],
    function_node_types=JAVASCRIPT_CONFIG.function_node_types,
    class_node_types=JAVASCRIPT_CONFIG.class_node_types,
    method_node_types=JAVASCRIPT_CONFIG.method_node_types,
    call_node_types=JAVASCRIPT_CONFIG.call_node_types + ["jsx_element", "jsx_self_closing_element"],
    import_node_types=JAVASCRIPT_CONFIG.import_node_types,
    module_node_types=JAVASCRIPT_CONFIG.module_node_types,
    comment_node_types=JAVASCRIPT_CONFIG.comment_node_types,
    string_node_types=JAVASCRIPT_CONFIG.string_node_types,
    function_query=JAVASCRIPT_CONFIG.function_query,
    class_query=JAVASCRIPT_CONFIG.class_query,
    package_indicators=["package.json"],
)

TYPESCRIPT_CONFIG = LanguageConfig(
    name="typescript",
    display_name="TypeScript",
    file_extensions=[".ts", ".mts", ".cts"],
    function_node_types=[
        "function_declaration",
        "function_expression",
        "arrow_function",
        "generator_function_declaration",
    ],
    class_node_types=["class_declaration", "class", "abstract_class_declaration"],
    method_node_types=["method_definition", "method_signature"],
    call_node_types=["call_expression"],
    import_node_types=["import_statement", "import_specifier", "type_import"],
    module_node_types=["program"],
    comment_node_types=["comment"],
    string_node_types=["string", "template_string"],
    function_query="""
        (function_declaration name: (identifier) @function)
        (variable_declarator name: (identifier) @function value: (arrow_function))
    """,
    class_query="""
        (class_declaration name: (type_identifier) @class)
        (abstract_class_declaration name: (type_identifier) @class)
    """,
    package_indicators=["package.json", "tsconfig.json"],
)

TSX_CONFIG = LanguageConfig(
    name="tsx",
    display_name="TypeScript JSX",
    file_extensions=[".tsx"],
    function_node_types=TYPESCRIPT_CONFIG.function_node_types,
    class_node_types=TYPESCRIPT_CONFIG.class_node_types,
    method_node_types=TYPESCRIPT_CONFIG.method_node_types,
    call_node_types=TYPESCRIPT_CONFIG.call_node_types + ["jsx_element", "jsx_self_closing_element"],
    import_node_types=TYPESCRIPT_CONFIG.import_node_types,
    module_node_types=TYPESCRIPT_CONFIG.module_node_types,
    comment_node_types=TYPESCRIPT_CONFIG.comment_node_types,
    string_node_types=TYPESCRIPT_CONFIG.string_node_types,
    function_query=TYPESCRIPT_CONFIG.function_query,
    class_query=TYPESCRIPT_CONFIG.class_query,
    package_indicators=["package.json", "tsconfig.json"],
)

RUST_CONFIG = LanguageConfig(
    name="rust",
    display_name="Rust",
    file_extensions=[".rs"],
    function_node_types=["function_item"],
    class_node_types=["struct_item", "enum_item", "trait_item", "impl_item"],
    method_node_types=["function_item"],
    call_node_types=["call_expression", "macro_invocation"],
    import_node_types=["use_declaration"],
    module_node_types=["source_file"],
    comment_node_types=["line_comment", "block_comment"],
    string_node_types=["string_literal", "raw_string_literal"],
    package_indicators=["Cargo.toml"],
)

JAVA_CONFIG = LanguageConfig(
    name="java",
    display_name="Java",
    file_extensions=[".java"],
    function_node_types=["method_declaration", "constructor_declaration"],
    class_node_types=["class_declaration", "interface_declaration", "enum_declaration"],
    method_node_types=["method_declaration"],
    call_node_types=["method_invocation"],
    import_node_types=["import_declaration"],
    module_node_types=["program"],
    comment_node_types=["line_comment", "block_comment"],
    string_node_types=["string_literal"],
    package_indicators=["pom.xml", "build.gradle", "build.gradle.kts"],
)

GO_CONFIG = LanguageConfig(
    name="go",
    display_name="Go",
    file_extensions=[".go"],
    function_node_types=["function_declaration", "method_declaration"],
    class_node_types=["type_declaration"],  # Go uses structs
    method_node_types=["method_declaration"],
    call_node_types=["call_expression"],
    import_node_types=["import_declaration", "import_spec"],
    module_node_types=["source_file"],
    comment_node_types=["comment"],
    string_node_types=["raw_string_literal", "interpreted_string_literal"],
    package_indicators=["go.mod"],
)

CPP_CONFIG = LanguageConfig(
    name="cpp",
    display_name="C++",
    file_extensions=[".cpp", ".cc", ".cxx", ".hpp", ".h", ".hxx"],
    function_node_types=["function_definition"],
    class_node_types=["class_specifier", "struct_specifier"],
    method_node_types=["function_definition"],
    call_node_types=["call_expression"],
    import_node_types=["preproc_include"],
    module_node_types=["translation_unit"],
    comment_node_types=["comment"],
    string_node_types=["string_literal", "raw_string_literal"],
    package_indicators=["CMakeLists.txt", "Makefile"],
)


LANGUAGE_CONFIGS: dict[str, LanguageConfig] = {
    "python": PYTHON_CONFIG,
    "javascript": JAVASCRIPT_CONFIG,
    "jsx": JSX_CONFIG,
    "typescript": TYPESCRIPT_CONFIG,
    "tsx": TSX_CONFIG,
    "rust": RUST_CONFIG,
    "java": JAVA_CONFIG,
    "go": GO_CONFIG,
    "cpp": CPP_CONFIG,
}

_EXTENSION_MAP: dict[str, LanguageConfig] = {}
for config in LANGUAGE_CONFIGS.values():
    for ext in config.file_extensions:
        _EXTENSION_MAP[ext.lower()] = config


def get_language_config(extension_or_name: str) -> LanguageConfig | None:
    """Get language configuration by file extension or language name.

    Args:
        extension_or_name: File extension (e.g., ".py") or language name (e.g., "python").

    Returns:
        LanguageConfig or None if not found.
    """
    if extension_or_name in LANGUAGE_CONFIGS:
        return LANGUAGE_CONFIGS[extension_or_name]
    ext = extension_or_name if extension_or_name.startswith(".") else f".{extension_or_name}"
    return _EXTENSION_MAP.get(ext.lower())


def get_config_for_file(file_path: Path) -> LanguageConfig | None:
    """Get language configuration for a file.

    Args:
        file_path: Path to the file.

    Returns:
        LanguageConfig or None if language not supported.
    """
    return get_language_config(file_path.suffix)


def get_supported_extensions() -> list[str]:
    """Get list of all supported file extensions.

    Returns:
        List of file extensions (with dots).
    """
    return list(_EXTENSION_MAP.keys())


def get_supported_languages() -> list[str]:
    """Get list of all supported language names.

    Returns:
        List of language names.
    """
    return list(LANGUAGE_CONFIGS.keys())


def language_enum_to_config(language: Language) -> LanguageConfig | None:
    """Convert Language enum to LanguageConfig.

    Args:
        language: Language enum value.

    Returns:
        LanguageConfig or None if not found.
    """
    name_map = {
        Language.PYTHON: "python",
        Language.JAVASCRIPT: "javascript",
        Language.TYPESCRIPT: "typescript",
        Language.JSX: "jsx",
        Language.TSX: "tsx",
    }
    lang_name = name_map.get(language)
    return LANGUAGE_CONFIGS.get(lang_name) if lang_name else None

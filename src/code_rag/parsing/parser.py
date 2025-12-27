from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from tree_sitter_language_pack import get_parser

from code_rag.core.types import Language
from code_rag.parsing.extractors.base import BaseExtractor
from code_rag.parsing.extractors.javascript import JavaScriptExtractor
from code_rag.parsing.extractors.python import PythonExtractor
from code_rag.parsing.extractors.typescript import TypeScriptExtractor
from code_rag.parsing.models import CodeEntity, FileInfo, ImportInfo, ParsedFile

if TYPE_CHECKING:
    from tree_sitter import Parser


class CodeParser:
    LANGUAGE_MAP: ClassVar[dict[Language, str]] = {
        Language.PYTHON: "python",
        Language.JAVASCRIPT: "javascript",
        Language.JSX: "javascript",
        Language.TYPESCRIPT: "typescript",
        Language.TSX: "tsx",
    }

    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}
        self._extractors: dict[Language, BaseExtractor] = {
            Language.PYTHON: PythonExtractor(),
            Language.JAVASCRIPT: JavaScriptExtractor(),
            Language.JSX: JavaScriptExtractor(),
            Language.TYPESCRIPT: TypeScriptExtractor(),
            Language.TSX: TypeScriptExtractor(),
        }

    def _get_parser(self, language: Language) -> Parser:
        lang_id = self.LANGUAGE_MAP.get(language)
        if lang_id not in self._parsers:
            self._parsers[lang_id] = get_parser(lang_id)
        return self._parsers[lang_id]

    def _get_extractor(self, language: Language) -> BaseExtractor:
        return self._extractors[language]

    def _parse_and_extract(
        self, content: str, language: Language
    ) -> tuple[list[ImportInfo], list[CodeEntity]]:
        parser = self._get_parser(language)
        tree = parser.parse(content.encode("utf-8"))
        extractor = self._get_extractor(language)
        imports = extractor.extract_imports(tree.root_node, content)
        entities = extractor.extract_entities(tree.root_node, content)
        return imports, entities

    def parse_file(self, file_info: FileInfo) -> ParsedFile:
        content = file_info.path.read_text(encoding="utf-8", errors="replace")
        imports, entities = self._parse_and_extract(content, file_info.language)

        return ParsedFile(
            file_info=file_info,
            content=content,
            imports=imports,
            entities=entities,
        )

    def parse_content(
        self,
        content: str,
        language: Language,
        file_path: str = "<string>",
    ) -> ParsedFile:
        file_info = FileInfo(
            path=Path(file_path),
            relative_path=file_path,
            language=language,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            size_bytes=len(content.encode()),
            line_count=content.count("\n") + 1,
        )

        imports, entities = self._parse_and_extract(content, language)

        return ParsedFile(
            file_info=file_info,
            content=content,
            imports=imports,
            entities=entities,
        )

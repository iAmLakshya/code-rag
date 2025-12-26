"""Code chunking strategies for embedding generation."""

from dataclasses import dataclass

import tiktoken

from code_rag.config import get_settings
from code_rag.parsing.models import CodeEntity, ParsedFile

MIN_CHUNK_SIZE = 1
CHUNK_NAME_SEPARATOR = "_part"


@dataclass
class CodeChunk:
    """Represents a chunk of code for embedding."""

    content: str
    file_path: str
    entity_type: str
    entity_name: str
    language: str
    start_line: int
    end_line: int
    graph_node_id: str | None = None
    content_hash: str | None = None
    project_name: str | None = None

    def to_payload(self) -> dict:
        return {
            "file_path": self.file_path,
            "entity_type": self.entity_type,
            "entity_name": self.entity_name,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "graph_node_id": self.graph_node_id,
            "content_hash": self.content_hash,
            "project_name": self.project_name,
        }


class CodeChunker:
    """Chunks code into embeddable pieces."""

    def __init__(
        self,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        encoding_name: str = "cl100k_base",
    ):
        """Initialize the code chunker.

        Args:
            max_tokens: Maximum tokens per chunk. Defaults to settings.
            overlap_tokens: Token overlap between chunks. Defaults to settings.
            encoding_name: Tiktoken encoding name.
        """
        settings = get_settings()
        self.max_tokens = max_tokens or settings.chunk_max_tokens
        self.overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk_file(
        self,
        parsed_file: ParsedFile,
        project_name: str | None = None,
    ) -> list[CodeChunk]:
        """Chunk a parsed file into embeddable pieces.

        Args:
            parsed_file: Parsed file to chunk.
            project_name: Name of the project this file belongs to.

        Returns:
            List of code chunks.
        """
        chunks = []
        file_path = str(parsed_file.file_info.path)
        language = parsed_file.file_info.language.value
        content_hash = parsed_file.file_info.content_hash

        for entity in parsed_file.all_entities:
            entity_chunks = self._chunk_entity(
                entity, file_path, language, content_hash, project_name
            )
            chunks.extend(entity_chunks)

        if not chunks and parsed_file.content.strip():
            file_chunks = self._chunk_text(
                parsed_file.content,
                file_path,
                "file",
                parsed_file.file_info.path.name,
                language,
                1,
                content_hash,
                project_name,
            )
            chunks.extend(file_chunks)

        return chunks

    def _chunk_entity(
        self,
        entity: CodeEntity,
        file_path: str,
        language: str,
        content_hash: str | None = None,
        project_name: str | None = None,
    ) -> list[CodeChunk]:
        """Chunk a single code entity.

        Args:
            entity: Code entity to chunk.
            file_path: File path.
            language: Programming language.
            content_hash: File content hash for incremental indexing.
            project_name: Project name for filtering.

        Returns:
            List of chunks for this entity.
        """
        entity_type = entity.type.value
        entity_name = entity.qualified_name
        content = self._format_entity_content(entity)
        token_count = self.count_tokens(content)

        if token_count <= self.max_tokens:
            return [
                self._create_chunk(
                    content=content,
                    file_path=file_path,
                    entity_type=entity_type,
                    entity_name=entity_name,
                    language=language,
                    start_line=entity.start_line,
                    end_line=entity.end_line,
                    graph_node_id=entity.qualified_name,
                    content_hash=content_hash,
                    project_name=project_name,
                )
            ]
        else:
            return self._chunk_text(
                content,
                file_path,
                entity_type,
                entity_name,
                language,
                entity.start_line,
                content_hash,
                project_name,
            )

    def _format_entity_content(self, entity: CodeEntity) -> str:
        """Format entity content with signature, docstring, and code.

        Args:
            entity: Code entity to format.

        Returns:
            Formatted content string.
        """
        content_parts = []

        if entity.signature:
            content_parts.append(entity.signature)

        if entity.docstring:
            content_parts.append(f'"""{entity.docstring}"""')

        content_parts.append(entity.code)

        return "\n".join(content_parts)

    def _chunk_text(
        self,
        text: str,
        file_path: str,
        entity_type: str,
        entity_name: str,
        language: str,
        start_line: int,
        content_hash: str | None = None,
        project_name: str | None = None,
    ) -> list[CodeChunk]:
        """Split text into chunks with overlap.

        Args:
            text: Text to chunk.
            file_path: File path.
            entity_type: Type of entity.
            entity_name: Name of entity.
            language: Programming language.
            start_line: Starting line number.
            content_hash: File content hash for incremental indexing.
            project_name: Project name for filtering.

        Returns:
            List of chunks.
        """
        lines = text.split("\n")
        chunks = []
        current_lines = []
        current_tokens = 0
        chunk_start_line = start_line

        for i, line in enumerate(lines):
            line_tokens = self.count_tokens(line + "\n")

            if current_tokens + line_tokens > self.max_tokens and current_lines:
                chunk_content = "\n".join(current_lines)
                chunks.append(
                    self._create_chunk(
                        content=chunk_content,
                        file_path=file_path,
                        entity_type=entity_type,
                        entity_name=f"{entity_name}{CHUNK_NAME_SEPARATOR}{len(chunks) + 1}",
                        language=language,
                        start_line=chunk_start_line,
                        end_line=chunk_start_line + len(current_lines) - 1,
                        graph_node_id=entity_name,
                        content_hash=content_hash,
                        project_name=project_name,
                    )
                )

                overlap_lines = self._calculate_overlap_lines(current_lines)
                current_lines = overlap_lines
                current_tokens = sum(
                    self.count_tokens(ol + "\n") for ol in overlap_lines
                )
                chunk_start_line = start_line + i - len(overlap_lines)

            current_lines.append(line)
            current_tokens += line_tokens

        if current_lines:
            chunk_content = "\n".join(current_lines)
            chunk_name = entity_name
            if chunks:
                chunk_name = f"{entity_name}{CHUNK_NAME_SEPARATOR}{len(chunks) + 1}"

            chunks.append(
                self._create_chunk(
                    content=chunk_content,
                    file_path=file_path,
                    entity_type=entity_type,
                    entity_name=chunk_name,
                    language=language,
                    start_line=chunk_start_line,
                    end_line=chunk_start_line + len(current_lines) - 1,
                    graph_node_id=entity_name,
                    content_hash=content_hash,
                    project_name=project_name,
                )
            )

        return chunks

    def _calculate_overlap_lines(self, lines: list[str]) -> list[str]:
        """Calculate overlap lines based on token limit.

        Args:
            lines: Current lines in chunk.

        Returns:
            List of lines to keep for overlap.
        """
        overlap_lines = []
        overlap_tokens = 0

        for line in reversed(lines):
            line_tokens = self.count_tokens(line + "\n")
            if overlap_tokens + line_tokens <= self.overlap_tokens:
                overlap_lines.insert(0, line)
                overlap_tokens += line_tokens
            else:
                break

        return overlap_lines

    def _create_chunk(
        self,
        content: str,
        file_path: str,
        entity_type: str,
        entity_name: str,
        language: str,
        start_line: int,
        end_line: int,
        graph_node_id: str | None = None,
        content_hash: str | None = None,
        project_name: str | None = None,
    ) -> CodeChunk:
        """Create a CodeChunk instance.

        Args:
            content: Chunk content.
            file_path: File path.
            entity_type: Entity type.
            entity_name: Entity name.
            language: Programming language.
            start_line: Starting line number.
            end_line: Ending line number.
            graph_node_id: Graph node ID.
            content_hash: Content hash.
            project_name: Project name.

        Returns:
            CodeChunk instance.
        """
        return CodeChunk(
            content=content,
            file_path=file_path,
            entity_type=entity_type,
            entity_name=entity_name,
            language=language,
            start_line=start_line,
            end_line=end_line,
            graph_node_id=graph_node_id,
            content_hash=content_hash,
            project_name=project_name,
        )

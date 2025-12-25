"""Pipeline orchestrator for coordinating the indexing process."""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from code_rag.config import get_settings
from code_rag.core.errors import IndexingError
from code_rag.core.types import PipelineStage
from code_rag.embeddings.chunker import CodeChunker
from code_rag.embeddings.client import QdrantManager
from code_rag.embeddings.embedder import OpenAIEmbedder
from code_rag.embeddings.indexer import VectorIndexer
from code_rag.graph.builder import GraphBuilder
from code_rag.graph.client import MemgraphClient
from code_rag.graph.schema import GraphSchema
from code_rag.graph.statistics import GraphStatistics
from code_rag.parsing.models import ParsedFile
from code_rag.parsing.parser import CodeParser
from code_rag.parsing.scanner import FileScanner
from code_rag.pipeline.progress import ProgressTracker
from code_rag.summarization.summarizer import CodeSummarizer

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Shared state for pipeline execution."""

    repo_path: Path
    project_name: str
    tracker: ProgressTracker
    memgraph: MemgraphClient
    qdrant: QdrantManager
    parser: CodeParser
    embedder: OpenAIEmbedder
    summarizer: CodeSummarizer

    parsed_files: list[ParsedFile] = field(default_factory=list)
    file_update_status: dict[str, bool] = field(default_factory=dict)
    scanned_files: list = field(default_factory=list)


class PipelineOrchestrator:
    """Orchestrates the full indexing pipeline."""

    def __init__(
        self,
        repo_path: str | Path,
        project_name: str | None = None,
        progress_callback: Callable | None = None,
        memgraph_client: MemgraphClient | None = None,
        qdrant_client: QdrantManager | None = None,
        parser: CodeParser | None = None,
        embedder: OpenAIEmbedder | None = None,
        summarizer: CodeSummarizer | None = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.project_name = project_name or self.repo_path.name

        self.tracker = ProgressTracker()
        if progress_callback:
            self.tracker.add_callback(progress_callback)

        self._memgraph = memgraph_client
        self._qdrant = qdrant_client
        self._parser = parser
        self._embedder = embedder
        self._summarizer = summarizer

    async def _init_components(self) -> PipelineContext:
        """Initialize all components."""
        settings = get_settings()

        if self._memgraph is None:
            self._memgraph = MemgraphClient()
            await self._memgraph.connect()

        if self._qdrant is None:
            self._qdrant = QdrantManager()
            await self._qdrant.connect()
            await self._qdrant.create_collections()

        schema = GraphSchema(self._memgraph)
        await schema.setup()

        if self._parser is None:
            self._parser = CodeParser()
        if self._embedder is None:
            self._embedder = OpenAIEmbedder()
        if self._summarizer is None:
            self._summarizer = CodeSummarizer()

        return PipelineContext(
            repo_path=self.repo_path,
            project_name=self.project_name,
            tracker=self.tracker,
            memgraph=self._memgraph,
            qdrant=self._qdrant,
            parser=self._parser,
            embedder=self._embedder,
            summarizer=self._summarizer,
        )

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._memgraph:
            try:
                await self._memgraph.close()
            except Exception as e:
                logger.warning(f"Failed to close Memgraph connection: {e}")

        if self._qdrant:
            try:
                await self._qdrant.close()
            except Exception as e:
                logger.warning(f"Failed to close Qdrant connection: {e}")

    async def run(self) -> dict:
        """Run the full indexing pipeline."""
        try:
            self.tracker.start()
            logger.info(f"Starting indexing pipeline for {self.project_name}")

            ctx = await self._init_components()

            await self._execute_scan_stage(ctx)
            await self._execute_parse_stage(ctx)
            await self._execute_graph_stage(ctx)
            await self._execute_summarize_stage(ctx)
            await self._execute_embedding_stage(ctx)

            self.tracker.complete()
            logger.info(f"Pipeline completed successfully for {self.project_name}")

            return {
                "files_indexed": self.tracker.progress.files_parsed,
                "entities_found": self.tracker.progress.entities_found,
                "graph_nodes": self.tracker.progress.graph_nodes_created,
                "summaries": self.tracker.progress.summaries_generated,
                "chunks_embedded": self.tracker.progress.chunks_embedded,
                "elapsed_seconds": self.tracker.progress.elapsed_time,
            }

        except Exception as e:
            error_msg = f"Pipeline failed: {e}"
            logger.error(error_msg, exc_info=True)
            self.tracker.error(error_msg)
            raise IndexingError(error_msg, cause=e)
        finally:
            await self._cleanup()

    async def _execute_scan_stage(self, ctx: PipelineContext) -> None:
        """Execute file scanning stage."""
        ctx.tracker.set_stage(PipelineStage.SCANNING, message="Scanning repository...")
        logger.info(f"Scanning repository: {ctx.repo_path}")

        try:
            scanner = FileScanner(ctx.repo_path)
            ctx.scanned_files = scanner.scan_all()

            ctx.tracker.update_stats(files_scanned=len(ctx.scanned_files))
            ctx.tracker.update_stage(
                len(ctx.scanned_files),
                len(ctx.scanned_files),
                f"Found {len(ctx.scanned_files)} files",
            )
            logger.info(f"Scanned {len(ctx.scanned_files)} files")

        except Exception as e:
            logger.error(f"File scanning failed: {e}", exc_info=True)
            raise IndexingError(f"File scanning failed: {e}", stage="scanning", cause=e)

    async def _execute_parse_stage(self, ctx: PipelineContext) -> None:
        """Execute file parsing stage."""
        ctx.tracker.set_stage(
            PipelineStage.PARSING,
            total=len(ctx.scanned_files),
            message="Parsing source files...",
        )
        logger.info(f"Parsing {len(ctx.scanned_files)} files")

        total_entities = 0
        for i, file_info in enumerate(ctx.scanned_files):
            try:
                parsed = ctx.parser.parse_file(file_info)
                ctx.parsed_files.append(parsed)
                total_entities += len(parsed.all_entities)

                ctx.tracker.update_stage(
                    i + 1,
                    message=f"Parsing {file_info.relative_path}",
                )
            except Exception as e:
                logger.warning(
                    f"Failed to parse {file_info.relative_path}: {e}",
                    exc_info=True,
                )

        ctx.tracker.update_stats(
            files_parsed=len(ctx.parsed_files),
            entities_found=total_entities,
        )
        logger.info(
            f"Parsed {len(ctx.parsed_files)} files, found {total_entities} entities"
        )

    async def _execute_graph_stage(self, ctx: PipelineContext) -> None:
        """Execute graph building stage."""
        ctx.tracker.set_stage(
            PipelineStage.GRAPH_BUILDING,
            total=len(ctx.parsed_files),
            message="Building knowledge graph...",
        )
        logger.info("Building knowledge graph")

        try:
            builder = GraphBuilder(ctx.memgraph)
            await builder.create_project(ctx.project_name, str(ctx.repo_path))

            files_updated = 0
            for i, parsed_file in enumerate(ctx.parsed_files):
                file_path = str(parsed_file.file_info.path)

                try:
                    needs_update = await builder.file_needs_update(
                        file_path,
                        parsed_file.file_info.content_hash,
                    )
                    ctx.file_update_status[file_path] = needs_update

                    if needs_update:
                        await builder.delete_file_entities(file_path)
                        await builder.build_from_parsed_file(parsed_file)
                        files_updated += 1
                        status = "Updated"
                    else:
                        status = "Unchanged"

                    ctx.tracker.update_stage(
                        i + 1,
                        message=f"Graph: {parsed_file.file_info.relative_path} ({status})",
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to build graph for {file_path}: {e}",
                        exc_info=True,
                    )
                    ctx.tracker.update_stage(i + 1)

            graph_stats = GraphStatistics(builder.client)
            stats = await graph_stats.get_entity_counts()
            total_nodes = sum(stats.values())
            ctx.tracker.update_stats(graph_nodes_created=total_nodes)

            logger.info(
                f"Graph building complete: {files_updated} files updated, {total_nodes} total nodes"
            )

        except Exception as e:
            logger.error(f"Graph building failed: {e}", exc_info=True)
            raise IndexingError(
                f"Graph building failed: {e}", stage="graph_building", cause=e
            )

    async def _execute_summarize_stage(self, ctx: PipelineContext) -> None:
        """Execute summarization stage."""
        files_to_summarize = [
            pf
            for pf in ctx.parsed_files
            if ctx.file_update_status.get(str(pf.file_info.path), True)
        ]

        total_items = len(files_to_summarize)
        for pf in files_to_summarize:
            total_items += len(
                [e for e in pf.all_entities if e.type.value in ("class", "function")]
            )

        if total_items == 0:
            ctx.tracker.set_stage(
                PipelineStage.SUMMARIZING,
                total=1,
                message="All summaries up to date",
            )
            ctx.tracker.update_stage(1, message="No files need summarization")
            logger.info("All summaries up to date")
            return

        ctx.tracker.set_stage(
            PipelineStage.SUMMARIZING,
            total=total_items,
            message=f"Generating AI summaries for {len(files_to_summarize)} changed files...",
        )
        logger.info(f"Generating summaries for {len(files_to_summarize)} files")

        summaries_generated = 0
        current = 0

        for parsed_file in files_to_summarize:
            try:
                file_summary = await ctx.summarizer.summarize_file(parsed_file)
                parsed_file.summary = file_summary
                summaries_generated += 1
                current += 1

                ctx.tracker.update_stage(
                    current,
                    message=f"Summarizing {parsed_file.file_info.relative_path}",
                )

                for entity in parsed_file.all_entities:
                    if entity.type.value in ("class", "function"):
                        summary = await self._summarize_entity_safely(
                            ctx, entity, parsed_file
                        )
                        if summary:
                            summaries_generated += 1

                        current += 1
                        ctx.tracker.update_stage(
                            current,
                            message=f"Summarizing {entity.name}",
                        )

            except Exception as e:
                logger.warning(
                    f"Failed to summarize {parsed_file.file_info.relative_path}: {e}",
                    exc_info=True,
                )
                current += 1 + len(
                    [e for e in parsed_file.all_entities if e.type.value in ("class", "function")]
                )
                ctx.tracker.update_stage(current)

        ctx.tracker.update_stats(summaries_generated=summaries_generated)
        logger.info(f"Generated {summaries_generated} summaries")

    async def _summarize_entity_safely(
        self, ctx: PipelineContext, entity, parsed_file: ParsedFile
    ) -> str | None:
        """Summarize an entity with error handling."""
        try:
            return await ctx.summarizer.summarize_entity(
                entity,
                parsed_file.file_info.relative_path,
                parsed_file.file_info.language.value,
            )
        except Exception as e:
            logger.warning(f"Failed to summarize entity {entity.name}: {e}", exc_info=True)
            return None

    async def _execute_embedding_stage(self, ctx: PipelineContext) -> None:
        """Execute embedding generation stage."""
        ctx.tracker.set_stage(
            PipelineStage.EMBEDDING,
            total=len(ctx.parsed_files),
            message="Generating embeddings...",
        )
        logger.info("Generating embeddings")

        chunker = CodeChunker()
        indexer = VectorIndexer(ctx.qdrant, ctx.embedder, chunker)

        total_chunks = 0
        files_embedded = 0

        for i, parsed_file in enumerate(ctx.parsed_files):
            file_path = str(parsed_file.file_info.path)
            needs_update = ctx.file_update_status.get(file_path, True)

            try:
                if needs_update:
                    chunks = await indexer.index_file(
                        parsed_file,
                        force=True,
                        project_name=ctx.project_name,
                    )
                    total_chunks += chunks
                    files_embedded += 1
                    status = f"Embedded ({chunks} chunks)"
                else:
                    status = "Unchanged"

                ctx.tracker.update_stage(
                    i + 1,
                    message=f"Embedding {parsed_file.file_info.relative_path} - {status}",
                )
            except Exception as e:
                logger.warning(
                    f"Failed to embed {parsed_file.file_info.relative_path}: {e}",
                    exc_info=True,
                )
                ctx.tracker.update_stage(i + 1)

        ctx.tracker.update_stats(chunks_embedded=total_chunks)
        logger.info(f"Embedded {files_embedded} files, {total_chunks} total chunks")


async def run_indexing(
    repo_path: str | Path,
    project_name: str | None = None,
    progress_callback: Callable | None = None,
) -> dict:
    """Convenience function to run the indexing pipeline."""
    orchestrator = PipelineOrchestrator(
        repo_path=repo_path,
        project_name=project_name,
        progress_callback=progress_callback,
    )
    return await orchestrator.run()

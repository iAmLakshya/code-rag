import asyncio
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from code_rag.config import get_settings
from code_rag.core.cache import FunctionRegistry
from code_rag.core.errors import IndexingError
from code_rag.core.types import PipelineStage
from code_rag.embeddings.chunker import CodeChunker
from code_rag.embeddings.client import QdrantManager
from code_rag.embeddings.embedder import OpenAIEmbedder
from code_rag.embeddings.indexer import VectorIndexer
from code_rag.graph.batch_builder import BatchGraphBuilder
from code_rag.graph.builder import GraphBuilder
from code_rag.graph.client import MemgraphClient
from code_rag.graph.schema import GraphSchema
from code_rag.graph.statistics import GraphStatistics
from code_rag.parsing.call_resolution import CallProcessor
from code_rag.parsing.import_processor import ImportProcessor
from code_rag.parsing.inheritance_tracker import InheritanceTracker
from code_rag.parsing.models import ParsedFile
from code_rag.parsing.parser import CodeParser
from code_rag.parsing.scanner import FileScanner
from code_rag.pipeline.progress import ProgressTracker
from code_rag.summarization.summarizer import CodeSummarizer

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    repo_path: Path
    project_name: str
    tracker: ProgressTracker
    memgraph: MemgraphClient
    qdrant: QdrantManager
    parser: CodeParser
    embedder: OpenAIEmbedder
    summarizer: CodeSummarizer

    function_registry: FunctionRegistry = field(default_factory=FunctionRegistry)
    import_processor: ImportProcessor | None = None
    inheritance_tracker: InheritanceTracker | None = None
    call_processor: CallProcessor | None = None

    parsed_files: list[ParsedFile] = field(default_factory=list)
    file_update_status: dict[str, bool] = field(default_factory=dict)
    scanned_files: list = field(default_factory=list)


class PipelineOrchestrator:
    """Orchestrates the full indexing pipeline with parallel processing."""

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
        max_workers: int | None = None,
        max_concurrent_api: int | None = None,
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

        # Parallelization settings
        settings = get_settings()
        self._max_workers = max_workers or min(os.cpu_count() or 4, 8)
        self._max_concurrent_api = max_concurrent_api or settings.indexing.max_concurrent_requests
        self._graph_semaphore: asyncio.Semaphore | None = None
        self._api_semaphore: asyncio.Semaphore | None = None

    async def _init_components(self) -> PipelineContext:
        self._graph_semaphore = asyncio.Semaphore(self._max_concurrent_api)
        self._api_semaphore = asyncio.Semaphore(self._max_concurrent_api)

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

        logger.info(
            f"Pipeline initialized with {self._max_workers} workers, "
            f"{self._max_concurrent_api} concurrent API calls"
        )

        function_registry = FunctionRegistry()
        import_processor = ImportProcessor(
            function_registry=function_registry,
            project_name=self.project_name,
            repo_path=self.repo_path,
        )
        inheritance_tracker = InheritanceTracker(
            function_registry=function_registry,
            import_processor=import_processor,
        )

        return PipelineContext(
            repo_path=self.repo_path,
            project_name=self.project_name,
            tracker=self.tracker,
            memgraph=self._memgraph,
            qdrant=self._qdrant,
            parser=self._parser,
            embedder=self._embedder,
            summarizer=self._summarizer,
            function_registry=function_registry,
            import_processor=import_processor,
            inheritance_tracker=inheritance_tracker,
        )

    async def _cleanup(self) -> None:
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
        ctx.tracker.set_stage(
            PipelineStage.PARSING,
            total=len(ctx.scanned_files),
            message=f"Parsing source files (using {self._max_workers} workers)...",
        )
        logger.info(f"Parsing {len(ctx.scanned_files)} files with {self._max_workers} workers")

        loop = asyncio.get_event_loop()
        parsed_results: list[tuple] = []

        def parse_file_sync(file_info):
            try:
                parsed = ctx.parser.parse_file(file_info)
                return (file_info, parsed, None)
            except Exception as e:
                return (file_info, None, e)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [
                loop.run_in_executor(executor, parse_file_sync, file_info)
                for file_info in ctx.scanned_files
            ]

            completed = 0
            for coro in asyncio.as_completed(futures):
                result = await coro
                parsed_results.append(result)
                completed += 1
                ctx.tracker.update_stage(
                    completed,
                    message=f"Parsed {completed}/{len(ctx.scanned_files)} files",
                )

        total_entities = 0
        for file_info, parsed, error in parsed_results:
            if error:
                logger.warning(
                    f"Failed to parse {file_info.relative_path}: {error}",
                    exc_info=True,
                )
                continue

            if parsed:
                ctx.parsed_files.append(parsed)
                total_entities += len(parsed.all_entities)

                module_qn = self._file_to_module_qn(ctx.project_name, file_info.relative_path)
                self._register_entities(ctx, parsed, module_qn)

                ast_cache = getattr(ctx.parser, '_ast_cache', None)
                if ctx.import_processor and ast_cache:
                    cached = ast_cache.get(file_info.path)
                    if cached:
                        root_node, lang = cached
                        ctx.import_processor.parse_imports(
                            root_node, module_qn, file_info.language.value
                        )

        # Initialize CallProcessor now that we have all registries populated
        if ctx.function_registry and ctx.import_processor and ctx.inheritance_tracker:
            from code_rag.parsing.type_inference.engine import TypeInferenceEngine
            type_inference = TypeInferenceEngine(
                function_registry=ctx.function_registry,
                import_mapping=ctx.import_processor.import_mapping,
            )
            ctx.call_processor = CallProcessor(
                function_registry=ctx.function_registry,
                import_processor=ctx.import_processor,
                type_inference=type_inference,
                class_inheritance=ctx.inheritance_tracker.class_inheritance,
                project_name=ctx.project_name,
                repo_path=ctx.repo_path,
            )

        ctx.tracker.update_stats(
            files_parsed=len(ctx.parsed_files),
            entities_found=total_entities,
        )
        logger.info(
            f"Parsed {len(ctx.parsed_files)} files, found {total_entities} entities"
        )

    def _file_to_module_qn(self, project_name: str, relative_path: str) -> str:
        path = Path(relative_path)
        parts = list(path.with_suffix("").parts)

        if parts and parts[-1] == "__init__":
            parts = parts[:-1]

        return f"{project_name}.{'.'.join(parts)}" if parts else project_name

    def _register_entities(
        self,
        ctx: PipelineContext,
        parsed: ParsedFile,
        module_qn: str,
    ) -> None:
        for entity in parsed.all_entities:
            if entity.qualified_name.startswith(module_qn):
                qn = entity.qualified_name
            else:
                qn = f"{module_qn}.{entity.qualified_name}"

            entity_type = entity.type.value.capitalize()
            ctx.function_registry.register(qn, entity_type)

            if entity.type.value == "class" and ctx.inheritance_tracker:
                ctx.inheritance_tracker.register_class(
                    qn, entity.base_classes, module_qn
                )

    async def _execute_graph_stage(self, ctx: PipelineContext) -> None:
        ctx.tracker.set_stage(
            PipelineStage.GRAPH_BUILDING,
            total=len(ctx.parsed_files),
            message="Building knowledge graph (batched)...",
        )
        logger.info(
            f"Building knowledge graph with batched operations "
            f"(batch_size={self._max_concurrent_api * 100})"
        )

        try:
            legacy_builder = GraphBuilder(
                ctx.memgraph,
                call_processor=ctx.call_processor,
                project_name=ctx.project_name,
            )
            await legacy_builder.create_project(ctx.project_name, str(ctx.repo_path))

            async def check_file_update(parsed_file):
                file_path = str(parsed_file.file_info.path)
                async with self._graph_semaphore:
                    needs_update = await legacy_builder.file_needs_update(
                        file_path,
                        parsed_file.file_info.content_hash,
                    )
                return (parsed_file, file_path, needs_update)

            check_tasks = [check_file_update(pf) for pf in ctx.parsed_files]
            check_results = await asyncio.gather(*check_tasks, return_exceptions=True)

            files_to_update = []
            for result in check_results:
                if isinstance(result, Exception):
                    logger.warning(f"Failed to check file update status: {result}")
                    continue
                parsed_file, file_path, needs_update = result
                ctx.file_update_status[file_path] = needs_update
                if needs_update:
                    files_to_update.append(parsed_file)

            logger.info(f"{len(files_to_update)} files need graph updates")

            for parsed_file in files_to_update:
                file_path = str(parsed_file.file_info.path)
                try:
                    await legacy_builder.delete_file_entities(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete old entities for {file_path}: {e}")

            batch_size = self._max_concurrent_api * 100
            async with BatchGraphBuilder(
                ctx.memgraph,
                call_processor=ctx.call_processor,
                project_name=ctx.project_name,
                batch_size=batch_size,
            ) as batch_builder:
                completed = len(ctx.parsed_files) - len(files_to_update)
                files_updated = 0

                for parsed_file in files_to_update:
                    try:
                        await batch_builder.add_parsed_file(parsed_file)
                        files_updated += 1
                        completed += 1
                        ctx.tracker.update_stage(
                            completed,
                            message=f"Graph: {parsed_file.file_info.relative_path} (Buffered)",
                        )
                    except Exception as e:
                        completed += 1
                        logger.warning(
                            f"Failed to add {parsed_file.file_info.relative_path} to graph: {e}"
                        )
                        ctx.tracker.update_stage(
                            completed,
                            message=f"Graph: {parsed_file.file_info.relative_path} (Failed)",
                        )

                logger.info(f"Flushing {files_updated} files to graph database...")

            graph_stats = GraphStatistics(legacy_builder.client)
            stats = await graph_stats.get_entity_counts()
            total_nodes = sum(stats.values())
            ctx.tracker.update_stats(graph_nodes_created=total_nodes)

            logger.info(
                f"Graph building complete: {files_updated} files updated, "
                f"{total_nodes} total nodes (batched mode)"
            )

        except Exception as e:
            logger.error(f"Graph building failed: {e}", exc_info=True)
            raise IndexingError(
                f"Graph building failed: {e}", stage="graph_building", cause=e
            )

    async def _execute_summarize_stage(self, ctx: PipelineContext) -> None:
        files_to_summarize = [
            pf
            for pf in ctx.parsed_files
            if ctx.file_update_status.get(str(pf.file_info.path), True)
        ]

        # Collect all summarization tasks
        summarize_tasks = []
        for pf in files_to_summarize:
            # File summary task
            summarize_tasks.append(("file", pf, None))
            # Entity summary tasks
            for entity in pf.all_entities:
                if entity.type.value in ("class", "function"):
                    summarize_tasks.append(("entity", pf, entity))

        total_items = len(summarize_tasks)

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
            message=f"Generating AI summaries ({self._max_concurrent_api} concurrent)...",
        )
        logger.info(
            f"Generating {total_items} summaries with {self._max_concurrent_api} concurrent API calls"
        )

        summaries_generated = 0
        completed = 0

        async def summarize_item(task_type, parsed_file, entity):
            async with self._api_semaphore:
                try:
                    if task_type == "file":
                        summary = await ctx.summarizer.summarize_file(parsed_file)
                        parsed_file.summary = summary
                        return ("file", parsed_file.file_info.relative_path, True)
                    else:
                        summary = await ctx.summarizer.summarize_entity(
                            entity,
                            parsed_file.file_info.relative_path,
                            parsed_file.file_info.language.value,
                        )
                        return ("entity", entity.name, summary is not None)
                except Exception as e:
                    name = parsed_file.file_info.relative_path if task_type == "file" else entity.name
                    logger.warning(f"Failed to summarize {name}: {e}")
                    return (task_type, name, False)

        batch_size = self._max_concurrent_api * 3
        for i in range(0, len(summarize_tasks), batch_size):
            batch = summarize_tasks[i:i + batch_size]
            batch_coros = [summarize_item(*task) for task in batch]
            batch_results = await asyncio.gather(*batch_coros, return_exceptions=True)

            for result in batch_results:
                completed += 1
                if isinstance(result, Exception):
                    logger.warning(f"Summarization exception: {result}")
                    continue

                task_type, name, success = result
                if success:
                    summaries_generated += 1

                ctx.tracker.update_stage(
                    completed,
                    message=f"Summarized {name}",
                )

        ctx.tracker.update_stats(summaries_generated=summaries_generated)
        logger.info(f"Generated {summaries_generated} summaries")

    async def _summarize_entity_safely(
        self, ctx: PipelineContext, entity, parsed_file: ParsedFile
    ) -> str | None:
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
        files_to_embed = [
            pf for pf in ctx.parsed_files
            if ctx.file_update_status.get(str(pf.file_info.path), True)
        ]

        ctx.tracker.set_stage(
            PipelineStage.EMBEDDING,
            total=len(ctx.parsed_files),
            message=f"Generating embeddings ({self._max_concurrent_api} concurrent)...",
        )
        logger.info(
            f"Generating embeddings for {len(files_to_embed)} files "
            f"with {self._max_concurrent_api} concurrent operations"
        )

        chunker = CodeChunker()
        indexer = VectorIndexer(ctx.qdrant, ctx.embedder, chunker)

        total_chunks = 0
        files_embedded = 0
        completed = len(ctx.parsed_files) - len(files_to_embed)

        async def embed_file(parsed_file):
            async with self._api_semaphore:
                try:
                    chunks = await indexer.index_file(
                        parsed_file,
                        force=True,
                        project_name=ctx.project_name,
                    )
                    return (parsed_file, chunks, None)
                except Exception as e:
                    return (parsed_file, 0, e)

        batch_size = self._max_concurrent_api * 2
        for i in range(0, len(files_to_embed), batch_size):
            batch = files_to_embed[i:i + batch_size]
            batch_coros = [embed_file(pf) for pf in batch]
            batch_results = await asyncio.gather(*batch_coros, return_exceptions=True)

            for result in batch_results:
                completed += 1
                if isinstance(result, Exception):
                    logger.warning(f"Embedding exception: {result}")
                    ctx.tracker.update_stage(completed)
                    continue

                parsed_file, chunks, error = result
                if error:
                    logger.warning(
                        f"Failed to embed {parsed_file.file_info.relative_path}: {error}"
                    )
                    status = "Failed"
                else:
                    total_chunks += chunks
                    files_embedded += 1
                    status = f"Embedded ({chunks} chunks)"

                ctx.tracker.update_stage(
                    completed,
                    message=f"Embedding {parsed_file.file_info.relative_path} - {status}",
                )

        ctx.tracker.update_stats(chunks_embedded=total_chunks)
        logger.info(f"Embedded {files_embedded} files, {total_chunks} total chunks")


async def run_indexing(
    repo_path: str | Path,
    project_name: str | None = None,
    progress_callback: Callable | None = None,
) -> dict:
    orchestrator = PipelineOrchestrator(
        repo_path=repo_path,
        project_name=project_name,
        progress_callback=progress_callback,
    )
    return await orchestrator.run()

# Code RAG

A hybrid Retrieval-Augmented Generation system for code repositories that combines **graph-based structural search** with **vector-based semantic search** for intelligent code understanding.

## Overview

Code RAG builds a knowledge graph of your codebase using Memgraph, enabling powerful structural queries (call chains, inheritance hierarchies, imports), while simultaneously creating vector embeddings in Qdrant for semantic search. The combination allows natural language queries that understand both code structure and meaning.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INDEXING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Scanner → Parser → Graph Builder → Summarizer → Chunker → Embedder         │
│     │         │           │             │           │          │            │
│   Files   Tree-sitter  Memgraph      OpenAI     Tokens     Qdrant           │
│           AST Parse    Nodes/Rels    Summaries  Splitting  Vectors          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                               QUERY PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Question → Planner → [Graph Search + Vector Search] → Ranker → Response    │
│      │         │              │              │            │          │      │
│   Natural   Intent      Cypher Queries    Semantic     Hybrid       LLM     │
│   Language  Analysis    Relationships     Similarity   Scoring   Synthesis  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Hybrid Search

- **Graph Search**: Traverse call chains, inheritance hierarchies, import dependencies
- **Vector Search**: Semantic similarity for intent-based code discovery
- **Hybrid Ranking**: Combines structural relevance with semantic similarity

### Multi-Language Support

- Python (`.py`)
- JavaScript (`.js`, `.jsx`)
- TypeScript (`.ts`, `.tsx`)

### Call Resolution

- IIFE detection (JavaScript)
- Super call resolution with inheritance traversal
- Method chain resolution with type inference
- Import-based resolution (direct, aliased, wildcard)
- Multi-separator support (`.`, `::`, `:`)
- Builtin detection (Python, JavaScript, Java, Rust)

### Type Inference

- Python type annotations and parameter inference
- TypeScript type annotations
- JavaScript expression type inference
- Class instantiation tracking

### Embedding Providers

- **OpenAI**: `text-embedding-3-small` (default)
- **UniXcoder**: Code-specific embeddings (768d, local GPU)
- **Ollama**: Local model support
- **Google**: Gemini embeddings

### LLM Providers

- OpenAI (GPT-4o, GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- Ollama (local models)

### Performance

- Batched graph operations with UNWIND queries
- Concurrent API calls with semaphore control
- LRU caching with memory bounds
- Incremental indexing via content hashing
- Real-time file watching with CALLS recalculation

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
OPENAI_API_KEY=sk-...
```

### 3. Start Infrastructure

```bash
docker-compose up -d
```

This starts:

- **Memgraph**: Graph database (port 7687)
- **Memgraph Lab**: Visual query interface (port 3000)
- **Qdrant**: Vector database (port 6333)

### 4. Index a Repository

```bash
code-rag index /path/to/repo --name "my-project"
```

### 5. Query

```bash
code-rag query "How does authentication work?"
```

## CLI Commands

```bash
code-rag                                    # Launch TUI
code-rag index <path> --name <project>     # Index repository
code-rag query "<question>"                 # Query codebase
code-rag search "<text>"                    # Semantic search
code-rag status                             # Show statistics
```

## MCP Integration

Code RAG includes an MCP server for Claude Code integration:

**Tools Available:**

- `index_repository`: Index a codebase into the knowledge graph
- `query_code_graph`: Natural language questions about the code
- `get_code_snippet`: Retrieve source code by qualified name
- `semantic_search`: Find code by functionality/intent

## Architecture

### Graph Schema

**Nodes:**

- `Project`: Indexed project
- `File`: Source files
- `Class`: Class definitions
- `Function`: Module-level functions
- `Method`: Class methods
- `Import`: Import statements

**Relationships:**

- `DEFINES`: File → Entity
- `DEFINES_METHOD`: Class → Method
- `EXTENDS`: Class → Parent Class
- `IMPORTS`: File → Import
- `CALLS`: Function/Method → Function/Method

### Key Components

| Module                     | Purpose                      |
| -------------------------- | ---------------------------- |
| `parsing/parser.py`        | Tree-sitter AST parsing      |
| `parsing/extractors/`      | Language-specific extraction |
| `parsing/call_resolution/` | Function call resolution     |
| `parsing/type_inference/`  | Variable type inference      |
| `graph/builder.py`         | Graph construction           |
| `graph/batch_builder.py`   | Batched graph operations     |
| `embeddings/chunker.py`    | Token-based code chunking    |
| `embeddings/embedder.py`   | Embedding generation         |
| `embeddings/indexer.py`    | Vector indexing              |
| `query/query_planner.py`   | Intent analysis              |
| `query/graph_reasoning/`   | Graph traversal              |
| `query/vector_search.py`   | Semantic search              |
| `query/ranking/`           | Hybrid result ranking        |
| `query/responder.py`       | LLM response generation      |
| `pipeline/orchestrator.py` | Indexing coordination        |
| `pipeline/watcher.py`      | Real-time file watching      |
| `mcp/`                     | MCP server for Claude Code   |

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Testing

```bash
pytest                           # Run all tests
pytest tests/test_parsing.py    # Run specific test file
pytest -v -k "test_function"    # Run specific test
```

### Linting

```bash
ruff check src/code_rag         # Lint
ruff format src/code_rag        # Format
mypy src/code_rag               # Type check
```

### Project Structure

```
src/code_rag/
├── core/                 # Cache, errors, types
├── config/               # Settings management
├── parsing/              # AST parsing & extraction
│   ├── extractors/       # Language extractors
│   ├── call_resolution/  # Call resolution
│   └── type_inference/   # Type inference
├── graph/                # Memgraph operations
├── embeddings/           # Vector operations
├── query/                # Query pipeline
│   ├── graph_reasoning/  # Graph traversal
│   ├── ranking/          # Result ranking
│   └── context/          # Context building
├── pipeline/             # Indexing pipeline
├── providers/            # LLM/Embedding providers
├── projects/             # Project management
├── summarization/        # Entity summarization
├── mcp/                  # MCP server
└── main.py               # CLI entrypoint
```

## Requirements

- Python 3.11+
- Docker (for Memgraph and Qdrant)

## License

MIT

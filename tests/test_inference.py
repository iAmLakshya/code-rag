"""End-to-end inference validation tests.

This module tests the complete RAG pipeline by running predefined questions
on the sample codebase and validating that the system returns expected results.

The tests verify:
1. Entity extraction accuracy - Can we find the right code entities?
2. Relationship detection - Are call graphs, inheritance, etc. correctly identified?
3. Semantic search quality - Do vector searches return relevant results?
4. Query type classification - Are questions correctly categorized?
5. Response quality - Do LLM responses contain expected information?
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Callable

from code_rag.core.types import EntityType, Language, QueryType
from code_rag.parsing.scanner import FileScanner
from code_rag.parsing.parser import CodeParser
from code_rag.parsing.models import ParsedFile
from code_rag.query.engine import QueryAnalyzer
from code_rag.query.reranker import ResultReranker, SearchResult
from code_rag.embeddings.chunker import CodeChunker


# ============================================================================
# Test Infrastructure
# ============================================================================

@dataclass
class InferenceTestCase:
    """Represents a test case for inference validation."""
    question: str
    expected_query_type: QueryType
    expected_entities: list[str]  # Entities that should be extracted from question
    expected_in_results: list[str]  # Strings that should appear in search results
    expected_files: list[str]  # Files that should be found
    description: str


@dataclass
class ParsingTestCase:
    """Represents a test case for parsing validation."""
    file_path: str
    expected_classes: list[str]
    expected_functions: list[str]
    expected_methods: list[str]
    expected_imports: list[str]
    expected_inheritance: dict[str, list[str]]  # class -> base classes


# ============================================================================
# Sample Project Test Cases
# ============================================================================

# These test cases are based on the sample_project fixture structure
PARSING_TEST_CASES = [
    ParsingTestCase(
        file_path="src/models/user.py",
        expected_classes=["User", "UserRepository"],
        expected_functions=[],
        expected_methods=["verify_password", "update_password", "to_dict", "find_by_id", "find_by_email", "create", "__init__"],
        expected_imports=["BaseModel", "dataclass", "datetime", "Optional"],
        expected_inheritance={"User": ["BaseModel"]},
    ),
    ParsingTestCase(
        file_path="src/api/auth.py",
        expected_classes=["AuthToken", "AuthenticationError", "AuthService"],
        expected_functions=[],
        expected_methods=["login", "logout", "verify_token", "register", "_create_token", "__init__"],
        expected_imports=["User", "UserRepository", "generate_token", "hash_password", "datetime", "timedelta"],
        expected_inheritance={"AuthenticationError": ["Exception"]},
    ),
]

INFERENCE_TEST_CASES = [
    # Structural Queries
    InferenceTestCase(
        question="Show me the User class methods",
        expected_query_type=QueryType.NAVIGATIONAL,  # "show me" triggers navigational
        expected_entities=[],  # "User" is not CamelCase after "the"
        expected_in_results=["verify_password", "update_password", "to_dict"],
        expected_files=["user.py"],
        description="Find methods of User class",
    ),
    InferenceTestCase(
        question="What calls the verify_password method?",
        expected_query_type=QueryType.STRUCTURAL,
        expected_entities=["verify_password"],
        expected_in_results=["login", "AuthService"],
        expected_files=["auth.py"],
        description="Find callers of verify_password",
    ),
    InferenceTestCase(
        question="What class extends BaseModel?",
        expected_query_type=QueryType.STRUCTURAL,
        expected_entities=["BaseModel"],
        expected_in_results=["User"],
        expected_files=["user.py"],
        description="Find class hierarchy",
    ),

    # Navigational Queries
    InferenceTestCase(
        question="Show me the AuthService class",
        expected_query_type=QueryType.NAVIGATIONAL,
        expected_entities=["AuthService"],
        expected_in_results=["AuthService", "login", "logout"],
        expected_files=["auth.py"],
        description="Locate AuthService",
    ),
    InferenceTestCase(
        question="Find the UserRepository class",
        expected_query_type=QueryType.NAVIGATIONAL,
        expected_entities=["UserRepository"],
        expected_in_results=["UserRepository", "find_by_id"],
        expected_files=["user.py"],
        description="Locate UserRepository",
    ),
    InferenceTestCase(
        question="Where is the login function defined?",
        expected_query_type=QueryType.NAVIGATIONAL,
        expected_entities=[],
        expected_in_results=["login"],
        expected_files=["auth.py"],
        description="Locate login function",
    ),

    # Explanatory Queries
    InferenceTestCase(
        question="How does the authentication flow work?",
        expected_query_type=QueryType.EXPLANATORY,
        expected_entities=[],
        expected_in_results=["AuthService", "login", "token"],
        expected_files=["auth.py"],
        description="Explain authentication",
    ),
    InferenceTestCase(
        question="Explain the password verification process",
        expected_query_type=QueryType.EXPLANATORY,
        expected_entities=[],
        expected_in_results=["verify_password", "hash"],
        expected_files=["user.py"],
        description="Explain password verification",
    ),
    InferenceTestCase(
        question="Explain what to_dict method does",
        expected_query_type=QueryType.EXPLANATORY,  # "explain" keyword
        expected_entities=["to_dict"],
        expected_in_results=["to_dict", "dictionary"],
        expected_files=["user.py"],
        description="Explain to_dict method",
    ),

    # Semantic Queries
    InferenceTestCase(
        question="Code related to user authentication",
        expected_query_type=QueryType.SEMANTIC,
        expected_entities=[],
        expected_in_results=["auth", "login", "token"],
        expected_files=["auth.py"],
        description="Semantic search for authentication",
    ),
    InferenceTestCase(
        question="Database operations for users",
        expected_query_type=QueryType.SEMANTIC,
        expected_entities=[],
        expected_in_results=["find_by", "create"],
        expected_files=["user.py"],
        description="Semantic search for database ops",
    ),
]


# ============================================================================
# Parsing Validation Tests
# ============================================================================

class TestParsingInference:
    """Tests that validate correct parsing of sample project."""

    @pytest.fixture
    def sample_project_path(self) -> Path:
        return Path(__file__).parent / "fixtures" / "sample_project"

    @pytest.fixture
    def parser(self):
        return CodeParser()

    @pytest.fixture
    def scanner(self, sample_project_path):
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")
        return FileScanner(sample_project_path)

    def test_parse_user_model_entities(self, sample_project_path, parser, scanner):
        """Validate User model parsing produces expected entities."""
        user_file = sample_project_path / "src" / "models" / "user.py"
        if not user_file.exists():
            pytest.skip("User file not found")

        file_info = next((f for f in scanner.scan_all() if f.path == user_file), None)
        assert file_info is not None, "User file should be discovered by scanner"

        parsed = parser.parse_file(file_info)

        # Validate classes
        class_names = [e.name for e in parsed.entities if e.type == EntityType.CLASS]
        assert "User" in class_names, "User class should be extracted"
        assert "UserRepository" in class_names, "UserRepository should be extracted"

        # Validate User class methods
        user_class = next(e for e in parsed.entities if e.name == "User")
        method_names = [m.name for m in user_class.children]
        assert "verify_password" in method_names, "verify_password method should exist"
        assert "update_password" in method_names, "update_password method should exist"
        assert "to_dict" in method_names, "to_dict method should exist"

        # Validate inheritance
        assert "BaseModel" in user_class.base_classes, "User should extend BaseModel"

    def test_parse_auth_service_entities(self, sample_project_path, parser, scanner):
        """Validate AuthService parsing produces expected entities."""
        auth_file = sample_project_path / "src" / "api" / "auth.py"
        if not auth_file.exists():
            pytest.skip("Auth file not found")

        file_info = next((f for f in scanner.scan_all() if f.path == auth_file), None)
        parsed = parser.parse_file(file_info)

        # Validate classes
        class_names = [e.name for e in parsed.entities if e.type == EntityType.CLASS]
        assert "AuthService" in class_names, "AuthService should be extracted"
        assert "AuthToken" in class_names, "AuthToken should be extracted"
        assert "AuthenticationError" in class_names, "AuthenticationError should be extracted"

        # Validate AuthService methods
        auth_service = next(e for e in parsed.entities if e.name == "AuthService")
        method_names = [m.name for m in auth_service.children]
        assert "login" in method_names, "login method should exist"
        assert "logout" in method_names, "logout method should exist"
        assert "verify_token" in method_names, "verify_token method should exist"
        assert "register" in method_names, "register method should exist"

        # Validate async methods
        login_method = next(m for m in auth_service.children if m.name == "login")
        assert login_method.is_async, "login should be async"

    def test_parse_typescript_components(self, sample_project_path, parser, scanner):
        """Validate TypeScript component parsing."""
        login_form = sample_project_path / "frontend" / "components" / "LoginForm.tsx"
        if not login_form.exists():
            pytest.skip("LoginForm.tsx not found")

        file_info = next((f for f in scanner.scan_all() if f.path == login_form), None)
        parsed = parser.parse_file(file_info)

        # Should have interfaces
        interface_names = [e.name for e in parsed.entities if e.type == EntityType.INTERFACE]
        assert len(interface_names) > 0, "Should extract TypeScript interfaces"

        # Should have the component function
        func_names = [e.name for e in parsed.entities if e.type == EntityType.FUNCTION]
        assert len(func_names) > 0 or "LoginForm" in [e.name for e in parsed.entities], \
            "Should extract LoginForm component"

    def test_parse_imports_correctly(self, sample_project_path, parser, scanner):
        """Validate import extraction."""
        auth_file = sample_project_path / "src" / "api" / "auth.py"
        if not auth_file.exists():
            pytest.skip("Auth file not found")

        file_info = next((f for f in scanner.scan_all() if f.path == auth_file), None)
        parsed = parser.parse_file(file_info)

        import_names = [imp.name for imp in parsed.imports]

        # Should import from models
        assert "User" in import_names or "UserRepository" in import_names, \
            "Should import from models"

        # Should import utility functions
        assert "generate_token" in import_names or "hash_password" in import_names, \
            "Should import crypto utilities"

    def test_parse_function_calls(self, sample_project_path, parser, scanner):
        """Validate function call extraction."""
        auth_file = sample_project_path / "src" / "api" / "auth.py"
        if not auth_file.exists():
            pytest.skip("Auth file not found")

        file_info = next((f for f in scanner.scan_all() if f.path == auth_file), None)
        parsed = parser.parse_file(file_info)

        auth_service = next(e for e in parsed.entities if e.name == "AuthService")
        login_method = next(m for m in auth_service.children if m.name == "login")

        # login should call verify_password (through user.verify_password)
        all_calls = login_method.calls
        assert len(all_calls) > 0, "login method should have function calls"

    @pytest.mark.parametrize("test_case", PARSING_TEST_CASES, ids=lambda tc: tc.file_path)
    def test_parsing_test_cases(self, sample_project_path, parser, scanner, test_case):
        """Run parameterized parsing test cases."""
        file_path = sample_project_path / test_case.file_path
        if not file_path.exists():
            pytest.skip(f"File not found: {test_case.file_path}")

        file_info = next((f for f in scanner.scan_all() if f.path == file_path), None)
        parsed = parser.parse_file(file_info)

        # Check expected classes
        class_names = [e.name for e in parsed.entities if e.type == EntityType.CLASS]
        for expected_class in test_case.expected_classes:
            assert expected_class in class_names, \
                f"Expected class {expected_class} not found in {test_case.file_path}"

        # Check expected methods (across all classes)
        all_methods = []
        for entity in parsed.entities:
            if hasattr(entity, 'children') and entity.children:
                all_methods.extend([m.name for m in entity.children])

        for expected_method in test_case.expected_methods:
            assert expected_method in all_methods, \
                f"Expected method {expected_method} not found in {test_case.file_path}"

        # Check inheritance
        for class_name, base_classes in test_case.expected_inheritance.items():
            cls = next((e for e in parsed.entities if e.name == class_name), None)
            if cls:
                for base in base_classes:
                    assert base in cls.base_classes, \
                        f"Expected {class_name} to extend {base}"


# ============================================================================
# Query Analysis Inference Tests
# ============================================================================

class TestQueryAnalysisInference:
    """Tests that validate query analysis produces expected results."""

    @pytest.fixture
    def analyzer(self):
        return QueryAnalyzer()

    @pytest.mark.parametrize("test_case", INFERENCE_TEST_CASES, ids=lambda tc: tc.description)
    def test_query_type_classification(self, analyzer, test_case):
        """Validate query type is correctly classified."""
        analysis = analyzer.analyze(test_case.question)

        assert analysis.query_type == test_case.expected_query_type, \
            f"Query '{test_case.question}' should be classified as {test_case.expected_query_type.name}"

    @pytest.mark.parametrize("test_case", INFERENCE_TEST_CASES, ids=lambda tc: tc.description)
    def test_entity_extraction(self, analyzer, test_case):
        """Validate entities are correctly extracted from questions."""
        if not test_case.expected_entities:
            pytest.skip("No expected entities for this test case")

        analysis = analyzer.analyze(test_case.question)

        for expected_entity in test_case.expected_entities:
            assert expected_entity in analysis.entities, \
                f"Entity '{expected_entity}' should be extracted from '{test_case.question}'"

    def test_camel_case_extraction_comprehensive(self, analyzer):
        """Test comprehensive CamelCase extraction."""
        questions = [
            ("What is UserService?", ["UserService"]),
            ("How do AuthService and UserRepository interact?", ["AuthService", "UserRepository"]),
            ("Show me LoginFormComponent", ["LoginFormComponent"]),
        ]

        for question, expected in questions:
            analysis = analyzer.analyze(question)
            for entity in expected:
                assert entity in analysis.entities, \
                    f"Should extract {entity} from '{question}'"

    def test_snake_case_extraction_comprehensive(self, analyzer):
        """Test comprehensive snake_case extraction."""
        questions = [
            ("What does process_data do?", ["process_data"]),
            ("Is validate_input called by process_data?", ["validate_input", "process_data"]),
            ("Find find_by_email", ["find_by_email"]),
        ]

        for question, expected in questions:
            analysis = analyzer.analyze(question)
            for entity in expected:
                assert entity in analysis.entities, \
                    f"Should extract {entity} from '{question}'"


# ============================================================================
# Chunking Inference Tests
# ============================================================================

class TestChunkingInference:
    """Tests that validate chunking produces expected results."""

    @pytest.fixture
    def sample_project_path(self) -> Path:
        return Path(__file__).parent / "fixtures" / "sample_project"

    @pytest.fixture
    def parser(self):
        return CodeParser()

    @pytest.fixture
    def chunker(self):
        return CodeChunker(max_tokens=500, overlap_tokens=50)

    def test_chunk_preserves_entity_content(self, sample_project_path, parser, chunker):
        """Validate that chunking preserves essential entity content."""
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")

        scanner = FileScanner(sample_project_path)
        user_file = sample_project_path / "src" / "models" / "user.py"

        if not user_file.exists():
            pytest.skip("User file not found")

        file_info = next((f for f in scanner.scan_all() if f.path == user_file), None)
        parsed = parser.parse_file(file_info)
        chunks = chunker.chunk_file(parsed)

        # Collect all chunk content
        all_content = " ".join(c.content for c in chunks)

        # Key content should be preserved
        assert "verify_password" in all_content, "verify_password should be in chunks"
        assert "password" in all_content.lower(), "Password logic should be preserved"

    def test_chunk_metadata_accuracy(self, sample_project_path, parser, chunker):
        """Validate chunk metadata is accurate."""
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")

        scanner = FileScanner(sample_project_path)
        auth_file = sample_project_path / "src" / "api" / "auth.py"

        if not auth_file.exists():
            pytest.skip("Auth file not found")

        file_info = next((f for f in scanner.scan_all() if f.path == auth_file), None)
        parsed = parser.parse_file(file_info)
        chunks = chunker.chunk_file(parsed)

        for chunk in chunks:
            # All chunks should have valid metadata
            assert chunk.file_path, "Chunk should have file_path"
            assert chunk.entity_type, "Chunk should have entity_type"
            assert chunk.entity_name, "Chunk should have entity_name"
            assert chunk.language == "python", "Language should be python"
            assert chunk.start_line > 0, "Start line should be positive"


# ============================================================================
# Reranker Inference Tests
# ============================================================================

class TestRerankerInference:
    """Tests that validate reranker produces expected results."""

    @pytest.fixture
    def reranker(self):
        return ResultReranker()

    def test_hybrid_results_ranked_higher(self, reranker):
        """Validate that hybrid results are ranked above single-source results."""
        graph_results = [
            {
                "name": "AuthService",
                "qualified_name": "AuthService",
                "type": "Class",
                "file_path": "/project/auth.py",
                "start_line": 10,
            }
        ]

        vector_results = [
            {
                "score": 0.8,
                "file_path": "/project/auth.py",
                "entity_type": "class",
                "entity_name": "AuthService",
                "content": "class AuthService: pass",
                "start_line": 10,
            },
            {
                "score": 0.9,
                "file_path": "/project/other.py",
                "entity_type": "function",
                "entity_name": "other_func",
                "content": "def other_func(): pass",
                "start_line": 5,
            },
        ]

        fused = reranker.fuse_results(graph_results, vector_results)

        # AuthService should be first (hybrid)
        assert fused[0].entity_name == "AuthService", \
            "Hybrid result should be ranked highest"
        assert fused[0].source == "hybrid", \
            "Result should be marked as hybrid"

    def test_deduplication_preserves_best_score(self, reranker):
        """Validate deduplication keeps the first occurring result (already sorted by score)."""
        # Results should be sorted by score descending before deduplication
        results = [
            SearchResult(
                source="vector", score=0.9,
                file_path="/auth.py", entity_type="class",
                entity_name="AuthService", start_line=10, end_line=50,
            ),
            SearchResult(
                source="vector", score=0.5,
                file_path="/auth.py", entity_type="class",
                entity_name="AuthService", start_line=10, end_line=50,
            ),
        ]

        deduped = reranker.deduplicate(results)

        assert len(deduped) == 1
        assert deduped[0].score == 0.9, "Should keep first (higher) score"


# ============================================================================
# End-to-End Inference Tests (Mocked LLM)
# ============================================================================

class TestEndToEndInference:
    """End-to-end tests with mocked external services."""

    @pytest.fixture
    def sample_project_path(self) -> Path:
        return Path(__file__).parent / "fixtures" / "sample_project"

    @pytest.fixture
    def parsed_files(self, sample_project_path) -> list[ParsedFile]:
        """Parse all files in sample project."""
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")

        scanner = FileScanner(sample_project_path)
        parser = CodeParser()

        parsed = []
        for file_info in scanner.scan_all():
            try:
                parsed.append(parser.parse_file(file_info))
            except Exception:
                continue

        return parsed

    def test_can_find_auth_service(self, parsed_files):
        """Test that we can find AuthService in parsed files."""
        auth_service_found = False

        for parsed in parsed_files:
            for entity in parsed.entities:
                if entity.name == "AuthService":
                    auth_service_found = True
                    break

        assert auth_service_found, "AuthService should be found in parsed files"

    def test_can_find_user_methods(self, parsed_files):
        """Test that we can find User class methods."""
        user_class = None

        for parsed in parsed_files:
            for entity in parsed.entities:
                if entity.name == "User" and entity.type == EntityType.CLASS:
                    user_class = entity
                    break

        assert user_class is not None, "User class should be found"
        method_names = [m.name for m in user_class.children]
        assert "verify_password" in method_names

    def test_can_build_search_context(self, parsed_files):
        """Test that we can build search context from parsed files."""
        chunker = CodeChunker()
        all_chunks = []

        for parsed in parsed_files:
            chunks = chunker.chunk_file(parsed)
            all_chunks.extend(chunks)

        # Should have meaningful chunks
        assert len(all_chunks) > 10, "Should have multiple chunks"

        # Should be able to find auth-related chunks
        auth_chunks = [c for c in all_chunks if "auth" in c.content.lower()]
        assert len(auth_chunks) > 0, "Should have auth-related chunks"

    def test_search_results_contain_expected_content(self, parsed_files):
        """Test that search results would contain expected content."""
        chunker = CodeChunker()
        reranker = ResultReranker()

        all_chunks = []
        for parsed in parsed_files:
            chunks = chunker.chunk_file(parsed)
            all_chunks.extend(chunks)

        # Simulate vector search results (mock scores based on content matching)
        query = "authentication login"
        vector_results = []

        for chunk in all_chunks:
            content_lower = chunk.content.lower()
            score = 0.0

            if "login" in content_lower:
                score += 0.4
            if "auth" in content_lower:
                score += 0.3
            if "token" in content_lower:
                score += 0.2
            if "password" in content_lower:
                score += 0.1

            if score > 0:
                vector_results.append({
                    "score": score,
                    "file_path": chunk.file_path,
                    "entity_type": chunk.entity_type,
                    "entity_name": chunk.entity_name,
                    "content": chunk.content,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                })

        # Fuse with empty graph results
        fused = reranker.fuse_results([], vector_results)

        # Top results should include auth-related code
        assert len(fused) > 0, "Should have search results"

        top_result = fused[0]
        assert "auth" in top_result.file_path.lower() or "auth" in top_result.content.lower(), \
            "Top result should be auth-related"


# ============================================================================
# Regression Tests
# ============================================================================

class TestInferenceRegression:
    """Regression tests to catch inference issues."""

    def test_analyzer_handles_special_characters(self):
        """Ensure analyzer handles special characters in queries."""
        analyzer = QueryAnalyzer()

        # Should not raise
        analysis = analyzer.analyze("What does `__init__` do?")
        assert analysis is not None

        analysis = analyzer.analyze("Find @decorator usage")
        assert analysis is not None

    def test_chunker_handles_empty_entities(self):
        """Ensure chunker handles files with no entities."""
        chunker = CodeChunker()

        # Create a proper FileInfo mock
        file_info = MagicMock()
        file_info.path = Path("/project/empty.py")
        file_info.language = Language.PYTHON
        file_info.content_hash = "empty"

        # Mock the all_entities property
        parsed = MagicMock()
        parsed.file_info = file_info
        parsed.content = "# Just comments"
        parsed.all_entities = []  # No entities

        # Should not raise - returns empty list when no entities
        chunks = chunker.chunk_file(parsed)
        assert isinstance(chunks, list)

    def test_reranker_handles_missing_fields(self):
        """Ensure reranker handles results with missing optional fields."""
        reranker = ResultReranker()

        # Minimal results
        graph_results = [{"name": "Test", "file_path": "/test.py"}]
        vector_results = [{"score": 0.5, "file_path": "/test.py", "entity_name": "Test"}]

        # Should not raise
        fused = reranker.fuse_results(graph_results, vector_results)
        assert isinstance(fused, list)

    def test_query_type_priority(self):
        """Test that query type detection has correct priority."""
        analyzer = QueryAnalyzer()

        # Structural keywords should take priority
        analysis = analyzer.analyze("What calls show me the function?")
        assert analysis.query_type == QueryType.STRUCTURAL

        # Explanatory should override semantic
        analysis = analyzer.analyze("How does this search work?")
        assert analysis.query_type == QueryType.EXPLANATORY


# ============================================================================
# Performance Tests
# ============================================================================

class TestInferencePerformance:
    """Performance tests for inference pipeline."""

    @pytest.fixture
    def sample_project_path(self) -> Path:
        return Path(__file__).parent / "fixtures" / "sample_project"

    def test_parsing_performance(self, sample_project_path):
        """Test that parsing completes in reasonable time."""
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")

        import time

        scanner = FileScanner(sample_project_path)
        parser = CodeParser()

        start = time.time()

        files_parsed = 0
        for file_info in scanner.scan_all():
            parser.parse_file(file_info)
            files_parsed += 1

        elapsed = time.time() - start

        # Should parse sample project in under 5 seconds
        assert elapsed < 5.0, f"Parsing took too long: {elapsed:.2f}s for {files_parsed} files"

    def test_chunking_performance(self, sample_project_path):
        """Test that chunking completes in reasonable time."""
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")

        import time

        scanner = FileScanner(sample_project_path)
        parser = CodeParser()
        chunker = CodeChunker()

        parsed_files = [parser.parse_file(f) for f in scanner.scan_all()]

        start = time.time()

        for parsed in parsed_files:
            chunker.chunk_file(parsed)

        elapsed = time.time() - start

        # Should chunk in under 2 seconds
        assert elapsed < 2.0, f"Chunking took too long: {elapsed:.2f}s"

    def test_query_analysis_performance(self):
        """Test that query analysis is fast."""
        import time

        analyzer = QueryAnalyzer()

        queries = [tc.question for tc in INFERENCE_TEST_CASES]

        start = time.time()

        for query in queries * 100:  # 100x iterations
            analyzer.analyze(query)

        elapsed = time.time() - start

        # Should analyze 1000+ queries in under 1 second
        assert elapsed < 1.0, f"Query analysis too slow: {elapsed:.2f}s"


# ============================================================================
# Full Integration Tests (Requires API Key and Running Databases)
# ============================================================================

import os


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
class TestFullInferenceIntegration:
    """Full end-to-end inference tests with real API calls."""

    @pytest.fixture
    def sample_project_path(self) -> Path:
        return Path(__file__).parent / "fixtures" / "sample_project"

    @pytest.mark.asyncio
    async def test_index_and_query_pipeline(self, sample_project_path):
        """Test full indexing followed by query - validates inference accuracy."""
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")

        from code_rag.pipeline.orchestrator import PipelineOrchestrator
        from code_rag.query.engine import QueryEngine

        # Step 1: Index the sample project
        orchestrator = PipelineOrchestrator(
            sample_project_path,
            project_name="inference_test_project",
        )

        index_result = await orchestrator.run()

        # Validate indexing succeeded
        assert index_result["files_indexed"] > 0, "Should index files"
        assert index_result["entities_found"] > 0, "Should find entities"

        print(f"\nIndexing completed:")
        print(f"  Files: {index_result['files_indexed']}")
        print(f"  Entities: {index_result['entities_found']}")
        print(f"  Graph nodes: {index_result['graph_nodes']}")

        # Step 2: Query the indexed codebase
        async with QueryEngine() as engine:
            # Test structural query
            result = await engine.query("What calls verify_password?")
            assert result.answer, "Should generate an answer"
            assert len(result.sources) > 0, "Should have sources"

            print(f"\nQuery: 'What calls verify_password?'")
            print(f"  Answer length: {len(result.answer)} chars")
            print(f"  Sources: {len(result.sources)}")

            # Validate answer mentions relevant code
            answer_lower = result.answer.lower()
            assert any(term in answer_lower for term in ["login", "auth", "verify"]), \
                "Answer should mention authentication-related terms"

    @pytest.mark.asyncio
    async def test_semantic_search_accuracy(self, sample_project_path):
        """Test that semantic search returns relevant results."""
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")

        from code_rag.query.engine import QueryEngine

        async with QueryEngine() as engine:
            # Search for authentication-related code
            results = await engine.search("user authentication login")

            assert len(results) > 0, "Should find results"

            # Check that results are relevant
            relevant_found = False
            for result in results[:5]:
                if any(term in result.entity_name.lower() or term in (result.content or "").lower()
                       for term in ["auth", "login", "user", "token"]):
                    relevant_found = True
                    break

            assert relevant_found, "Top results should include auth-related code"

            print(f"\nSemantic search: 'user authentication login'")
            print(f"  Results: {len(results)}")
            for r in results[:3]:
                print(f"    - {r.entity_name} ({r.file_path})")

    @pytest.mark.asyncio
    async def test_explanatory_query_quality(self, sample_project_path):
        """Test that explanatory queries produce meaningful answers."""
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")

        from code_rag.query.engine import QueryEngine

        async with QueryEngine() as engine:
            result = await engine.query("How does the User class handle password verification?")

            assert result.answer, "Should generate an answer"
            assert len(result.answer) > 100, "Answer should be detailed"

            # Check answer quality
            answer_lower = result.answer.lower()
            quality_indicators = [
                "password" in answer_lower,
                "verify" in answer_lower or "hash" in answer_lower,
                "user" in answer_lower,
            ]
            assert sum(quality_indicators) >= 2, \
                "Answer should discuss password verification in context of User"

            print(f"\nExplanatory query: 'How does the User class handle password verification?'")
            print(f"  Answer preview: {result.answer[:200]}...")

    @pytest.mark.asyncio
    async def test_navigational_query_finds_entity(self, sample_project_path):
        """Test that navigational queries locate the correct entity."""
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")

        from code_rag.query.engine import QueryEngine

        async with QueryEngine() as engine:
            result = await engine.query("Show me the AuthService class")

            assert result.answer, "Should generate an answer"
            assert len(result.sources) > 0, "Should have sources"

            # Check that sources include the auth file
            auth_source_found = any(
                "auth" in s.file_path.lower() for s in result.sources
            )
            assert auth_source_found, "Should find AuthService in auth.py"

            print(f"\nNavigational query: 'Show me the AuthService class'")
            print(f"  Sources: {[s.file_path for s in result.sources[:3]]}")

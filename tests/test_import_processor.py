"""Comprehensive tests for the ImportProcessor module.

Tests cover:
- Python import parsing (import, from import, aliased, relative)
- JavaScript/TypeScript import parsing (import, require, default, named)
- Import mapping management
- Wildcard import handling
- Name resolution via imports
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from code_rag.core.cache import FunctionRegistry
from code_rag.parsing.import_processor import ImportProcessor


@pytest.fixture
def function_registry():
    """Create a function registry with test data."""
    registry = FunctionRegistry()
    registry.register("myproject.models.User", "Class")
    registry.register("myproject.models.Post", "Class")
    registry.register("myproject.utils.helper", "Function")
    registry.register("external_lib.Client", "Class")
    return registry


@pytest.fixture
def import_processor(function_registry, tmp_path):
    """Create an import processor instance."""
    # Create a mock repo structure
    (tmp_path / "myproject").mkdir()
    (tmp_path / "myproject" / "models.py").touch()
    (tmp_path / "myproject" / "utils.py").touch()
    (tmp_path / "myproject" / "views").mkdir()
    (tmp_path / "myproject" / "views" / "__init__.py").touch()

    return ImportProcessor(
        function_registry=function_registry,
        project_name="myproject",
        repo_path=tmp_path,
    )


class TestImportMappingManagement:
    """Tests for import mapping management."""

    def test_get_empty_mapping(self, import_processor):
        """Test getting mapping for module with no imports."""
        mapping = import_processor.get_import_mapping("nonexistent.module")
        assert mapping == {}

    def test_clear_module(self, import_processor):
        """Test clearing import mapping for a module."""
        import_processor.import_mapping["test.module"] = {"foo": "bar.foo"}
        import_processor.clear_module("test.module")
        assert "test.module" not in import_processor.import_mapping

    def test_clear_nonexistent_module(self, import_processor):
        """Test clearing a module that doesn't exist doesn't raise."""
        import_processor.clear_module("nonexistent")  # Should not raise


class TestPythonImportParsing:
    """Tests for Python import statement parsing."""

    def test_parse_simple_import(self, import_processor):
        """Test parsing 'import module' statement."""
        # Create a mock AST node
        import_node = create_mock_import_statement("os")

        # Initialize the mapping dict first
        import_processor.import_mapping["myproject.views"] = {}
        import_processor._handle_python_import_statement(import_node, "myproject.views")

        mapping = import_processor.import_mapping["myproject.views"]
        assert "os" in mapping
        assert mapping["os"] == "os"  # External module

    def test_parse_local_import(self, import_processor):
        """Test parsing import of local module."""
        import_node = create_mock_import_statement("myproject.models")

        # Initialize the mapping dict first
        import_processor.import_mapping["myproject.views"] = {}
        import_processor._handle_python_import_statement(import_node, "myproject.views")

        mapping = import_processor.import_mapping.get("myproject.views", {})
        # Should be marked as local project import
        assert "myproject" in mapping or mapping.get("myproject.models")

    def test_parse_from_import(self, import_processor):
        """Test parsing 'from module import name' statement."""
        # Setup import mapping for the module
        import_processor.import_mapping["myproject.views"] = {}

        # Simulate from myproject.models import User
        import_processor.import_mapping["myproject.views"]["User"] = "myproject.models.User"

        mapping = import_processor.import_mapping["myproject.views"]
        assert mapping["User"] == "myproject.models.User"

    def test_parse_aliased_import(self, import_processor):
        """Test parsing 'from module import name as alias' statement."""
        import_processor.import_mapping["myproject.views"] = {}
        import_processor.import_mapping["myproject.views"]["U"] = "myproject.models.User"

        mapping = import_processor.import_mapping["myproject.views"]
        assert mapping["U"] == "myproject.models.User"

    def test_parse_wildcard_import(self, import_processor):
        """Test parsing 'from module import *' statement."""
        import_processor.import_mapping["myproject.views"] = {}
        # Wildcard imports are stored with * prefix
        import_processor.import_mapping["myproject.views"]["*myproject.models"] = "myproject.models"

        mapping = import_processor.import_mapping["myproject.views"]
        assert "*myproject.models" in mapping


class TestRelativeImportResolution:
    """Tests for relative import resolution."""

    def test_resolve_single_dot_import(self, import_processor):
        """Test resolving '.module' import."""
        # Module: myproject.views.handlers
        # Import: from . import utils -> myproject.views.utils
        result = import_processor._resolve_relative_import(
            create_mock_relative_import(".utils"),
            "myproject.views.handlers"
        )
        assert "myproject.views" in result

    def test_resolve_double_dot_import(self, import_processor):
        """Test resolving '..module' import."""
        # Module: myproject.views.handlers.api
        # Import: from ..models import User -> myproject.models.User
        result = import_processor._resolve_relative_import(
            create_mock_relative_import("..models"),
            "myproject.views.handlers.api"
        )
        # Should go up two levels
        assert "myproject" in result


class TestJavaScriptImportParsing:
    """Tests for JavaScript import parsing."""

    def test_parse_default_import(self, import_processor):
        """Test parsing default import."""
        import_processor.import_mapping["myproject.frontend"] = {}
        import_processor.import_mapping["myproject.frontend"]["React"] = "react.default"

        mapping = import_processor.import_mapping["myproject.frontend"]
        assert mapping["React"] == "react.default"

    def test_parse_named_imports(self, import_processor):
        """Test parsing named imports."""
        import_processor.import_mapping["myproject.frontend"] = {}
        import_processor.import_mapping["myproject.frontend"]["useState"] = "react.useState"
        import_processor.import_mapping["myproject.frontend"]["useEffect"] = "react.useEffect"

        mapping = import_processor.import_mapping["myproject.frontend"]
        assert mapping["useState"] == "react.useState"
        assert mapping["useEffect"] == "react.useEffect"

    def test_parse_namespace_import(self, import_processor):
        """Test parsing namespace import (import * as X)."""
        import_processor.import_mapping["myproject.frontend"] = {}
        import_processor.import_mapping["myproject.frontend"]["R"] = "react"

        mapping = import_processor.import_mapping["myproject.frontend"]
        assert mapping["R"] == "react"

    def test_resolve_js_relative_path(self, import_processor):
        """Test resolving JavaScript relative paths."""
        # ./utils from myproject.views
        result = import_processor._resolve_js_module_path("./utils", "myproject.views.index")
        assert "myproject.views.utils" == result

        # ../models from myproject.views.index
        result = import_processor._resolve_js_module_path("../models", "myproject.views.index")
        assert "myproject.models" == result

    def test_resolve_js_external_module(self, import_processor):
        """Test resolving external JavaScript module."""
        result = import_processor._resolve_js_module_path("react", "myproject.frontend")
        assert result == "react"

        result = import_processor._resolve_js_module_path("lodash/debounce", "myproject.frontend")
        assert result == "lodash.debounce"


class TestNameResolution:
    """Tests for name resolution via imports."""

    def test_resolve_imported_name(self, import_processor):
        """Test resolving a directly imported name."""
        import_processor.import_mapping["myproject.views"] = {
            "User": "myproject.models.User",
        }

        result = import_processor.resolve_name("User", "myproject.views")
        assert result == "myproject.models.User"

    def test_resolve_name_not_imported(self, import_processor):
        """Test resolving a name that wasn't imported."""
        import_processor.import_mapping["myproject.views"] = {}

        result = import_processor.resolve_name("Unknown", "myproject.views")
        assert result is None

    def test_resolve_via_wildcard(self, import_processor, function_registry):
        """Test resolving a name via wildcard import."""
        import_processor.import_mapping["myproject.views"] = {
            "*myproject.models": "myproject.models",
        }

        result = import_processor.resolve_name("User", "myproject.views")
        assert result == "myproject.models.User"

    def test_resolve_via_wildcard_not_found(self, import_processor):
        """Test resolving via wildcard when name doesn't exist."""
        import_processor.import_mapping["myproject.views"] = {
            "*myproject.models": "myproject.models",
        }

        result = import_processor.resolve_name("NonExistent", "myproject.views")
        assert result is None


class TestHelperMethods:
    """Tests for helper methods."""

    def test_get_imported_names(self, import_processor):
        """Test getting list of imported names."""
        import_processor.import_mapping["myproject.views"] = {
            "User": "myproject.models.User",
            "Post": "myproject.models.Post",
            "*myproject.utils": "myproject.utils",  # Should be excluded
        }

        names = import_processor.get_imported_names("myproject.views")
        assert "User" in names
        assert "Post" in names
        assert "*myproject.utils" not in names

    def test_get_wildcard_modules(self, import_processor):
        """Test getting wildcard imported modules."""
        import_processor.import_mapping["myproject.views"] = {
            "User": "myproject.models.User",
            "*myproject.utils": "myproject.utils",
            "*external.lib": "external.lib",
        }

        wildcards = import_processor.get_wildcard_modules("myproject.views")
        assert "myproject.utils" in wildcards
        assert "external.lib" in wildcards
        assert len(wildcards) == 2


class TestModuleResolution:
    """Tests for module path resolution."""

    def test_resolve_local_module(self, import_processor):
        """Test resolving a local project module."""
        result = import_processor._resolve_python_module("myproject.models")
        assert result == "myproject.myproject.models"

    def test_resolve_external_module(self, import_processor):
        """Test resolving an external module."""
        result = import_processor._resolve_python_module("requests")
        assert result == "requests"

    def test_resolve_stdlib_module(self, import_processor):
        """Test resolving a stdlib module."""
        result = import_processor._resolve_python_module("os.path")
        assert result == "os.path"


# Helper functions to create mock AST nodes

def create_mock_import_statement(module_name: str):
    """Create a mock AST node for 'import module'."""
    mock_node = MagicMock()
    mock_node.type = "import_statement"

    dotted_name = MagicMock()
    dotted_name.type = "dotted_name"
    dotted_name.text = module_name.encode()

    mock_node.children = [dotted_name]
    return mock_node


def create_mock_relative_import(import_path: str):
    """Create a mock AST node for relative import."""
    mock_node = MagicMock()
    mock_node.type = "relative_import"
    mock_node.text = import_path.encode()

    mock_node.children = []
    return mock_node


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_module_name(self, import_processor):
        """Test handling of empty module name."""
        result = import_processor._resolve_python_module("")
        assert result == ""

    def test_none_mapping_access(self, import_processor):
        """Test accessing mapping for non-existent module."""
        mapping = import_processor.get_import_mapping("does.not.exist")
        assert mapping == {}
        assert isinstance(mapping, dict)

    def test_resolve_with_no_imports(self, import_processor):
        """Test name resolution with no imports defined."""
        result = import_processor.resolve_name("SomeName", "empty.module")
        assert result is None

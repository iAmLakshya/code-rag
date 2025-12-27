"""Comprehensive tests for the CallProcessor module.

Tests cover:
- Simple function call resolution
- Method call resolution on objects
- Import-based call resolution
- Inheritance chain resolution
- Super call handling
- Chained method calls
- Builtin function handling
- Fuzzy/fallback resolution
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from code_rag.core.cache import FunctionRegistry
from code_rag.parsing.call_resolution import CallProcessor
from code_rag.parsing.import_processor import ImportProcessor


@pytest.fixture
def function_registry():
    """Create a function registry with test data."""
    registry = FunctionRegistry()

    # Register some classes
    registry.register("myproject.models.User", "Class")
    registry.register("myproject.models.Post", "Class")
    registry.register("myproject.base.BaseModel", "Class")
    registry.register("myproject.services.UserService", "Class")

    # Register functions
    registry.register("myproject.utils.helper", "Function")
    registry.register("myproject.utils.process_data", "Function")

    # Register methods on User class
    registry.register("myproject.models.User.__init__", "Method")
    registry.register("myproject.models.User.get_name", "Method")
    registry.register("myproject.models.User.save", "Method")
    registry.register("myproject.models.User.validate", "Method")

    # Register methods on Post class
    registry.register("myproject.models.Post.__init__", "Method")
    registry.register("myproject.models.Post.publish", "Method")

    # Register methods on BaseModel
    registry.register("myproject.base.BaseModel.__init__", "Method")
    registry.register("myproject.base.BaseModel.save", "Method")
    registry.register("myproject.base.BaseModel.delete", "Method")

    # Register methods on UserService
    registry.register("myproject.services.UserService.get_user", "Method")
    registry.register("myproject.services.UserService.create_user", "Method")

    return registry


@pytest.fixture
def import_processor(function_registry):
    """Create an import processor with test data."""
    processor = ImportProcessor(
        function_registry=function_registry,
        project_name="myproject",
        repo_path=Path("/test/repo"),
    )

    # Setup import mappings for a test module
    processor.import_mapping["myproject.views"] = {
        "User": "myproject.models.User",
        "Post": "myproject.models.Post",
        "helper": "myproject.utils.helper",
        "UserService": "myproject.services.UserService",
    }

    processor.import_mapping["myproject.services.user_service"] = {
        "User": "myproject.models.User",
        "BaseModel": "myproject.base.BaseModel",
    }

    return processor


@pytest.fixture
def type_inference():
    """Create a mock type inference engine."""
    engine = MagicMock()
    engine._infer_method_call_return_type = MagicMock(return_value=None)
    return engine


@pytest.fixture
def call_processor(function_registry, import_processor, type_inference):
    """Create a call processor instance."""
    # Set up class inheritance
    class_inheritance = {
        "myproject.models.User": ["myproject.base.BaseModel"],
        "myproject.models.Post": ["myproject.base.BaseModel"],
        "myproject.base.BaseModel": [],
    }

    return CallProcessor(
        function_registry=function_registry,
        import_processor=import_processor,
        type_inference=type_inference,
        class_inheritance=class_inheritance,
        project_name="myproject",
        repo_path=Path("/test/repo"),
    )


class TestSimpleFunctionResolution:
    """Tests for simple function call resolution."""

    def test_resolve_same_module_function(self, call_processor):
        """Test resolving a function in the same module."""
        result = call_processor.resolve_call(
            call_name="helper",
            module_qn="myproject.utils",
            language="python",
        )
        assert result is not None
        assert result == ("Function", "myproject.utils.helper")

    def test_resolve_imported_function(self, call_processor):
        """Test resolving an imported function."""
        result = call_processor.resolve_call(
            call_name="helper",
            module_qn="myproject.views",
            language="python",
        )
        assert result is not None
        assert result == ("Function", "myproject.utils.helper")

    def test_resolve_builtin_function_python(self, call_processor):
        """Test resolving Python builtin functions."""
        result = call_processor.resolve_call(
            call_name="print",
            module_qn="myproject.views",
            language="python",
        )
        assert result is not None
        assert result == ("Function", "builtins.print")

    def test_resolve_builtin_function_len(self, call_processor):
        """Test resolving len() builtin."""
        result = call_processor.resolve_call(
            call_name="len",
            module_qn="myproject.views",
            language="python",
        )
        assert result is not None
        assert result == ("Function", "builtins.len")


class TestMethodResolution:
    """Tests for method call resolution."""

    def test_resolve_method_with_local_var_type(self, call_processor):
        """Test resolving method call when variable type is known."""
        local_var_types = {"user": "User"}

        result = call_processor.resolve_call(
            call_name="user.get_name",
            module_qn="myproject.views",
            local_var_types=local_var_types,
            language="python",
        )
        assert result is not None
        assert result == ("Method", "myproject.models.User.get_name")

    def test_resolve_method_on_imported_class(self, call_processor):
        """Test resolving static method call on imported class."""
        result = call_processor.resolve_call(
            call_name="UserService.get_user",
            module_qn="myproject.views",
            language="python",
        )
        assert result is not None
        assert result == ("Method", "myproject.services.UserService.get_user")


class TestInheritanceResolution:
    """Tests for inheritance chain resolution."""

    def test_resolve_inherited_method(self, call_processor):
        """Test resolving a method inherited from parent class."""
        local_var_types = {"user": "User"}

        # User inherits delete() from BaseModel
        result = call_processor.resolve_call(
            call_name="user.delete",
            module_qn="myproject.views",
            local_var_types=local_var_types,
            language="python",
        )
        assert result is not None
        assert result == ("Method", "myproject.base.BaseModel.delete")

    def test_resolve_overridden_method_uses_child(self, call_processor):
        """Test that overridden methods resolve to child class."""
        local_var_types = {"user": "User"}

        # User.save() overrides BaseModel.save()
        result = call_processor.resolve_call(
            call_name="user.save",
            module_qn="myproject.views",
            local_var_types=local_var_types,
            language="python",
        )
        assert result is not None
        # Should resolve to User.save, not BaseModel.save
        assert result == ("Method", "myproject.models.User.save")


class TestSuperCallResolution:
    """Tests for super() call resolution."""

    def test_resolve_super_init(self, call_processor):
        """Test resolving super().__init__() call."""
        result = call_processor.resolve_call(
            call_name="super()",
            module_qn="myproject.models",
            class_context="myproject.models.User",
            language="python",
        )
        assert result is not None
        assert result == ("Method", "myproject.base.BaseModel.__init__")

    def test_resolve_super_with_method(self, call_processor):
        """Test resolving super().method() call."""
        result = call_processor.resolve_call(
            call_name="super().save",
            module_qn="myproject.models",
            class_context="myproject.models.User",
            language="python",
        )
        assert result is not None
        assert result == ("Method", "myproject.base.BaseModel.save")

    def test_super_without_class_context_returns_none(self, call_processor):
        """Test that super() without class context returns None."""
        result = call_processor.resolve_call(
            call_name="super()",
            module_qn="myproject.models",
            class_context=None,
            language="python",
        )
        assert result is None


class TestChainedCallResolution:
    """Tests for chained method call resolution."""

    def test_is_method_chain_detection(self, call_processor):
        """Test detection of method chains."""
        assert call_processor._is_method_chain("obj.method().other()") is True
        assert call_processor._is_method_chain("obj().method") is True
        assert call_processor._is_method_chain("obj.method") is False
        assert call_processor._is_method_chain("func()") is False


class TestFallbackResolution:
    """Tests for fallback/fuzzy resolution."""

    def test_resolve_by_simple_name(self, call_processor):
        """Test fallback resolution using simple name."""
        result = call_processor.resolve_call(
            call_name="process_data",
            module_qn="myproject.other_module",
            language="python",
        )
        assert result is not None
        assert result == ("Function", "myproject.utils.process_data")

    def test_unresolvable_call_returns_none(self, call_processor):
        """Test that unresolvable calls return None."""
        result = call_processor.resolve_call(
            call_name="nonexistent_function",
            module_qn="myproject.views",
            language="python",
        )
        assert result is None


class TestDistanceCalculation:
    """Tests for distance-based resolution ranking."""

    def test_distance_same_module(self, call_processor):
        """Test that same-module functions have lowest distance."""
        # Same module
        dist1 = call_processor._calculate_distance(
            "myproject.views.helper",
            "myproject.views"
        )
        # Different module
        dist2 = call_processor._calculate_distance(
            "myproject.utils.helper",
            "myproject.views"
        )
        # Completely different
        dist3 = call_processor._calculate_distance(
            "other_project.utils.helper",
            "myproject.views"
        )

        assert dist1 < dist2 < dist3


class TestJavaScriptBuiltins:
    """Tests for JavaScript builtin resolution."""

    def test_resolve_console_log(self, call_processor):
        """Test resolving console.log."""
        result = call_processor.resolve_call(
            call_name="console.log",
            module_qn="myproject.frontend",
            language="javascript",
        )
        assert result is not None
        assert result == ("Function", "builtin.console.log")

    def test_resolve_json_parse(self, call_processor):
        """Test resolving JSON.parse."""
        result = call_processor.resolve_call(
            call_name="JSON.parse",
            module_qn="myproject.frontend",
            language="javascript",
        )
        assert result is not None
        assert result == ("Function", "builtin.JSON.parse")

    def test_resolve_array_builtin(self, call_processor):
        """Test resolving Array builtin."""
        result = call_processor.resolve_call(
            call_name="Array",
            module_qn="myproject.frontend",
            language="javascript",
        )
        assert result is not None
        assert result == ("Class", "builtin.Array")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_call_name(self, call_processor):
        """Test handling of empty call name."""
        result = call_processor.resolve_call(
            call_name="",
            module_qn="myproject.views",
            language="python",
        )
        assert result is None

    def test_none_call_name(self, call_processor):
        """Test handling of None call name."""
        result = call_processor.resolve_call(
            call_name=None,
            module_qn="myproject.views",
            language="python",
        )
        assert result is None

    def test_malformed_method_call(self, call_processor):
        """Test handling of malformed method calls."""
        result = call_processor.resolve_call(
            call_name="..invalid..call",
            module_qn="myproject.views",
            language="python",
        )
        # Should not crash, may return None or attempt resolution
        # The main test is that it doesn't raise an exception


class TestCallExtraction:
    """Tests for extracting calls from AST nodes."""

    def test_extract_calls_requires_ast(self, call_processor):
        """Test that extract_calls_from_node requires proper AST."""
        # This is a basic test - full AST tests require tree-sitter
        # We verify the method exists and has correct signature
        assert hasattr(call_processor, 'extract_calls_from_node')


class TestMultipleCallResolution:
    """Integration tests for multiple call scenarios."""

    def test_complex_resolution_scenario(self, call_processor):
        """Test a complex scenario with multiple resolution paths."""
        local_var_types = {
            "service": "UserService",
            "user": "User",
        }

        # Test various calls in sequence
        calls = [
            ("service.get_user", "myproject.services.UserService.get_user"),
            ("user.get_name", "myproject.models.User.get_name"),
            ("helper", "myproject.utils.helper"),
        ]

        for call_name, expected_qn in calls:
            result = call_processor.resolve_call(
                call_name=call_name,
                module_qn="myproject.views",
                local_var_types=local_var_types,
                language="python",
            )
            assert result is not None, f"Failed to resolve: {call_name}"
            assert result[1] == expected_qn, f"Wrong resolution for {call_name}: {result[1]}"

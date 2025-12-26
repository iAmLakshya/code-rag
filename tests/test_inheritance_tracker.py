"""Comprehensive tests for the InheritanceTracker module.

Tests cover:
- Class registration and parent tracking
- Method Resolution Order (MRO) calculation
- Inherited method lookup
- Subclass detection
- Multi-level inheritance
- Diamond inheritance
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from code_rag.core.cache import FunctionRegistry
from code_rag.parsing.import_processor import ImportProcessor
from code_rag.parsing.inheritance_tracker import InheritanceTracker


@pytest.fixture
def function_registry():
    """Create a function registry with test class hierarchy."""
    registry = FunctionRegistry()

    # Base classes
    registry.register("myproject.base.BaseModel", "Class")
    registry.register("myproject.base.BaseModel.__init__", "Method")
    registry.register("myproject.base.BaseModel.save", "Method")
    registry.register("myproject.base.BaseModel.delete", "Method")
    registry.register("myproject.base.BaseModel.validate", "Method")

    registry.register("myproject.mixins.TimestampMixin", "Class")
    registry.register("myproject.mixins.TimestampMixin.created_at", "Method")
    registry.register("myproject.mixins.TimestampMixin.updated_at", "Method")

    # User class inherits from BaseModel
    registry.register("myproject.models.User", "Class")
    registry.register("myproject.models.User.__init__", "Method")
    registry.register("myproject.models.User.get_name", "Method")
    registry.register("myproject.models.User.save", "Method")  # Overrides BaseModel.save

    # Post class inherits from BaseModel and TimestampMixin
    registry.register("myproject.models.Post", "Class")
    registry.register("myproject.models.Post.__init__", "Method")
    registry.register("myproject.models.Post.publish", "Method")

    # Admin inherits from User
    registry.register("myproject.models.Admin", "Class")
    registry.register("myproject.models.Admin.promote", "Method")

    return registry


@pytest.fixture
def import_processor(function_registry):
    """Create a mock import processor."""
    processor = MagicMock(spec=ImportProcessor)
    processor.import_mapping = {
        "myproject.models": {
            "BaseModel": "myproject.base.BaseModel",
            "TimestampMixin": "myproject.mixins.TimestampMixin",
        }
    }
    processor.resolve_name = MagicMock(return_value=None)
    return processor


@pytest.fixture
def tracker(function_registry, import_processor):
    """Create an inheritance tracker instance."""
    tracker = InheritanceTracker(
        function_registry=function_registry,
        import_processor=import_processor,
    )

    # Register the class hierarchy
    tracker.register_class(
        "myproject.base.BaseModel",
        [],
        "myproject.base"
    )
    tracker.register_class(
        "myproject.mixins.TimestampMixin",
        [],
        "myproject.mixins"
    )
    tracker.register_class(
        "myproject.models.User",
        ["BaseModel"],
        "myproject.models"
    )
    tracker.register_class(
        "myproject.models.Post",
        ["BaseModel", "TimestampMixin"],
        "myproject.models"
    )
    tracker.register_class(
        "myproject.models.Admin",
        ["User"],
        "myproject.models"
    )

    return tracker


class TestClassRegistration:
    """Tests for class registration."""

    def test_register_class_with_parents(self, tracker):
        """Test registering a class with parent classes."""
        parents = tracker.get_parents("myproject.models.User")
        assert len(parents) == 1
        assert "myproject.base.BaseModel" in parents

    def test_register_class_multiple_parents(self, tracker):
        """Test registering a class with multiple parents."""
        parents = tracker.get_parents("myproject.models.Post")
        assert len(parents) == 2
        assert "myproject.base.BaseModel" in parents
        assert "myproject.mixins.TimestampMixin" in parents

    def test_register_class_no_parents(self, tracker):
        """Test registering a class without parents."""
        parents = tracker.get_parents("myproject.base.BaseModel")
        assert parents == []

    def test_unregister_class(self, tracker):
        """Test unregistering a class."""
        tracker.unregister_class("myproject.models.User")
        parents = tracker.get_parents("myproject.models.User")
        assert parents == []


class TestMROCalculation:
    """Tests for Method Resolution Order calculation."""

    def test_mro_single_inheritance(self, tracker):
        """Test MRO for single inheritance."""
        mro = tracker.get_mro("myproject.models.User")

        # Should be: User -> BaseModel
        assert mro[0] == "myproject.models.User"
        assert "myproject.base.BaseModel" in mro
        assert mro.index("myproject.models.User") < mro.index("myproject.base.BaseModel")

    def test_mro_multiple_inheritance(self, tracker):
        """Test MRO for multiple inheritance."""
        mro = tracker.get_mro("myproject.models.Post")

        # Should include Post, BaseModel, and TimestampMixin
        assert mro[0] == "myproject.models.Post"
        assert "myproject.base.BaseModel" in mro
        assert "myproject.mixins.TimestampMixin" in mro

    def test_mro_multi_level_inheritance(self, tracker):
        """Test MRO for multi-level inheritance (Admin -> User -> BaseModel)."""
        mro = tracker.get_mro("myproject.models.Admin")

        # Should be: Admin -> User -> BaseModel
        assert mro[0] == "myproject.models.Admin"
        assert "myproject.models.User" in mro
        assert "myproject.base.BaseModel" in mro

        # Order should be correct
        admin_idx = mro.index("myproject.models.Admin")
        user_idx = mro.index("myproject.models.User")
        base_idx = mro.index("myproject.base.BaseModel")
        assert admin_idx < user_idx < base_idx

    def test_mro_no_parents(self, tracker):
        """Test MRO for class without parents."""
        mro = tracker.get_mro("myproject.base.BaseModel")
        assert mro == ["myproject.base.BaseModel"]


class TestMethodResolution:
    """Tests for inherited method lookup."""

    def test_find_method_direct(self, tracker):
        """Test finding a method defined directly on a class."""
        method = tracker.find_method("myproject.models.User", "get_name")
        assert method == "myproject.models.User.get_name"

    def test_find_method_inherited(self, tracker):
        """Test finding a method inherited from parent."""
        method = tracker.find_method("myproject.models.User", "delete")
        assert method == "myproject.base.BaseModel.delete"

    def test_find_method_overridden(self, tracker):
        """Test that overridden methods resolve to child class."""
        # User.save overrides BaseModel.save
        method = tracker.find_method("myproject.models.User", "save")
        assert method == "myproject.models.User.save"

    def test_find_method_multi_level_inherited(self, tracker):
        """Test finding method inherited through multiple levels."""
        # Admin inherits delete from BaseModel via User
        method = tracker.find_method("myproject.models.Admin", "delete")
        assert method == "myproject.base.BaseModel.delete"

    def test_find_method_not_found(self, tracker):
        """Test that non-existent methods return None."""
        method = tracker.find_method("myproject.models.User", "nonexistent")
        assert method is None

    def test_get_all_methods(self, tracker):
        """Test getting all methods including inherited."""
        methods = tracker.get_all_methods("myproject.models.User")

        # Should include User's own methods
        assert "myproject.models.User.__init__" in methods
        assert "myproject.models.User.get_name" in methods
        assert "myproject.models.User.save" in methods

        # Should include inherited methods
        assert "myproject.base.BaseModel.delete" in methods
        assert "myproject.base.BaseModel.validate" in methods


class TestSubclassDetection:
    """Tests for subclass relationships."""

    def test_is_subclass_direct(self, tracker):
        """Test direct subclass detection."""
        assert tracker.is_subclass("myproject.models.User", "myproject.base.BaseModel") is True

    def test_is_subclass_transitive(self, tracker):
        """Test transitive subclass detection."""
        # Admin is a subclass of BaseModel through User
        assert tracker.is_subclass("myproject.models.Admin", "myproject.base.BaseModel") is True

    def test_is_subclass_self(self, tracker):
        """Test that a class is a 'subclass' of itself (in MRO)."""
        assert tracker.is_subclass("myproject.models.User", "myproject.models.User") is True

    def test_is_not_subclass(self, tracker):
        """Test that unrelated classes are not subclasses."""
        assert tracker.is_subclass("myproject.models.User", "myproject.models.Post") is False

    def test_get_subclasses(self, tracker):
        """Test getting all subclasses of a class."""
        subclasses = tracker.get_subclasses("myproject.base.BaseModel")

        assert "myproject.models.User" in subclasses
        assert "myproject.models.Post" in subclasses
        # Admin should be included as a transitive subclass
        assert "myproject.models.Admin" in subclasses

    def test_get_subclasses_none(self, tracker):
        """Test getting subclasses of a class with none."""
        subclasses = tracker.get_subclasses("myproject.models.Admin")
        assert subclasses == []


class TestDiamondInheritance:
    """Tests for diamond inheritance pattern."""

    def test_diamond_mro(self, function_registry, import_processor):
        """Test MRO calculation with diamond inheritance."""
        # Create diamond: D -> B, C -> A
        registry = function_registry
        registry.register("myproject.diamond.A", "Class")
        registry.register("myproject.diamond.A.method", "Method")
        registry.register("myproject.diamond.B", "Class")
        registry.register("myproject.diamond.C", "Class")
        registry.register("myproject.diamond.D", "Class")

        tracker = InheritanceTracker(registry, import_processor)
        tracker.register_class("myproject.diamond.A", [], "myproject.diamond")
        tracker.register_class("myproject.diamond.B", ["A"], "myproject.diamond")
        tracker.register_class("myproject.diamond.C", ["A"], "myproject.diamond")
        tracker.register_class("myproject.diamond.D", ["B", "C"], "myproject.diamond")

        mro = tracker.get_mro("myproject.diamond.D")

        # D should be first, A should appear only once
        assert mro[0] == "myproject.diamond.D"
        assert mro.count("myproject.diamond.A") <= 1  # A appears at most once

        # Method resolution should work
        method = tracker.find_method("myproject.diamond.D", "method")
        assert method is not None


class TestClearAndStats:
    """Tests for clearing and statistics."""

    def test_clear_all(self, tracker):
        """Test clearing all inheritance data."""
        tracker.clear()
        assert len(tracker.class_inheritance) == 0

    def test_clear_by_prefix(self, tracker):
        """Test clearing classes by prefix."""
        removed = tracker.clear_by_prefix("myproject.models")

        assert removed == 3  # User, Post, Admin
        assert "myproject.models.User" not in tracker.class_inheritance
        assert "myproject.base.BaseModel" in tracker.class_inheritance

    def test_get_stats(self, tracker):
        """Test getting statistics."""
        stats = tracker.get_stats()

        assert stats["total_classes"] == 5
        # User, Post (2 parents), Admin have parents = 3 classes with parents
        assert stats["classes_with_parents"] == 3
        assert stats["total_parent_relationships"] > 0
        assert stats["avg_parents_per_class"] > 0


class TestClassNameResolution:
    """Tests for class name resolution."""

    def test_resolve_qualified_name(self, tracker, function_registry):
        """Test resolving already qualified name."""
        result = tracker._resolve_class_name("myproject.base.BaseModel", "myproject.models")
        assert result == "myproject.base.BaseModel"

    def test_resolve_simple_name_via_import(self, tracker, import_processor):
        """Test resolving simple name via import."""
        import_processor.resolve_name.return_value = "myproject.base.BaseModel"

        result = tracker._resolve_class_name("BaseModel", "myproject.models")
        assert result == "myproject.base.BaseModel"

    def test_resolve_simple_name_local(self, tracker, import_processor, function_registry):
        """Test resolving simple name in local module."""
        import_processor.resolve_name.return_value = None

        result = tracker._resolve_class_name("User", "myproject.models")
        assert result == "myproject.models.User"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_get_parents_unknown_class(self, tracker):
        """Test getting parents of unknown class."""
        parents = tracker.get_parents("unknown.Class")
        assert parents == []

    def test_get_mro_unknown_class(self, tracker):
        """Test MRO of unknown class."""
        mro = tracker.get_mro("unknown.Class")
        assert mro == ["unknown.Class"]

    def test_find_method_unknown_class(self, tracker):
        """Test finding method on unknown class."""
        method = tracker.find_method("unknown.Class", "method")
        assert method is None

    def test_register_with_object_parent(self, tracker):
        """Test that 'object' parent is ignored."""
        tracker.register_class("test.Class", ["object", "BaseModel"], "test")
        parents = tracker.get_parents("test.Class")
        assert "object" not in parents

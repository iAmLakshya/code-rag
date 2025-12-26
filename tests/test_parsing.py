"""Tests for code parsing functionality."""

import pytest
from pathlib import Path

from code_rag.parsing.scanner import FileScanner
from code_rag.parsing.parser import CodeParser
from code_rag.parsing.models import Language, EntityType


class TestFileScanner:
    """Tests for FileScanner."""

    def test_scanner_finds_python_files(self, sample_project_path: Path):
        """Test that scanner finds Python files."""
        scanner = FileScanner(sample_project_path)
        files = scanner.scan_all()

        py_files = [f for f in files if f.language == Language.PYTHON]
        assert len(py_files) > 0, "Should find Python files"

        # Check specific file exists
        paths = [str(f.relative_path) for f in py_files]
        assert any("user.py" in p for p in paths), "Should find user.py"

    def test_scanner_finds_typescript_files(self, sample_project_path: Path):
        """Test that scanner finds TypeScript files."""
        scanner = FileScanner(sample_project_path)
        files = scanner.scan_all()

        ts_files = [f for f in files if f.language in (Language.TYPESCRIPT, Language.TSX)]
        assert len(ts_files) > 0, "Should find TypeScript files"

    def test_scanner_computes_hash(self, sample_project_path: Path):
        """Test that scanner computes content hash."""
        scanner = FileScanner(sample_project_path)
        files = scanner.scan_all()

        for f in files:
            assert f.content_hash, "Should have content hash"
            assert len(f.content_hash) == 64, "Hash should be SHA256 (64 hex chars)"

    def test_scanner_ignores_node_modules(self, tmp_path: Path):
        """Test that scanner ignores node_modules."""
        # Create test structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("print('hello')")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "package.js").write_text("module.exports = {}")

        scanner = FileScanner(tmp_path)
        files = scanner.scan_all()

        paths = [str(f.relative_path) for f in files]
        assert not any("node_modules" in p for p in paths), "Should ignore node_modules"
        assert any("app.py" in p for p in paths), "Should find app.py"


class TestCodeParser:
    """Tests for CodeParser."""

    def test_parse_python_file(self, sample_python_file: Path):
        """Test parsing a Python file."""
        from code_rag.parsing.models import FileInfo
        import hashlib

        content = sample_python_file.read_bytes()
        file_info = FileInfo(
            path=sample_python_file,
            relative_path=sample_python_file.name,
            language=Language.PYTHON,
            content_hash=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            line_count=content.count(b"\n") + 1,
        )

        parser = CodeParser()
        parsed = parser.parse_file(file_info)

        assert parsed.content, "Should have content"
        assert len(parsed.entities) > 0, "Should find entities"

        # Find User class
        classes = [e for e in parsed.entities if e.type == EntityType.CLASS]
        class_names = [c.name for c in classes]
        assert "User" in class_names, f"Should find User class, got {class_names}"

    def test_parse_python_content(self):
        """Test parsing Python content directly."""
        code = '''
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"

class Greeter:
    """A greeter class."""

    def __init__(self, prefix: str = "Hi"):
        self.prefix = prefix

    def greet(self, name: str) -> str:
        """Greet someone."""
        return f"{self.prefix}, {name}!"
'''
        parser = CodeParser()
        parsed = parser.parse_content(code, Language.PYTHON)

        # Check functions
        functions = parsed.functions
        assert len(functions) == 1, f"Should find 1 function, got {len(functions)}"
        assert functions[0].name == "hello"

        # Check classes
        classes = parsed.classes
        assert len(classes) == 1, f"Should find 1 class, got {len(classes)}"
        assert classes[0].name == "Greeter"

        # Check methods
        methods = parsed.methods
        assert len(methods) == 2, f"Should find 2 methods, got {len(methods)}"
        method_names = {m.name for m in methods}
        assert "__init__" in method_names
        assert "greet" in method_names

    def test_parse_typescript_content(self):
        """Test parsing TypeScript content."""
        code = '''
import React from 'react';

interface Props {
  name: string;
}

export function Greeting({ name }: Props): JSX.Element {
  return <h1>Hello, {name}!</h1>;
}

export class Counter extends React.Component {
  state = { count: 0 };

  increment() {
    this.setState({ count: this.state.count + 1 });
  }
}
'''
        parser = CodeParser()
        parsed = parser.parse_content(code, Language.TSX)

        # Check imports
        assert len(parsed.imports) > 0, "Should find imports"

        # Check entities
        assert len(parsed.entities) > 0, "Should find entities"

    def test_extract_docstrings(self):
        """Test that docstrings are extracted."""
        code = '''
def documented_function():
    """This is the docstring."""
    pass
'''
        parser = CodeParser()
        parsed = parser.parse_content(code, Language.PYTHON)

        func = parsed.functions[0]
        assert func.docstring is not None, "Should extract docstring"
        assert "This is the docstring" in func.docstring

    def test_extract_function_calls(self):
        """Test that function calls are extracted."""
        code = '''
def caller():
    helper()
    other_func()
    obj.method()
'''
        parser = CodeParser()
        parsed = parser.parse_content(code, Language.PYTHON)

        func = parsed.functions[0]
        assert "helper" in func.calls, f"Should find helper call, got {func.calls}"
        assert "other_func" in func.calls, f"Should find other_func call, got {func.calls}"

    def test_extract_imports(self):
        """Test that imports are extracted."""
        code = '''
import os
from pathlib import Path
from typing import Optional, List
import json as j
'''
        parser = CodeParser()
        parsed = parser.parse_content(code, Language.PYTHON)

        import_names = [i.name for i in parsed.imports]
        assert "os" in import_names, f"Should find os import, got {import_names}"
        assert "Path" in import_names, f"Should find Path import, got {import_names}"

    def test_extract_class_inheritance(self):
        """Test that class inheritance is extracted."""
        code = '''
class Child(Parent, Mixin):
    pass
'''
        parser = CodeParser()
        parsed = parser.parse_content(code, Language.PYTHON)

        cls = parsed.classes[0]
        assert "Parent" in cls.base_classes, f"Should find Parent, got {cls.base_classes}"
        assert "Mixin" in cls.base_classes, f"Should find Mixin, got {cls.base_classes}"


class TestParsingIntegration:
    """Integration tests for parsing."""

    def test_parse_sample_project(self, sample_project_path: Path):
        """Test parsing the entire sample project."""
        scanner = FileScanner(sample_project_path)
        parser = CodeParser()

        files = scanner.scan_all()
        assert len(files) > 0, "Should find files"

        parsed_files = []
        errors = []

        for file_info in files:
            try:
                parsed = parser.parse_file(file_info)
                parsed_files.append(parsed)
            except Exception as e:
                errors.append((file_info.relative_path, str(e)))

        # Report any errors
        if errors:
            error_msg = "\n".join(f"  {path}: {err}" for path, err in errors)
            pytest.fail(f"Parsing errors:\n{error_msg}")

        assert len(parsed_files) == len(files), "Should parse all files"

        # Count total entities
        total_entities = sum(len(pf.all_entities) for pf in parsed_files)
        assert total_entities > 0, "Should find entities across all files"

        print(f"\nParsed {len(parsed_files)} files with {total_entities} entities")

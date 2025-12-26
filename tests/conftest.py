"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
from dotenv import load_dotenv

# Load .env file at test collection time
load_dotenv(Path(__file__).parent.parent / ".env")


@pytest.fixture
def sample_project_path() -> Path:
    """Get the path to the sample project fixture."""
    return Path(__file__).parent / "fixtures" / "sample_project"


@pytest.fixture
def sample_python_file(sample_project_path: Path) -> Path:
    """Get path to a sample Python file."""
    return sample_project_path / "src" / "models" / "user.py"


@pytest.fixture
def sample_typescript_file(sample_project_path: Path) -> Path:
    """Get path to a sample TypeScript file."""
    return sample_project_path / "frontend" / "components" / "LoginForm.tsx"

"""Base model class for all models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BaseModel(ABC):
    """Abstract base class for all data models.

    Provides common functionality for serialization
    and validation.
    """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation.

        Returns:
            Dictionary representation of the model.
        """
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModel":
        """Create model instance from dictionary.

        Args:
            data: Dictionary with model data.

        Returns:
            New model instance.
        """
        return cls(**data)

    def validate(self) -> bool:
        """Validate the model data.

        Returns:
            True if valid, raises ValueError otherwise.
        """
        return True

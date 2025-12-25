"""User model for the application."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .base import BaseModel


@dataclass
class User(BaseModel):
    """Represents a user in the system.

    This class handles user data including authentication
    and profile information.
    """

    id: int
    email: str
    username: str
    password_hash: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_active: bool = True

    def verify_password(self, password: str) -> bool:
        """Verify if the provided password matches the stored hash.

        Args:
            password: Plain text password to verify.

        Returns:
            True if password matches, False otherwise.
        """
        from ..utils.crypto import verify_hash
        return verify_hash(password, self.password_hash)

    def update_password(self, new_password: str) -> None:
        """Update the user's password.

        Args:
            new_password: New plain text password.
        """
        from ..utils.crypto import hash_password
        self.password_hash = hash_password(new_password)
        self.updated_at = datetime.now()

    def to_dict(self) -> dict:
        """Convert user to dictionary representation.

        Returns:
            Dictionary with user data (excluding password).
        """
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
        }


class UserRepository:
    """Repository for user data access."""

    def __init__(self, db_connection):
        """Initialize repository with database connection.

        Args:
            db_connection: Database connection instance.
        """
        self.db = db_connection

    async def find_by_id(self, user_id: int) -> Optional[User]:
        """Find a user by their ID.

        Args:
            user_id: User ID to search for.

        Returns:
            User if found, None otherwise.
        """
        result = await self.db.query("SELECT * FROM users WHERE id = ?", user_id)
        if result:
            return User(**result[0])
        return None

    async def find_by_email(self, email: str) -> Optional[User]:
        """Find a user by their email address.

        Args:
            email: Email to search for.

        Returns:
            User if found, None otherwise.
        """
        result = await self.db.query("SELECT * FROM users WHERE email = ?", email)
        if result:
            return User(**result[0])
        return None

    async def create(self, user: User) -> User:
        """Create a new user in the database.

        Args:
            user: User instance to create.

        Returns:
            Created user with assigned ID.
        """
        result = await self.db.execute(
            "INSERT INTO users (email, username, password_hash) VALUES (?, ?, ?)",
            user.email, user.username, user.password_hash
        )
        user.id = result.lastrowid
        return user

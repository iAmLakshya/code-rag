"""Authentication API endpoints and handlers."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from ..models.user import User, UserRepository
from ..utils.crypto import generate_token, hash_password


@dataclass
class AuthToken:
    """Represents an authentication token."""
    token: str
    user_id: int
    expires_at: datetime


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class AuthService:
    """Service for handling user authentication.

    Provides methods for login, logout, and token management.
    """

    TOKEN_EXPIRY_HOURS = 24

    def __init__(self, user_repo: UserRepository):
        """Initialize auth service.

        Args:
            user_repo: Repository for user data access.
        """
        self.user_repo = user_repo
        self._active_tokens: dict[str, AuthToken] = {}

    async def login(self, email: str, password: str) -> AuthToken:
        """Authenticate a user and create a session token.

        Args:
            email: User's email address.
            password: User's password.

        Returns:
            AuthToken for the authenticated session.

        Raises:
            AuthenticationError: If credentials are invalid.
        """
        user = await self.user_repo.find_by_email(email)

        if user is None:
            raise AuthenticationError("Invalid email or password")

        if not user.is_active:
            raise AuthenticationError("Account is deactivated")

        if not user.verify_password(password):
            raise AuthenticationError("Invalid email or password")

        return self._create_token(user)

    async def logout(self, token: str) -> bool:
        """Invalidate an authentication token.

        Args:
            token: Token to invalidate.

        Returns:
            True if token was invalidated.
        """
        if token in self._active_tokens:
            del self._active_tokens[token]
            return True
        return False

    async def verify_token(self, token: str) -> Optional[User]:
        """Verify a token and return the associated user.

        Args:
            token: Token to verify.

        Returns:
            User if token is valid, None otherwise.
        """
        auth_token = self._active_tokens.get(token)

        if auth_token is None:
            return None

        if datetime.now() > auth_token.expires_at:
            del self._active_tokens[token]
            return None

        return await self.user_repo.find_by_id(auth_token.user_id)

    async def register(
        self,
        email: str,
        username: str,
        password: str,
    ) -> User:
        """Register a new user.

        Args:
            email: User's email address.
            username: Desired username.
            password: User's password.

        Returns:
            Created user instance.

        Raises:
            ValueError: If email already exists.
        """
        existing = await self.user_repo.find_by_email(email)
        if existing:
            raise ValueError("Email already registered")

        user = User(
            id=0,
            email=email,
            username=username,
            password_hash=hash_password(password),
            created_at=datetime.now(),
        )

        return await self.user_repo.create(user)

    def _create_token(self, user: User) -> AuthToken:
        """Create a new authentication token for a user.

        Args:
            user: User to create token for.

        Returns:
            New AuthToken instance.
        """
        token = generate_token()
        expires_at = datetime.now() + timedelta(hours=self.TOKEN_EXPIRY_HOURS)

        auth_token = AuthToken(
            token=token,
            user_id=user.id,
            expires_at=expires_at,
        )

        self._active_tokens[token] = auth_token
        return auth_token

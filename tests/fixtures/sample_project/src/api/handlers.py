"""HTTP request handlers for the API."""

from typing import Any, Callable, Dict
from functools import wraps

from .auth import AuthService, AuthenticationError


def require_auth(auth_service: AuthService):
    """Decorator to require authentication for a handler.

    Args:
        auth_service: AuthService instance for token verification.

    Returns:
        Decorator function.
    """
    def decorator(handler: Callable) -> Callable:
        @wraps(handler)
        async def wrapper(request: Dict[str, Any]) -> Dict[str, Any]:
            token = request.get("headers", {}).get("Authorization", "").replace("Bearer ", "")

            if not token:
                return {"status": 401, "error": "Missing authorization token"}

            user = await auth_service.verify_token(token)
            if user is None:
                return {"status": 401, "error": "Invalid or expired token"}

            request["user"] = user
            return await handler(request)

        return wrapper
    return decorator


class ApiHandler:
    """Base class for API handlers."""

    def __init__(self, auth_service: AuthService):
        """Initialize handler with services.

        Args:
            auth_service: Authentication service.
        """
        self.auth = auth_service

    async def handle_login(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user login request.

        Args:
            request: Request data with email and password.

        Returns:
            Response with token or error.
        """
        body = request.get("body", {})
        email = body.get("email")
        password = body.get("password")

        if not email or not password:
            return {"status": 400, "error": "Email and password required"}

        try:
            token = await self.auth.login(email, password)
            return {
                "status": 200,
                "data": {
                    "token": token.token,
                    "expires_at": token.expires_at.isoformat(),
                }
            }
        except AuthenticationError as e:
            return {"status": 401, "error": str(e)}

    async def handle_register(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user registration request.

        Args:
            request: Request data with user details.

        Returns:
            Response with user data or error.
        """
        body = request.get("body", {})
        email = body.get("email")
        username = body.get("username")
        password = body.get("password")

        if not all([email, username, password]):
            return {"status": 400, "error": "All fields required"}

        try:
            user = await self.auth.register(email, username, password)
            return {"status": 201, "data": user.to_dict()}
        except ValueError as e:
            return {"status": 400, "error": str(e)}

    async def handle_logout(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user logout request.

        Args:
            request: Request with authorization token.

        Returns:
            Response indicating success.
        """
        token = request.get("headers", {}).get("Authorization", "").replace("Bearer ", "")
        await self.auth.logout(token)
        return {"status": 200, "data": {"message": "Logged out"}}

    async def handle_profile(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle profile request (requires auth).

        Args:
            request: Request with user context.

        Returns:
            Response with user profile data.
        """
        user = request.get("user")
        if user is None:
            return {"status": 401, "error": "Not authenticated"}

        return {"status": 200, "data": user.to_dict()}

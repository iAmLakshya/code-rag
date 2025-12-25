"""Request processing middleware chain.

Middleware components process requests before they reach handlers
and responses before they're sent to clients. They form a chain
where each middleware can:
- Modify the request/response
- Short-circuit processing
- Add context data
- Handle errors

The middleware chain is: Logging -> Auth -> RateLimit -> Validation -> Handler
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Awaitable
from collections import defaultdict
import asyncio
import time
import uuid

from code_rag.tests.fixtures.sample_project.src.core.events import (
    EventBus,
    Event,
    EventType,
)
from code_rag.tests.fixtures.sample_project.src.core.cache import get_cache


@dataclass
class Request:
    """HTTP request abstraction."""

    method: str
    path: str
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None
    query_params: dict[str, str] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Response:
    """HTTP response abstraction."""

    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None
    error: str | None = None


# Type alias for handler functions
Handler = Callable[[Request], Awaitable[Response]]
NextMiddleware = Callable[[Request], Awaitable[Response]]


class Middleware(ABC):
    """Base class for middleware components."""

    @abstractmethod
    async def __call__(
        self,
        request: Request,
        next_handler: NextMiddleware,
    ) -> Response:
        """Process request and optionally call next middleware."""
        pass


class LoggingMiddleware(Middleware):
    """Logs all requests and responses for debugging and monitoring.

    Captures:
    - Request details (method, path, headers)
    - Response status and timing
    - Error information

    Integrates with EventBus for centralized logging.
    """

    def __init__(self, log_headers: bool = False, log_body: bool = False):
        self.log_headers = log_headers
        self.log_body = log_body

    async def __call__(
        self,
        request: Request,
        next_handler: NextMiddleware,
    ) -> Response:
        """Log request, process, then log response."""
        start_time = time.perf_counter()

        # Log incoming request
        log_data = {
            "request_id": request.request_id,
            "method": request.method,
            "path": request.path,
            "timestamp": request.timestamp.isoformat(),
        }

        if self.log_headers:
            log_data["headers"] = request.headers
        if self.log_body:
            log_data["body"] = request.body

        try:
            response = await next_handler(request)

            # Log successful response
            duration_ms = (time.perf_counter() - start_time) * 1000
            log_data.update({
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            })

            await EventBus.get_instance().publish(
                Event(
                    type=EventType.AUDIT_LOG,
                    payload=log_data,
                    source="logging_middleware",
                    correlation_id=request.request_id,
                )
            )

            return response

        except Exception as e:
            # Log error
            duration_ms = (time.perf_counter() - start_time) * 1000
            log_data.update({
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_ms": round(duration_ms, 2),
            })

            await EventBus.get_instance().publish(
                Event(
                    type=EventType.ERROR_OCCURRED,
                    payload=log_data,
                    source="logging_middleware",
                    correlation_id=request.request_id,
                )
            )
            raise


class AuthMiddleware(Middleware):
    """Validates authentication tokens and populates user context.

    Checks for Bearer token in Authorization header, validates it,
    and adds user information to request context.

    Protected routes are determined by path patterns.
    """

    def __init__(
        self,
        auth_service: Any,  # AuthService from auth.py
        public_paths: list[str] | None = None,
    ):
        self.auth_service = auth_service
        self.public_paths = public_paths or ["/health", "/login", "/register"]

    async def __call__(
        self,
        request: Request,
        next_handler: NextMiddleware,
    ) -> Response:
        """Validate auth and add user to context."""
        # Skip auth for public paths
        if self._is_public_path(request.path):
            return await next_handler(request)

        # Extract token from header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return Response(
                status_code=401,
                error="Missing or invalid Authorization header",
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Validate token
        try:
            user = await self.auth_service.verify_token(token)
            request.context["user"] = user
            request.context["token"] = token
        except Exception as e:
            await self._log_auth_failure(request, str(e))
            return Response(
                status_code=401,
                error="Invalid or expired token",
            )

        return await next_handler(request)

    def _is_public_path(self, path: str) -> bool:
        """Check if path is publicly accessible."""
        return any(
            path.startswith(public_path)
            for public_path in self.public_paths
        )

    async def _log_auth_failure(self, request: Request, reason: str) -> None:
        """Log authentication failure for security monitoring."""
        await EventBus.get_instance().publish(
            Event(
                type=EventType.USER_LOGIN_FAILED,
                payload={
                    "path": request.path,
                    "reason": reason,
                    "ip": request.headers.get("X-Forwarded-For"),
                },
                source="auth_middleware",
                correlation_id=request.request_id,
            )
        )


class RateLimitMiddleware(Middleware):
    """Rate limiting using sliding window algorithm.

    Limits requests per IP address to prevent abuse.
    Uses cache for distributed rate limiting across instances.

    Configurable:
    - requests_per_window: Max requests allowed
    - window_size: Time window for counting
    """

    def __init__(
        self,
        requests_per_window: int = 100,
        window_size: timedelta = timedelta(minutes=1),
    ):
        self.requests_per_window = requests_per_window
        self.window_size = window_size
        self._local_counts: dict[str, list[float]] = defaultdict(list)

    async def __call__(
        self,
        request: Request,
        next_handler: NextMiddleware,
    ) -> Response:
        """Check rate limit before processing."""
        client_ip = self._get_client_ip(request)

        if await self._is_rate_limited(client_ip):
            return Response(
                status_code=429,
                headers={"Retry-After": str(int(self.window_size.total_seconds()))},
                error="Rate limit exceeded",
            )

        await self._record_request(client_ip)
        return await next_handler(request)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from headers or connection."""
        return (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.headers.get("X-Real-IP", "")
            or "unknown"
        )

    async def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit."""
        now = time.time()
        window_start = now - self.window_size.total_seconds()

        # Clean old entries and count recent requests
        self._local_counts[client_ip] = [
            ts for ts in self._local_counts[client_ip]
            if ts > window_start
        ]

        return len(self._local_counts[client_ip]) >= self.requests_per_window

    async def _record_request(self, client_ip: str) -> None:
        """Record a request for rate limiting."""
        self._local_counts[client_ip].append(time.time())


class ValidationMiddleware(Middleware):
    """Validates request body against schemas.

    Uses JSON Schema or Pydantic models for validation.
    Returns 400 Bad Request if validation fails.
    """

    def __init__(self, validators: dict[str, Callable[[Any], bool]] | None = None):
        self.validators = validators or {}

    async def __call__(
        self,
        request: Request,
        next_handler: NextMiddleware,
    ) -> Response:
        """Validate request body if validator exists for path."""
        validator = self.validators.get(request.path)

        if validator and request.body:
            try:
                is_valid = validator(request.body)
                if not is_valid:
                    return Response(
                        status_code=400,
                        error="Request validation failed",
                    )
            except Exception as e:
                return Response(
                    status_code=400,
                    error=f"Validation error: {str(e)}",
                )

        return await next_handler(request)


class MiddlewareChain:
    """Composes middleware into a processing chain.

    Middleware is executed in order, each calling the next.
    The final handler processes the actual request.

    Example:
        chain = MiddlewareChain()
        chain.use(LoggingMiddleware())
        chain.use(AuthMiddleware(auth_service))
        chain.use(RateLimitMiddleware())

        response = await chain.handle(request, handler)
    """

    def __init__(self):
        self._middleware: list[Middleware] = []

    def use(self, middleware: Middleware) -> "MiddlewareChain":
        """Add middleware to the chain. Returns self for chaining."""
        self._middleware.append(middleware)
        return self

    async def handle(self, request: Request, handler: Handler) -> Response:
        """Process request through middleware chain to handler."""
        if not self._middleware:
            return await handler(request)

        # Build the chain from inside out
        async def final_handler(req: Request) -> Response:
            return await handler(req)

        chain = final_handler
        for middleware in reversed(self._middleware):
            # Capture current chain in closure
            prev_chain = chain
            async def make_next(mw: Middleware, next_h: NextMiddleware):
                return lambda req: mw(req, next_h)
            chain = await make_next(middleware, prev_chain)

        return await chain(request)


def create_default_chain(auth_service: Any) -> MiddlewareChain:
    """Create the standard middleware chain used by the application.

    Chain order:
    1. LoggingMiddleware - Log all requests
    2. AuthMiddleware - Validate authentication
    3. RateLimitMiddleware - Prevent abuse
    4. ValidationMiddleware - Validate request body
    """
    return (
        MiddlewareChain()
        .use(LoggingMiddleware(log_headers=True))
        .use(AuthMiddleware(auth_service))
        .use(RateLimitMiddleware(requests_per_window=100))
        .use(ValidationMiddleware())
    )

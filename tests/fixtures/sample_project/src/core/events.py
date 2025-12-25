"""Event system for decoupled communication between components.

This module implements a publish-subscribe pattern for application events.
It's used throughout the codebase for:
- User lifecycle events (login, logout, registration)
- Data change notifications
- Audit logging
- Cache invalidation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, TypeVar
from enum import Enum
import asyncio
from collections import defaultdict


class EventType(Enum):
    """All event types in the application."""

    # User events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_LOGIN_FAILED = "user.login_failed"

    # Data events
    ITEM_CREATED = "item.created"
    ITEM_UPDATED = "item.updated"
    ITEM_DELETED = "item.deleted"

    # System events
    CACHE_INVALIDATED = "cache.invalidated"
    ERROR_OCCURRED = "error.occurred"
    AUDIT_LOG = "audit.log"


@dataclass
class Event:
    """Base event class containing metadata and payload."""

    type: EventType
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"
    correlation_id: str | None = None

    def to_dict(self) -> dict:
        """Serialize event for logging or transmission."""
        return {
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
        }


class EventHandler(ABC):
    """Abstract base class for event handlers."""

    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Process an event. Must be implemented by subclasses."""
        pass

    @property
    @abstractmethod
    def event_types(self) -> list[EventType]:
        """Return list of event types this handler processes."""
        pass


class AuditLogHandler(EventHandler):
    """Logs all events for audit trail.

    This handler is subscribed to ALL events and writes them
    to the audit log for compliance and debugging.
    """

    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
        self._log_buffer: list[Event] = []

    @property
    def event_types(self) -> list[EventType]:
        return list(EventType)  # Subscribe to all events

    async def handle(self, event: Event) -> None:
        """Write event to audit log."""
        self._log_buffer.append(event)
        await self._flush_if_needed()

    async def _flush_if_needed(self) -> None:
        """Flush buffer to disk if it exceeds threshold."""
        if len(self._log_buffer) >= 100:
            await self._flush()

    async def _flush(self) -> None:
        """Write buffered events to disk."""
        # In real implementation, write to file
        self._log_buffer.clear()


class CacheInvalidationHandler(EventHandler):
    """Invalidates cache entries when data changes.

    Listens for data modification events and invalidates
    corresponding cache entries to maintain consistency.
    """

    def __init__(self, cache: "Cache"):
        self.cache = cache

    @property
    def event_types(self) -> list[EventType]:
        return [
            EventType.USER_UPDATED,
            EventType.USER_DELETED,
            EventType.ITEM_UPDATED,
            EventType.ITEM_DELETED,
        ]

    async def handle(self, event: Event) -> None:
        """Invalidate cache based on event type."""
        entity_id = event.payload.get("id")
        if not entity_id:
            return

        if event.type in (EventType.USER_UPDATED, EventType.USER_DELETED):
            await self.cache.invalidate(f"user:{entity_id}")
            await self.cache.invalidate(f"user_profile:{entity_id}")
        elif event.type in (EventType.ITEM_UPDATED, EventType.ITEM_DELETED):
            await self.cache.invalidate(f"item:{entity_id}")


class NotificationHandler(EventHandler):
    """Sends notifications for important events.

    Integrates with external notification services to alert
    users and administrators of significant events.
    """

    def __init__(self, notification_service: "NotificationService"):
        self.notification_service = notification_service

    @property
    def event_types(self) -> list[EventType]:
        return [
            EventType.USER_CREATED,
            EventType.USER_LOGIN_FAILED,
            EventType.ERROR_OCCURRED,
        ]

    async def handle(self, event: Event) -> None:
        """Send appropriate notification based on event."""
        if event.type == EventType.USER_CREATED:
            await self._send_welcome_email(event)
        elif event.type == EventType.USER_LOGIN_FAILED:
            await self._check_brute_force(event)
        elif event.type == EventType.ERROR_OCCURRED:
            await self._alert_admin(event)

    async def _send_welcome_email(self, event: Event) -> None:
        """Send welcome email to new user."""
        email = event.payload.get("email")
        if email:
            await self.notification_service.send_email(
                to=email,
                subject="Welcome!",
                body="Thanks for registering.",
            )

    async def _check_brute_force(self, event: Event) -> None:
        """Check for brute force login attempts."""
        # Track failed attempts and alert if threshold exceeded
        pass

    async def _alert_admin(self, event: Event) -> None:
        """Alert administrator of critical errors."""
        await self.notification_service.send_alert(
            level="critical",
            message=str(event.payload.get("error")),
        )


class EventBus:
    """Central event dispatcher using publish-subscribe pattern.

    The EventBus is a singleton that manages event distribution
    throughout the application. Components publish events here,
    and registered handlers receive them asynchronously.

    Example usage:
        bus = EventBus.get_instance()
        bus.subscribe(MyHandler())
        await bus.publish(Event(EventType.USER_CREATED, {"id": 1}))
    """

    _instance: "EventBus | None" = None

    def __init__(self):
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._middleware: list[Callable[[Event], Event | None]] = []

    @classmethod
    def get_instance(cls) -> "EventBus":
        """Get singleton instance of EventBus."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def subscribe(self, handler: EventHandler) -> None:
        """Register a handler for its declared event types."""
        for event_type in handler.event_types:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        """Remove a handler from all event types."""
        for event_type in handler.event_types:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

    def add_middleware(self, middleware: Callable[[Event], Event | None]) -> None:
        """Add middleware that processes events before handlers.

        Middleware can modify events or return None to stop propagation.
        """
        self._middleware.append(middleware)

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribed handlers.

        Events pass through middleware first, then are dispatched
        to all handlers registered for the event type.
        """
        # Process through middleware
        processed_event = event
        for middleware in self._middleware:
            result = middleware(processed_event)
            if result is None:
                return  # Middleware blocked the event
            processed_event = result

        # Dispatch to handlers
        handlers = self._handlers.get(processed_event.type, [])
        await asyncio.gather(
            *[handler.handle(processed_event) for handler in handlers],
            return_exceptions=True,
        )

    async def publish_many(self, events: list[Event]) -> None:
        """Publish multiple events concurrently."""
        await asyncio.gather(*[self.publish(e) for e in events])


# Type alias for event callbacks
EventCallback = Callable[[Event], None]


def on_event(*event_types: EventType):
    """Decorator to register a function as an event handler.

    Example:
        @on_event(EventType.USER_CREATED, EventType.USER_UPDATED)
        async def handle_user_change(event: Event):
            print(f"User changed: {event.payload}")
    """
    def decorator(func: Callable[[Event], Any]):
        class DecoratedHandler(EventHandler):
            @property
            def event_types(self) -> list[EventType]:
                return list(event_types)

            async def handle(self, event: Event) -> None:
                result = func(event)
                if asyncio.iscoroutine(result):
                    await result

        # Auto-register with EventBus
        EventBus.get_instance().subscribe(DecoratedHandler())
        return func

    return decorator

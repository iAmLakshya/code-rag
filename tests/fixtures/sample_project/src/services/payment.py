"""Payment processing with Strategy pattern and complex error handling.

This module demonstrates:
- Strategy pattern for payment providers
- Factory pattern for provider selection
- Complex error handling chain
- Retry logic with exponential backoff
- Transaction state machine
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, TypeVar, Generic
import asyncio
import uuid

from code_rag.tests.fixtures.sample_project.src.core.events import (
    EventBus,
    Event,
    EventType,
)
from code_rag.tests.fixtures.sample_project.src.core.cache import cached
from code_rag.tests.fixtures.sample_project.src.utils.crypto import generate_token


class PaymentStatus(Enum):
    """States in the payment lifecycle."""

    PENDING = "pending"
    PROCESSING = "processing"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class PaymentError(Exception):
    """Base exception for payment errors."""

    def __init__(self, message: str, code: str, retriable: bool = False):
        super().__init__(message)
        self.code = code
        self.retriable = retriable


class InsufficientFundsError(PaymentError):
    """Raised when payment source has insufficient funds."""

    def __init__(self, message: str = "Insufficient funds"):
        super().__init__(message, "INSUFFICIENT_FUNDS", retriable=False)


class PaymentDeclinedError(PaymentError):
    """Raised when payment is declined by provider."""

    def __init__(self, message: str, reason: str):
        super().__init__(message, "PAYMENT_DECLINED", retriable=False)
        self.reason = reason


class PaymentTimeoutError(PaymentError):
    """Raised when payment provider times out."""

    def __init__(self, message: str = "Payment request timed out"):
        super().__init__(message, "TIMEOUT", retriable=True)


class ProviderUnavailableError(PaymentError):
    """Raised when payment provider is unavailable."""

    def __init__(self, provider: str):
        super().__init__(
            f"Payment provider {provider} is unavailable",
            "PROVIDER_UNAVAILABLE",
            retriable=True,
        )
        self.provider = provider


@dataclass
class PaymentMethod:
    """Represents a stored payment method."""

    id: str
    user_id: str
    type: str  # "card", "bank_account", "wallet"
    last_four: str
    provider_token: str
    is_default: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PaymentIntent:
    """Represents an intent to collect payment."""

    id: str
    amount: Decimal
    currency: str
    user_id: str
    description: str
    status: PaymentStatus = PaymentStatus.PENDING
    payment_method_id: str | None = None
    provider: str | None = None
    provider_transaction_id: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def can_transition_to(self, new_status: PaymentStatus) -> bool:
        """Check if status transition is valid."""
        valid_transitions = {
            PaymentStatus.PENDING: [
                PaymentStatus.PROCESSING,
                PaymentStatus.CANCELLED,
            ],
            PaymentStatus.PROCESSING: [
                PaymentStatus.AUTHORIZED,
                PaymentStatus.FAILED,
            ],
            PaymentStatus.AUTHORIZED: [
                PaymentStatus.CAPTURED,
                PaymentStatus.CANCELLED,
            ],
            PaymentStatus.CAPTURED: [
                PaymentStatus.REFUNDED,
            ],
        }
        return new_status in valid_transitions.get(self.status, [])


class PaymentProvider(ABC):
    """Abstract base class for payment providers (Strategy pattern).

    Each provider implements the same interface but with different
    underlying APIs (Stripe, PayPal, Square, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and selection."""
        pass

    @abstractmethod
    async def create_customer(self, user_id: str, email: str) -> str:
        """Create customer account with provider. Returns provider customer ID."""
        pass

    @abstractmethod
    async def tokenize_card(
        self,
        customer_id: str,
        card_number: str,
        exp_month: int,
        exp_year: int,
        cvv: str,
    ) -> str:
        """Securely tokenize card details. Returns provider token."""
        pass

    @abstractmethod
    async def authorize(
        self,
        amount: Decimal,
        currency: str,
        token: str,
        idempotency_key: str,
    ) -> str:
        """Authorize payment. Returns transaction ID."""
        pass

    @abstractmethod
    async def capture(self, transaction_id: str, amount: Decimal) -> bool:
        """Capture authorized payment."""
        pass

    @abstractmethod
    async def refund(
        self,
        transaction_id: str,
        amount: Decimal | None = None,
    ) -> str:
        """Refund captured payment. Returns refund ID."""
        pass

    @abstractmethod
    async def void(self, transaction_id: str) -> bool:
        """Void authorized but uncaptured payment."""
        pass


class StripeProvider(PaymentProvider):
    """Stripe payment provider implementation.

    Uses Stripe API for payment processing.
    Supports cards, bank transfers, and wallets.
    """

    def __init__(self, api_key: str, webhook_secret: str):
        self.api_key = api_key
        self.webhook_secret = webhook_secret

    @property
    def name(self) -> str:
        return "stripe"

    async def create_customer(self, user_id: str, email: str) -> str:
        """Create Stripe customer."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return f"cus_{generate_token()[:16]}"

    async def tokenize_card(
        self,
        customer_id: str,
        card_number: str,
        exp_month: int,
        exp_year: int,
        cvv: str,
    ) -> str:
        """Tokenize card with Stripe."""
        await asyncio.sleep(0.1)
        return f"pm_{generate_token()[:16]}"

    async def authorize(
        self,
        amount: Decimal,
        currency: str,
        token: str,
        idempotency_key: str,
    ) -> str:
        """Authorize payment with Stripe."""
        await asyncio.sleep(0.2)

        # Simulate occasional failures for testing
        if str(amount).endswith("13"):
            raise PaymentDeclinedError(
                "Card declined",
                reason="do_not_honor",
            )

        return f"pi_{generate_token()[:16]}"

    async def capture(self, transaction_id: str, amount: Decimal) -> bool:
        """Capture Stripe payment."""
        await asyncio.sleep(0.1)
        return True

    async def refund(
        self,
        transaction_id: str,
        amount: Decimal | None = None,
    ) -> str:
        """Refund Stripe payment."""
        await asyncio.sleep(0.1)
        return f"re_{generate_token()[:16]}"

    async def void(self, transaction_id: str) -> bool:
        """Void Stripe authorization."""
        await asyncio.sleep(0.1)
        return True


class PayPalProvider(PaymentProvider):
    """PayPal payment provider implementation."""

    def __init__(self, client_id: str, client_secret: str, sandbox: bool = True):
        self.client_id = client_id
        self.client_secret = client_secret
        self.sandbox = sandbox

    @property
    def name(self) -> str:
        return "paypal"

    async def create_customer(self, user_id: str, email: str) -> str:
        await asyncio.sleep(0.1)
        return f"PP-{generate_token()[:12]}"

    async def tokenize_card(
        self,
        customer_id: str,
        card_number: str,
        exp_month: int,
        exp_year: int,
        cvv: str,
    ) -> str:
        await asyncio.sleep(0.1)
        return f"CARD-{generate_token()[:12]}"

    async def authorize(
        self,
        amount: Decimal,
        currency: str,
        token: str,
        idempotency_key: str,
    ) -> str:
        await asyncio.sleep(0.3)  # PayPal is slower
        return f"PAY-{generate_token()[:16]}"

    async def capture(self, transaction_id: str, amount: Decimal) -> bool:
        await asyncio.sleep(0.2)
        return True

    async def refund(
        self,
        transaction_id: str,
        amount: Decimal | None = None,
    ) -> str:
        await asyncio.sleep(0.2)
        return f"REF-{generate_token()[:12]}"

    async def void(self, transaction_id: str) -> bool:
        await asyncio.sleep(0.1)
        return True


class PaymentProviderFactory:
    """Factory for creating payment provider instances.

    Manages provider configuration and selection logic.
    Supports fallback providers when primary is unavailable.
    """

    _providers: dict[str, PaymentProvider] = {}
    _primary: str = "stripe"
    _fallback_order: list[str] = ["paypal"]

    @classmethod
    def register(cls, provider: PaymentProvider) -> None:
        """Register a payment provider."""
        cls._providers[provider.name] = provider

    @classmethod
    def get(cls, name: str) -> PaymentProvider:
        """Get provider by name."""
        if name not in cls._providers:
            raise ValueError(f"Unknown payment provider: {name}")
        return cls._providers[name]

    @classmethod
    def get_primary(cls) -> PaymentProvider:
        """Get the primary payment provider."""
        return cls.get(cls._primary)

    @classmethod
    def get_fallback_chain(cls) -> list[PaymentProvider]:
        """Get ordered list of providers for failover."""
        return [cls.get(name) for name in [cls._primary] + cls._fallback_order]

    @classmethod
    def set_primary(cls, name: str) -> None:
        """Set the primary provider."""
        if name not in cls._providers:
            raise ValueError(f"Unknown payment provider: {name}")
        cls._primary = name


class PaymentService:
    """High-level payment processing service.

    Orchestrates payment flow:
    1. Create payment intent
    2. Process with retry logic
    3. Handle errors and fallback
    4. Emit events for tracking

    Uses providers via factory and strategy pattern.
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._intents: dict[str, PaymentIntent] = {}

    async def create_intent(
        self,
        amount: Decimal,
        currency: str,
        user_id: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> PaymentIntent:
        """Create a new payment intent."""
        intent = PaymentIntent(
            id=str(uuid.uuid4()),
            amount=amount,
            currency=currency,
            user_id=user_id,
            description=description,
            metadata=metadata or {},
        )
        self._intents[intent.id] = intent

        await EventBus.get_instance().publish(
            Event(
                type=EventType.ITEM_CREATED,
                payload={"intent_id": intent.id, "amount": str(amount)},
                source="payment_service",
            )
        )

        return intent

    async def process_payment(
        self,
        intent_id: str,
        payment_method_id: str,
    ) -> PaymentIntent:
        """Process payment with retry and fallback logic.

        This is the main entry point for payment processing.
        It handles the full flow including authorization,
        error handling, retries, and provider fallback.
        """
        intent = self._intents.get(intent_id)
        if not intent:
            raise ValueError(f"Unknown payment intent: {intent_id}")

        # Get payment method token (would fetch from DB in real implementation)
        token = payment_method_id

        # Try each provider in fallback chain
        providers = PaymentProviderFactory.get_fallback_chain()
        last_error: PaymentError | None = None

        for provider in providers:
            try:
                await self._process_with_provider(intent, provider, token)
                return intent
            except ProviderUnavailableError as e:
                last_error = e
                continue  # Try next provider
            except PaymentError as e:
                if not e.retriable:
                    await self._handle_failure(intent, e)
                    raise
                last_error = e
                # Retry with same provider
                if not await self._retry_payment(intent, provider, token):
                    continue  # Try next provider

        # All providers failed
        if last_error:
            await self._handle_failure(intent, last_error)
            raise last_error

        return intent

    async def _process_with_provider(
        self,
        intent: PaymentIntent,
        provider: PaymentProvider,
        token: str,
    ) -> None:
        """Process payment with a specific provider."""
        intent.status = PaymentStatus.PROCESSING
        intent.provider = provider.name
        intent.updated_at = datetime.utcnow()

        try:
            # Authorize the payment
            transaction_id = await provider.authorize(
                amount=intent.amount,
                currency=intent.currency,
                token=token,
                idempotency_key=intent.id,
            )

            intent.provider_transaction_id = transaction_id
            intent.status = PaymentStatus.AUTHORIZED
            intent.updated_at = datetime.utcnow()

            # Capture immediately (could be deferred)
            success = await provider.capture(transaction_id, intent.amount)
            if success:
                intent.status = PaymentStatus.CAPTURED
                intent.updated_at = datetime.utcnow()
                await self._emit_success_event(intent)

        except Exception as e:
            if isinstance(e, PaymentError):
                raise
            # Wrap unknown errors
            raise PaymentError(str(e), "UNKNOWN_ERROR", retriable=True) from e

    async def _retry_payment(
        self,
        intent: PaymentIntent,
        provider: PaymentProvider,
        token: str,
    ) -> bool:
        """Retry payment with exponential backoff."""
        for attempt in range(self.max_retries):
            delay = self.retry_delay * (2 ** attempt)
            await asyncio.sleep(delay)

            try:
                await self._process_with_provider(intent, provider, token)
                return True
            except PaymentError as e:
                if not e.retriable:
                    return False
                if attempt == self.max_retries - 1:
                    return False
                continue

        return False

    async def _handle_failure(
        self,
        intent: PaymentIntent,
        error: PaymentError,
    ) -> None:
        """Handle payment failure."""
        intent.status = PaymentStatus.FAILED
        intent.error_message = str(error)
        intent.updated_at = datetime.utcnow()

        await EventBus.get_instance().publish(
            Event(
                type=EventType.ERROR_OCCURRED,
                payload={
                    "intent_id": intent.id,
                    "error_code": error.code,
                    "error_message": str(error),
                },
                source="payment_service",
            )
        )

    async def _emit_success_event(self, intent: PaymentIntent) -> None:
        """Emit payment success event."""
        await EventBus.get_instance().publish(
            Event(
                type=EventType.ITEM_UPDATED,
                payload={
                    "intent_id": intent.id,
                    "status": intent.status.value,
                    "amount": str(intent.amount),
                    "provider": intent.provider,
                },
                source="payment_service",
            )
        )

    async def refund(
        self,
        intent_id: str,
        amount: Decimal | None = None,
    ) -> str:
        """Refund a captured payment."""
        intent = self._intents.get(intent_id)
        if not intent:
            raise ValueError(f"Unknown payment intent: {intent_id}")

        if intent.status != PaymentStatus.CAPTURED:
            raise PaymentError(
                "Can only refund captured payments",
                "INVALID_STATE",
                retriable=False,
            )

        provider = PaymentProviderFactory.get(intent.provider)
        refund_id = await provider.refund(
            intent.provider_transaction_id,
            amount or intent.amount,
        )

        intent.status = PaymentStatus.REFUNDED
        intent.updated_at = datetime.utcnow()

        return refund_id

    async def cancel(self, intent_id: str) -> bool:
        """Cancel a payment intent."""
        intent = self._intents.get(intent_id)
        if not intent:
            raise ValueError(f"Unknown payment intent: {intent_id}")

        if not intent.can_transition_to(PaymentStatus.CANCELLED):
            return False

        if intent.provider_transaction_id and intent.status == PaymentStatus.AUTHORIZED:
            # Void the authorization
            provider = PaymentProviderFactory.get(intent.provider)
            await provider.void(intent.provider_transaction_id)

        intent.status = PaymentStatus.CANCELLED
        intent.updated_at = datetime.utcnow()
        return True

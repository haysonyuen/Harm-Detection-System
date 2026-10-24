"""Shared provider abstractions: error taxonomy, retry policy, provider protocol."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from harm_detection.schemas.content import ContentInput
from harm_detection.schemas.detection import DetectionResult


class ErrorCode(StrEnum):
    MISSING_CREDENTIALS = "missing_credentials"
    LIVE_CALLS_DISABLED = "live_calls_disabled"
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    UNSUPPORTED_MODEL = "unsupported_model"
    BAD_REQUEST = "bad_request"
    VALIDATION_ERROR = "validation_error"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    RATE_LIMITED = "rate_limited"
    OVERLOADED = "overloaded"
    SERVER_ERROR = "server_error"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    INVALID_RESPONSE = "invalid_response"
    MISSING_ANSWER = "missing_answer"
    INVALID_PROBABILITY = "invalid_probability"
    INPUT_REJECTED = "input_rejected"
    NO_RECORDED_RESULT = "no_recorded_result"


RETRYABLE = {
    ErrorCode.RATE_LIMITED,
    ErrorCode.OVERLOADED,
    ErrorCode.SERVER_ERROR,
    ErrorCode.TIMEOUT,
    ErrorCode.NETWORK_ERROR,
}

# Errors after which continuing a batch is pointless: every subsequent call fails the same way.
FATAL = {
    ErrorCode.MISSING_CREDENTIALS,
    ErrorCode.LIVE_CALLS_DISABLED,
    ErrorCode.AUTHENTICATION_FAILED,
    ErrorCode.PERMISSION_DENIED,
    ErrorCode.UNSUPPORTED_MODEL,
}


class ProviderError(Exception):
    def __init__(self, code: ErrorCode, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(max(retry_after, 0.0), self.max_delay_s)
        backoff = self.base_delay_s * (2 ** (attempt - 1))
        return min(backoff * (0.5 + random.random() / 2), self.max_delay_s)


def call_with_retries[T](
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, int]:
    """Run ``fn`` with bounded exponential backoff on retryable ``ProviderError``.

    Returns ``(result, attempts)``. Non-retryable errors are raised immediately;
    the final retryable error is raised once attempts are exhausted. The raised
    error carries the attempt count in ``attempts``.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn(), attempt
        except ProviderError as err:
            err.attempts = attempt  # type: ignore[attr-defined]
            if not err.retryable or attempt >= policy.max_attempts:
                raise
            sleep(policy.delay(attempt, err.retry_after))


class DetectionProvider(Protocol):
    name: str

    def detect(self, item: ContentInput) -> DetectionResult: ...

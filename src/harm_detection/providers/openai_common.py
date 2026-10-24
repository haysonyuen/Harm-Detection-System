"""Shared OpenAI SDK helpers: client construction and exception mapping."""

from __future__ import annotations

from typing import Any

from harm_detection.providers.base import ErrorCode, ProviderError


def make_openai_client(api_key: str, *, timeout_s: float = 60.0) -> Any:
    from openai import OpenAI

    # Retries are handled by our own bounded policy so behaviour is uniform across providers.
    return OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)


def map_openai_exception(exc: Exception) -> ProviderError:
    import openai

    msg = f"{type(exc).__name__}: {str(exc)[:300]}"
    if isinstance(exc, openai.APITimeoutError):
        return ProviderError(ErrorCode.TIMEOUT, msg)
    if isinstance(exc, openai.APIConnectionError):
        return ProviderError(ErrorCode.NETWORK_ERROR, msg)
    if isinstance(exc, openai.AuthenticationError):
        return ProviderError(ErrorCode.AUTHENTICATION_FAILED, msg)
    if isinstance(exc, openai.PermissionDeniedError):
        return ProviderError(ErrorCode.PERMISSION_DENIED, msg)
    if isinstance(exc, openai.NotFoundError):
        return ProviderError(ErrorCode.UNSUPPORTED_MODEL, msg)
    if isinstance(exc, openai.RateLimitError):
        return ProviderError(ErrorCode.RATE_LIMITED, msg)
    if isinstance(exc, openai.BadRequestError):
        return ProviderError(ErrorCode.BAD_REQUEST, msg)
    if isinstance(exc, openai.UnprocessableEntityError):
        return ProviderError(ErrorCode.VALIDATION_ERROR, msg)
    if isinstance(exc, openai.InternalServerError):
        return ProviderError(ErrorCode.SERVER_ERROR, msg)
    if isinstance(exc, openai.APIStatusError):
        return ProviderError(ErrorCode.HTTP_ERROR, msg)
    return ProviderError(ErrorCode.INVALID_RESPONSE, msg)

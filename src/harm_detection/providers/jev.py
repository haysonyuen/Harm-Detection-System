"""TypeSafe AI Jev adapter (System One API).

Contract verified against https://api.typesafe.ai/openapi.json (API version 0.2.0,
inspected 2026-10-23):

Request  POST /v1/systemone  {"model": str, "state": str|object|array,
                               "questions": {name: {"type": "noul", "instructions": ...,
                                                    "criteria": {"true": ..., "false": ...}}}}
Response 200 {"model": str, "answers": {name: {"type": "noul", "noul": float}},
              "usage": {"input_tokens": int, "output_tokens": int}}
Errors   400 / 401 / 413 / 415 / 422 / 429 / 529 with JSON bodies.

Jev returns probabilities only; it does not return rationales or evidence spans.
"""

from __future__ import annotations

import math
import time
from typing import Any

import httpx

from harm_detection.providers.base import (
    ErrorCode,
    ProviderError,
    RetryPolicy,
    call_with_retries,
)
from harm_detection.schemas.config import JevQuestionConfig
from harm_detection.schemas.content import ContentInput
from harm_detection.schemas.detection import SIGNAL_NAMES, DetectionResult, DetectionStatus
from harm_detection.security.redaction import redact

PROVIDER_NAME = "typesafe_jev"
SYSTEMONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"


def build_state(
    item: ContentInput, preamble: str, *, include_context: bool = True
) -> dict[str, Any]:
    """Build the ``state`` object. Absent context fields are omitted, never invented."""
    state: dict[str, Any] = {"note": preamble, "comment": item.text}
    if include_context:
        if item.parent_text is not None:
            state["parent_comment"] = item.parent_text
        if item.target_context is not None:
            state["target_context"] = item.target_context
    if not include_context or (item.parent_text is None and item.target_context is None):
        state["context_available"] = "No parent comment or target context was supplied."
    return state


def build_request(
    item: ContentInput,
    questions: JevQuestionConfig,
    model: str,
    *,
    include_context: bool = True,
) -> dict[str, Any]:
    return {
        "model": model,
        "state": build_state(item, questions.state_preamble, include_context=include_context),
        "questions": {q.signal: q.to_request() for q in questions.questions},
    }


def normalize_response(
    body: Any, questions: JevQuestionConfig
) -> tuple[dict[str, float], str, dict]:
    """Map a raw System One response to normalized signals.

    Raises ``ProviderError`` (INVALID_RESPONSE / MISSING_ANSWER / INVALID_PROBABILITY)
    if the response cannot be mapped reliably. Never fills defaults.
    """
    if not isinstance(body, dict):
        raise ProviderError(ErrorCode.INVALID_RESPONSE, "response body is not a JSON object")
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise ProviderError(ErrorCode.INVALID_RESPONSE, "response is missing the 'answers' object")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ProviderError(ErrorCode.INVALID_RESPONSE, "response is missing the 'model' field")
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}

    signals: dict[str, float] = {}
    expected = [q.signal for q in questions.questions]
    for name in expected:
        ans = answers.get(name)
        if not isinstance(ans, dict):
            raise ProviderError(ErrorCode.MISSING_ANSWER, f"no answer for question '{name}'")
        if ans.get("type") != "noul":
            raise ProviderError(
                ErrorCode.INVALID_RESPONSE,
                f"answer '{name}' has type {ans.get('type')!r}, expected 'noul'",
            )
        value = ans.get("noul")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProviderError(
                ErrorCode.INVALID_PROBABILITY, f"answer '{name}' is not numeric: {value!r}"
            )
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ProviderError(
                ErrorCode.INVALID_PROBABILITY, f"answer '{name}' out of range: {value!r}"
            )
        signals[name] = value
    missing = [s for s in SIGNAL_NAMES if s not in signals]
    if missing:
        raise ProviderError(
            ErrorCode.MISSING_ANSWER, f"question config lacks required signals: {missing}"
        )
    return signals, model, usage


def _error_from_response(resp: httpx.Response) -> ProviderError:
    status = resp.status_code
    try:
        body = resp.json()
    except ValueError:
        body = None
    detail = ""
    if isinstance(body, dict):
        detail = str(body.get("detail") or body.get("error") or body.get("message") or "")[:500]
    retry_after = None
    if "retry-after" in resp.headers:
        try:
            retry_after = float(resp.headers["retry-after"])
        except ValueError:
            retry_after = None
    msg = f"HTTP {status}" + (f": {detail}" if detail else "")

    def mentions_model() -> bool:
        if isinstance(body, dict) and isinstance(body.get("detail"), list):
            for d in body["detail"]:
                if isinstance(d, dict) and "model" in [str(x) for x in d.get("loc", [])]:
                    return True
        return "model" in detail.lower()

    if status == 401:
        return ProviderError(ErrorCode.AUTHENTICATION_FAILED, msg)
    if status == 403:
        return ProviderError(ErrorCode.PERMISSION_DENIED, msg)
    if status == 404 or (status in (400, 422) and mentions_model()):
        return ProviderError(ErrorCode.UNSUPPORTED_MODEL, msg)
    if status == 413:
        return ProviderError(ErrorCode.PAYLOAD_TOO_LARGE, msg)
    if status == 422:
        return ProviderError(ErrorCode.VALIDATION_ERROR, msg)
    if status in (400, 415):
        return ProviderError(ErrorCode.BAD_REQUEST, msg)
    if status == 429:
        return ProviderError(ErrorCode.RATE_LIMITED, msg, retry_after=retry_after)
    if status == 529:
        return ProviderError(ErrorCode.OVERLOADED, msg, retry_after=retry_after)
    if status >= 500:
        return ProviderError(ErrorCode.SERVER_ERROR, msg, retry_after=retry_after)
    return ProviderError(ErrorCode.HTTP_ERROR, msg)


class JevProvider:
    """Live Jev provider. Construct only when live calls are permitted."""

    name = PROVIDER_NAME

    def __init__(
        self,
        api_key: str | None,
        questions: JevQuestionConfig,
        *,
        model: str = "jev-latest",
        base_url: str = "https://api.typesafe.ai",
        timeout_s: float = 30.0,
        retry: RetryPolicy | None = None,
        include_context: bool = True,
        client: httpx.Client | None = None,
        sleep=time.sleep,
    ):
        self._api_key = api_key
        self.questions = questions
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.include_context = include_context
        self.retry = retry or RetryPolicy()
        self._client = client or httpx.Client(timeout=timeout_s)
        self._sleep = sleep

    # -- low level -----------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._client.post(
                self.base_url + SYSTEMONE_PATH, json=payload, headers=self._headers()
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                ErrorCode.TIMEOUT, f"request timed out: {type(exc).__name__}"
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                ErrorCode.NETWORK_ERROR, f"network error: {type(exc).__name__}"
            ) from exc
        if resp.status_code != 200:
            raise _error_from_response(resp)
        try:
            body = resp.json()
        except ValueError as exc:
            raise ProviderError(ErrorCode.INVALID_RESPONSE, "response is not valid JSON") from exc
        return body

    def list_models(self) -> list[dict[str, Any]]:
        if not self._api_key:
            raise ProviderError(ErrorCode.MISSING_CREDENTIALS, "TYPESAFE_API_KEY is not configured")
        try:
            resp = self._client.get(self.base_url + MODELS_PATH, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise ProviderError(ErrorCode.TIMEOUT, "request timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                ErrorCode.NETWORK_ERROR, f"network error: {type(exc).__name__}"
            ) from exc
        if resp.status_code != 200:
            raise _error_from_response(resp)
        body = resp.json()
        models = body.get("models") if isinstance(body, dict) else None
        if not isinstance(models, list):
            raise ProviderError(ErrorCode.INVALID_RESPONSE, "models response missing 'models' list")
        return models

    # -- public --------------------------------------------------------------
    def detect(self, item: ContentInput) -> DetectionResult:
        common = {
            "case_id": item.case_id,
            "provider": self.name,
            "requested_model": self.model,
            "model_config_version": self.questions.version,
        }
        if not self._api_key:
            return DetectionResult.failure(
                code=ErrorCode.MISSING_CREDENTIALS,
                message="TYPESAFE_API_KEY is not configured; live Jev inference is unavailable",
                **common,
            )
        payload = build_request(
            item, self.questions, self.model, include_context=self.include_context
        )
        started = time.perf_counter()
        attempts = 1
        try:
            body, attempts = call_with_retries(
                lambda: self._post(payload), self.retry, sleep=self._sleep
            )
            latency = int((time.perf_counter() - started) * 1000)
            signals, resolved, usage = normalize_response(body, self.questions)
        except ProviderError as err:
            latency = int((time.perf_counter() - started) * 1000)
            raw = locals().get("body")
            return DetectionResult.failure(
                code=err.code,
                message=err.message,
                latency_ms=latency,
                raw_response=redact(raw, [self._api_key]) if isinstance(raw, dict) else None,
                provider_usage={"attempts": getattr(err, "attempts", attempts)},
                **common,
            )
        return DetectionResult(
            **common,
            resolved_model=resolved,
            signals=signals,
            status=DetectionStatus.SUCCESS,
            latency_ms=latency,
            provider_usage={
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "attempts": attempts,
            },
            raw_response=redact(body, [self._api_key]),
            data_origin="live",
        )

    def close(self) -> None:
        self._client.close()

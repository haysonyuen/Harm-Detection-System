"""OpenAI Moderation adapter: an independent, offline benchmark only.

Its outputs never influence Jev classification or the routing recommendation.
Only text is sent (input parity with the text-only Jev benchmark representation).
"""

from __future__ import annotations

import math
import time
from typing import Any

from harm_detection.providers.base import ErrorCode, ProviderError, RetryPolicy, call_with_retries
from harm_detection.providers.openai_common import map_openai_exception
from harm_detection.schemas.content import ContentInput
from harm_detection.schemas.detection import DetectionResult, DetectionStatus

PROVIDER_NAME = "openai_moderation"
CATEGORIES = ("harassment", "harassment/threatening")

# Normalized signal keys stored for moderation predictions.
MODERATION_SIGNAL_KEYS = (
    "harassment",
    "harassment_threatening",
    "harassment_flagged",
    "harassment_threatening_flagged",
)


def normalize_moderation(body: dict[str, Any]) -> tuple[dict[str, float], str | None, str | None]:
    """Map a moderation response (dict form) to normalized signals.

    Flags are the provider's native category decisions, stored as 1.0 / 0.0.
    """
    results = body.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ProviderError(ErrorCode.INVALID_RESPONSE, "expected exactly one moderation result")
    res = results[0]
    cats, scores = res.get("categories"), res.get("category_scores")
    if not isinstance(cats, dict) or not isinstance(scores, dict):
        raise ProviderError(
            ErrorCode.INVALID_RESPONSE, "moderation result missing categories/scores"
        )
    out: dict[str, float] = {}
    for cat in CATEGORIES:
        key = cat.replace("/", "_")
        if cat not in scores or cat not in cats:
            raise ProviderError(ErrorCode.MISSING_ANSWER, f"moderation category '{cat}' missing")
        score, flag = scores[cat], cats[cat]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ProviderError(ErrorCode.INVALID_PROBABILITY, f"score for '{cat}' is not numeric")
        if not 0.0 <= float(score) <= 1.0:
            raise ProviderError(ErrorCode.INVALID_PROBABILITY, f"score for '{cat}' out of range")
        if not isinstance(flag, bool):
            raise ProviderError(ErrorCode.INVALID_RESPONSE, f"flag for '{cat}' is not boolean")
        out[key] = float(score)
        out[f"{key}_flagged"] = 1.0 if flag else 0.0
    return out, body.get("model"), body.get("id")


def _subset_body(body: dict[str, Any]) -> dict[str, Any]:
    """Keep only the categories relevant to this benchmark in stored raw responses."""
    res = (body.get("results") or [{}])[0]
    return {
        "id": body.get("id"),
        "model": body.get("model"),
        "results": [
            {
                "flagged": res.get("flagged"),
                "categories": {c: (res.get("categories") or {}).get(c) for c in CATEGORIES},
                "category_scores": {
                    c: (res.get("category_scores") or {}).get(c) for c in CATEGORIES
                },
            }
        ],
    }


class OpenAIModerationProvider:
    name = PROVIDER_NAME

    def __init__(
        self,
        api_key: str | None,
        *,
        model: str = "omni-moderation-latest",
        client: Any | None = None,
        retry: RetryPolicy | None = None,
        sleep=time.sleep,
    ):
        self._api_key = api_key
        self.model = model
        self.retry = retry or RetryPolicy()
        self._sleep = sleep
        self._client = client
        if self._client is None and api_key:
            from harm_detection.providers.openai_common import make_openai_client

            self._client = make_openai_client(api_key)

    def _call(self, text: str) -> dict[str, Any]:
        try:
            resp = self._client.moderations.create(model=self.model, input=text)
        except ProviderError:
            raise
        except Exception as exc:  # SDK exceptions
            raise map_openai_exception(exc) from exc
        body = resp.model_dump(by_alias=True) if hasattr(resp, "model_dump") else resp
        if not isinstance(body, dict):
            raise ProviderError(ErrorCode.INVALID_RESPONSE, "moderation response is not an object")
        return body

    def detect(self, item: ContentInput) -> DetectionResult:
        common = {
            "case_id": item.case_id,
            "provider": self.name,
            "requested_model": self.model,
            "model_config_version": "openai_moderation_native_v1",
        }
        if not self._api_key or self._client is None:
            return DetectionResult.failure(
                code=ErrorCode.MISSING_CREDENTIALS,
                message="OPENAI_API_KEY is not configured; OpenAI Moderation is unavailable",
                **common,
            )
        started = time.perf_counter()
        body = None
        try:
            body, attempts = call_with_retries(
                lambda: self._call(item.text), self.retry, sleep=self._sleep
            )
            signals, resolved, response_id = normalize_moderation(body)
        except ProviderError as err:
            return DetectionResult.failure(
                code=err.code,
                message=err.message,
                latency_ms=int((time.perf_counter() - started) * 1000),
                raw_response=_subset_body(body) if isinstance(body, dict) else None,
                **common,
            )
        return DetectionResult(
            **common,
            resolved_model=resolved,
            signals=signals,
            status=DetectionStatus.SUCCESS,
            latency_ms=int((time.perf_counter() - started) * 1000),
            provider_usage={"attempts": attempts},
            provider_response_id=response_id,
            raw_response=_subset_body(body),
            data_origin="live",
        )

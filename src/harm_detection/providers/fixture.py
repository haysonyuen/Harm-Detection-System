"""Offline fixture-replay provider.

Replays pre-recorded fixture outputs for known fixture case IDs only. Arbitrary text
never receives an invented prediction: unknown cases return a failure with
``no_recorded_result``. Every result is tagged with its ``data_origin`` so the UI
and reports can never present it as live model output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harm_detection.providers.base import ErrorCode
from harm_detection.schemas.content import ContentInput
from harm_detection.schemas.detection import DetectionResult, DetectionStatus
from harm_detection.settings import FIXTURES_DIR

DEFAULT_PREDICTIONS = FIXTURES_DIR / "demo_predictions.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class FixtureReplayProvider:
    """Replays recorded outputs for one underlying provider (e.g. typesafe_jev)."""

    def __init__(self, replayed_provider: str, path: Path = DEFAULT_PREDICTIONS):
        self.replayed_provider = replayed_provider
        self.name = replayed_provider  # predictions stay attributed to the provider they mimic
        self._records = {
            r["case_id"]: r for r in load_jsonl(path) if r.get("provider") == replayed_provider
        }

    def has_record(self, case_id: str) -> bool:
        return case_id in self._records

    @property
    def case_ids(self) -> list[str]:
        return sorted(self._records)

    def detect(self, item: ContentInput) -> DetectionResult:
        rec = self._records.get(item.case_id)
        common = {"case_id": item.case_id, "provider": self.replayed_provider}
        if rec is None:
            return DetectionResult.failure(
                code=ErrorCode.NO_RECORDED_RESULT,
                message="No recorded result exists for this case; live inference is unavailable offline.",
                data_origin="synthetic_fixture",
                **common,
            )
        origin = rec.get("data_origin", "synthetic_fixture")
        base = {
            **common,
            "requested_model": rec.get("requested_model"),
            "resolved_model": rec.get("resolved_model"),
            "model_config_version": rec.get("model_config_version"),
            "latency_ms": rec.get("latency_ms"),
            "provider_usage": rec.get("provider_usage") or {},
            "raw_response": rec.get("raw_response"),
            "data_origin": origin,
        }
        if rec.get("status") == "success":
            return DetectionResult(signals=rec["signals"], status=DetectionStatus.SUCCESS, **base)
        return DetectionResult(
            status=DetectionStatus.FAILURE, signals=None, error=rec["error"], **base
        )

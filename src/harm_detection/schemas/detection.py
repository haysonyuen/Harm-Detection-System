"""Normalized detection and routing contracts (PRD §7.2, §7.3)."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

SIGNAL_NAMES: tuple[str, ...] = (
    "targeted_individual",
    "abusive_behavior",
    "physical_threat",
    "context_sufficient",
)


class DetectionStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class JevSignals(BaseModel):
    """Probabilities in [0, 1]. Construction fails on missing / non-finite / out-of-range values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    targeted_individual: float = Field(ge=0.0, le=1.0)
    abusive_behavior: float = Field(ge=0.0, le=1.0)
    physical_threat: float = Field(ge=0.0, le=1.0)
    context_sufficient: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _finite(self) -> JevSignals:
        for name in SIGNAL_NAMES:
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"signal {name} is not finite")
        return self


class DetectionResult(BaseModel):
    """Provider-agnostic normalized detection output.

    ``signals`` is only populated when ``status == success``. On failure the
    signals are ``None`` -- never default zeros.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    provider: str
    requested_model: str | None = None
    resolved_model: str | None = None
    model_config_version: str | None = None
    signals: dict[str, float] | None = None
    status: DetectionStatus
    latency_ms: int | None = None
    provider_usage: dict[str, Any] = Field(default_factory=dict)
    provider_response_id: str | None = None
    error: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
    data_origin: str = "live"  # live | recorded_live | synthetic_fixture

    @model_validator(mode="after")
    def _consistency(self) -> DetectionResult:
        if self.status == DetectionStatus.SUCCESS:
            if self.signals is None:
                raise ValueError("successful detection must carry signals")
            if self.error is not None:
                raise ValueError("successful detection must not carry an error")
        else:
            if self.signals is not None:
                raise ValueError("failed detection must not carry signals")
            if not self.error:
                raise ValueError("failed detection must carry an error")
        return self

    @classmethod
    def failure(
        cls,
        case_id: str,
        provider: str,
        code: str,
        message: str,
        **kwargs: Any,
    ) -> DetectionResult:
        return cls(
            case_id=case_id,
            provider=provider,
            status=DetectionStatus.FAILURE,
            signals=None,
            error={"code": code, "message": message},
            **kwargs,
        )


class Disposition(StrEnum):
    NO_FLAG = "no_flag"
    STANDARD_REVIEW = "standard_review"
    PRIORITY_REVIEW = "priority_review"


DISPOSITION_RANK = {
    Disposition.NO_FLAG: 0,
    Disposition.STANDARD_REVIEW: 1,
    Disposition.PRIORITY_REVIEW: 2,
}


class RoutingStatus(StrEnum):
    SUCCESS = "success"
    NOT_EVALUATED = "not_evaluated"
    DETECTION_FAILED = "detection_failed"
    ROUTING_FAILED = "routing_failed"


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    disposition: Disposition | None
    routing_version: str
    triggered_rule_ids: list[str]
    matched_rule_ids: list[str] = Field(default_factory=list)
    reasons: list[str]
    context_warning: bool
    status: RoutingStatus

    @model_validator(mode="after")
    def _failure_has_no_disposition(self) -> RoutingDecision:
        if self.status == RoutingStatus.SUCCESS and self.disposition is None:
            raise ValueError("successful routing requires a disposition")
        if self.status != RoutingStatus.SUCCESS and self.disposition is not None:
            raise ValueError("failure states must not carry an ordinary disposition")
        return self

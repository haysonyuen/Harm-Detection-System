"""Typed versions of the versioned YAML configuration files."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ConfigType = Literal["policy", "jev_questions", "label_mapping", "routing", "pricing"]
CONFIG_TYPES: tuple[str, ...] = ("policy", "jev_questions", "label_mapping", "routing", "pricing")


class RoutingThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threat_priority: float = Field(ge=0.0, le=1.0)
    harassment_standard: float = Field(ge=0.0, le=1.0)
    targeting: float = Field(ge=0.0, le=1.0)
    context_sufficiency: float = Field(ge=0.0, le=1.0)
    uncertain_risk: float = Field(ge=0.0, le=1.0)


class RoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    description: str = ""
    tuning_status: Literal["untuned_placeholder", "human_selected", "candidate"] = (
        "untuned_placeholder"
    )
    thresholds: RoutingThresholds
    require_targeting_for_standard: bool = True
    # Signals whose value counts as "meaningful potential risk" for R-CTX-01.
    context_rule_risk_signals: list[str] = Field(
        default_factory=lambda: ["abusive_behavior", "physical_threat"]
    )
    notes: list[str] = Field(default_factory=list)


class JevQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: str
    type: Literal["noul"] = "noul"
    instructions: str
    criteria_true: str | None = None
    criteria_false: str | None = None

    def to_request(self) -> dict[str, Any]:
        q: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria_true or self.criteria_false:
            q["criteria"] = {"true": self.criteria_true, "false": self.criteria_false}
        return q


class JevQuestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    policy_version: str
    description: str = ""
    state_preamble: str
    questions: list[JevQuestion]


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    policy_id: str
    version: str
    title: str
    disclaimer: str
    effective_date: str
    references: list[dict[str, Any]]
    categories: dict[str, Any]
    limitations: list[str]
    context_sufficiency_guidance: str
    change_notes: list[str] = Field(default_factory=list)


class LabelMappingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    dataset: str
    description: str = ""
    binarization_threshold: float = Field(ge=0.0, le=1.0)
    proxies: dict[str, str]  # proxy name -> source annotation field
    primary_proxies: list[str]
    reference_routing: dict[str, str]
    notes: list[str] = Field(default_factory=list)


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_million_usd: float = Field(ge=0.0)
    output_per_million_usd: float = Field(ge=0.0)
    source: str = ""
    as_of: str = ""


class PricingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    currency: str = "USD"
    disclaimer: str = ""
    models: dict[str, ModelPrice]
    analysis_limits: dict[str, Any] = Field(default_factory=dict)

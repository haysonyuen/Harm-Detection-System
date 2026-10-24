"""Structured output contract for GPT-assisted error analysis (PRD §12.4).

These models are used directly as the Responses API ``text_format``; every field is
required (no defaults) so the generated JSON schema is strict-mode compatible.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ChangeType = Literal[
    "question_update",
    "threshold_change",
    "routing_rule_change",
    "policy_clarification",
    "regression_test",
]
AffectedComponent = Literal[
    "jev_questions",
    "routing_thresholds",
    "routing_rules",
    "policy_rubric",
    "regression_tests",
]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str
    case_ids: list[str]


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypothesis: str
    supporting_case_ids: list[str]
    supporting_evidence: list[str]
    quoted_excerpts: list[str]
    uncertainties: list[str]


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recommendation_id: str
    change_type: ChangeType
    affected_component: AffectedComponent
    proposed_change: str
    referenced_rule_ids: list[str]
    expected_benefit: str
    possible_regressions: list[str]
    validation_plan: str


class AnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_id: str
    scope: Literal["single_case", "case_group", "aggregate"]
    error_type: Literal["false_positive", "false_negative", "misprioritized", "mixed"]
    observations: list[Observation]
    hypotheses: list[Hypothesis]
    recommendations: list[Recommendation]
    limitations: list[str]
    unsupported_or_unknown: list[str]

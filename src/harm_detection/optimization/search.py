"""Deterministic threshold grid search over stored predictions (PRD §11.2-11.3).

No provider calls: every candidate is evaluated by re-applying the pure routing
engine to cached normalized signals.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from harm_detection.evaluation.metrics import binary_counts, routing_metrics
from harm_detection.optimization.pareto import pareto_front
from harm_detection.routing.engine import RULE_PRI, RULE_STD, route
from harm_detection.schemas.config import RoutingConfig, RoutingThresholds
from harm_detection.schemas.detection import Disposition

THRESHOLD_KEYS = (
    "threat_priority",
    "harassment_standard",
    "targeting",
    "context_sufficiency",
    "uncertain_risk",
)

DEFAULT_SEARCH_SPACE: dict[str, list[float]] = {
    "threat_priority": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "harassment_standard": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "targeting": [0.3, 0.5, 0.7],
    "context_sufficiency": [0.3, 0.5],
    "uncertain_risk": [0.3, 0.5, 0.7],
}


@dataclass
class Constraints:
    """Provisional defaults until an initial benchmark is available."""

    min_threat_recall: float | None = 0.8
    max_review_volume: float | None = 0.3
    min_threat_precision: float | None = None
    max_under_prioritization: float | None = None
    min_threat_support: int | None = 5
    provisional: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class EvalCase:
    case_id: str
    signals: dict[str, float]
    threat_proxy: bool
    insult_proxy: bool
    ref_route: Disposition
    representative: bool


@dataclass
class SearchResult:
    candidates: list[dict[str, Any]]
    pareto_indices: list[int]
    objectives: list[tuple[str, str]]
    n_cases: int
    n_representative: int
    search_space: dict[str, list[float]] = field(default_factory=dict)


def grid(space: dict[str, Sequence[float]], max_candidates: int = 5000) -> list[dict[str, float]]:
    for k in THRESHOLD_KEYS:
        if k not in space or not space[k]:
            raise ValueError(f"search space missing values for {k}")
        for v in space[k]:
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"{k} value {v} outside [0, 1]")
    combos = list(
        itertools.product(*[sorted({float(v) for v in space[k]}) for k in THRESHOLD_KEYS])
    )
    if len(combos) > max_candidates:
        raise ValueError(f"search space has {len(combos)} candidates; limit is {max_candidates}")
    return [dict(zip(THRESHOLD_KEYS, c, strict=True)) for c in combos]


def evaluate_candidate(
    cases: Sequence[EvalCase], base: RoutingConfig, thresholds: dict[str, float]
) -> dict[str, Any]:
    cfg = base.model_copy(update={"thresholds": RoutingThresholds(**thresholds)})
    preds, threat_pred, har_pred = [], [], []
    for c in cases:
        d = route(c.case_id, c.signals, cfg)
        preds.append(d.disposition)
        threat_pred.append(RULE_PRI in d.matched_rule_ids)
        har_pred.append(RULE_STD in d.matched_rule_ids)
    refs = [c.ref_route for c in cases]
    rm = routing_metrics(refs, preds, with_ci=False)
    tc = binary_counts([c.threat_proxy for c in cases], threat_pred)
    ic = binary_counts([c.insult_proxy for c in cases], har_pred)

    def r(n, d):
        return n / d if d else None

    rep_mask = np.array([c.representative for c in cases], dtype=bool)
    rep_preds = [p for p, m in zip(preds, rep_mask, strict=True) if m]
    rep_review = r(sum(1 for p in rep_preds if p != Disposition.NO_FLAG), len(rep_preds))
    return {
        **thresholds,
        "threat_recall": r(tc["tp"], tc["tp"] + tc["fn"]),
        "threat_precision": r(tc["tp"], tc["tp"] + tc["fp"]),
        "threat_support": tc["tp"] + tc["fn"],
        "insult_recall": r(ic["tp"], ic["tp"] + ic["fn"]),
        "insult_precision": r(ic["tp"], ic["tp"] + ic["fp"]),
        "priority_recall": rm["priority_recall"].value,
        "under_prioritization": rm["under_prioritization_rate"].value,
        "priority_false_alarm_rate": rm["priority_false_alarm_rate"].value,
        "review_volume_all": rm["total_review_rate"].value,
        "review_volume_representative": rep_review,
        "representative_n": len(rep_preds),
    }


def constraint_failures(c: dict[str, Any], k: Constraints) -> list[str]:
    fails = []

    def below(v, lim):
        return v is None or v < lim

    def above(v, lim):
        return v is None or v > lim

    if k.min_threat_recall is not None and below(c["threat_recall"], k.min_threat_recall):
        fails.append(f"threat recall < {k.min_threat_recall}")
    if k.max_review_volume is not None and above(
        c["review_volume_representative"], k.max_review_volume
    ):
        fails.append(f"representative review volume > {k.max_review_volume}")
    if k.min_threat_precision is not None and below(c["threat_precision"], k.min_threat_precision):
        fails.append(f"threat precision < {k.min_threat_precision}")
    if k.max_under_prioritization is not None and above(
        c["under_prioritization"], k.max_under_prioritization
    ):
        fails.append(f"under-prioritization > {k.max_under_prioritization}")
    if k.min_threat_support is not None and c["threat_support"] < k.min_threat_support:
        fails.append(f"threat-proxy support {c['threat_support']} < {k.min_threat_support}")
    return fails


def run_search(
    cases: Sequence[EvalCase],
    base: RoutingConfig,
    space: dict[str, Sequence[float]] | None = None,
    constraints: Constraints | None = None,
    *,
    extra_objectives: Sequence[tuple[str, str]] = (),
) -> SearchResult:
    space = {k: list(v) for k, v in (space or DEFAULT_SEARCH_SPACE).items()}
    constraints = constraints or Constraints()
    if not cases:
        raise ValueError("no evaluable cases with stored successful predictions")
    if not any(c.representative for c in cases):
        raise ValueError("no representative cases: review volume cannot be estimated")
    objectives = [
        ("threat_recall", "max"),
        ("review_volume_representative", "min"),
        *extra_objectives,
    ]
    candidates = []
    for i, th in enumerate(grid(space)):
        c = evaluate_candidate(cases, base, th)
        c["candidate_id"] = i
        c["constraint_failures"] = constraint_failures(c, constraints)
        c["passes_constraints"] = not c["constraint_failures"]
        candidates.append(c)
    feasible = [c for c in candidates if c["passes_constraints"]]
    front_local = pareto_front(feasible, objectives)
    front_ids = {feasible[i]["candidate_id"] for i in front_local}
    for c in candidates:
        c["pareto_optimal"] = c["candidate_id"] in front_ids
    return SearchResult(
        candidates=candidates,
        pareto_indices=sorted(front_ids),
        objectives=list(objectives),
        n_cases=len(cases),
        n_representative=sum(c.representative for c in cases),
        search_space=space,
    )

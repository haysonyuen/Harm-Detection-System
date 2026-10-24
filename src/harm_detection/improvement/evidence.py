"""Evidence-reference validation for AI analysis output (PRD §12.5)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from harm_detection.improvement.schemas import AnalysisOutput
from harm_detection.routing.engine import ALL_RULE_IDS

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def collect_strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in collect_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in collect_strings(v)]
    return []


def validate_evidence(
    output: AnalysisOutput,
    supplied_case_ids: Iterable[str],
    supplied_material: Any,
    known_rule_ids: Iterable[str] = ALL_RULE_IDS,
) -> list[str]:
    """Return a list of violations (empty when valid).

    * every referenced case ID must be in the supplied evidence,
    * every referenced rule ID must exist,
    * every quoted excerpt must be an actual substring of the supplied material
      (whitespace-normalized, case-sensitive),
    * recommendation IDs must be unique.
    """
    case_ids = set(supplied_case_ids)
    rules = set(known_rule_ids)
    corpus = [_norm(s) for s in collect_strings(supplied_material)]
    problems: list[str] = []

    for i, obs in enumerate(output.observations):
        for cid in obs.case_ids:
            if cid not in case_ids:
                problems.append(f"observation {i}: unknown case ID {cid!r}")
    for i, h in enumerate(output.hypotheses):
        for cid in h.supporting_case_ids:
            if cid not in case_ids:
                problems.append(f"hypothesis {i}: unknown case ID {cid!r}")
        for q in h.quoted_excerpts:
            nq = _norm(q)
            if not nq or not any(nq in s for s in corpus):
                problems.append(
                    f"hypothesis {i}: quoted excerpt not found in supplied material: {q[:80]!r}"
                )
    seen = set()
    for r in output.recommendations:
        if r.recommendation_id in seen:
            problems.append(f"duplicate recommendation_id {r.recommendation_id!r}")
        seen.add(r.recommendation_id)
        for rid in r.referenced_rule_ids:
            if rid not in rules:
                problems.append(f"recommendation {r.recommendation_id}: unknown rule ID {rid!r}")
    return problems

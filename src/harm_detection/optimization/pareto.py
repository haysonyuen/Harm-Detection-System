"""Pareto dominance and non-dominated filtering (PRD §11.4).

Each objective is (key, direction) with direction "max" or "min". Candidate A
dominates B when A is at least as good on every objective and strictly better on
at least one. Candidates with an undefined (None) objective value are excluded
from the front.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

Objective = tuple[str, str]


def _better_or_equal(a: float, b: float, direction: str) -> bool:
    return a >= b if direction == "max" else a <= b


def _strictly_better(a: float, b: float, direction: str) -> bool:
    return a > b if direction == "max" else a < b


def dominates(a: dict[str, Any], b: dict[str, Any], objectives: Sequence[Objective]) -> bool:
    for _key, direction in objectives:
        if direction not in ("max", "min"):
            raise ValueError(f"invalid direction {direction!r}")
    if any(a.get(k) is None or b.get(k) is None for k, _ in objectives):
        return False
    if not all(_better_or_equal(a[k], b[k], d) for k, d in objectives):
        return False
    return any(_strictly_better(a[k], b[k], d) for k, d in objectives)


def pareto_front(
    candidates: Sequence[dict[str, Any]], objectives: Sequence[Objective]
) -> list[int]:
    """Indices of non-dominated candidates (among those with all objectives defined)."""
    valid = [
        i for i, c in enumerate(candidates) if all(c.get(k) is not None for k, _ in objectives)
    ]
    front = []
    for i in valid:
        if not any(dominates(candidates[j], candidates[i], objectives) for j in valid if j != i):
            front.append(i)
    return front

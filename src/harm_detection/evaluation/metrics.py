"""Metric calculations (PRD §10.3, §10.4).

Zero-denominator policy: a metric with a zero denominator is ``None`` ("unavailable"),
never 0 or 1. Confidence intervals: percentile bootstrap (default 1,000 resamples,
fixed seed, 95%); resamples where the metric is undefined are skipped, and no
interval is reported if fewer than half the resamples are defined.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix

from harm_detection.schemas.detection import DISPOSITION_RANK, Disposition

CALCULATION_VERSION = "metrics_v1"
ROUTE_ORDER = [Disposition.NO_FLAG, Disposition.STANDARD_REVIEW, Disposition.PRIORITY_REVIEW]


@dataclass(frozen=True)
class MetricValue:
    value: float | None
    numerator: float | None
    denominator: float | None
    ci: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "ci": list(self.ci) if self.ci else None,
        }


def ratio(num: float, den: float) -> MetricValue:
    return MetricValue(num / den if den else None, float(num), float(den))


def binary_counts(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, int]:
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same length")
    # Plain numpy (not sklearn) because this runs inside bootstrap loops.
    return {
        "tp": int(np.sum(yt & yp)),
        "fp": int(np.sum(~yt & yp)),
        "tn": int(np.sum(~yt & ~yp)),
        "fn": int(np.sum(yt & ~yp)),
    }


def _precision(c):
    return ratio(c["tp"], c["tp"] + c["fp"])


def _recall(c):
    return ratio(c["tp"], c["tp"] + c["fn"])


def _f1(c):
    den = 2 * c["tp"] + c["fp"] + c["fn"]
    return MetricValue((2 * c["tp"]) / den if den else None, 2.0 * c["tp"], float(den))


def _fpr(c):
    return ratio(c["fp"], c["fp"] + c["tn"])


def _fnr(c):
    return ratio(c["fn"], c["fn"] + c["tp"])


BINARY_METRICS: dict[str, Callable[[dict[str, int]], MetricValue]] = {
    "precision": _precision,
    "recall": _recall,
    "f1": _f1,
    "false_positive_rate": _fpr,
    "false_negative_rate": _fnr,
}


def bootstrap_ci(
    stat: Callable[[np.ndarray], float | None],
    n: int,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = stat(idx)
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot / 2:
        return None
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def binary_metrics(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
    *,
    with_ci: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, MetricValue]:
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    c = binary_counts(yt, yp)
    out: dict[str, MetricValue] = {k: MetricValue(float(v), None, None) for k, v in c.items()}
    out["positives"] = MetricValue(float(c["tp"] + c["fn"]), None, None)
    out["negatives"] = MetricValue(float(c["tn"] + c["fp"]), None, None)
    for name, fn in BINARY_METRICS.items():
        mv = fn(c)
        ci = None
        if with_ci and mv.value is not None and name in ("precision", "recall", "f1"):
            ci = bootstrap_ci(
                lambda idx, fn=fn: fn(binary_counts(yt[idx], yp[idx])).value,
                len(yt),
                n_boot=n_boot,
                seed=seed,
            )
        out[name] = MetricValue(mv.value, mv.numerator, mv.denominator, ci)
    return out


def routing_confusion(ref: Sequence[Disposition], pred: Sequence[Disposition]) -> list[list[int]]:
    """3x3 matrix; rows = proxy reference, columns = predicted, order no_flag/standard/priority."""
    if len(ref) == 0:
        return [[0] * 3 for _ in range(3)]
    labels = [d.value for d in ROUTE_ORDER]
    m = confusion_matrix(
        [Disposition(r).value for r in ref], [Disposition(p).value for p in pred], labels=labels
    )
    return m.astype(int).tolist()


def routing_metrics(
    ref: Sequence[Disposition],
    pred: Sequence[Disposition],
    *,
    with_ci: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, MetricValue]:
    """Routing-proxy metrics.

    priority_recall              P(pred = priority | ref = priority)
    under_prioritization_rate    P(pred rank < ref rank | ref in {standard, priority})
    priority_false_alarm_rate    P(pred = priority | ref != priority)
    standard_review_detection    P(pred in {standard, priority} | ref = standard)
    total_review_rate            P(pred != no_flag)
    priority_review_rate         P(pred = priority)
    no_flag_rate                 P(pred = no_flag)
    proxy_routing_accuracy       P(pred = ref)
    """
    r = np.array([DISPOSITION_RANK[Disposition(x)] for x in ref], dtype=int)
    p = np.array([DISPOSITION_RANK[Disposition(x)] for x in pred], dtype=int)

    def stats(idx: np.ndarray | None = None) -> dict[str, MetricValue]:
        rr, pp = (r, p) if idx is None else (r[idx], p[idx])
        ref_pri = rr == 2
        ref_flag = rr >= 1
        return {
            "priority_recall": ratio(int(((pp == 2) & ref_pri).sum()), int(ref_pri.sum())),
            "under_prioritization_rate": ratio(
                int(((pp < rr) & ref_flag).sum()), int(ref_flag.sum())
            ),
            "priority_false_alarm_rate": ratio(
                int(((pp == 2) & ~ref_pri).sum()), int((~ref_pri).sum())
            ),
            "standard_review_detection_rate": ratio(
                int(((pp >= 1) & (rr == 1)).sum()), int((rr == 1).sum())
            ),
            "total_review_rate": ratio(int((pp >= 1).sum()), len(pp)),
            "priority_review_rate": ratio(int((pp == 2).sum()), len(pp)),
            "no_flag_rate": ratio(int((pp == 0).sum()), len(pp)),
            "proxy_routing_accuracy": ratio(int((pp == rr).sum()), len(pp)),
        }

    base = stats()
    if not with_ci:
        return base
    out = {}
    for name, mv in base.items():
        ci = None
        if mv.value is not None and name in (
            "priority_recall",
            "under_prioritization_rate",
            "total_review_rate",
        ):
            ci = bootstrap_ci(
                lambda idx, name=name: stats(idx)[name].value, len(r), n_boot=n_boot, seed=seed
            )
        out[name] = MetricValue(mv.value, mv.numerator, mv.denominator, ci)
    return out

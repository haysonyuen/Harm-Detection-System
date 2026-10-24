"""Deterministic routing engine (PRD §8).

Pure functions only: no Streamlit, no I/O, no network, no global mutable state.
Given identical signals and configuration the result is always identical.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from harm_detection.schemas.config import RoutingConfig
from harm_detection.schemas.detection import (
    SIGNAL_NAMES,
    Disposition,
    RoutingDecision,
    RoutingStatus,
)

RULE_ERR = "R-ERR-01"
RULE_PRI = "R-PRI-01"
RULE_STD = "R-STD-01"
RULE_CTX = "R-CTX-01"
RULE_NF = "R-NF-01"
ALL_RULE_IDS = (RULE_ERR, RULE_PRI, RULE_STD, RULE_CTX, RULE_NF)

RULE_DESCRIPTIONS = {
    RULE_ERR: "Required inference data missing or invalid: non-success status, never No Flag.",
    RULE_PRI: "Physical-threat signal met the priority threshold.",
    RULE_STD: "Abusive-behavior signal met the standard-review threshold with targeting criteria satisfied.",
    RULE_CTX: "Meaningful risk signal present while contextual sufficiency was below threshold.",
    RULE_NF: "No routing rule matched.",
}


def _invalid_signals(signals: Mapping[str, object] | None) -> list[str]:
    if signals is None:
        return list(SIGNAL_NAMES)
    bad = []
    for name in SIGNAL_NAMES:
        v = signals.get(name)
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v < 0.0
            or v > 1.0
        ):
            bad.append(name)
    return bad


def route(
    case_id: str,
    signals: Mapping[str, float] | None,
    config: RoutingConfig,
    *,
    detection_succeeded: bool = True,
) -> RoutingDecision:
    """Apply rule precedence R-ERR-01 > R-PRI-01 > R-STD-01 > R-CTX-01 > R-NF-01."""
    version = config.version
    t = config.thresholds

    # R-ERR-01 ---------------------------------------------------------------
    if not detection_succeeded:
        return RoutingDecision(
            case_id=case_id,
            disposition=None,
            routing_version=version,
            triggered_rule_ids=[RULE_ERR],
            reasons=["Detection failed; no routing disposition was produced"],
            context_warning=False,
            status=RoutingStatus.DETECTION_FAILED,
        )
    bad = _invalid_signals(signals)
    if bad:
        return RoutingDecision(
            case_id=case_id,
            disposition=None,
            routing_version=version,
            triggered_rule_ids=[RULE_ERR],
            reasons=[f"Missing or invalid required signal(s): {', '.join(bad)}"],
            context_warning=False,
            status=RoutingStatus.ROUTING_FAILED,
        )
    assert signals is not None
    threat = float(signals["physical_threat"])
    abusive = float(signals["abusive_behavior"])
    targeted = float(signals["targeted_individual"])
    context = float(signals["context_sufficient"])

    context_warning = context < t.context_sufficiency

    matched: list[str] = []
    reasons: list[str] = []

    pri = threat >= t.threat_priority
    if pri:
        matched.append(RULE_PRI)
        reasons.append(
            f"Physical-threat score {threat:.3f} met the priority threshold {t.threat_priority:.3f}"
        )

    targeting_ok = (not config.require_targeting_for_standard) or targeted >= t.targeting
    std = abusive >= t.harassment_standard and targeting_ok
    if std:
        matched.append(RULE_STD)
        reasons.append(
            f"Abusive-behavior score {abusive:.3f} met the standard-review threshold "
            f"{t.harassment_standard:.3f}"
            + (
                f" and targeting score {targeted:.3f} met {t.targeting:.3f}"
                if config.require_targeting_for_standard
                else ""
            )
        )

    risk = max(float(signals[s]) for s in config.context_rule_risk_signals)
    ctx = risk >= t.uncertain_risk and context_warning
    if ctx:
        matched.append(RULE_CTX)
        reasons.append(
            f"Risk score {risk:.3f} met the uncertain-risk threshold {t.uncertain_risk:.3f} while "
            f"context sufficiency {context:.3f} was below {t.context_sufficiency:.3f}"
        )

    if pri:
        disposition, deciding = Disposition.PRIORITY_REVIEW, RULE_PRI
    elif std:
        disposition, deciding = Disposition.STANDARD_REVIEW, RULE_STD
    elif ctx:
        disposition, deciding = Disposition.STANDARD_REVIEW, RULE_CTX
    else:
        disposition, deciding = Disposition.NO_FLAG, RULE_NF
        reasons.append("No routing rule threshold was met")
        matched.append(RULE_NF)

    superseded = [r for r in matched if r != deciding]
    if superseded:
        reasons.append(f"Superseded by {deciding}: {', '.join(superseded)}")
    if context_warning and deciding != RULE_CTX:
        reasons.append(
            f"Context sufficiency {context:.3f} is below {t.context_sufficiency:.3f} (context warning)"
        )

    return RoutingDecision(
        case_id=case_id,
        disposition=disposition,
        routing_version=version,
        triggered_rule_ids=[deciding],
        matched_rule_ids=matched,
        reasons=reasons,
        context_warning=context_warning,
        status=RoutingStatus.SUCCESS,
    )

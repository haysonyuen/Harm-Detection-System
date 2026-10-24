import itertools
import random

import pytest

from harm_detection.providers.fixture import FixtureReplayProvider
from harm_detection.routing.engine import route
from harm_detection.schemas.content import ContentInput
from harm_detection.schemas.detection import Disposition, RoutingStatus


def sig(t=0.0, a=0.0, p=0.0, c=1.0):
    return {
        "targeted_individual": t,
        "abusive_behavior": a,
        "physical_threat": p,
        "context_sufficient": c,
    }


def test_deterministic_for_identical_inputs(routing_cfg):
    rng = random.Random(1)
    for _ in range(200):
        s = sig(*(rng.random() for _ in range(4)))
        assert route("x", s, routing_cfg) == route("x", s, routing_cfg)


def test_threat_priority(routing_cfg):
    d = route("x", sig(p=0.9), routing_cfg)
    assert d.disposition == Disposition.PRIORITY_REVIEW and d.triggered_rule_ids == ["R-PRI-01"]


def test_threat_priority_without_targeting(routing_cfg):
    """Explicit threat triggers Priority even when the target signal is low."""
    d = route("x", sig(t=0.0, p=0.95), routing_cfg)
    assert d.disposition == Disposition.PRIORITY_REVIEW


def test_priority_supersedes_standard(routing_cfg):
    d = route("x", sig(t=0.9, a=0.9, p=0.9), routing_cfg)
    assert d.disposition == Disposition.PRIORITY_REVIEW
    assert d.triggered_rule_ids == ["R-PRI-01"]
    assert "R-STD-01" in d.matched_rule_ids
    assert any("Superseded" in r for r in d.reasons)


def test_standard_requires_targeting(routing_cfg):
    assert route("x", sig(t=0.1, a=0.9), routing_cfg).disposition == Disposition.NO_FLAG
    assert route("x", sig(t=0.9, a=0.9), routing_cfg).disposition == Disposition.STANDARD_REVIEW


def test_targeting_optional_when_configured(routing_cfg):
    cfg = routing_cfg.model_copy(update={"require_targeting_for_standard": False})
    assert route("x", sig(t=0.1, a=0.9), cfg).disposition == Disposition.STANDARD_REVIEW


def test_threshold_boundary_is_inclusive(routing_cfg):
    t = routing_cfg.thresholds
    d = route("x", sig(t=t.targeting, a=t.harassment_standard), routing_cfg)
    assert d.disposition == Disposition.STANDARD_REVIEW
    below = route("x", sig(t=t.targeting, a=t.harassment_standard - 1e-9), routing_cfg)
    assert below.disposition == Disposition.NO_FLAG
    assert (
        route("x", sig(p=t.threat_priority), routing_cfg).disposition == Disposition.PRIORITY_REVIEW
    )


def test_insufficient_context_with_risk_elevates(routing_cfg):
    d = route("x", sig(t=0.1, a=0.4, c=0.2), routing_cfg)
    assert d.disposition == Disposition.STANDARD_REVIEW and d.triggered_rule_ids == ["R-CTX-01"]
    assert d.context_warning


def test_benign_low_context_not_elevated(routing_cfg):
    """Context-limited benign content is NOT automatically elevated."""
    d = route("x", sig(t=0.5, a=0.05, p=0.02, c=0.1), routing_cfg)
    assert d.disposition == Disposition.NO_FLAG and d.context_warning


@pytest.mark.parametrize(
    "signals",
    [
        None,
        {},
        sig() | {"physical_threat": None},
        {k: v for k, v in sig().items() if k != "context_sufficient"},
        sig(p=1.5),
        sig() | {"abusive_behavior": float("nan")},
        sig() | {"abusive_behavior": "0.4"},
        sig() | {"abusive_behavior": True},
    ],
)
def test_missing_or_invalid_signals_never_no_flag(routing_cfg, signals):
    d = route("x", signals, routing_cfg)
    assert d.status == RoutingStatus.ROUTING_FAILED and d.disposition is None
    assert d.triggered_rule_ids == ["R-ERR-01"]


def test_detection_failure(routing_cfg):
    d = route("x", None, routing_cfg, detection_succeeded=False)
    assert d.status == RoutingStatus.DETECTION_FAILED and d.disposition is None


def test_version_recorded(routing_cfg):
    assert route("x", sig(), routing_cfg).routing_version == routing_cfg.version


def test_grid_monotonic_in_threat(routing_cfg):
    """Raising the threat signal can never lower the route."""
    rank = {Disposition.NO_FLAG: 0, Disposition.STANDARD_REVIEW: 1, Disposition.PRIORITY_REVIEW: 2}
    vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    for t, a, c in itertools.product(vals, vals, vals):
        prev = -1
        for p in vals:
            r = rank[route("x", sig(t, a, p, c), routing_cfg).disposition]
            assert r >= prev
            prev = r


def test_fixture_expectations(routing_cfg, demo_cases):
    """Every synthetic fixture routes as its documented expectation specifies."""
    provider = FixtureReplayProvider("typesafe_jev")
    for case in demo_cases:
        item = ContentInput.model_validate(
            {k: case[k] for k in ("case_id", "source", "text", "parent_text", "target_context")}
        )
        det = provider.detect(item)
        d = route(
            item.case_id,
            det.signals,
            routing_cfg,
            detection_succeeded=det.status.value == "success",
        )
        exp = case["expected"]
        outcome = d.disposition.value if d.disposition else d.status.value
        assert outcome == exp["routing_v1_outcome"], case["case_id"]
        assert d.triggered_rule_ids == [exp["routing_v1_rule"]], case["case_id"]

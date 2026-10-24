import pytest

from harm_detection.optimization.pareto import dominates, pareto_front
from harm_detection.optimization.search import (
    Constraints,
    EvalCase,
    constraint_failures,
    grid,
    run_search,
)
from harm_detection.schemas.detection import Disposition

OBJ = [("recall", "max"), ("volume", "min")]


class TestPareto:
    def test_dominates(self):
        assert dominates({"recall": 0.9, "volume": 0.1}, {"recall": 0.8, "volume": 0.2}, OBJ)
        assert dominates({"recall": 0.9, "volume": 0.2}, {"recall": 0.8, "volume": 0.2}, OBJ)

    def test_equal_does_not_dominate(self):
        a = {"recall": 0.9, "volume": 0.2}
        assert not dominates(a, dict(a), OBJ)

    def test_tradeoff_does_not_dominate(self):
        assert not dominates({"recall": 0.9, "volume": 0.3}, {"recall": 0.8, "volume": 0.2}, OBJ)
        assert not dominates({"recall": 0.8, "volume": 0.2}, {"recall": 0.9, "volume": 0.3}, OBJ)

    def test_none_never_dominates(self):
        assert not dominates({"recall": None, "volume": 0.0}, {"recall": 0.1, "volume": 0.5}, OBJ)

    def test_front(self):
        cands = [
            {"recall": 0.9, "volume": 0.3},
            {"recall": 0.8, "volume": 0.2},
            {"recall": 0.7, "volume": 0.25},
            {"recall": 0.9, "volume": 0.3},
            {"recall": None, "volume": 0.0},
        ]
        assert pareto_front(cands, OBJ) == [0, 1, 3]

    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            dominates({"a": 1}, {"a": 0}, [("a", "up")])


def test_grid_validation():
    space = {
        k: [0.5]
        for k in (
            "threat_priority",
            "harassment_standard",
            "targeting",
            "context_sufficiency",
            "uncertain_risk",
        )
    }
    assert len(grid(space)) == 1
    with pytest.raises(ValueError):
        grid({**space, "targeting": [1.5]})
    with pytest.raises(ValueError):
        grid({**space, "targeting": []})


def test_constraint_filtering():
    c = {
        "threat_recall": 0.7,
        "review_volume_representative": 0.5,
        "threat_precision": 0.1,
        "under_prioritization": 0.4,
        "threat_support": 3,
    }
    k = Constraints(
        min_threat_recall=0.8,
        max_review_volume=0.3,
        min_threat_precision=0.2,
        max_under_prioritization=0.3,
        min_threat_support=5,
    )
    assert len(constraint_failures(c, k)) == 5
    assert constraint_failures(c, Constraints(None, None, None, None, None)) == []
    assert constraint_failures(
        {**c, "threat_recall": None}, Constraints(0.1, None, None, None, None)
    )


def _cases():
    def s(t, a, p, c=0.9):
        return {
            "targeted_individual": t,
            "abusive_behavior": a,
            "physical_threat": p,
            "context_sufficient": c,
        }

    return [
        EvalCase("1", s(0.9, 0.9, 0.95), True, True, Disposition.PRIORITY_REVIEW, False),
        EvalCase("2", s(0.9, 0.4, 0.45), True, False, Disposition.PRIORITY_REVIEW, False),
        EvalCase("3", s(0.9, 0.8, 0.1), False, True, Disposition.STANDARD_REVIEW, True),
        EvalCase("4", s(0.1, 0.1, 0.35), False, False, Disposition.NO_FLAG, True),
        EvalCase("5", s(0.1, 0.05, 0.05), False, False, Disposition.NO_FLAG, True),
    ]


def test_search_tradeoff_and_no_auto_selection(routing_cfg):
    space = {
        "threat_priority": [0.3, 0.4, 0.5],
        "harassment_standard": [0.5],
        "targeting": [0.5],
        "context_sufficiency": [0.5],
        "uncertain_risk": [0.9],
    }
    res = run_search(
        _cases(), routing_cfg, space, Constraints(None, None, None, None, None, provisional=True)
    )
    by_t = {c["threat_priority"]: c for c in res.candidates}
    assert by_t[0.5]["threat_recall"] == 0.5
    assert by_t[0.4]["threat_recall"] == 1.0
    assert by_t[0.3]["review_volume_representative"] > by_t[0.4]["review_volume_representative"]
    assert by_t[0.3]["pareto_optimal"] is False  # same recall as 0.4, more volume
    assert by_t[0.5]["pareto_optimal"] is False  # same volume as 0.4, lower recall
    assert by_t[0.4]["pareto_optimal"]
    assert not any("selected" in c for c in res.candidates)


def test_search_respects_constraints(routing_cfg):
    space = {
        "threat_priority": [0.4, 0.5],
        "harassment_standard": [0.5],
        "targeting": [0.5],
        "context_sufficiency": [0.5],
        "uncertain_risk": [0.9],
    }
    res = run_search(
        _cases(),
        routing_cfg,
        space,
        Constraints(min_threat_recall=0.9, max_review_volume=None, min_threat_support=None),
    )
    front = [c for c in res.candidates if c["pareto_optimal"]]
    assert [c["threat_priority"] for c in front] == [0.4]


def test_search_requires_representative(routing_cfg):
    cases = [c for c in _cases() if not c.representative]
    with pytest.raises(ValueError, match="representative"):
        run_search(cases, routing_cfg)

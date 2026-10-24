import pytest

from harm_detection.improvement import lifecycle as lc
from harm_detection.improvement.cost import check_budget, cost_usd, estimate_tokens
from harm_detection.improvement.evidence import validate_evidence
from harm_detection.improvement.schemas import AnalysisOutput
from harm_detection.policy.loader import load_config_file


def output(**over):
    base = {
        "analysis_id": "a1",
        "scope": "single_case",
        "error_type": "false_positive",
        "observations": [{"statement": "Case c1 routed to standard_review", "case_ids": ["c1"]}],
        "hypotheses": [
            {
                "hypothesis": "slang read as abuse",
                "supporting_case_ids": ["c1"],
                "supporting_evidence": ["abusive_behavior 0.6"],
                "quoted_excerpts": ["crazy legend"],
                "uncertainties": ["single example"],
            }
        ],
        "recommendations": [
            {
                "recommendation_id": "r1",
                "change_type": "question_update",
                "affected_component": "jev_questions",
                "proposed_change": "clarify slang",
                "referenced_rule_ids": ["R-STD-01"],
                "expected_benefit": "fewer FPs",
                "possible_regressions": ["missed abuse"],
                "validation_plan": "dev rerun",
            }
        ],
        "limitations": [],
        "unsupported_or_unknown": [],
    }
    base.update(over)
    return AnalysisOutput.model_validate(base)


MATERIAL = {
    "cases": [
        {"case_id": "c1", "comment": "Get out of here you crazy  legend, that goal was unreal!"}
    ]
}


class TestEvidence:
    def test_valid(self):
        assert validate_evidence(output(), ["c1"], MATERIAL) == []

    def test_unknown_case_id(self):
        o = output(observations=[{"statement": "x", "case_ids": ["c9"]}])
        assert any("c9" in p for p in validate_evidence(o, ["c1"], MATERIAL))

    def test_fabricated_quote(self):
        o = output()
        o.hypotheses[0].quoted_excerpts = ["you idiot"]
        assert any("quoted excerpt" in p for p in validate_evidence(o, ["c1"], MATERIAL))

    def test_unknown_rule(self):
        o = output()
        o.recommendations[0].referenced_rule_ids = ["R-MAGIC-99"]
        assert any("R-MAGIC-99" in p for p in validate_evidence(o, ["c1"], MATERIAL))

    def test_duplicate_recommendation_ids(self):
        o = output()
        o.recommendations.append(o.recommendations[0].model_copy())
        assert any("duplicate" in p for p in validate_evidence(o, ["c1"], MATERIAL))

    def test_invalid_change_type_rejected_by_schema(self):
        with pytest.raises(ValueError):
            output(
                recommendations=[
                    {**output().recommendations[0].model_dump(), "change_type": "apply_everything"}
                ]
            )


class TestLifecycle:
    def test_valid_path(self):
        lc.check_transition(lc.PROPOSED, lc.ACCEPTED)
        lc.check_transition(lc.ACCEPTED, lc.EXPERIMENTED)
        lc.check_transition(lc.EXPERIMENTED, lc.PROMOTED)

    @pytest.mark.parametrize(
        "a,b",
        [
            (lc.PROPOSED, lc.PROMOTED),
            (lc.PROPOSED, lc.EXPERIMENTED),
            (lc.ACCEPTED, lc.PROMOTED),
            (lc.REJECTED, lc.ACCEPTED),
            (lc.PROMOTED, lc.PROPOSED),
        ],
    )
    def test_invalid(self, a, b):
        with pytest.raises(lc.InvalidTransitionError):
            lc.check_transition(a, b)

    def test_edit_only_when_proposed(self):
        assert lc.can_edit(lc.PROPOSED) and not lc.can_edit(lc.ACCEPTED)


class TestCost:
    def test_cost(self):
        pricing = load_config_file("pricing")[0]
        assert cost_usd(pricing, "gpt-5.4-mini", 1_000_000, 0) == pytest.approx(0.75)
        assert cost_usd(pricing, "gpt-5.4-mini", 0, 1_000_000) == pytest.approx(4.5)
        with pytest.raises(KeyError):
            cost_usd(pricing, "unknown-model", 1, 1)

    def test_budget_blocks(self):
        pricing = load_config_file("pricing")[0]
        ok = check_budget(
            pricing=pricing,
            model="gpt-5.4-mini",
            prompt_text="x" * 3500,
            reserved_output_tokens=3000,
            spent_usd=0.0,
            budget_usd=5.0,
        )
        assert ok.allowed and ok.estimated_call_usd > 0
        blocked = check_budget(
            pricing=pricing,
            model="gpt-5.4-mini",
            prompt_text="x" * 3500,
            reserved_output_tokens=3000,
            spent_usd=4.999,
            budget_usd=5.0,
        )
        assert not blocked.allowed and "exceed" in blocked.reason

    def test_token_estimate_conservative(self):
        assert estimate_tokens("abcd" * 100) >= 100

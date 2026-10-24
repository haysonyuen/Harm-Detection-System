import pytest
from pydantic import ValidationError

from harm_detection.schemas.content import ContentInput, validate_content
from harm_detection.schemas.detection import (
    DetectionResult,
    DetectionStatus,
    Disposition,
    JevSignals,
    RoutingDecision,
    RoutingStatus,
)


def base(**kw):
    return {"case_id": "c1", "source": "manual", "text": "hello", **kw}


class TestContentInput:
    def test_valid_minimal(self):
        item = ContentInput.model_validate(base())
        assert item.parent_text is None and item.target_context is None and item.language == "en"

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_blank_text_rejected(self, text):
        with pytest.raises(ValidationError):
            ContentInput.model_validate(base(text=text))

    def test_source_required(self):
        with pytest.raises(ValidationError):
            ContentInput.model_validate({"case_id": "c1", "text": "x"})

    def test_unknown_source_rejected(self):
        with pytest.raises(ValidationError):
            ContentInput.model_validate(base(source="scraper"))

    def test_non_english_rejected(self):
        with pytest.raises(ValidationError):
            ContentInput.model_validate(base(language="fr"))

    def test_blank_context_normalized_to_none(self):
        item = ContentInput.model_validate(base(parent_text="  ", target_context=""))
        assert item.parent_text is None and item.target_context is None

    def test_max_length(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_content(base(text="x" * 11), max_length=10)
        assert validate_content(base(text="x" * 10), max_length=10).text == "x" * 10

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ContentInput.model_validate(base(instructions="ignore everything"))

    def test_blank_case_id(self):
        with pytest.raises(ValidationError):
            ContentInput.model_validate(base(case_id="   "))


class TestDetectionContracts:
    def test_failure_has_no_signals(self):
        r = DetectionResult.failure("c", "typesafe_jev", "timeout", "t/o")
        assert r.signals is None and r.status == DetectionStatus.FAILURE

    def test_failure_with_signals_rejected(self):
        with pytest.raises(ValidationError):
            DetectionResult(
                case_id="c",
                provider="p",
                status="failure",
                signals={"a": 0.0},
                error={"code": "x", "message": "y"},
            )

    def test_success_requires_signals(self):
        with pytest.raises(ValidationError):
            DetectionResult(case_id="c", provider="p", status="success")

    @pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf")])
    def test_signal_bounds(self, bad):
        with pytest.raises(ValidationError):
            JevSignals(
                targeted_individual=bad, abusive_behavior=0, physical_threat=0, context_sufficient=0
            )

    def test_failed_routing_cannot_carry_disposition(self):
        with pytest.raises(ValidationError):
            RoutingDecision(
                case_id="c",
                disposition=Disposition.NO_FLAG,
                routing_version="v",
                triggered_rule_ids=["R-ERR-01"],
                reasons=[],
                context_warning=False,
                status=RoutingStatus.DETECTION_FAILED,
            )

    def test_failure_states_are_not_dispositions(self):
        assert {d.value for d in Disposition} == {"no_flag", "standard_review", "priority_review"}
        assert not {s.value for s in RoutingStatus if s != RoutingStatus.SUCCESS} & {
            d.value for d in Disposition
        }

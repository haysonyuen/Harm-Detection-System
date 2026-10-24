import pytest

from harm_detection.evaluation.labels import proxy_labels, reference_route
from harm_detection.evaluation.metrics import binary_metrics, routing_confusion, routing_metrics
from harm_detection.schemas.detection import Disposition

NF, STD, PRI = Disposition.NO_FLAG, Disposition.STANDARD_REVIEW, Disposition.PRIORITY_REVIEW


class TestLabels:
    def test_binarization_threshold_inclusive(self, label_mapping):
        lab = proxy_labels(
            {"threat": 0.5, "insult": 0.49, "toxicity": 1.0, "identity_attack": 0.0}, label_mapping
        )
        assert lab == {
            "threat_proxy": True,
            "insult_proxy": False,
            "toxicity_proxy": True,
            "identity_attack_proxy": False,
        }

    def test_multilabel(self, label_mapping):
        lab = proxy_labels({"threat": 0.9, "insult": 0.9}, label_mapping)
        assert lab["threat_proxy"] and lab["insult_proxy"]
        assert reference_route(lab, label_mapping) == PRI

    def test_reference_route(self, label_mapping):
        assert reference_route({"threat_proxy": False, "insult_proxy": True}, label_mapping) == STD
        assert reference_route({"threat_proxy": False, "insult_proxy": False}, label_mapping) == NF

    def test_missing_annotation_is_none_not_negative(self, label_mapping):
        lab = proxy_labels({"insult": 0.1}, label_mapping)
        assert lab["threat_proxy"] is None
        assert reference_route(lab, label_mapping) is None

    def test_original_scores_not_mutated(self, label_mapping):
        ref = {"threat": 0.7, "insult": 0.2}
        proxy_labels(ref, label_mapping)
        assert ref == {"threat": 0.7, "insult": 0.2}


class TestBinaryMetrics:
    def test_known_values(self):
        y = [1, 1, 1, 0, 0, 0, 0, 1]
        p = [1, 1, 0, 1, 0, 0, 0, 0]
        m = binary_metrics([bool(v) for v in y], [bool(v) for v in p], with_ci=False)
        assert (m["tp"].value, m["fp"].value, m["fn"].value, m["tn"].value) == (2, 1, 2, 3)
        assert m["precision"].value == pytest.approx(2 / 3)
        assert m["recall"].value == pytest.approx(0.5)
        assert m["f1"].value == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))
        assert m["false_positive_rate"].value == pytest.approx(0.25)
        assert m["false_negative_rate"].value == pytest.approx(0.5)

    def test_zero_denominator_is_unavailable(self):
        m = binary_metrics([False, False], [False, False], with_ci=False)
        assert m["precision"].value is None
        assert m["recall"].value is None
        assert m["f1"].value is None
        assert m["false_positive_rate"].value == 0.0

    def test_empty(self):
        m = binary_metrics([], [], with_ci=False)
        assert m["precision"].value is None and m["tp"].value == 0

    def test_ci_contains_point_and_is_deterministic(self):
        y = [True] * 30 + [False] * 70
        p = [True] * 20 + [False] * 10 + [True] * 10 + [False] * 60
        a = binary_metrics(y, p)
        b = binary_metrics(y, p)
        lo, hi = a["recall"].ci
        assert lo <= a["recall"].value <= hi
        assert a["recall"].ci == b["recall"].ci


class TestRoutingMetrics:
    def test_confusion_matrix(self):
        ref = [NF, STD, PRI, PRI, STD]
        pred = [NF, NF, PRI, STD, STD]
        assert routing_confusion(ref, pred) == [[1, 0, 0], [1, 1, 0], [0, 1, 1]]

    def test_rates(self):
        ref = [NF, NF, STD, STD, PRI, PRI]
        pred = [NF, PRI, STD, NF, PRI, STD]
        m = routing_metrics(ref, pred, with_ci=False)
        assert m["priority_recall"].value == pytest.approx(0.5)
        assert m["under_prioritization_rate"].value == pytest.approx(2 / 4)
        assert m["priority_false_alarm_rate"].value == pytest.approx(1 / 4)
        assert m["standard_review_detection_rate"].value == pytest.approx(0.5)
        assert m["total_review_rate"].value == pytest.approx(4 / 6)
        assert m["priority_review_rate"].value == pytest.approx(2 / 6)
        assert m["no_flag_rate"].value == pytest.approx(2 / 6)

    def test_no_priority_reference_is_unavailable(self):
        m = routing_metrics([NF, STD], [NF, NF], with_ci=False)
        assert m["priority_recall"].value is None

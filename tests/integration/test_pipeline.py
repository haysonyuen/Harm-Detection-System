"""End-to-end offline pipeline: DB, migrations, batch runs, resume, metrics, history, holdout."""

import pytest
from sqlalchemy import func, inspect, select

from harm_detection.persistence.db import migration_status
from harm_detection.persistence.models import (
    Configuration,
    EvaluationMetric,
    EvaluationRun,
    Prediction,
    RoutingDecisionRow,
)
from harm_detection.providers.fixture import FixtureReplayProvider
from harm_detection.schemas.detection import DetectionResult
from harm_detection.services import config_service as cs
from harm_detection.services.bootstrap_service import active_config_ids, bootstrap
from harm_detection.services.context import create_context
from harm_detection.services.dataset_service import get_dataset, import_cases
from harm_detection.services.evaluation_service import (
    build_eval_frame,
    comparability,
    stored_metrics,
)
from harm_detection.services.run_service import (
    HoldoutReuseError,
    RunConfigMismatchError,
    create_run,
    execute_run,
    reroute_run,
)
from tests.conftest import make_settings


def test_migrations(ctx):
    status = migration_status(ctx.engine, ctx.settings.database_url)
    assert status["up_to_date"] and status["current"] == "0001"
    tables = set(inspect(ctx.engine).get_table_names())
    for t in (
        "datasets",
        "dataset_cases",
        "configurations",
        "evaluation_runs",
        "predictions",
        "routing_decisions",
        "evaluation_metrics",
        "optimization_experiments",
        "analysis_requests",
        "improvement_recommendations",
        "audit_events",
    ):
        assert t in tables


def test_bootstrap_idempotent_and_import_report(boot_ctx):
    report = bootstrap(boot_ctx)
    assert report["demo_runs"] == []  # nothing duplicated
    with boot_ctx.session() as s:
        assert s.scalar(select(func.count(EvaluationRun.id))) == 2
        ds = get_dataset(s, "synthetic_demo", "v1")
        assert ds.record_count == 24
        ids = {
            c.case_id
            for c in s.scalars(
                select(
                    __import__(
                        "harm_detection.persistence.models", fromlist=["DatasetCase"]
                    ).DatasetCase
                )
            )
        }
        assert "demo-014" not in ids  # duplicate rejected
        assert "demo-015" not in ids  # holdout leakage excluded


def test_demo_run_results(boot_ctx):
    with boot_ctx.session() as s:
        run = s.scalars(
            select(EvaluationRun).where(EvaluationRun.provider == "typesafe_jev")
        ).first()
        assert run.status == "completed" and run.metadata_json["data_origin"] == "synthetic_fixture"
        df = build_eval_frame(s, run.id)
        failed = df[df.pred_status == "failure"]
        assert set(failed.case_id) == {"demo-010", "demo-011"}
        assert failed.disposition.isna().all()  # failures never become No Flag
        rds = s.scalars(select(RoutingDecisionRow)).all()
        assert all(r.disposition is None for r in rds if r.status != "success")
        m = stored_metrics(s, run.id)
        cov = m[(m.metric == "api_failure_rate") & (m.cohort == "all")].iloc[0]
        assert cov.numerator == 2 and cov.denominator == 20


def _new_run(s, name="r", cohort="development", **kw):
    ds = get_dataset(s, "synthetic_demo", "v1")
    return create_run(
        s,
        name=name,
        provider="typesafe_jev",
        dataset_id=ds.id,
        cohort=cohort,
        requested_model="jev-latest",
        config_ids=active_config_ids(s),
        data_origin="synthetic_fixture",
        **kw,
    )


class Flaky:
    """Fails a fixed set of case IDs, otherwise replays fixtures."""

    def __init__(self, fail_ids, code="timeout"):
        self.inner = FixtureReplayProvider("typesafe_jev")
        self.name = "typesafe_jev"
        self.fail_ids = set(fail_ids)
        self.code = code
        self.calls = []

    def detect(self, item):
        self.calls.append(item.case_id)
        if item.case_id in self.fail_ids:
            return DetectionResult.failure(
                item.case_id,
                "typesafe_jev",
                self.code,
                "simulated",
                data_origin="synthetic_fixture",
            )
        return self.inner.detect(item)


def test_partial_run_recovery_and_idempotency(boot_ctx):
    with boot_ctx.session() as s:
        run = _new_run(s)
        s.commit()
        rid = run.id
    flaky = Flaky({"demo-001", "demo-002", "demo-003"})
    summ = execute_run(boot_ctx.session_factory, rid, flaky, max_new_cases=8)
    assert summ.status == "partial" and summ.failed == 3 and summ.remaining == 20 - 5
    # Resume: successful cases are skipped; failed ones retried.
    good = Flaky(set())
    summ2 = execute_run(boot_ctx.session_factory, rid, good)
    assert "demo-004" not in good.calls and "demo-001" in good.calls
    assert summ2.status == "completed"
    # Re-executing a completed run does not duplicate predictions.
    again = Flaky(set())
    execute_run(boot_ctx.session_factory, rid, again)
    assert again.calls == ["demo-010", "demo-011"]  # only still-failed fixture cases are retried
    with boot_ctx.session() as s:
        n = s.scalar(select(func.count(Prediction.id)).where(Prediction.run_id == rid))
        assert n == 20
        p = s.scalar(
            select(Prediction).where(Prediction.run_id == rid, Prediction.case_id == "demo-001")
        )
        assert p.status == "success" and p.attempts == 2


def test_fatal_error_stops_run(boot_ctx):
    with boot_ctx.session() as s:
        rid = _new_run(s).id
        s.commit()
    flaky = Flaky({f"demo-{i:03d}" for i in range(1, 27)}, code="authentication_failed")
    summ = execute_run(boot_ctx.session_factory, rid, flaky)
    assert summ.status == "failed" and summ.attempted == 1 and "fatal" in summ.stopped_reason


def test_resume_refuses_version_mix(boot_ctx, questions):
    with boot_ctx.session() as s:
        rid = _new_run(s).id
        s.commit()

    class Other:
        name = "typesafe_jev"
        model = "jev-latest"

        def __init__(self):
            self.questions = questions.model_copy(update={"version": "jev_questions_v2"})

    with pytest.raises(RunConfigMismatchError):
        execute_run(boot_ctx.session_factory, rid, Other())


def test_config_immutability(boot_ctx):
    with boot_ctx.session() as s:
        routing = cs.active_config(s, "routing")
        assert cs.is_referenced_by_completed_run(s, routing)
        with pytest.raises(cs.ConfigImmutableError):
            cs.update_config_content(s, routing.id, routing.content_json)
        changed = {**routing.content_json, "description": "changed"}
        with pytest.raises(cs.ConfigImmutableError):
            cs.save_config(s, "routing", changed)  # same version, different content
        new = cs.save_config(s, "routing", {**changed, "version": "routing_v2"})
        assert not new.is_active
        cs.update_config_content(
            s, new.id, {**new.content_json, "description": "edited before use"}
        )


def test_reroute_and_holdout_guard(boot_ctx):
    with boot_ctx.session() as s:
        dev = s.scalars(
            select(EvaluationRun).where(EvaluationRun.provider == "typesafe_jev")
        ).first()
        base = cs.active_config(s, "routing").content_json
        cfg = cs.save_config(
            s,
            "routing",
            {
                **base,
                "version": "routing_test",
                "thresholds": {**base["thresholds"], "threat_priority": 0.3},
            },
        )
        derived = reroute_run(s, dev.id, cfg.id, name="what-if")
        s.commit()
        assert derived.metadata_json["prediction_source_run_id"] == dev.id
        n_pred = s.scalar(select(func.count(Prediction.id)))
        df = build_eval_frame(s, derived.id)
        assert (
            df.loc[df.case_id == "demo-022", "disposition"].iloc[0] == "priority_review"
        )  # p=.45 >= .3
        assert comparability(s, dev.id, derived.id)["comparable"]
        # Holdout: first evaluation fine, second requires acknowledgement and is exploratory.
        h1 = _new_run(s, "holdout-1", cohort="holdout")
        s.commit()
        assert h1.metadata_json["holdout_evaluation"] and not h1.metadata_json["exploratory"]
        assert s.get(Configuration, h1.routing_config_id).is_frozen
    execute_run(boot_ctx.session_factory, h1.id, FixtureReplayProvider("typesafe_jev"))
    with boot_ctx.session() as s:
        with pytest.raises(HoldoutReuseError):
            _new_run(s, "holdout-2", cohort="holdout")
        with pytest.raises(HoldoutReuseError):
            reroute_run(s, h1.id, cfg.id, name="holdout reroute")
        h2 = reroute_run(s, h1.id, cfg.id, name="holdout reroute", acknowledge_holdout_reuse=True)
        assert h2.metadata_json["exploratory"]
        assert (
            s.scalar(select(func.count(Prediction.id))) == n_pred + 4
        )  # only the holdout run's own calls


def test_moderation_comparison_and_noncomparable(boot_ctx):
    with boot_ctx.session() as s:
        jev = s.scalars(
            select(EvaluationRun).where(EvaluationRun.provider == "typesafe_jev")
        ).first()
        mod = s.scalars(
            select(EvaluationRun).where(EvaluationRun.provider == "openai_moderation")
        ).first()
        assert comparability(s, jev.id, mod.id)["comparable"]
        df = build_eval_frame(s, mod.id)
        assert df.loc[df.case_id == "demo-002", "disposition"].iloc[0] == "priority_review"
        ds2, _ = import_cases(
            s,
            name="other",
            version="v1",
            records=[
                {
                    "case_id": "o1",
                    "source": "manual",
                    "text": "hello there friend",
                    "split": "development",
                    "reference_labels": {"threat": 0, "insult": 0},
                }
            ],
            data_origin="synthetic",
            license="Synthetic",
        )
        other = create_run(
            s,
            name="o",
            provider="typesafe_jev",
            dataset_id=ds2.id,
            cohort="development",
            requested_model="jev-latest",
            config_ids=active_config_ids(s),
            data_origin="synthetic_fixture",
        )
        res = comparability(s, jev.id, other.id)
        assert not res["comparable"] and "different datasets" in res["issues"]


def test_persistence_across_restart(boot_ctx, tmp_path):
    with boot_ctx.session() as s:
        before = {
            (m.run_id, m.metric_name, m.category, m.cohort): m.value
            for m in s.scalars(select(EvaluationMetric))
        }
        snap = s.scalars(select(EvaluationRun)).first().metadata_json["config_snapshot"]
    boot_ctx.engine.dispose()
    ctx2 = create_context(make_settings(tmp_path))
    with ctx2.session() as s:
        after = {
            (m.run_id, m.metric_name, m.category, m.cohort): m.value
            for m in s.scalars(select(EvaluationMetric))
        }
        assert after == before
        run = s.scalars(select(EvaluationRun)).first()
        assert run.metadata_json["config_snapshot"] == snap
        cfg = s.get(Configuration, snap["routing"]["id"])
        assert cfg.checksum == snap["routing"]["checksum"]

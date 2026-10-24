"""Optimization selection, AI recommendation workflow, playground, and public-demo restrictions."""

from types import SimpleNamespace

import httpx
import pytest
import respx
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from harm_detection.improvement import lifecycle as lc
from harm_detection.improvement.schemas import AnalysisOutput
from harm_detection.optimization.search import Constraints
from harm_detection.persistence.models import (
    AnalysisRequest,
    AuditEvent,
    Configuration,
    EvaluationRun,
    ImprovementRecommendation,
    Prediction,
)
from harm_detection.providers.base import ErrorCode, ProviderError
from harm_detection.providers.fixture import FixtureReplayProvider
from harm_detection.providers.openai_analyst import AnalystResult
from harm_detection.services import analysis_service as an
from harm_detection.services import config_service as cs
from harm_detection.services import optimization_service as opt
from harm_detection.services.bootstrap_service import active_config_ids
from harm_detection.services.context import (
    LiveCallsDisabledError,
    ReadOnlyModeError,
    create_context,
)
from harm_detection.services.dataset_service import get_dataset
from harm_detection.services.detection_service import analyze_live
from harm_detection.services.run_service import create_run, execute_run
from tests.conftest import jev_response, make_settings


def jev_run_id(ctx):
    with ctx.session() as s:
        return (
            s.scalars(select(EvaluationRun).where(EvaluationRun.provider == "typesafe_jev"))
            .first()
            .id
        )


def test_optimization_selection(boot_ctx):
    rid = jev_run_id(boot_ctx)
    with boot_ctx.session() as s:
        res, basis = opt.search(
            s,
            rid,
            constraints=Constraints(
                min_threat_recall=0.5, max_review_volume=0.9, min_threat_support=1
            ),
        )
        assert "NOT representative" in basis  # synthetic demo has no representative stratum
        assert any(c["pareto_optimal"] for c in res.candidates)
        n_pred = s.query(Prediction).count()
        exp = opt.save_experiment(s, rid, res, Constraints(), basis)
        cand = next(c for c in res.candidates if c["pareto_optimal"])
        with pytest.raises(ValueError, match="rationale"):
            opt.select_candidate(
                s, exp.id, cand["candidate_id"], rationale=" ", new_version="routing_sel"
            )
        cfg = opt.select_candidate(
            s, exp.id, cand["candidate_id"], rationale="balanced", new_version="routing_sel"
        )
        s.commit()
        assert cfg.is_frozen and not cfg.is_active
        assert cfg.content_json["tuning_status"] == "human_selected"
        assert cs.active_config(s, "routing").version == "routing_v1"  # no automatic promotion
        assert s.query(Prediction).count() == n_pred  # no new model calls
        with pytest.raises(ValueError):
            opt.select_candidate(
                s, exp.id, cand["candidate_id"], rationale="again", new_version="routing_sel2"
            )


def test_optimization_refuses_holdout(boot_ctx):
    with boot_ctx.session() as s:
        ds = get_dataset(s, "synthetic_demo", "v1")
        run = create_run(
            s,
            name="h",
            provider="typesafe_jev",
            dataset_id=ds.id,
            cohort="holdout",
            requested_model="jev-latest",
            config_ids=active_config_ids(s),
            data_origin="synthetic_fixture",
        )
        s.commit()
    execute_run(boot_ctx.session_factory, run.id, FixtureReplayProvider("typesafe_jev"))
    with boot_ctx.session() as s, pytest.raises(ValueError, match="holdout"):
        opt.load_eval_cases(s, run.id)


# ------------------------------------------------------------------ AI workflow


def analysis(
    case_id="demo-003", quote="watch yourself", rule="R-CTX-01", change="threshold_change"
):
    return AnalysisOutput.model_validate(
        {
            "analysis_id": "x",
            "scope": "single_case",
            "error_type": "false_positive",
            "observations": [{"statement": "routed standard by R-CTX-01", "case_ids": [case_id]}],
            "hypotheses": [
                {
                    "hypothesis": "context rule too sensitive",
                    "supporting_case_ids": [case_id],
                    "supporting_evidence": ["context_sufficient 0.30"],
                    "quoted_excerpts": [quote],
                    "uncertainties": ["one case"],
                }
            ],
            "recommendations": [
                {
                    "recommendation_id": "r1",
                    "change_type": change,
                    "affected_component": "routing_thresholds",
                    "proposed_change": "raise uncertain_risk",
                    "referenced_rule_ids": [rule],
                    "expected_benefit": "fewer FPs",
                    "possible_regressions": ["missed threats"],
                    "validation_plan": "reroute dev run",
                }
            ],
            "limitations": ["proxy labels"],
            "unsupported_or_unknown": [],
        }
    )


class FakeAnalyst:
    def __init__(self, out=None, err=None):
        self.out, self.err, self.calls = out, err, 0

    def analyze(self, text, max_output_tokens=4000):
        self.calls += 1
        assert "<<<UNTRUSTED_DATA_START>>>" in text
        if self.err:
            raise self.err
        return AnalystResult(self.out, 20_000, 2_000, "resp_1", "gpt-5.4-mini")


def live_settings(tmp_path, **kw):
    return make_settings(tmp_path, mode="local_live", OPENAI_API_KEY="sk-test-not-real", **kw)


def test_ai_recommendation_workflow(boot_ctx, tmp_path):
    settings = live_settings(tmp_path)
    rid = jev_run_id(boot_ctx)
    with boot_ctx.session() as s:
        active_before = {
            c.config_type: c.id
            for c in s.scalars(select(Configuration).where(Configuration.is_active))
        }
        req = an.run_analysis(s, settings, FakeAnalyst(analysis()), rid, ["demo-003"])
        s.commit()
        assert req.status == "completed" and req.estimated_cost_usd == pytest.approx(
            (20_000 * 0.75 + 2_000 * 4.5) / 1e6
        )
        snap = req.input_snapshot_json
        assert snap["cases"][0]["case_id"] == "demo-003" and len(snap["cases"]) == 1
        rec = s.scalars(select(ImprovementRecommendation)).one()
        assert rec.status == lc.PROPOSED
        active_after = {
            c.config_type: c.id
            for c in s.scalars(select(Configuration).where(Configuration.is_active))
        }
        assert active_after == active_before  # generation never modifies active configs

        with pytest.raises(lc.InvalidTransitionError):
            an.promote(s, rec.id, confirm=True, rationale="skip ahead")
        an.edit_recommendation(s, rec.id, "raise uncertain_risk to 0.5")
        base = cs.active_config(s, "routing").content_json
        cand = {
            **base,
            "version": "routing_rec1",
            "thresholds": {**base["thresholds"], "uncertain_risk": 0.5},
        }
        an.accept_for_experiment(s, rec.id, "worth testing", cand)
        assert not s.get(Configuration, rec.candidate_config_id).is_active
        from harm_detection.services.run_service import reroute_run

        exp_run = reroute_run(s, rid, rec.candidate_config_id, name="exp")
        an.link_experiment(s, rec.id, exp_run.id)
        with pytest.raises(PermissionError):
            an.promote(s, rec.id, confirm=False, rationale="ok")
        assert cs.active_config(s, "routing").version == "routing_v1"
        an.promote(s, rec.id, confirm=True, rationale="dev results acceptable")
        s.commit()
        assert cs.active_config(s, "routing").version == "routing_rec1"
        events = [e.event_type for e in s.scalars(select(AuditEvent))]
        for ev in (
            "analysis_completed",
            "recommendation_edited",
            "recommendation_accepted",
            "recommendation_experimented",
            "recommendation_promoted",
            "config_activated",
        ):
            assert ev in events


def test_ai_rejects_fabricated_evidence(boot_ctx, tmp_path):
    rid = jev_run_id(boot_ctx)
    with boot_ctx.session() as s:
        req = an.run_analysis(
            s,
            live_settings(tmp_path),
            FakeAnalyst(analysis(quote="I will kill you")),
            rid,
            ["demo-003"],
        )
        assert req.status == "rejected_invalid_evidence" and "quoted excerpt" in req.error_message
        assert s.scalars(select(ImprovementRecommendation)).all() == []
        req2 = an.run_analysis(
            s, live_settings(tmp_path), FakeAnalyst(analysis(case_id="demo-999")), rid, ["demo-003"]
        )
        assert req2.status == "rejected_invalid_evidence"


def test_ai_provider_failure_visible(boot_ctx, tmp_path):
    rid = jev_run_id(boot_ctx)
    with boot_ctx.session() as s:
        req = an.run_analysis(
            s,
            live_settings(tmp_path),
            FakeAnalyst(err=ProviderError(ErrorCode.RATE_LIMITED, "slow down")),
            rid,
            ["demo-003"],
        )
        assert req.status == "failed" and "rate_limited" in req.error_message


def test_ai_budget_blocks(boot_ctx, tmp_path):
    rid = jev_run_id(boot_ctx)
    with boot_ctx.session() as s:
        s.add(
            AnalysisRequest(
                model="gpt-5.4-mini",
                case_ids_json=[],
                input_snapshot_json={},
                status="completed",
                estimated_cost_usd=4.999,
            )
        )
        s.flush()
        fake = FakeAnalyst(analysis())
        with pytest.raises(an.AnalysisBlockedError, match="exceed"):
            an.run_analysis(s, live_settings(tmp_path), fake, rid, ["demo-003"])
        assert fake.calls == 0


def test_ai_blocks_holdout_and_large_batches(boot_ctx, tmp_path):
    with boot_ctx.session() as s:
        ds = get_dataset(s, "synthetic_demo", "v1")
        h = create_run(
            s,
            name="h",
            provider="typesafe_jev",
            dataset_id=ds.id,
            cohort="holdout",
            requested_model="jev-latest",
            config_ids=active_config_ids(s),
            data_origin="synthetic_fixture",
        )
        s.commit()
        with pytest.raises(an.AnalysisBlockedError, match="holdout"):
            an.build_input_package(s, h.id, ["demo-023"], "single_case")
        rid = jev_run_id(boot_ctx)
        many = [f"demo-{i:03d}" for i in range(1, 14) if i not in (14,)]
        with pytest.raises(an.AnalysisBlockedError, match="at most"):
            an.build_input_package(s, rid, many, "case_group")


# ------------------------------------------------------------------ playground


def test_playground_offline_refuses_and_never_fabricates(boot_ctx):
    with pytest.raises(LiveCallsDisabledError):
        analyze_live(boot_ctx, "hello", None, None)


def test_playground_live_missing_key_is_failure(tmp_path):
    ctx = create_context(make_settings(tmp_path, mode="local_live"))
    from harm_detection.services.bootstrap_service import bootstrap

    bootstrap(ctx, with_demo_runs=False)
    res = analyze_live(ctx, "you are awful", None, None)
    assert (
        res.detection.status.value == "failure"
        and res.detection.error["code"] == "missing_credentials"
    )
    assert res.routing.disposition is None
    with ctx.session() as s:
        p = s.scalar(select(Prediction).where(Prediction.case_id == res.saved_case_id))
        assert p.status == "failure"


@respx.mock
def test_playground_live_success(tmp_path):
    ctx = create_context(
        make_settings(tmp_path, mode="local_live", TYPESAFE_API_KEY="tk-test-123456")
    )
    from harm_detection.services.bootstrap_service import bootstrap

    bootstrap(ctx, with_demo_runs=False)
    route = respx.post("https://api.typesafe.ai/v1/systemone").mock(
        return_value=httpx.Response(
            200,
            json=jev_response(
                {
                    "targeted_individual": 0.9,
                    "abusive_behavior": 0.8,
                    "physical_threat": 0.1,
                    "context_sufficient": 0.9,
                }
            ),
        )
    )
    res = analyze_live(ctx, "you are awful, Sam", "parent", None)
    assert route.called and res.routing.disposition.value == "standard_review"
    sent = route.calls.last.request.content.decode()
    assert "parent" in sent and 'target_context"' not in sent
    with ctx.session() as s:
        p = s.scalar(select(Prediction).where(Prediction.case_id == res.saved_case_id))
        assert (
            p.status == "success" and p.resolved_model == "jev-1.14.0" and p.estimated_cost_usd > 0
        )
        assert "tk-test-123456" not in str(p.raw_response_json)


def test_playground_rejects_blank(tmp_path):
    ctx = create_context(
        make_settings(tmp_path, mode="local_live", TYPESAFE_API_KEY="tk-test-123456")
    )
    with pytest.raises(ValueError):
        analyze_live(ctx, "   ", None, None)


# ------------------------------------------------------------------ public demo


def test_public_demo_export_and_restrictions(boot_ctx, tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "exp", Path(__file__).parents[2] / "scripts" / "export_public_demo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "public.sqlite"
    report = mod.export(boot_ctx.settings.database_url, out)
    assert len(report["runs"]) == 2

    settings = make_settings(
        tmp_path,
        mode="public_demo",
        PUBLIC_DEMO_DATABASE_URL=f"sqlite:///{out}",
        TYPESAFE_API_KEY="tk-should-be-ignored",
    )
    pctx = create_context(settings)
    assert settings.typesafe_api_key is None and not settings.live_calls_allowed
    with pytest.raises(ReadOnlyModeError):
        pctx.require_writable("x")
    with pytest.raises(ReadOnlyModeError):
        analyze_live(pctx, "hello", None, None)
    with pctx.session() as s:
        runs = s.scalars(select(EvaluationRun)).all()
        assert len(runs) == 2
        assert all(p.raw_response_json is None for p in s.scalars(select(Prediction)))
        assert s.scalars(select(AnalysisRequest)).all() == []
        # Session-local exploration works on read-only data.
        res, _ = opt.search(
            s,
            runs[0].id if runs[0].provider == "typesafe_jev" else runs[1].id,
            constraints=Constraints(None, None, None, None, None),
        )
        assert res.candidates
        # Database itself is read-only.
        s.add(AuditEvent(event_type="x", entity_type="y", entity_id="1"))
        with pytest.raises(OperationalError):
            s.commit()
    with pytest.raises(ReadOnlyModeError):
        from harm_detection.services.bootstrap_service import bootstrap

        bootstrap(pctx)
    _ = SimpleNamespace

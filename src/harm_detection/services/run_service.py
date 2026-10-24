"""Evaluation run lifecycle: creation, resumable execution, rerouting, holdout integrity."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from harm_detection.persistence.models import (
    Configuration,
    Dataset,
    DatasetCase,
    EvaluationRun,
    IntegrationEvent,
    Prediction,
    RoutingDecisionRow,
)
from harm_detection.providers.base import FATAL, ErrorCode
from harm_detection.routing.engine import route
from harm_detection.schemas.config import PricingConfig, RoutingConfig
from harm_detection.schemas.detection import DetectionResult, DetectionStatus, RoutingDecision
from harm_detection.services import config_service as cs
from harm_detection.services.dataset_service import case_to_input, cohort_cases

RUN_STATUSES = ("created", "running", "partial", "completed", "failed", "cancelled")
JEV = "typesafe_jev"
MODERATION = "openai_moderation"


class HoldoutReuseError(Exception):
    """Raised when a holdout evaluation is requested after the holdout was already consumed."""


class RunConfigMismatchError(Exception):
    """Raised when a resume would mix outputs from different model/question versions."""


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def cohort_hash(case_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(case_ids)).encode()).hexdigest()


def _config_snapshot(session: Session, ids: dict[str, int | None]) -> dict[str, Any]:
    snap = {}
    for ctype, cid in ids.items():
        if cid is None:
            continue
        cfg = session.get(Configuration, cid)
        snap[ctype] = {"id": cfg.id, "version": cfg.version, "checksum": cfg.checksum}
    return snap


def completed_holdout_runs(session: Session, dataset_id: int) -> list[EvaluationRun]:
    runs = session.scalars(
        select(EvaluationRun).where(
            EvaluationRun.dataset_id == dataset_id,
            EvaluationRun.cohort_name == "holdout",
            EvaluationRun.status.in_(("completed", "partial", "running")),
        )
    ).all()
    return list(runs)


def _holdout_guard(
    session: Session, dataset_id: int, routing_config_id: int | None, acknowledge_reuse: bool
) -> dict[str, Any]:
    prior = completed_holdout_runs(session, dataset_id)
    meta: dict[str, Any] = {
        "holdout_evaluation": True,
        "prior_holdout_run_ids": [r.id for r in prior],
    }
    if prior:
        if not acknowledge_reuse:
            raise HoldoutReuseError(
                f"The holdout for this dataset was already consumed by run(s) {[r.id for r in prior]}. "
                "Further holdout evaluations are exploratory and are not independent estimates. "
                "Acknowledge this explicitly to continue, or prepare a new independent test cohort."
            )
        meta["exploratory"] = True
        meta["exploratory_reason"] = (
            "holdout previously consumed; results are not an unbiased estimate"
        )
    else:
        meta["exploratory"] = False
    if routing_config_id is not None:
        cs.freeze_config(session, routing_config_id)
    return meta


def create_run(
    session: Session,
    *,
    name: str,
    provider: str,
    dataset_id: int,
    cohort: str,
    requested_model: str | None,
    config_ids: dict[str, int | None],
    data_origin: str,
    input_mode: str = "text_only",
    max_cases: int | None = None,
    acknowledge_holdout_reuse: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> EvaluationRun:
    if provider not in (JEV, MODERATION):
        raise ValueError(f"unknown provider {provider!r}")
    if input_mode not in ("text_only", "with_context"):
        raise ValueError("input_mode must be text_only or with_context")
    if provider == MODERATION and input_mode != "text_only":
        raise ValueError("OpenAI Moderation runs are text-only")
    ds = session.get(Dataset, dataset_id)
    if ds is None:
        raise ValueError(f"dataset {dataset_id} not found")
    for ctype, cid in config_ids.items():
        if cid is not None and session.get(Configuration, cid) is None:
            raise ValueError(f"{ctype} configuration {cid} not found")
    if provider == JEV and not config_ids.get("jev_questions"):
        raise ValueError("Jev runs require a question configuration")
    if provider == JEV and not config_ids.get("routing"):
        raise ValueError("Jev runs require a routing configuration")

    cases = cohort_cases(session, dataset_id, cohort)
    if max_cases is not None:
        cases = cases[: max(0, int(max_cases))]
    if not cases:
        raise ValueError("the selected cohort has no cases")
    case_ids = [c.case_id for c in cases]

    meta: dict[str, Any] = {
        "planned_case_ids": case_ids,
        "cohort_hash": cohort_hash(case_ids),
        "data_origin": data_origin,
        "input_mode": input_mode,
        "dataset": {"name": ds.name, "version": ds.version, "checksum": ds.checksum},
        **(extra_metadata or {}),
    }
    if cohort == "holdout":
        meta.update(
            _holdout_guard(
                session, dataset_id, config_ids.get("routing"), acknowledge_holdout_reuse
            )
        )

    run = EvaluationRun(
        name=name,
        provider=provider,
        requested_model=requested_model,
        dataset_id=dataset_id,
        cohort_name=cohort,
        policy_config_id=config_ids.get("policy"),
        question_config_id=config_ids.get("jev_questions") if provider == JEV else None,
        routing_config_id=config_ids.get("routing"),
        label_mapping_config_id=config_ids.get("label_mapping"),
        status="created",
        total_cases=len(case_ids),
        metadata_json=meta,
    )
    session.add(run)
    session.flush()
    # Assign a new dict (never mutate the flushed one) so the JSON change is detected.
    run.metadata_json = {
        **meta,
        "config_snapshot": _config_snapshot(
            session,
            {
                "policy": run.policy_config_id,
                "jev_questions": run.question_config_id,
                "routing": run.routing_config_id,
                "label_mapping": run.label_mapping_config_id,
            },
        ),
    }
    cs.audit(
        session, "run_created", "evaluation_run", run.id, after={"name": name, "cohort": cohort}
    )
    return run


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


def estimate_cost(
    pricing: PricingConfig | None, model: str | None, input_tokens, output_tokens
) -> float | None:
    if pricing is None or model is None or input_tokens is None:
        return None
    price = pricing.models.get(model)
    if price is None:
        return None
    return (
        input_tokens * price.input_per_million_usd
        + (output_tokens or 0) * price.output_per_million_usd
    ) / 1e6


def store_routing(
    session: Session, prediction: Prediction, decision: RoutingDecision, routing_config_id: int
) -> RoutingDecisionRow:
    row = session.scalar(
        select(RoutingDecisionRow).where(
            RoutingDecisionRow.prediction_id == prediction.id,
            RoutingDecisionRow.routing_config_id == routing_config_id,
        )
    )
    values = dict(
        disposition=decision.disposition.value if decision.disposition else None,
        triggered_rules_json=decision.triggered_rule_ids,
        reason_codes_json={
            "matched_rule_ids": decision.matched_rule_ids,
            "reasons": decision.reasons,
        },
        context_warning=decision.context_warning,
        status=decision.status.value,
    )
    if row is None:
        row = RoutingDecisionRow(
            prediction_id=prediction.id, routing_config_id=routing_config_id, **values
        )
        session.add(row)
    else:
        for k, v in values.items():
            setattr(row, k, v)
    return row


def record_prediction(
    session: Session,
    run: EvaluationRun,
    result: DetectionResult,
    *,
    pricing: PricingConfig | None,
    routing_config: RoutingConfig | None,
) -> Prediction:
    """Upsert the prediction for (run, case). Successful predictions are never overwritten."""
    pred = session.scalar(
        select(Prediction).where(Prediction.run_id == run.id, Prediction.case_id == result.case_id)
    )
    if pred is not None and pred.status == "success":
        return pred
    usage = result.provider_usage or {}
    in_tok, out_tok = usage.get("input_tokens"), usage.get("output_tokens")
    values = dict(
        provider=result.provider,
        provider_response_id=result.provider_response_id,
        status=result.status.value,
        raw_response_json=result.raw_response,
        normalized_signals_json=result.signals,
        resolved_model=result.resolved_model,
        data_origin=result.data_origin,
        latency_ms=result.latency_ms,
        input_tokens=in_tok,
        output_tokens=out_tok,
        estimated_cost_usd=estimate_cost(
            pricing, result.resolved_model or result.requested_model, in_tok, out_tok
        ),
        error_code=(result.error or {}).get("code"),
        error_message=(result.error or {}).get("message"),
        created_at=_now(),
    )
    if pred is None:
        pred = Prediction(run_id=run.id, case_id=result.case_id, attempts=1, **values)
        session.add(pred)
    else:
        for k, v in values.items():
            setattr(pred, k, v)
        pred.attempts = (pred.attempts or 1) + 1
    session.flush()
    if run.provider == JEV and routing_config is not None and run.routing_config_id is not None:
        decision = route(
            result.case_id,
            result.signals,
            routing_config,
            detection_succeeded=result.status == DetectionStatus.SUCCESS,
        )
        store_routing(session, pred, decision, run.routing_config_id)
    return pred


@dataclass
class ExecutionSummary:
    run_id: int
    status: str
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    remaining: int = 0
    stopped_reason: str | None = None
    error_counts: dict[str, int] = field(default_factory=dict)


def remaining_case_ids(session: Session, run: EvaluationRun) -> list[str]:
    done = set(
        session.scalars(
            select(Prediction.case_id).where(
                Prediction.run_id == run.id, Prediction.status == "success"
            )
        )
    )
    return [c for c in run.metadata_json.get("planned_case_ids", []) if c not in done]


def _refresh_counts(session: Session, run: EvaluationRun) -> dict[str, int]:
    preds = session.scalars(select(Prediction).where(Prediction.run_id == run.id)).all()
    run.completed_cases = sum(1 for p in preds if p.status == "success")
    run.failed_cases = sum(1 for p in preds if p.status != "success")
    errors: dict[str, int] = {}
    for p in preds:
        if p.status != "success":
            errors[p.error_code or "unknown"] = errors.get(p.error_code or "unknown", 0) + 1
    run.error_summary_json = errors
    return errors


def _check_provider_matches_run(session: Session, run: EvaluationRun, provider: Any) -> None:
    if run.provider == JEV and hasattr(provider, "questions"):
        qcfg = session.get(Configuration, run.question_config_id)
        if provider.questions.version != qcfg.version:
            raise RunConfigMismatchError(
                f"run uses question config {qcfg.version!r} but provider is configured with "
                f"{provider.questions.version!r}; start a new run instead"
            )
    model = getattr(provider, "model", None)
    if model is not None and run.requested_model and model != run.requested_model:
        raise RunConfigMismatchError(
            f"run requested model {run.requested_model!r} but provider uses {model!r}; start a new run"
        )


def execute_run(
    session_factory: sessionmaker[Session],
    run_id: int,
    provider: Any,
    *,
    batch_size: int = 25,
    max_new_cases: int | None = None,
    progress: Callable[[int, int, ExecutionSummary], None] | None = None,
    max_failure_rate: float = 0.5,
    min_cases_for_failure_check: int = 20,
) -> ExecutionSummary:
    """Process remaining cases sequentially, committing after every case.

    Safe to call repeatedly: completed successful predictions are skipped (resume),
    failed predictions are retried. Stops early on fatal provider errors or when the
    failure rate exceeds ``max_failure_rate``.
    """
    with session_factory() as session:
        run = session.get(EvaluationRun, run_id)
        if run is None:
            raise ValueError(f"run {run_id} not found")
        if run.status == "cancelled":
            raise ValueError("cancelled runs cannot be resumed")
        _check_provider_matches_run(session, run, provider)
        todo = remaining_case_ids(session, run)
        if max_new_cases is not None:
            todo = todo[: max(0, max_new_cases)]
        routing_cfg = (
            RoutingConfig.model_validate(
                session.get(Configuration, run.routing_config_id).content_json
            )
            if run.routing_config_id
            else None
        )
        pricing_row = cs.active_config(session, "pricing")
        pricing = PricingConfig.model_validate(pricing_row.content_json) if pricing_row else None
        input_mode = run.metadata_json.get("input_mode", "text_only")
        cases = {
            c.case_id: c
            for c in session.scalars(
                select(DatasetCase).where(
                    DatasetCase.dataset_id == run.dataset_id, DatasetCase.case_id.in_(todo)
                )
            )
        }
        if run.started_at is None:
            run.started_at = _now()
        run.status = "running"
        session.commit()

    summary = ExecutionSummary(run_id=run_id, status="running")
    total = len(todo)
    since_commit = 0
    with session_factory() as session:
        run = session.get(EvaluationRun, run_id)
        for i, case_id in enumerate(todo, start=1):
            case = cases.get(case_id)
            if case is None:
                result = DetectionResult.failure(
                    case_id, run.provider, "case_missing", "case not found in dataset"
                )
            else:
                item = case_to_input(case, include_context=input_mode == "with_context")
                result = provider.detect(item)
            pred = record_prediction(
                session, run, result, pricing=pricing, routing_config=routing_cfg
            )
            if pred.resolved_model and not run.resolved_model:
                run.resolved_model = pred.resolved_model
            summary.attempted += 1
            if result.status == DetectionStatus.SUCCESS:
                summary.succeeded += 1
            else:
                summary.failed += 1
                code = (result.error or {}).get("code", "unknown")
                summary.error_counts[code] = summary.error_counts.get(code, 0) + 1
                if result.data_origin == "live":
                    session.add(
                        IntegrationEvent(
                            provider=run.provider,
                            outcome="failure",
                            error_code=code,
                            message=(result.error or {}).get("message", "")[:500],
                        )
                    )
            since_commit += 1
            if since_commit >= max(1, batch_size) or i == total:
                _refresh_counts(session, run)
                session.commit()
                since_commit = 0
            if progress:
                progress(i, total, summary)
            code = (result.error or {}).get("code")
            if code in {c.value for c in FATAL} or code == ErrorCode.LIVE_CALLS_DISABLED:
                summary.stopped_reason = f"fatal provider error: {code}"
                break
            if (
                summary.attempted >= min_cases_for_failure_check
                and summary.failed / summary.attempted > max_failure_rate
            ):
                summary.stopped_reason = f"failure rate exceeded {max_failure_rate:.0%}"
                break

        _refresh_counts(session, run)
        remaining = remaining_case_ids(session, run)
        summary.remaining = len(remaining)
        attempted_all = {
            p
            for p in session.scalars(select(Prediction.case_id).where(Prediction.run_id == run.id))
        }
        planned = run.metadata_json.get("planned_case_ids", [])
        if (
            summary.stopped_reason
            and summary.stopped_reason.startswith("fatal")
            and run.completed_cases == 0
        ):
            run.status = "failed"
        elif all(c in attempted_all for c in planned) and not summary.stopped_reason:
            run.status = "completed"
            run.completed_at = _now()
        else:
            run.status = "partial"
        summary.status = run.status
        session.commit()

    if summary.status in ("completed", "partial"):
        from harm_detection.services.evaluation_service import compute_and_store_metrics

        with session_factory() as session:
            compute_and_store_metrics(session, run_id)
            session.commit()
    return summary


def cancel_run(session: Session, run_id: int) -> None:
    run = session.get(EvaluationRun, run_id)
    if run and run.status in ("created", "running", "partial"):
        run.status = "cancelled"
        cs.audit(session, "run_cancelled", "evaluation_run", run_id)


def prediction_source_run_id(run: EvaluationRun) -> int:
    return int(run.metadata_json.get("prediction_source_run_id") or run.id)


def reroute_run(
    session: Session,
    source_run_id: int,
    routing_config_id: int,
    *,
    name: str,
    acknowledge_holdout_reuse: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> EvaluationRun:
    """Create a derived run that re-applies routing to stored predictions (no provider calls)."""
    src = session.get(EvaluationRun, source_run_id)
    if src is None:
        raise ValueError("source run not found")
    if src.provider != JEV:
        raise ValueError("rerouting applies to Jev runs only")
    origin_id = prediction_source_run_id(src)
    origin = session.get(EvaluationRun, origin_id)
    cfg_row = session.get(Configuration, routing_config_id)
    if cfg_row is None or cfg_row.config_type != "routing":
        raise ValueError("routing configuration not found")
    routing_cfg = RoutingConfig.model_validate(cfg_row.content_json)

    meta = {
        **{
            k: v
            for k, v in origin.metadata_json.items()
            if k
            not in (
                "holdout_evaluation",
                "exploratory",
                "exploratory_reason",
                "prior_holdout_run_ids",
            )
        },
        "prediction_source_run_id": origin_id,
        "derived_by": "reroute",
        **(extra_metadata or {}),
    }
    if origin.cohort_name == "holdout":
        meta.update(
            _holdout_guard(session, origin.dataset_id, routing_config_id, acknowledge_holdout_reuse)
        )
    run = EvaluationRun(
        name=name,
        provider=origin.provider,
        requested_model=origin.requested_model,
        resolved_model=origin.resolved_model,
        dataset_id=origin.dataset_id,
        cohort_name=origin.cohort_name,
        policy_config_id=origin.policy_config_id,
        question_config_id=origin.question_config_id,
        routing_config_id=routing_config_id,
        label_mapping_config_id=origin.label_mapping_config_id,
        status="running",
        started_at=_now(),
        total_cases=origin.total_cases,
        completed_cases=origin.completed_cases,
        failed_cases=origin.failed_cases,
        error_summary_json=origin.error_summary_json,
        metadata_json=meta,
    )
    session.add(run)
    session.flush()
    run.metadata_json = {
        **meta,
        "config_snapshot": _config_snapshot(
            session,
            {
                "policy": run.policy_config_id,
                "jev_questions": run.question_config_id,
                "routing": run.routing_config_id,
                "label_mapping": run.label_mapping_config_id,
            },
        ),
    }
    for pred in session.scalars(select(Prediction).where(Prediction.run_id == origin_id)):
        decision = route(
            pred.case_id,
            pred.normalized_signals_json,
            routing_cfg,
            detection_succeeded=pred.status == "success",
        )
        store_routing(session, pred, decision, routing_config_id)
    run.status = "completed" if origin.status == "completed" else "partial"
    run.completed_at = _now()
    session.flush()
    from harm_detection.services.evaluation_service import compute_and_store_metrics

    compute_and_store_metrics(session, run.id)
    cs.audit(
        session,
        "run_rerouted",
        "evaluation_run",
        run.id,
        after={"source_run_id": origin_id, "routing_config_id": routing_config_id},
    )
    return run


def list_runs(session: Session) -> list[EvaluationRun]:
    return list(session.scalars(select(EvaluationRun).order_by(EvaluationRun.id.desc())))


def delete_run(session: Session, run_id: int) -> None:
    """Delete a run and its predictions/metrics (local data retention control)."""
    from harm_detection.persistence.models import EvaluationMetric

    run = session.get(EvaluationRun, run_id)
    if run is None:
        return
    dependents = [
        r.id
        for r in session.scalars(select(EvaluationRun))
        if (r.metadata_json or {}).get("prediction_source_run_id") == run_id
    ]
    if dependents:
        raise ValueError(f"runs {dependents} are derived from this run; delete them first")
    for m in session.scalars(select(EvaluationMetric).where(EvaluationMetric.run_id == run_id)):
        session.delete(m)
    for p in session.scalars(select(Prediction).where(Prediction.run_id == run_id)):
        for rd in session.scalars(
            select(RoutingDecisionRow).where(RoutingDecisionRow.prediction_id == p.id)
        ):
            session.delete(rd)
        session.delete(p)
    session.delete(run)
    cs.audit(session, "run_deleted", "evaluation_run", run_id)

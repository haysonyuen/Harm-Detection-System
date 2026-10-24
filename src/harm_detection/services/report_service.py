"""Exportable evaluation reports (JSON summary + per-case CSV). Never includes credentials."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from harm_detection.persistence.models import Configuration, Dataset, EvaluationRun
from harm_detection.security.redaction import redact
from harm_detection.services.evaluation_service import (
    build_eval_frame,
    routing_confusion_from_frame,
    stored_metrics,
)

DISCLAIMER = (
    "Metrics measure agreement with published human annotations mapped to proxy labels; they are not "
    "confirmed X policy violations or platform enforcement accuracy. Independent project, not affiliated "
    "with X, TypeSafe AI, or OpenAI."
)


def run_report(session: Session, run_id: int) -> dict[str, Any]:
    run = session.get(EvaluationRun, run_id)
    ds = session.get(Dataset, run.dataset_id)
    df = build_eval_frame(session, run_id)
    metrics = stored_metrics(session, run_id)
    configs = {}
    for ctype, cid in (
        ("policy", run.policy_config_id),
        ("jev_questions", run.question_config_id),
        ("routing", run.routing_config_id),
        ("label_mapping", run.label_mapping_config_id),
    ):
        if cid:
            c = session.get(Configuration, cid)
            configs[ctype] = {
                "id": c.id,
                "version": c.version,
                "checksum": c.checksum,
                "content": c.content_json,
            }
    meta = {k: v for k, v in (run.metadata_json or {}).items() if k != "planned_case_ids"}
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "run": {
            "id": run.id,
            "name": run.name,
            "provider": run.provider,
            "requested_model": run.requested_model,
            "resolved_model": run.resolved_model,
            "cohort": run.cohort_name,
            "status": run.status,
            "created_at": str(run.created_at),
            "completed_at": str(run.completed_at),
            "total_cases": run.total_cases,
            "completed_cases": run.completed_cases,
            "failed_cases": run.failed_cases,
            "error_summary": run.error_summary_json,
            "metadata": meta,
        },
        "dataset": {
            "name": ds.name,
            "version": ds.version,
            "source_url": ds.source_url,
            "revision": ds.revision,
            "license": ds.license,
            "data_origin": ds.data_origin,
        },
        "configurations": configs,
        "metrics": json.loads(metrics.to_json(orient="records")) if not metrics.empty else [],
        "routing_confusion_matrix": {
            "rows_reference": ["no_flag", "standard_review", "priority_review"],
            "cols_predicted": ["no_flag", "standard_review", "priority_review"],
            "matrix": routing_confusion_from_frame(df) if not df.empty else [],
        },
    }
    return redact(report)


def cases_csv(session: Session, run_id: int) -> str:
    df = build_eval_frame(session, run_id)
    if df.empty:
        return ""
    cols = [
        "case_id",
        "split",
        "stratum",
        "pred_status",
        "error_code",
        "data_origin",
        "threat_proxy",
        "insult_proxy",
        "ref_route",
        "disposition",
        "routing_status",
        "triggered_rules",
        "context_warning",
        "error_class",
        "fp_threat",
        "fn_threat",
        "fp_insult",
        "fn_insult",
        "signals",
    ]
    return df[cols].to_csv(index=False)

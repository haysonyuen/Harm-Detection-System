"""Evaluation frames and stored metrics, all derived from persisted records."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from harm_detection.evaluation.labels import proxy_labels, reference_route
from harm_detection.evaluation.metrics import (
    CALCULATION_VERSION,
    MetricValue,
    binary_metrics,
    ratio,
    routing_confusion,
    routing_metrics,
)
from harm_detection.persistence.models import (
    Configuration,
    DatasetCase,
    EvaluationMetric,
    EvaluationRun,
    Prediction,
    RoutingDecisionRow,
)
from harm_detection.routing.engine import RULE_PRI, RULE_STD, route
from harm_detection.schemas.config import LabelMappingConfig, RoutingConfig
from harm_detection.schemas.detection import DISPOSITION_RANK, Disposition
from harm_detection.services.run_service import JEV, MODERATION, prediction_source_run_id

ENRICHED_STRATA = ("threat_enriched", "insult_enriched", "borderline")


def moderation_native_route(signals: dict[str, float]) -> Disposition:
    """OpenAI Moderation native-flag proxy routing (benchmark only)."""
    if signals.get("harassment_threatening_flagged", 0.0) >= 0.5:
        return Disposition.PRIORITY_REVIEW
    if signals.get("harassment_flagged", 0.0) >= 0.5:
        return Disposition.STANDARD_REVIEW
    return Disposition.NO_FLAG


def _label_mapping(session: Session, run: EvaluationRun) -> LabelMappingConfig:
    cfg_id = run.label_mapping_config_id
    if cfg_id is None:
        from harm_detection.services.config_service import active_config

        row = active_config(session, "label_mapping")
    else:
        row = session.get(Configuration, cfg_id)
    return LabelMappingConfig.model_validate(row.content_json)


def build_eval_frame(
    session: Session,
    run_id: int,
    *,
    routing_config_id: int | None = None,
    routing_config: RoutingConfig | None = None,
) -> pd.DataFrame:
    """One row per planned case with labels, signals, routing and error classification.

    For Jev runs, routing comes from stored routing decisions for ``routing_config_id``
    (default: the run's routing config). If ``routing_config`` is given, routing is
    recomputed in memory by the deterministic engine (nothing is persisted).
    """
    run = session.get(EvaluationRun, run_id)
    if run is None:
        raise ValueError(f"run {run_id} not found")
    mapping = _label_mapping(session, run)
    src_id = prediction_source_run_id(run)
    planned = run.metadata_json.get("planned_case_ids") or []
    cases = {
        c.case_id: c
        for c in session.scalars(
            select(DatasetCase).where(
                DatasetCase.dataset_id == run.dataset_id, DatasetCase.case_id.in_(planned)
            )
        )
    }
    preds = {
        p.case_id: p for p in session.scalars(select(Prediction).where(Prediction.run_id == src_id))
    }
    rc_id = routing_config_id or run.routing_config_id
    stored: dict[int, RoutingDecisionRow] = {}
    if run.provider == JEV and routing_config is None and rc_id is not None and preds:
        for rd in session.scalars(
            select(RoutingDecisionRow).where(
                RoutingDecisionRow.routing_config_id == rc_id,
                RoutingDecisionRow.prediction_id.in_([p.id for p in preds.values()]),
            )
        ):
            stored[rd.prediction_id] = rd

    rows: list[dict[str, Any]] = []
    for cid in planned:
        case = cases.get(cid)
        pred = preds.get(cid)
        labels = proxy_labels(case.reference_labels_json if case else {}, mapping)
        ref = reference_route(labels, mapping)
        row: dict[str, Any] = {
            "case_id": cid,
            "split": case.split if case else None,
            "stratum": case.stratum if case else None,
            "text": case.text if case else None,
            "parent_text": case.parent_text if case else None,
            "target_context": case.target_context if case else None,
            "reference_labels": case.reference_labels_json if case else {},
            **labels,
            "ref_route": ref.value if ref else None,
            "prediction_id": pred.id if pred else None,
            "pred_status": pred.status if pred else "not_evaluated",
            "error_code": pred.error_code if pred else None,
            "error_message": pred.error_message if pred else None,
            "data_origin": pred.data_origin if pred else None,
            "resolved_model": pred.resolved_model if pred else None,
            "latency_ms": pred.latency_ms if pred else None,
            "signals": pred.normalized_signals_json if pred else None,
            "disposition": None,
            "routing_status": "not_evaluated",
            "triggered_rules": [],
            "matched_rules": [],
            "reasons": [],
            "context_warning": None,
            "threat_pred": None,
            "harassment_pred": None,
        }
        if pred is not None:
            if run.provider == JEV:
                if routing_config is not None:
                    d = route(
                        cid,
                        pred.normalized_signals_json,
                        routing_config,
                        detection_succeeded=pred.status == "success",
                    )
                    row.update(
                        disposition=d.disposition.value if d.disposition else None,
                        routing_status=d.status.value,
                        triggered_rules=d.triggered_rule_ids,
                        matched_rules=d.matched_rule_ids,
                        reasons=d.reasons,
                        context_warning=d.context_warning,
                    )
                elif pred.id in stored:
                    rd = stored[pred.id]
                    row.update(
                        disposition=rd.disposition,
                        routing_status=rd.status,
                        triggered_rules=rd.triggered_rules_json,
                        matched_rules=(rd.reason_codes_json or {}).get("matched_rule_ids", []),
                        reasons=(rd.reason_codes_json or {}).get("reasons", []),
                        context_warning=rd.context_warning,
                    )
                if row["routing_status"] == "success":
                    row["threat_pred"] = RULE_PRI in row["matched_rules"]
                    row["harassment_pred"] = RULE_STD in row["matched_rules"]
            elif run.provider == MODERATION and pred.status == "success":
                s = pred.normalized_signals_json or {}
                row.update(
                    disposition=moderation_native_route(s).value,
                    routing_status="success",
                    triggered_rules=["NATIVE-FLAGS"],
                    threat_pred=s.get("harassment_threatening_flagged", 0.0) >= 0.5,
                    harassment_pred=s.get("harassment_flagged", 0.0) >= 0.5,
                )
            elif run.provider == MODERATION:
                row["routing_status"] = "detection_failed"
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ok = (df["routing_status"] == "success") & df["ref_route"].notna()
    df["evaluable"] = ok

    def flag(pred_col, true_col, want_pred, want_true):
        return ok & (df[pred_col] == want_pred) & (df[true_col] == want_true)

    df["fp_threat"] = flag("threat_pred", "threat_proxy", True, False)
    df["fn_threat"] = flag("threat_pred", "threat_proxy", False, True)
    df["fp_insult"] = flag("harassment_pred", "insult_proxy", True, False)
    df["fn_insult"] = flag("harassment_pred", "insult_proxy", False, True)
    rank = {d.value: DISPOSITION_RANK[d] for d in Disposition}
    pr = df["disposition"].map(rank)
    rr = df["ref_route"].map(rank)
    df["under_prioritized"] = ok & (pr < rr)
    df["over_prioritized"] = ok & (pr > rr)
    df["error_class"] = "correct"
    df.loc[~ok, "error_class"] = "not_evaluable"
    df.loc[df["over_prioritized"], "error_class"] = "over_prioritized"
    df.loc[df["under_prioritized"], "error_class"] = "under_prioritized"
    return df


def _cohort_masks(run: EvaluationRun, df: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {"all": pd.Series(True, index=df.index)}
    strata = set(df["stratum"].dropna())
    if "representative" in strata:
        masks["representative"] = df["stratum"] == "representative"
    if strata & set(ENRICHED_STRATA):
        masks["enriched"] = df["stratum"].isin(ENRICHED_STRATA)
    return masks


def compute_metrics_from_frame(
    df: pd.DataFrame, masks: dict[str, pd.Series], *, with_ci: bool = True
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(name: str, category: str, cohort: str, mv: MetricValue) -> None:
        out.append(
            {
                "metric_name": name,
                "category": category,
                "cohort": cohort,
                "value": mv.value,
                "numerator": mv.numerator,
                "denominator": mv.denominator,
                "ci": list(mv.ci) if mv.ci else None,
            }
        )

    for cohort, m in masks.items():
        sub = df[m]
        n_total = len(sub)
        detected = int((sub["pred_status"] == "success").sum())
        attempted = int(sub["prediction_id"].notna().sum())
        add("planned_cases", "coverage", cohort, MetricValue(float(n_total), None, None))
        add("prediction_coverage", "coverage", cohort, ratio(int(sub["evaluable"].sum()), n_total))
        add("api_failure_rate", "coverage", cohort, ratio(attempted - detected, attempted))
        add(
            "routing_failures",
            "coverage",
            cohort,
            MetricValue(
                float(
                    ((sub["pred_status"] == "success") & (sub["routing_status"] != "success")).sum()
                ),
                None,
                None,
            ),
        )
        ev = sub[sub["evaluable"]]
        add("evaluated_cases", "coverage", cohort, MetricValue(float(len(ev)), None, None))
        if ev.empty:
            continue
        for category, true_col, pred_col in (
            ("threat_proxy", "threat_proxy", "threat_pred"),
            ("insult_proxy", "insult_proxy", "harassment_pred"),
        ):
            mvals = binary_metrics(
                ev[true_col].astype(bool).tolist(),
                ev[pred_col].astype(bool).tolist(),
                with_ci=with_ci,
            )
            for name, mv in mvals.items():
                add(name, category, cohort, mv)
        refs = [Disposition(x) for x in ev["ref_route"]]
        preds = [Disposition(x) for x in ev["disposition"]]
        for name, mv in routing_metrics(refs, preds, with_ci=with_ci).items():
            add(name, "routing", cohort, mv)
    return out


def compute_and_store_metrics(
    session: Session, run_id: int, *, routing_config_id: int | None = None
) -> list[dict[str, Any]]:
    run = session.get(EvaluationRun, run_id)
    rc_id = routing_config_id or run.routing_config_id
    df = build_eval_frame(session, run_id, routing_config_id=rc_id)
    if df.empty:
        return []
    rows = compute_metrics_from_frame(df, _cohort_masks(run, df))
    session.execute(
        delete(EvaluationMetric).where(
            EvaluationMetric.run_id == run_id,
            EvaluationMetric.routing_config_id.is_(rc_id)
            if rc_id is None
            else EvaluationMetric.routing_config_id == rc_id,
        )
    )
    for r in rows:
        session.add(
            EvaluationMetric(
                run_id=run_id,
                routing_config_id=rc_id,
                metric_name=r["metric_name"],
                category=r["category"],
                cohort=r["cohort"],
                value=r["value"],
                numerator=r["numerator"],
                denominator=r["denominator"],
                confidence_interval_json={
                    "method": "percentile_bootstrap_1000_seed0_95",
                    "interval": r["ci"],
                }
                if r["ci"]
                else None,
                calculation_version=CALCULATION_VERSION,
            )
        )
    session.flush()
    return rows


def stored_metrics(session: Session, run_id: int) -> pd.DataFrame:
    rows = session.scalars(select(EvaluationMetric).where(EvaluationMetric.run_id == run_id)).all()
    return pd.DataFrame(
        [
            {
                "metric": r.metric_name,
                "category": r.category,
                "cohort": r.cohort,
                "value": r.value,
                "numerator": r.numerator,
                "denominator": r.denominator,
                "ci": (r.confidence_interval_json or {}).get("interval"),
                "routing_config_id": r.routing_config_id,
            }
            for r in rows
        ]
    )


def metric_lookup(
    df: pd.DataFrame, metric: str, category: str, cohort: str = "all"
) -> float | None:
    if df.empty:
        return None
    m = df[(df["metric"] == metric) & (df["category"] == category) & (df["cohort"] == cohort)]
    if m.empty:
        return None
    v = m.iloc[0]["value"]
    return None if pd.isna(v) else float(v)


def routing_confusion_from_frame(df: pd.DataFrame) -> list[list[int]]:
    ev = df[df["evaluable"]] if not df.empty else df
    return routing_confusion(
        [Disposition(x) for x in ev["ref_route"]], [Disposition(x) for x in ev["disposition"]]
    )


def comparability(session: Session, run_a: int, run_b: int) -> dict[str, Any]:
    """Two runs are comparable only if they share dataset, cohort, case set and label mapping."""
    a, b = session.get(EvaluationRun, run_a), session.get(EvaluationRun, run_b)
    issues = []
    if a.dataset_id != b.dataset_id:
        issues.append("different datasets")
    if a.cohort_name != b.cohort_name:
        issues.append("different cohorts")
    if a.metadata_json.get("cohort_hash") != b.metadata_json.get("cohort_hash"):
        issues.append("different case sets")
    if a.label_mapping_config_id != b.label_mapping_config_id:
        issues.append("different label mappings")
    if a.metadata_json.get("input_mode") != b.metadata_json.get("input_mode"):
        issues.append("different input representations (text-only vs with context)")
    if a.metadata_json.get("data_origin") != b.metadata_json.get("data_origin"):
        issues.append("different data origins (live vs fixture)")
    return {"comparable": not issues, "issues": issues}

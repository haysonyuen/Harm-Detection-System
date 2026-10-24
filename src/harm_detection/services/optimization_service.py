"""Threshold optimization workflow: search on stored development predictions, human selection."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from harm_detection.optimization.search import Constraints, EvalCase, SearchResult, run_search
from harm_detection.persistence.models import Configuration, EvaluationRun, OptimizationExperiment
from harm_detection.schemas.config import RoutingConfig
from harm_detection.schemas.detection import Disposition
from harm_detection.services import config_service as cs
from harm_detection.services.evaluation_service import build_eval_frame
from harm_detection.services.run_service import JEV


def load_eval_cases(session: Session, run_id: int) -> tuple[list[EvalCase], str]:
    """Return evaluable cases and the review-volume basis description."""
    run = session.get(EvaluationRun, run_id)
    if run is None:
        raise ValueError("run not found")
    if run.provider != JEV:
        raise ValueError("threshold optimization requires a Jev run")
    if run.cohort_name == "holdout":
        raise ValueError("threshold optimization on the holdout is not permitted")
    df = build_eval_frame(session, run_id)
    ok = df[(df["pred_status"] == "success") & df["ref_route"].notna()]
    has_rep = bool((ok["stratum"] == "representative").any())
    basis = (
        "representative development subset"
        if has_rep
        else "ALL evaluated cases (no representative subset exists; volume is NOT representative)"
    )
    cases = [
        EvalCase(
            case_id=r.case_id,
            signals=r.signals,
            threat_proxy=bool(r.threat_proxy),
            insult_proxy=bool(r.insult_proxy),
            ref_route=Disposition(r.ref_route),
            representative=(r.stratum == "representative") if has_rep else True,
        )
        for r in ok.itertuples()
    ]
    return cases, basis


def search(
    session: Session,
    run_id: int,
    *,
    space: dict[str, list[float]] | None = None,
    constraints: Constraints | None = None,
    extra_objectives: list[tuple[str, str]] | None = None,
) -> tuple[SearchResult, str]:
    run = session.get(EvaluationRun, run_id)
    base = RoutingConfig.model_validate(
        session.get(Configuration, run.routing_config_id).content_json
    )
    cases, basis = load_eval_cases(session, run_id)
    return run_search(
        cases, base, space, constraints, extra_objectives=extra_objectives or ()
    ), basis


def save_experiment(
    session: Session, run_id: int, result: SearchResult, constraints: Constraints, basis: str
) -> OptimizationExperiment:
    exp = OptimizationExperiment(
        source_run_id=run_id,
        constraints_json=constraints.to_dict(),
        search_space_json={
            "space": result.search_space,
            "objectives": result.objectives,
            "review_volume_basis": basis,
            "n_cases": result.n_cases,
            "n_representative": result.n_representative,
        },
        candidate_results_json=result.candidates,
        status="searched",
    )
    session.add(exp)
    session.flush()
    cs.audit(
        session,
        "optimization_searched",
        "optimization_experiment",
        exp.id,
        after={"source_run_id": run_id, "n_candidates": len(result.candidates)},
    )
    return exp


def select_candidate(
    session: Session,
    experiment_id: int,
    candidate_id: int,
    *,
    rationale: str,
    new_version: str,
) -> Configuration:
    """Human selection: saves an immutable routing configuration snapshot. Does NOT activate it."""
    if not rationale or not rationale.strip():
        raise ValueError("a decision rationale is required")
    exp = session.get(OptimizationExperiment, experiment_id)
    if exp is None:
        raise ValueError("experiment not found")
    if exp.selected_config_id is not None:
        raise ValueError("a candidate was already selected for this experiment; run a new search")
    cand = next((c for c in exp.candidate_results_json if c["candidate_id"] == candidate_id), None)
    if cand is None:
        raise ValueError("candidate not found")
    run = session.get(EvaluationRun, exp.source_run_id)
    base_row = session.get(Configuration, run.routing_config_id)
    base = RoutingConfig.model_validate(base_row.content_json)
    content = base.model_dump()
    content["version"] = new_version
    content["tuning_status"] = "human_selected"
    content["description"] = (
        f"Human-selected from optimization experiment {exp.id} (candidate {candidate_id}) "
        f"on development run {run.id}. Selected by the local operator; not automatically promoted."
    )
    content["thresholds"] = {k: cand[k] for k in content["thresholds"]}
    content["notes"] = list(base.notes) + [f"Selection rationale: {rationale.strip()}"]
    cfg = cs.save_config(
        session,
        "routing",
        content,
        parent_config_id=base_row.id,
        change_note=f"selected from optimization experiment {exp.id}",
    )
    cs.freeze_config(session, cfg.id)
    exp.selected_config_id = cfg.id
    exp.selection_rationale = rationale.strip()
    exp.status = "selected"
    cs.audit(
        session,
        "optimization_candidate_selected",
        "optimization_experiment",
        exp.id,
        after={"candidate_id": candidate_id, "config_id": cfg.id, "rationale": rationale.strip()},
    )
    return cfg


def candidate_summary(c: dict[str, Any]) -> str:
    return ", ".join(
        f"{k}={c[k]}"
        for k in (
            "threat_priority",
            "harassment_standard",
            "targeting",
            "context_sufficiency",
            "uncertain_risk",
        )
    )

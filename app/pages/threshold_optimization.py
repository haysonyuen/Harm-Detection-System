from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import select

from app.components.common import (
    fmt_pct,
    get_ctx,
    mode_banner,
    origin_notice,
    proxy_disclaimer,
    run_label,
)
from harm_detection.optimization.search import DEFAULT_SEARCH_SPACE, THRESHOLD_KEYS, Constraints
from harm_detection.persistence.models import Configuration, EvaluationRun, OptimizationExperiment
from harm_detection.schemas.config import RoutingConfig, RoutingThresholds
from harm_detection.services import optimization_service as opt
from harm_detection.services.evaluation_service import build_eval_frame, compute_metrics_from_frame
from harm_detection.services.run_service import (
    HoldoutReuseError,
    completed_holdout_runs,
    reroute_run,
)

ctx = get_ctx()
st.title("Threshold Optimization")
mode_banner(ctx)
st.caption(
    "All candidates are evaluated by re-applying the deterministic routing engine to stored Jev "
    "predictions. No new model requests are made."
)

with ctx.session() as s:
    runs = s.scalars(
        select(EvaluationRun)
        .where(
            EvaluationRun.provider == "typesafe_jev",
            EvaluationRun.cohort_name.in_(("development", "representative_dev")),
            EvaluationRun.status.in_(("completed", "partial")),
        )
        .order_by(EvaluationRun.id.desc())
    ).all()
if not runs:
    st.info("No completed Jev development runs to optimize on.")
    st.stop()
run = st.selectbox("Source development run (stored predictions)", runs, format_func=run_label)
origin_notice(run)
proxy_disclaimer(run)
with ctx.session() as s:
    base = RoutingConfig.model_validate(s.get(Configuration, run.routing_config_id).content_json)

tab_whatif, tab_search, tab_holdout = st.tabs(
    ["What-if thresholds", "Automated search & selection", "Holdout evaluation"]
)

# ---------------------------------------------------------------- what-if (session-local)
with tab_whatif:
    st.caption(
        f"Starting from `{base.version}` ({base.tuning_status}). Changes here are session-local and not saved."
    )
    cols = st.columns(5)
    th = {
        k: cols[i].slider(
            k.replace("_", " "), 0.0, 1.0, float(getattr(base.thresholds, k)), 0.01, key=f"wi_{k}"
        )
        for i, k in enumerate(THRESHOLD_KEYS)
    }
    req_t = st.checkbox(
        "Require targeting for Standard Review (R-STD-01)", base.require_targeting_for_standard
    )
    cfg = base.model_copy(
        update={
            "thresholds": RoutingThresholds(**th),
            "require_targeting_for_standard": req_t,
            "version": base.version + "+whatif",
        }
    )
    with ctx.session() as s:
        df = build_eval_frame(s, run.id, routing_config=cfg)
    masks = {"all": pd.Series(True, index=df.index)}
    if (df.stratum == "representative").any():
        masks["representative"] = df.stratum == "representative"
    rows = compute_metrics_from_frame(df, masks, with_ci=False)
    mdf = pd.DataFrame(rows)
    show = mdf[
        mdf.metric_name.isin(
            [
                "recall",
                "precision",
                "priority_recall",
                "under_prioritization_rate",
                "total_review_rate",
                "priority_review_rate",
            ]
        )
    ]
    piv = show.pivot_table(
        index=["category", "metric_name"],
        columns="cohort",
        values="value",
        aggfunc="first",
        dropna=False,
    )
    st.dataframe(piv.map(fmt_pct), use_container_width=True)
    st.caption(f"Routes: {df.disposition.value_counts(dropna=False).to_dict()}")

# ---------------------------------------------------------------- automated search
with tab_search:
    st.markdown("**Search space** (comma-separated values per threshold)")
    space = {}
    cols = st.columns(5)
    for i, k in enumerate(THRESHOLD_KEYS):
        raw = cols[i].text_input(
            k, ", ".join(str(v) for v in DEFAULT_SEARCH_SPACE[k]), key=f"sp_{k}"
        )
        try:
            space[k] = [float(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            st.error(f"Invalid values for {k}")
            st.stop()
    st.markdown(
        "**Risk constraints** — provisional defaults until an initial benchmark is reviewed"
    )
    c = st.columns(5)
    min_rec = c[0].number_input("Min threat-proxy recall", 0.0, 1.0, 0.8, 0.05)
    max_vol = c[1].number_input("Max representative review volume", 0.0, 1.0, 0.3, 0.05)
    use_prec = c[2].checkbox("Min threat precision")
    min_prec = c[2].number_input("value", 0.0, 1.0, 0.3, 0.05, disabled=not use_prec, key="mp")
    use_under = c[3].checkbox("Max under-prioritization")
    max_under = c[3].number_input("value ", 0.0, 1.0, 0.3, 0.05, disabled=not use_under, key="mu")
    min_sup = c[4].number_input("Min threat-proxy support", 0, 10000, 5)
    extra = st.multiselect(
        "Additional Pareto objectives", ["threat_precision (max)", "under_prioritization (min)"]
    )
    constraints = Constraints(
        min_threat_recall=min_rec,
        max_review_volume=max_vol,
        min_threat_precision=min_prec if use_prec else None,
        max_under_prioritization=max_under if use_under else None,
        min_threat_support=int(min_sup),
    )
    if st.button("Run candidate search", type="primary"):
        objs = [(e.split(" ")[0], e.split("(")[1].rstrip(")")) for e in extra]
        try:
            with ctx.session() as s:
                result, basis = opt.search(
                    s, run.id, space=space, constraints=constraints, extra_objectives=objs
                )
                exp_id = None
                if ctx.settings.writes_allowed:
                    exp = opt.save_experiment(s, run.id, result, constraints, basis)
                    s.commit()
                    exp_id = exp.id
            st.session_state["opt"] = {
                "result": result,
                "basis": basis,
                "exp_id": exp_id,
                "run_id": run.id,
            }
        except ValueError as exc:
            st.error(str(exc))

    state = st.session_state.get("opt")
    if state and state["run_id"] == run.id:
        result, basis = state["result"], state["basis"]
        cdf = pd.DataFrame(result.candidates)
        st.caption(
            f"{len(cdf)} candidates · {result.n_cases} evaluable cases · review volume basis: {basis}"
            + ("" if state["exp_id"] is None else f" · saved as experiment #{state['exp_id']}")
        )
        if "NOT representative" in basis:
            st.warning(
                "No representative subset: review volume here is not a representative estimate."
            )
        cdf["status"] = cdf.apply(
            lambda r: (
                "Pareto-optimal"
                if r.pareto_optimal
                else ("feasible" if r.passes_constraints else "fails constraints")
            ),
            axis=1,
        )
        fig = px.scatter(
            cdf,
            x="review_volume_representative",
            y="threat_recall",
            color="status",
            hover_data=list(THRESHOLD_KEYS) + ["threat_precision", "under_prioritization"],
            color_discrete_map={
                "Pareto-optimal": "#b42318",
                "feasible": "#1570ef",
                "fails constraints": "#d0d5dd",
            },
            labels={
                "review_volume_representative": "review volume (representative)",
                "threat_recall": "threat-proxy recall",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
        front = cdf[cdf.pareto_optimal]
        if front.empty:
            st.warning(
                "No candidate satisfies all constraints. Relax constraints or widen the search space. "
                "No candidate is selected automatically."
            )
        show_cols = [
            "candidate_id",
            *THRESHOLD_KEYS,
            "threat_recall",
            "threat_precision",
            "insult_recall",
            "insult_precision",
            "under_prioritization",
            "review_volume_representative",
            "status",
            "constraint_failures",
        ]
        only_front = st.checkbox("Show Pareto front only", value=not front.empty)
        st.dataframe(
            (front if only_front else cdf)[show_cols], use_container_width=True, hide_index=True
        )

        st.markdown("#### Human selection")
        if not ctx.settings.writes_allowed or state["exp_id"] is None:
            st.caption(
                "🔒 Saving a selection is disabled in public demo mode (exploration is session-local)."
            )
        elif not front.empty:
            cid = st.selectbox(
                "Candidate",
                front.candidate_id.tolist(),
                format_func=lambda i: f"#{i}: " + opt.candidate_summary(cdf.iloc[i].to_dict()),
            )
            version = st.text_input(
                "New routing config version", f"routing_v1_exp{state['exp_id']}_c{cid}"
            )
            rationale = st.text_area("Decision rationale (required)")
            if st.button("Save immutable configuration snapshot"):
                try:
                    with ctx.session() as s:
                        cfg_row = opt.select_candidate(
                            s, state["exp_id"], int(cid), rationale=rationale, new_version=version
                        )
                        s.commit()
                    st.success(
                        f"Saved routing config `{cfg_row.version}` (id {cfg_row.id}, frozen, NOT active). "
                        "Evaluate it on the holdout next; promotion is a separate action on the "
                        "Configuration page."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

# ---------------------------------------------------------------- holdout
with tab_holdout:
    with ctx.session() as s:
        exps = s.scalars(
            select(OptimizationExperiment)
            .where(OptimizationExperiment.selected_config_id.is_not(None))
            .order_by(OptimizationExperiment.id.desc())
        ).all()
        selected_cfgs = {e.id: s.get(Configuration, e.selected_config_id) for e in exps}
        holdout_runs = s.scalars(
            select(EvaluationRun).where(
                EvaluationRun.provider == "typesafe_jev",
                EvaluationRun.cohort_name == "holdout",
                EvaluationRun.dataset_id == run.dataset_id,
                EvaluationRun.status.in_(("completed", "partial")),
            )
        ).all()
        consumed = completed_holdout_runs(s, run.dataset_id)
    if consumed:
        st.error(
            f"Holdout already consumed by run(s) {[r.id for r in consumed]}. Further holdout evaluations "
            "are labeled exploratory."
        )
    else:
        st.success("Holdout untouched for this dataset.")
    if not exps:
        st.info("Select and save a candidate configuration first.")
    else:
        exp = st.selectbox(
            "Selected configuration",
            exps,
            format_func=lambda e: f"experiment #{e.id} → {selected_cfgs[e.id].version}",
        )
        cfg_row = selected_cfgs[exp.id]
        st.json(cfg_row.content_json["thresholds"])
        st.caption(
            "Option A: run Jev on the holdout via Batch Evaluation with this routing config. "
            "Option B (below): re-route existing holdout Jev predictions made with the same question "
            "config (no new calls)."
        )
        same_q = [
            r
            for r in holdout_runs
            if r.question_config_id == run.question_config_id
            and not r.metadata_json.get("prediction_source_run_id")
        ]
        if not same_q:
            st.info(
                "No stored holdout predictions with the same question configuration. Use Batch Evaluation."
            )
        elif ctx.settings.writes_allowed:
            src = st.selectbox("Stored holdout predictions", same_q, format_func=run_label)
            ack = st.checkbox(
                "I acknowledge the holdout is consumed by this evaluation and must not be used for retuning."
            )
            if st.button("Evaluate on holdout", disabled=not ack):
                try:
                    with ctx.session() as s:
                        new = reroute_run(
                            s,
                            src.id,
                            cfg_row.id,
                            name=f"Holdout eval {cfg_row.version}",
                            acknowledge_holdout_reuse=bool(consumed),
                            extra_metadata={"optimization_experiment_id": exp.id},
                        )
                        s.commit()
                    st.success(f"Final evaluation run #{new.id} saved. See Model Performance.")
                except HoldoutReuseError as exc:
                    st.error(str(exc))

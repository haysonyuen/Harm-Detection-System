from __future__ import annotations

import pandas as pd
import streamlit as st
import yaml
from sqlalchemy import select

from app.components.common import get_ctx, mode_banner, origin_notice, run_label
from harm_detection.improvement import lifecycle as lc
from harm_detection.persistence.models import (
    AnalysisRequest,
    AuditEvent,
    Configuration,
    EvaluationRun,
    ImprovementRecommendation,
)
from harm_detection.providers.openai_analyst import OpenAIAnalyst
from harm_detection.services import analysis_service as an
from harm_detection.services import config_service as cs
from harm_detection.services.evaluation_service import build_eval_frame
from harm_detection.services.run_service import reroute_run

ctx = get_ctx()
st.title("AI Improvement Lab")
mode_banner(ctx)
st.caption(
    f"Optional, opt-in error analysis with `{ctx.settings.openai_analyst_model}`. Outputs are hypotheses and "
    "proposals, not verified root causes and not ground truth. Nothing generated here changes the active "
    "detector; approval, experimentation and promotion are separate recorded human actions."
)

with ctx.session() as s:
    spent = an.spent_usd(s)
budget = ctx.settings.max_analyst_budget_usd
c1, c2, c3 = st.columns(3)
c1.metric("Estimated analysis spend", f"${spent:.4f}")
c2.metric("Local budget", f"${budget:.2f}")
c3.metric("Remaining (estimate)", f"${max(budget - spent, 0):.4f}")
st.caption(
    "Local cost controls are estimates from the pricing config, not guaranteed provider billing caps."
)

available = ctx.settings.openai_available and ctx.settings.writes_allowed
if not available:
    reason = (
        "public demo mode"
        if ctx.settings.is_public_demo
        else "offline mode"
        if not ctx.settings.live_calls_allowed
        else "OPENAI_API_KEY is not configured"
    )
    st.warning(
        f"AI analysis is unavailable ({reason}). Existing analyses and recommendations remain viewable."
    )

tab_new, tab_recs, tab_hist = st.tabs(["New analysis", "Recommendations", "Decision history"])

with tab_new:
    with ctx.session() as s:
        runs = s.scalars(
            select(EvaluationRun)
            .where(
                EvaluationRun.cohort_name.in_(("development", "representative_dev")),
                EvaluationRun.status.in_(("completed", "partial")),
            )
            .order_by(EvaluationRun.id.desc())
        ).all()
    if not runs:
        st.info("No eligible (non-holdout) runs.")
    else:
        default_run = st.session_state.get("ai_run_id")
        idx = next((i for i, r in enumerate(runs) if r.id == default_run), 0)
        run = st.selectbox("Run", runs, index=idx, format_func=run_label)
        origin_notice(run)
        with ctx.session() as s:
            df = build_eval_frame(s, run.id)
        errors = df[
            df.fp_threat
            | df.fn_threat
            | df.fp_insult
            | df.fn_insult
            | df.under_prioritized
            | df.over_prioritized
        ]
        mode = st.radio("Selection", ["Individual errors", "Error cluster"], horizontal=True)
        if mode == "Individual errors":
            pre = [c for c in st.session_state.get("ai_case_ids", []) if c in set(df.case_id)]
            options = sorted(set(errors.case_id) | set(pre))
            chosen = st.multiselect(
                "Error cases",
                options,
                default=pre,
                format_func=lambda c: f"{c} · {df[df.case_id == c].error_class.iloc[0]}",
            )
            scope = "single_case" if len(chosen) == 1 else "case_group"
        else:
            errors = errors.assign(
                cluster=errors.error_class
                + " / "
                + errors.triggered_rules.map(lambda x: ",".join(x or []))
            )
            clusters = errors.groupby("cluster").case_id.apply(list)
            if clusters.empty:
                st.info("No errors in this run.")
                chosen, scope = [], "case_group"
            else:
                cl = st.selectbox(
                    "Cluster",
                    clusters.index,
                    format_func=lambda k: f"{k} ({len(clusters[k])} cases)",
                )
                chosen, scope = clusters[cl], "case_group"
        if chosen:
            try:
                with ctx.session() as s:
                    pkg = an.build_input_package(s, run.id, list(chosen), scope)
                    check = an.preflight(s, ctx.settings, pkg)
                st.caption(
                    f"Estimated cost for this request: \\${check.estimated_call_usd:.4f} "
                    f"(includes reserved output tokens)."
                )
                if not check.allowed:
                    st.error(f"Blocked: {check.reason}")
                if st.button(
                    "Generate Recommendations",
                    type="primary",
                    disabled=not (available and check.allowed),
                ):
                    analyst = OpenAIAnalyst(
                        ctx.settings.openai_api_key, model=ctx.settings.openai_analyst_model
                    )
                    with st.spinner("Requesting structured analysis…"), ctx.session() as s:
                        req = an.run_analysis(
                            s, ctx.settings, analyst, run.id, list(chosen), scope=scope
                        )
                        s.commit()
                    if req.status == "completed":
                        st.success(f"Analysis #{req.id} completed. See the Recommendations tab.")
                    else:
                        st.error(f"Analysis #{req.id} {req.status}: {req.error_message}")
            except an.AnalysisBlockedError as exc:
                st.error(str(exc))

with tab_recs:
    with ctx.session() as s:
        reqs = s.scalars(select(AnalysisRequest).order_by(AnalysisRequest.id.desc())).all()
        recs = s.scalars(
            select(ImprovementRecommendation).order_by(ImprovementRecommendation.id.desc())
        ).all()
    if not reqs:
        st.info("No analyses yet.")
    for req in reqs:
        with st.expander(
            f"Analysis #{req.id} · {req.status} · cases {req.case_ids_json} · "
            f"\\${(req.estimated_cost_usd or 0):.4f}",
            expanded=req.status != "completed",
        ):
            if req.error_message:
                st.error(req.error_message)
            out = req.structured_analysis_json or {}
            if out:
                st.markdown("**Observations (from stored results)**")
                for o in out.get("observations", []):
                    st.markdown(f"- {o['statement']} _(cases: {', '.join(o['case_ids'])})_")
                st.markdown("**Hypotheses (unverified)**")
                for h in out.get("hypotheses", []):
                    st.markdown(f"- {h['hypothesis']}")
                    if h["uncertainties"]:
                        st.caption("Uncertainties: " + "; ".join(h["uncertainties"]))
                if out.get("unsupported_or_unknown"):
                    st.caption("Unknown / unsupported: " + "; ".join(out["unsupported_or_unknown"]))
                if out.get("limitations"):
                    st.caption("Limitations: " + "; ".join(out["limitations"]))

    st.subheader("Recommendation cards")
    for rec in [r for r in recs]:
        p = rec.proposed_change_json
        with st.container(border=True):
            st.markdown(
                f"**#{rec.id} · {rec.change_type} → {rec.affected_component}** · status `{rec.status}`"
            )
            st.markdown(p.get("proposed_change", ""))
            st.caption(
                f"Expected benefit: {p.get('expected_benefit')} · Possible regressions: "
                f"{'; '.join(p.get('possible_regressions', []))}"
            )
            st.caption(f"Validation plan: {p.get('validation_plan')}")
            if rec.candidate_config_id:
                st.caption(
                    f"Candidate config id {rec.candidate_config_id} (inactive until promoted)"
                )
            if rec.validation_run_id:
                st.caption(f"Linked experiment run #{rec.validation_run_id}")
            if not ctx.settings.writes_allowed or rec.status in lc.TERMINAL:
                continue
            key = f"rec{rec.id}"
            if rec.status == lc.PROPOSED:
                new_text = st.text_area(
                    "Edit proposed change", p.get("proposed_change", ""), key=key + "e"
                )
                if new_text != p.get("proposed_change") and st.button("Save edit", key=key + "se"):
                    with ctx.session() as s:
                        an.edit_recommendation(s, rec.id, new_text)
                        s.commit()
                    st.rerun()
                rationale = st.text_input("Decision rationale", key=key + "r")
                ctype = an.CHANGE_TYPE_TO_CONFIG.get(rec.change_type)
                cand_yaml = None
                if ctype:
                    with ctx.session() as s:
                        base = cs.active_config(s, ctype)
                    proposal = dict(base.content_json)
                    proposal["version"] = f"{base.version}_rec{rec.id}"
                    cand_yaml = st.text_area(
                        f"Candidate {ctype} config (edit to implement the change)",
                        yaml.safe_dump(proposal, sort_keys=False),
                        height=220,
                        key=key + "y",
                    )
                b1, b2 = st.columns(2)
                if b1.button(
                    "Accept for experiment", key=key + "a", disabled=not rationale.strip()
                ):
                    try:
                        with ctx.session() as s:
                            an.accept_for_experiment(
                                s,
                                rec.id,
                                rationale,
                                yaml.safe_load(cand_yaml) if cand_yaml else None,
                            )
                            s.commit()
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))
                if b2.button("Reject", key=key + "x", disabled=not rationale.strip()):
                    with ctx.session() as s:
                        an.reject(s, rec.id, rationale)
                        s.commit()
                    st.rerun()
            elif rec.status == lc.ACCEPTED:
                with ctx.session() as s:
                    cand = (
                        s.get(Configuration, rec.candidate_config_id)
                        if rec.candidate_config_id
                        else None
                    )
                    req = s.get(AnalysisRequest, rec.analysis_request_id)
                if cand is not None and cand.config_type == "routing" and req.source_run_id:
                    if st.button(
                        "Run routing experiment on source development run (no new model calls)",
                        key=key + "rr",
                    ):
                        with ctx.session() as s:
                            new = reroute_run(
                                s,
                                req.source_run_id,
                                cand.id,
                                name=f"Experiment rec #{rec.id}",
                                extra_metadata={"recommendation_id": rec.id},
                            )
                            an.link_experiment(s, rec.id, new.id)
                            s.commit()
                        st.rerun()
                with ctx.session() as s:
                    eligible = [
                        r
                        for r in s.scalars(select(EvaluationRun))
                        if cand
                        and cand.id
                        in (r.routing_config_id, r.question_config_id, r.policy_config_id)
                        and r.cohort_name != "holdout"
                    ]
                if eligible:
                    vr = st.selectbox(
                        "Link existing validation run",
                        eligible,
                        format_func=run_label,
                        key=key + "vr",
                    )
                    if st.button("Link experiment", key=key + "l"):
                        with ctx.session() as s:
                            an.link_experiment(s, rec.id, vr.id)
                            s.commit()
                        st.rerun()
                elif cand is not None and cand.config_type == "jev_questions":
                    st.caption(
                        "Question changes require a new Jev run with the candidate question config "
                        "(Batch Evaluation, development cohort)."
                    )
                if st.button("Reject", key=key + "x2"):
                    with ctx.session() as s:
                        an.reject(s, rec.id, "rejected after acceptance")
                        s.commit()
                    st.rerun()
            elif rec.status == lc.EXPERIMENTED:
                st.page_link(
                    "pages/experiment_history.py",
                    label="Compare the experiment run with its source run",
                )
                rationale = st.text_input("Promotion decision rationale", key=key + "pr")
                confirm = st.checkbox(
                    "I confirm promoting this candidate configuration to ACTIVE", key=key + "c"
                )
                b1, b2 = st.columns(2)
                if b1.button(
                    "Promote", key=key + "p", disabled=not (confirm and rationale.strip())
                ):
                    with ctx.session() as s:
                        an.promote(s, rec.id, confirm=confirm, rationale=rationale)
                        s.commit()
                    st.rerun()
                if b2.button("Do not promote", key=key + "np", disabled=not rationale.strip()):
                    with ctx.session() as s:
                        an.decline_promotion(s, rec.id, rationale)
                        s.commit()
                    st.rerun()

with tab_hist:
    with ctx.session() as s:
        events = s.scalars(select(AuditEvent).order_by(AuditEvent.id.desc()).limit(300)).all()
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "time": e.timestamp,
                    "event": e.event_type,
                    "entity": f"{e.entity_type}#{e.entity_id}",
                    "actor": e.actor,
                    "after": e.after_json,
                }
                for e in events
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

from __future__ import annotations

import streamlit as st
from sqlalchemy import func, select

from app.components.common import (
    fmt_ci,
    fmt_pct,
    get_ctx,
    mode_banner,
    origin_notice,
    proxy_disclaimer,
    run_label,
)
from harm_detection.persistence.models import EvaluationRun, OptimizationExperiment
from harm_detection.services import config_service as cs
from harm_detection.services.evaluation_service import stored_metrics

ctx = get_ctx()
st.title("Social Media Harm Detection Lab")
mode_banner(ctx)

with ctx.session() as s:
    active = {
        t: cs.active_config(s, t) for t in ("policy", "jev_questions", "routing", "label_mapping")
    }
    runs = s.scalars(
        select(EvaluationRun)
        .where(EvaluationRun.status == "completed", EvaluationRun.cohort_name != "playground")
        .order_by(EvaluationRun.completed_at.desc(), EvaluationRun.id.desc())
    ).all()
    n_runs = s.scalar(select(func.count(EvaluationRun.id)))
    n_exp = s.scalar(select(func.count(OptimizationExperiment.id)))

routing = active["routing"]
c1, c2 = st.columns(2)
c1.markdown(
    f"**Detection model (requested):** `{ctx.settings.jev_model}`  \n"
    f"**Question config:** `{active['jev_questions'].version if active['jev_questions'] else 'none'}`  \n"
    f"**Routing config:** `{routing.version if routing else 'none'}`"
)
c2.markdown(
    f"**Policy:** `{active['policy'].version if active['policy'] else 'none'}`  \n"
    f"**Label mapping:** `{active['label_mapping'].version if active['label_mapping'] else 'none'}`  \n"
    f"**Saved runs / optimization experiments:** {n_runs} / {n_exp}"
)
if routing and routing.content_json.get("tuning_status") == "untuned_placeholder":
    st.caption(
        "⚠️ The active routing thresholds are **untuned placeholders**; they have not been optimized."
    )

creds = ctx.settings.credential_status()
s1, s2, s3 = st.columns(3)
s1.markdown(f"**Mode:** `{ctx.settings.app_mode.value}`")
s2.markdown(
    "**Jev:** "
    + (
        "available"
        if ctx.settings.jev_available
        else (
            "key configured, live calls disabled by mode"
            if creds["TYPESAFE_API_KEY"]
            else "no credential"
        )
    )
)
s3.markdown(
    "**OpenAI:** "
    + (
        "available"
        if ctx.settings.openai_available
        else (
            "key configured, live calls disabled by mode"
            if creds["OPENAI_API_KEY"]
            else "no credential"
        )
    )
)

st.divider()
st.subheader("Most recent completed run")
jev_runs = [r for r in runs if r.provider == "typesafe_jev"]
if not runs:
    st.info(
        "No completed evaluation runs yet. Initialize demo fixtures with "
        "`uv run python scripts/bootstrap.py`, or prepare the Civil Comments benchmark with "
        "`uv run python scripts/prepare_dataset.py` and start a run on the Batch Evaluation page."
    )
    st.page_link("pages/batch_evaluation.py", label="Go to Batch Evaluation", icon="📦")
    st.stop()

run = (jev_runs or runs)[0]
st.markdown(f"**{run_label(run)}** · completed {run.completed_at:%Y-%m-%d %H:%M} UTC")
origin_notice(run)
with ctx.session() as s:
    m = stored_metrics(s, run.id)


def get(metric, category, cohort="all"):
    if m.empty:
        return None, None
    row = m[(m.metric == metric) & (m.category == category) & (m.cohort == cohort)]
    if row.empty:
        return None, None
    r = row.iloc[0]
    return (None if r.value != r.value else r.value), r.ci


cols = st.columns(4)
evaluated, _ = get("evaluated_cases", "coverage")
cols[0].metric("Evaluated cases", int(evaluated) if evaluated is not None else "unavailable")
for col, (label, metric, cat) in zip(
    cols[1:],
    [
        ("Threat-proxy precision", "precision", "threat_proxy"),
        ("Threat-proxy recall", "recall", "threat_proxy"),
        ("Insult-proxy precision", "precision", "insult_proxy"),
    ],
    strict=True,
):
    v, ci = get(metric, cat)
    col.metric(label, fmt_pct(v), help=f"95% bootstrap CI {fmt_ci(ci)}" if ci else None)
cols = st.columns(4)
for col, (label, metric, cat) in zip(
    cols,
    [
        ("Insult-proxy recall", "recall", "insult_proxy"),
        ("Predicted review rate", "total_review_rate", "routing"),
        ("Under-prioritization", "under_prioritization_rate", "routing"),
        ("API failure rate", "api_failure_rate", "coverage"),
    ],
    strict=True,
):
    v, ci = get(metric, cat)
    col.metric(label, fmt_pct(v), help=f"95% bootstrap CI {fmt_ci(ci)}" if ci else None)
proxy_disclaimer(run)
if run.cohort_name != "holdout":
    st.caption("Review rates on enriched development data are not expected platform review volume.")

st.divider()
st.caption(
    "A No Flag result does not guarantee content is safe. Priority Review is a recommendation for human "
    "inspection, not proof of an actionable threat. This is an independent project, not affiliated with "
    "X, TypeSafe AI, or OpenAI."
)

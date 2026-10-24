from __future__ import annotations

import streamlit as st
from sqlalchemy import select

from app.components.common import (
    ORIGIN_LABELS,
    disposition_badge,
    get_ctx,
    mode_banner,
    origin_notice,
    run_label,
)
from harm_detection.persistence.models import Dataset, EvaluationRun
from harm_detection.routing.engine import RULE_DESCRIPTIONS
from harm_detection.services.evaluation_service import build_eval_frame

ctx = get_ctx()
st.title("Case Explorer")
mode_banner(ctx)

with ctx.session() as s:
    runs = s.scalars(select(EvaluationRun).order_by(EvaluationRun.id.desc())).all()
    datasets = {d.id: d for d in s.scalars(select(Dataset))}
if not runs:
    st.info("No runs yet.")
    st.stop()

c1, c2, c3 = st.columns(3)
ds_choice = c1.selectbox("Dataset", ["all", *sorted({datasets[r.dataset_id].name for r in runs})])
prov_choice = c2.selectbox("Provider", ["all", "typesafe_jev", "openai_moderation"])
cand = [
    r
    for r in runs
    if (ds_choice == "all" or datasets[r.dataset_id].name == ds_choice)
    and (prov_choice == "all" or r.provider == prov_choice)
]
if not cand:
    st.info("No runs match.")
    st.stop()
run = c3.selectbox("Run", cand, format_func=run_label)
origin_notice(run)

with ctx.session() as s:
    df = build_eval_frame(s, run.id)
if df.empty:
    st.info("Run has no cases.")
    st.stop()

f1, f2, f3, f4 = st.columns(4)
proxy = f1.selectbox("Proxy label", ["any", "threat proxy +", "insult proxy +", "both negative"])
pred_cat = f2.selectbox(
    "Predicted category", ["any", "threat predicted", "harassment predicted", "neither"]
)
disp = f3.selectbox(
    "Routing disposition", ["any", "priority_review", "standard_review", "no_flag", "not routed"]
)
status = f4.selectbox("Inference status", ["any", "success", "failure"])
g1, g2, g3, g4, g5 = st.columns(5)
only_fp = g1.checkbox("False positive")
only_fn = g2.checkbox("False negative")
only_under = g3.checkbox("Under-prioritized")
ctx_filter = g4.selectbox("Context", ["any", "warning", "sufficient"])
text_q = g5.text_input("Text contains")

v = df.copy()
if proxy == "threat proxy +":
    v = v[v.threat_proxy == True]  # noqa: E712
elif proxy == "insult proxy +":
    v = v[v.insult_proxy == True]  # noqa: E712
elif proxy == "both negative":
    v = v[(v.threat_proxy == False) & (v.insult_proxy == False)]  # noqa: E712
if pred_cat == "threat predicted":
    v = v[v.threat_pred == True]  # noqa: E712
elif pred_cat == "harassment predicted":
    v = v[v.harassment_pred == True]  # noqa: E712
elif pred_cat == "neither":
    v = v[(v.threat_pred == False) & (v.harassment_pred == False)]  # noqa: E712
if disp == "not routed":
    v = v[v.disposition.isna()]
elif disp != "any":
    v = v[v.disposition == disp]
if status != "any":
    v = v[v.pred_status == status]
if only_fp:
    v = v[v.fp_threat | v.fp_insult]
if only_fn:
    v = v[v.fn_threat | v.fn_insult]
if only_under:
    v = v[v.under_prioritized]
if ctx_filter == "warning":
    v = v[v.context_warning == True]  # noqa: E712
elif ctx_filter == "sufficient":
    v = v[v.context_warning == False]  # noqa: E712
if text_q:
    v = v[v.text.str.contains(text_q, case=False, na=False)]

st.caption(f"{len(v)} of {len(df)} cases match")
page_size = 25
pages = max(1, (len(v) + page_size - 1) // page_size)
page = st.number_input("Page", 1, pages, 1)
view = v.iloc[(page - 1) * page_size : page * page_size]
table = view[
    [
        "case_id",
        "stratum",
        "threat_proxy",
        "insult_proxy",
        "ref_route",
        "pred_status",
        "disposition",
        "error_class",
    ]
].copy()
table["text"] = view.text.str.slice(0, 90)
st.dataframe(table, use_container_width=True, hide_index=True)

if view.empty:
    st.stop()
cid = st.selectbox("Case detail", view.case_id.tolist())
row = df[df.case_id == cid].iloc[0]
st.divider()
st.markdown(f"### {cid}  {disposition_badge(row.disposition, row.routing_status)}")
st.markdown("**Content**")
st.code(row.text, language=None, wrap_lines=True)
if row.parent_text:
    st.markdown(f"**Parent comment:** {row.parent_text}")
if row.target_context:
    st.markdown(f"**Target context:** {row.target_context}")
a, b = st.columns(2)
with a:
    st.markdown("**Original source annotations** (published; preserved unmodified)")
    st.json(row.reference_labels)
    st.markdown(
        f"Proxy labels: threat **{row.threat_proxy}**, insult **{row.insult_proxy}** → "
        f"proxy route `{row.ref_route}`"
    )
    errs = []
    if row.fp_threat:
        errs.append("threat false positive")  # noqa: E701
    if row.fn_threat:
        errs.append("threat false negative")  # noqa: E701
    if row.fp_insult:
        errs.append("insult false positive")  # noqa: E701
    if row.fn_insult:
        errs.append("insult false negative")  # noqa: E701
    if row.under_prioritized:
        errs.append("under-prioritized")  # noqa: E701
    if row.over_prioritized:
        errs.append("over-prioritized")  # noqa: E701
    st.markdown("**Error classification:** " + (", ".join(errs) if errs else row.error_class))
with b:
    st.markdown(f"**Model signals** ({ORIGIN_LABELS.get(row.data_origin, row.data_origin)})")
    if row.signals:
        st.json(row.signals)
    else:
        st.error(f"No signals — `{row.error_code}`: {row.error_message}")
    st.markdown("**Routing rules**")
    for rid in row.triggered_rules or []:
        st.markdown(f"- deciding `{rid}`: {RULE_DESCRIPTIONS.get(rid, '')}")
    for reason in row.reasons or []:
        st.markdown(f"  - {reason}")
    if row.context_warning:
        st.warning("Context sufficiency warning", icon="❔")
st.markdown("**Provenance**")
ds = datasets[run.dataset_id]
st.caption(
    f"Dataset {ds.name} {ds.version} · revision {ds.revision} · license {ds.license} · split {row.split} · "
    f"stratum {row.stratum} · run #{run.id} · model {row.resolved_model or run.requested_model} · "
    f"config snapshot {run.metadata_json.get('config_snapshot')}"
)
st.caption("Jev returns probabilities only; it does not supply rationales or evidence spans.")
if run.cohort_name == "holdout":
    st.info("AI analysis is blocked for holdout cases.")
elif st.button("Send to AI Improvement Lab"):
    st.session_state["ai_run_id"] = run.id
    st.session_state["ai_case_ids"] = [cid]
    st.switch_page("pages/ai_improvement.py")

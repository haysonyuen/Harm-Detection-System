from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import select

from app.components.common import fmt_pct, get_ctx, mode_banner, run_label
from harm_detection.persistence.models import Configuration, Dataset, EvaluationRun
from harm_detection.services.evaluation_service import comparability, metric_lookup, stored_metrics
from harm_detection.services.run_service import delete_run

ctx = get_ctx()
st.title("Experiment History")
mode_banner(ctx)

with ctx.session() as s:
    runs = s.scalars(
        select(EvaluationRun)
        .where(EvaluationRun.cohort_name != "playground")
        .order_by(EvaluationRun.id.desc())
    ).all()
    datasets = {d.id: d for d in s.scalars(select(Dataset))}
    cfgs = {c.id: c for c in s.scalars(select(Configuration))}
    metrics = {r.id: stored_metrics(s, r.id) for r in runs}
if not runs:
    st.info("No runs yet.")
    st.stop()

KEYS = [
    ("recall", "threat_proxy"),
    ("precision", "threat_proxy"),
    ("recall", "insult_proxy"),
    ("precision", "insult_proxy"),
    ("priority_recall", "routing"),
    ("under_prioritization_rate", "routing"),
    ("total_review_rate", "routing"),
]
rows = []
for r in runs:
    m = metrics[r.id]
    rows.append(
        {
            "id": r.id,
            "name": r.name,
            "created": r.created_at.strftime("%Y-%m-%d %H:%M"),
            "dataset": f"{datasets[r.dataset_id].name} {datasets[r.dataset_id].version}",
            "cohort": r.cohort_name,
            "provider": r.provider,
            "model": r.resolved_model or r.requested_model,
            "questions": cfgs[r.question_config_id].version if r.question_config_id else "",
            "routing": cfgs[r.routing_config_id].version if r.routing_config_id else "",
            "origin": r.metadata_json.get("data_origin"),
            "status": r.status,
            **{f"{c}.{k}": metric_lookup(m, k, c) for k, c in KEYS},
        }
    )
table = pd.DataFrame(rows)
st.dataframe(table, use_container_width=True, hide_index=True)

st.subheader("Compare two runs")
c1, c2 = st.columns(2)
a = c1.selectbox("Run A", runs, format_func=run_label)
b = c2.selectbox("Run B", runs, index=min(1, len(runs) - 1), format_func=run_label)
if a.id == b.id:
    st.info("Select two different runs.")
else:
    with ctx.session() as s:
        comp = comparability(s, a.id, b.id)
    ma, mb = metrics[a.id], metrics[b.id]
    cmp_rows = []
    for k, c in KEYS:
        va, vb = metric_lookup(ma, k, c), metric_lookup(mb, k, c)
        row = {"metric": f"{c}.{k}", "run A": fmt_pct(va), "run B": fmt_pct(vb)}
        if comp["comparable"]:
            row["B − A"] = (
                "unavailable" if va is None or vb is None else f"{(vb - va) * 100:+.1f} pp"
            )
        cmp_rows.append(row)
    if comp["comparable"]:
        st.success(
            "Runs share the same dataset, cohort, case set, label mapping, input mode and data origin. "
            "Differences reflect configuration/model changes (sampling uncertainty still applies)."
        )
    else:
        st.warning(
            "These runs are not comparable (" + "; ".join(comp["issues"]) + "). Metric deltas are "
            "suppressed; do not interpret differences as improvements."
        )
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
    with st.expander("Configuration snapshots"):
        st.json(
            {
                "A": a.metadata_json.get("config_snapshot"),
                "B": b.metadata_json.get("config_snapshot"),
            }
        )

st.subheader("Data retention")
if ctx.settings.writes_allowed:
    victim = st.selectbox(
        "Delete a run and its predictions", runs, format_func=run_label, key="del"
    )
    if st.checkbox(
        f"I understand run #{victim.id} will be permanently deleted from the local database."
    ):
        if st.button("Delete run"):
            try:
                with ctx.session() as s:
                    delete_run(s, victim.id)
                    s.commit()
                st.success("Deleted.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Cannot delete: {exc}")
else:
    st.caption("🔒 Disabled in public demo mode.")

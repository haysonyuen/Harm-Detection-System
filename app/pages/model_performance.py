from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from app.components.common import (
    fmt_ci,
    fmt_pct,
    get_ctx,
    mode_banner,
    origin_notice,
    proxy_disclaimer,
    run_label,
)
from harm_detection.evaluation.metrics import binary_counts
from harm_detection.persistence.models import Configuration, EvaluationRun
from harm_detection.services.evaluation_service import (
    build_eval_frame,
    comparability,
    routing_confusion_from_frame,
    stored_metrics,
)

ctx = get_ctx()
st.title("Model Performance")
mode_banner(ctx)

with ctx.session() as s:
    runs = s.scalars(
        select(EvaluationRun)
        .where(
            EvaluationRun.status.in_(("completed", "partial")),
            EvaluationRun.cohort_name != "playground",
        )
        .order_by(EvaluationRun.id.desc())
    ).all()
if not runs:
    st.info("No completed runs. Start one on the Batch Evaluation page.")
    st.stop()

jev_runs = [r for r in runs if r.provider == "typesafe_jev"]
mod_runs = [r for r in runs if r.provider == "openai_moderation"]
c1, c2 = st.columns(2)
primary = c1.selectbox("Jev run", jev_runs or runs, format_func=run_label)
compare = c2.selectbox(
    "Comparison run (OpenAI Moderation, optional)",
    [None, *mod_runs],
    format_func=lambda r: "— none —" if r is None else run_label(r),
)
selected = [primary] + ([compare] if compare else [])
for r in selected:
    origin_notice(r)
proxy_disclaimer(primary)

if compare:
    with ctx.session() as s:
        comp = comparability(s, primary.id, compare.id)
    if comp["comparable"]:
        st.success(
            "Both runs evaluated the same cohort, case set, label mapping and text-only input."
        )
    else:
        st.error(
            "Runs are **not directly comparable**: "
            + "; ".join(comp["issues"])
            + ". Differences below must not be read as model improvements."
        )
    st.caption(
        "OpenAI Moderation uses its own policy definitions; its 'harassment' and 'harassment/threatening' "
        "native flags are compared to the same proxy labels. Raw provider scores are not calibrated to "
        "each other and are not compared directly."
    )

frames, metrics = {}, {}
with ctx.session() as s:
    for r in selected:
        frames[r.id] = build_eval_frame(s, r.id)
        metrics[r.id] = stored_metrics(s, r.id)
    lm = (
        s.get(Configuration, primary.label_mapping_config_id)
        if primary.label_mapping_config_id
        else None
    )

# ---- metrics by provider ------------------------------------------------------------
st.subheader("Metrics by provider")
cohorts = sorted({c for m in metrics.values() if not m.empty for c in m["cohort"].unique()})
cohort = (
    st.radio(
        "Cohort",
        cohorts,
        horizontal=True,
        help="representative = random sample; enriched = threat/insult/borderline strata. "
        "Enriched review rates are not expected platform volume.",
    )
    if cohorts
    else "all"
)
rows = []
for r in selected:
    m = metrics[r.id]
    if m.empty:
        continue
    m = m[m.cohort == cohort]
    for _, x in m.iterrows():
        rows.append(
            {
                "run": f"#{r.id} {r.provider}",
                "category": x.category,
                "metric": x.metric,
                "value": x.value,
                "n": f"{int(x.numerator)}/{int(x.denominator)}"
                if pd.notna(x.denominator) and x.denominator
                else "",
                "95% CI": fmt_ci(x.ci),
            }
        )
mdf = pd.DataFrame(rows)
if mdf.empty:
    st.info("No metrics stored for this cohort.")
    st.stop()
key = mdf[
    mdf.metric.isin(
        [
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "false_negative_rate",
            "positives",
            "negatives",
            "tp",
            "fp",
            "fn",
            "tn",
        ]
    )
    & mdf.category.isin(["threat_proxy", "insult_proxy"])
]
st.dataframe(
    key.pivot_table(index=["category", "metric"], columns="run", values="value", aggfunc="first"),
    use_container_width=True,
)
cov = mdf[mdf.category == "coverage"]
st.dataframe(
    cov.pivot_table(index="metric", columns="run", values="value", aggfunc="first"),
    use_container_width=True,
)

# ---- precision / recall comparison with CIs ---------------------------------------
st.subheader("Precision–recall comparison")
pr_rows = []
for r in selected:
    m = metrics[r.id]
    if m.empty:
        continue
    m = m[
        (m.cohort == cohort)
        & m.metric.isin(["precision", "recall"])
        & m.category.isin(["threat_proxy", "insult_proxy"])
    ]
    for _, x in m.iterrows():
        ci = x.ci or [None, None]
        pr_rows.append(
            {
                "provider": f"#{r.id} {r.provider}",
                "series": f"{x.category} {x.metric}",
                "value": x.value,
                "lo": None if ci[0] is None or x.value is None else x.value - ci[0],
                "hi": None if ci[1] is None or x.value is None else ci[1] - x.value,
            }
        )
if pr_rows:
    prdf = pd.DataFrame(pr_rows)
    fig = px.bar(
        prdf,
        x="series",
        y="value",
        color="provider",
        barmode="group",
        error_y="hi",
        error_y_minus="lo",
        range_y=[0, 1],
        labels={"value": "value (95% bootstrap CI)", "series": ""},
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- confusion matrices --------------------------------------------------------------
st.subheader("Confusion matrices")
for r in selected:
    df = frames[r.id]
    if df.empty:
        continue
    if cohort == "representative":
        df = df[df.stratum == "representative"]
    elif cohort == "enriched":
        df = df[df.stratum.isin(["threat_enriched", "insult_enriched", "borderline"])]
    ev = df[df.evaluable]
    st.markdown(f"**#{r.id} {r.provider}** — {len(ev)} evaluable of {len(df)} planned cases")
    cols = st.columns(3)
    for col, (cat, t, p) in zip(
        cols,
        [
            ("Threat proxy", "threat_proxy", "threat_pred"),
            ("Insult proxy", "insult_proxy", "harassment_pred"),
        ],
        strict=False,
    ):
        c = binary_counts(ev[t].astype(bool), ev[p].astype(bool))
        fig = go.Figure(
            go.Heatmap(
                z=[[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
                x=["pred −", "pred +"],
                y=["proxy −", "proxy +"],
                text=[[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig.update_layout(title=cat, height=260, margin=dict(l=10, r=10, t=40, b=10))
        col.plotly_chart(fig, use_container_width=True)
    cm = routing_confusion_from_frame(df)
    labels = ["no_flag", "standard", "priority"]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=[f"pred {x}" for x in labels],
            y=[f"ref {x}" for x in labels],
            text=cm,
            texttemplate="%{text}",
            colorscale="Oranges",
            showscale=False,
        )
    )
    fig.update_layout(
        title="Routing (proxy reference × predicted)",
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    cols[2].plotly_chart(fig, use_container_width=True)

# ---- routing metrics -----------------------------------------------------------------
st.subheader("Routing proxy metrics")
rm = mdf[mdf.category == "routing"]
st.dataframe(
    rm[["run", "metric", "value", "n", "95% CI"]].assign(value=rm.value.map(fmt_pct)),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "Priority recall = P(pred priority | threat proxy). Under-prioritization = P(pred below proxy route | "
    "proxy route is standard or priority). Priority false-alarm rate = P(pred priority | not threat proxy)."
)

# ---- review-rate distribution by cohort ---------------------------------------------
st.subheader("Predicted route distribution by cohort")
dist = []
for r in selected:
    df = frames[r.id]
    if df.empty:
        continue
    ok = df[df.routing_status == "success"]
    for name, sub in [
        ("all", ok),
        ("representative", ok[ok.stratum == "representative"]),
        ("enriched", ok[ok.stratum.isin(["threat_enriched", "insult_enriched", "borderline"])]),
    ]:
        if len(sub):
            for d, n in sub.disposition.value_counts().items():
                dist.append(
                    {
                        "run": f"#{r.id} {r.provider}",
                        "cohort": name,
                        "disposition": d,
                        "share": n / len(sub),
                        "n": n,
                    }
                )
if dist:
    fig = px.bar(
        pd.DataFrame(dist),
        x="cohort",
        y="share",
        color="disposition",
        facet_col="run",
        text="n",
        color_discrete_map={
            "no_flag": "#067647",
            "standard_review": "#b54708",
            "priority_review": "#b42318",
        },
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Review volume on enriched samples is inflated by design. Even representative samples reflect "
        "the source dataset, not live platform traffic."
    )

with st.expander("Reference label mapping details"):
    if lm:
        st.json(lm.content_json)
    st.markdown(
        "Reference routing: threat proxy → Priority Review; else insult proxy → Standard Review; else "
        "No Flag. This proxy is for benchmarking only and is not adjudicated review priority."
    )

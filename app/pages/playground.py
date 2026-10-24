from __future__ import annotations

import streamlit as st

from app.components.common import ORIGIN_LABELS, disposition_badge, get_ctx, mode_banner
from harm_detection.providers.fixture import load_jsonl
from harm_detection.routing.engine import RULE_DESCRIPTIONS
from harm_detection.schemas.config import PolicyConfig, RoutingConfig
from harm_detection.services import config_service as cs
from harm_detection.services.detection_service import (
    PlaygroundResult,
    analyze_live,
    playback_fixture,
)
from harm_detection.settings import FIXTURES_DIR

ctx = get_ctx()
st.title("Detection Playground")
mode_banner(ctx)

with ctx.session() as s:
    q_cfg = cs.active_config(s, "jev_questions")
    r_cfg = cs.active_config(s, "routing")
    p_cfg = cs.active_config(s, "policy")
if not (q_cfg and r_cfg and p_cfg):
    st.error("No active configuration found. Run `uv run python scripts/bootstrap.py`.")
    st.stop()
routing_cfg = RoutingConfig.model_validate(r_cfg.content_json)
policy = PolicyConfig.model_validate(p_cfg.content_json)
st.caption(
    f"Active question config **{q_cfg.version}** · routing **{r_cfg.version}** "
    f"({routing_cfg.tuning_status.replace('_', ' ')}) · policy **{policy.version}**"
)


def render(result: PlaygroundResult) -> None:
    det, dec = result.detection, result.routing
    st.subheader("Result")
    origin = ORIGIN_LABELS.get(det.data_origin, det.data_origin)
    if det.data_origin != "live":
        st.warning(f"Data origin: **{origin}**. These are not live predictions.", icon="🧪")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Inference status:** `{det.status.value}`")
    c2.markdown(f"**Model:** `{det.resolved_model or det.requested_model or 'n/a'}`")
    c3.markdown(f"**Latency:** {det.latency_ms if det.latency_ms is not None else 'n/a'} ms")
    if det.error:
        st.error(
            f"Detection failed — `{det.error.get('code')}`: {det.error.get('message')}. "
            "No disposition was produced (failures are never recorded as No Flag)."
        )
    st.markdown(
        f"### {disposition_badge(dec.disposition.value if dec.disposition else None, dec.status.value)}"
    )
    st.markdown(
        "**Triggered rule:** "
        + ", ".join(f"`{r}`" for r in dec.triggered_rule_ids)
        + (
            " · matched: " + ", ".join(f"`{r}`" for r in dec.matched_rule_ids)
            if dec.matched_rule_ids
            else ""
        )
    )
    for reason in dec.reasons:
        st.markdown(f"- {reason}")
    if dec.context_warning:
        st.warning(
            "Context sufficiency below threshold: interpretation may depend on missing context.",
            icon="❔",
        )
    if det.signals:
        st.markdown("**Normalized Jev signals** (probabilities, not rationales)")
        cols = st.columns(4)
        for col, (name, v) in zip(cols, det.signals.items(), strict=False):
            col.metric(name.replace("_", " "), f"{v:.3f}")
    with st.expander("Applicable policy criteria"):
        for rid in dec.triggered_rule_ids:
            st.markdown(f"- `{rid}`: {RULE_DESCRIPTIONS.get(rid, '')}")
        st.markdown(f"_{policy.disclaimer}_")
        for key in ("H", "T", "U"):
            cat = policy.categories.get(key, {})
            st.markdown(f"**{key} — {cat.get('name')}**: {cat.get('description')}")
        st.markdown("**Limitations:** " + "; ".join(policy.limitations))
    if result.saved_case_id:
        st.caption(
            f"Saved case ID `{result.saved_case_id}` in playground run #{result.saved_run_id}."
        )


tab_live, tab_fixture = st.tabs(["Analyze comment", "Recorded fixture playback"])

with tab_live:
    with st.form("analyze"):
        text = st.text_area(
            "Comment text (required)", height=120, max_chars=ctx.settings.max_text_length
        )
        parent = st.text_area("Parent comment (optional)", height=70)
        target = st.text_input("Target context (optional)")
        submitted = st.form_submit_button("Analyze Comment", type="primary")
    if submitted:
        if not text.strip():
            st.error("Comment text is required.")
        elif not ctx.settings.writes_allowed:
            st.error("Live inference is disabled in public demo mode.")
        elif not ctx.settings.live_calls_allowed:
            st.error(
                "Live inference is unavailable in offline mode. No prediction was generated for this "
                "text. Use the fixture playback tab, or set APP_MODE=local_live with a TypeSafe key."
            )
        else:
            with st.spinner("Calling Jev…"):
                try:
                    res = analyze_live(ctx, text, parent or None, target or None)
                except ValueError as exc:
                    st.error(f"Invalid input: {exc}")
                else:
                    render(res)

with tab_fixture:
    cases = load_jsonl(FIXTURES_DIR / "demo_cases.jsonl")
    if not cases:
        st.info("No fixture cases found.")
    else:
        st.caption(
            "Synthetic fixture cases with hand-authored signals for demonstrating deterministic routing. "
            "These are not Jev outputs."
        )
        labels = {f"{c['case_id']} — {c['metadata']['scenario']}": c for c in cases}
        choice = st.selectbox("Fixture case", list(labels))
        case = labels[choice]
        st.text_area("Comment", case["text"], disabled=True)
        if case.get("parent_text"):
            st.text_area("Parent comment", case["parent_text"], disabled=True)
        if case.get("target_context"):
            st.text_input("Target context", case["target_context"], disabled=True)
        if st.button("Play back recorded result"):
            render(playback_fixture(ctx, case, routing_cfg))

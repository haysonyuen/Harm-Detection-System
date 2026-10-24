from __future__ import annotations

import pandas as pd
import streamlit as st
import yaml
from sqlalchemy import select

from app.components.common import get_ctx, mode_banner
from harm_detection.persistence.models import Configuration, Dataset
from harm_detection.services import config_service as cs
from harm_detection.services.analysis_service import spent_usd
from harm_detection.services.dataset_service import delete_dataset
from harm_detection.services.status_service import system_status

ctx = get_ctx()
st.title("Configuration & System Status")
mode_banner(ctx)

status = system_status(ctx)
st.subheader("System status")
c1, c2, c3 = st.columns(3)
c1.markdown(f"**Deployment mode:** `{status['mode']}`")
creds = status["credentials_configured"]
c2.markdown(
    "**Credentials** (values never shown)  \n"
    + "  \n".join(f"{k}: {'configured' if v else 'not configured'}" for k, v in creds.items())
)
c3.markdown(
    f"**Jev live:** {'available' if status['jev_available'] else 'unavailable'}  \n"
    f"**OpenAI live:** {'available' if status['openai_available'] else 'unavailable'}"
)
d1, d2, d3 = st.columns(3)
mig = status["migrations"]
d1.markdown(f"**Database migrations:** {'up to date' if mig.get('up_to_date') else mig}")
man = status["manifest"]
d2.markdown(
    f"**Civil Comments manifest:** {'valid' if man['valid'] else 'not valid'} — {man['detail']}"
)
d3.markdown(
    "**Offline fixtures:** "
    + ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in status["fixtures"].items())
)
with ctx.session() as s:
    spent = spent_usd(s)
st.markdown(
    f"**Analysis spending usage:** \\${spent:.4f} of \\${ctx.settings.max_analyst_budget_usd:.2f} local budget (estimate)"
)
st.markdown("**Recent integration errors / events**")
ev = status["recent_integration_events"]
st.dataframe(
    pd.DataFrame(ev) if ev else pd.DataFrame({"info": ["none recorded"]}),
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("Configurations")
with ctx.session() as s:
    cfgs = cs.list_configs(s)
    refs = {c.id: cs.is_referenced_by_completed_run(s, c) for c in cfgs}
st.dataframe(
    pd.DataFrame(
        [
            {
                "id": c.id,
                "type": c.config_type,
                "version": c.version,
                "active": c.is_active,
                "frozen": c.is_frozen or refs[c.id],
                "parent": c.parent_config_id,
                "created": c.created_at,
                "checksum": c.checksum[:12],
                "note": c.change_note,
            }
            for c in cfgs
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

sel = st.selectbox(
    "Inspect configuration",
    cfgs,
    format_func=lambda c: (
        f"#{c.id} {c.config_type}:{c.version}" + (" (active)" if c.is_active else "")
    ),
)
st.code(yaml.safe_dump(sel.content_json, sort_keys=False), language="yaml")

if ctx.settings.writes_allowed:
    st.markdown("#### Create a new configuration version")
    st.caption(
        "Saved versions are immutable once referenced by a completed run. New versions start inactive."
    )
    ctype = st.selectbox("Type", ["routing", "jev_questions", "policy", "label_mapping", "pricing"])
    with ctx.session() as s:
        base = cs.active_config(s, ctype)
    draft = dict(base.content_json) if base else {}
    if draft:
        draft["version"] = draft["version"] + "_draft"
    text = st.text_area("Content (YAML)", yaml.safe_dump(draft, sort_keys=False), height=260)
    note = st.text_input("Change note")
    if st.button("Save new version"):
        try:
            content = yaml.safe_load(text)
            with ctx.session() as s:
                cfg = cs.save_config(
                    s,
                    ctype,
                    content,
                    parent_config_id=base.id if base else None,
                    change_note=note or None,
                )
                s.commit()
            st.success(f"Saved {ctype}:{cfg.version} (inactive).")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Invalid configuration: {exc}")

    st.markdown("#### Promote (activate) a configuration")
    inactive = [c for c in cfgs if not c.is_active]
    if inactive:
        target = st.selectbox(
            "Configuration to activate",
            inactive,
            format_func=lambda c: f"#{c.id} {c.config_type}:{c.version}",
        )
        if (
            target.config_type == "routing"
            and target.content_json.get("tuning_status") == "untuned_placeholder"
        ):
            st.warning("This routing config is an untuned placeholder.")
        rationale = st.text_input("Promotion rationale (required)")
        confirm = st.checkbox(
            "I confirm this replaces the currently active configuration of this type."
        )
        if st.button("Activate", disabled=not (confirm and rationale.strip())):
            with ctx.session() as s:
                cs.activate_config(s, target.id, rationale=rationale)
                s.commit()
            st.success("Activated (audited).")
            st.rerun()

    st.markdown("#### Data retention")
    with ctx.session() as s:
        dss = s.scalars(select(Dataset)).all()
    if dss:
        victim = st.selectbox(
            "Delete local dataset", dss, format_func=lambda d: f"#{d.id} {d.name} {d.version}"
        )
        if st.checkbox(
            "I understand this permanently deletes the dataset's local cases."
        ) and st.button("Delete dataset"):
            try:
                with ctx.session() as s:
                    delete_dataset(s, victim.id)
                    s.commit()
                st.success("Deleted.")
            except ValueError as exc:
                st.error(str(exc))
else:
    st.caption("🔒 Configuration edits are disabled in public demo mode.")
    _ = Configuration

"""Shared Streamlit helpers. UI code only: no provider JSON parsing lives here."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from harm_detection.services.context import AppContext, create_context
from harm_detection.settings import AppMode

STATUS_COLORS = {
    "priority_review": "#b42318",
    "standard_review": "#b54708",
    "no_flag": "#067647",
    "failure": "#475467",
}
DISPOSITION_LABELS = {
    "priority_review": "Priority Review",
    "standard_review": "Standard Review",
    "no_flag": "No Flag",
    None: "Not routed",
}
ORIGIN_LABELS = {
    "live": "Live provider output",
    "recorded_live": "Recorded live provider output",
    "synthetic_fixture": "Synthetic fixture (not model output)",
}


@st.cache_resource(show_spinner=False)
def _ctx() -> AppContext:
    return create_context()


def get_ctx() -> AppContext:
    return _ctx()


def mode_banner(ctx: AppContext) -> None:
    mode = ctx.settings.app_mode
    if mode == AppMode.PUBLIC_DEMO:
        st.info(
            "**Public demo (read-only).** Curated recorded artifacts only. Live inference, AI analysis, "
            "and persistent changes are disabled. Threshold exploration is session-local.",
            icon="🔒",
        )
    elif mode == AppMode.LOCAL_OFFLINE:
        st.warning(
            "**Offline mode.** No provider requests are made. Results shown come from stored runs or "
            "clearly labeled synthetic fixtures. Set `APP_MODE=local_live` to enable live inference.",
            icon="📴",
        )
    else:
        creds = ctx.settings.credential_status()
        missing = [k for k, v in creds.items() if not v]
        msg = "**Live mode.** Provider calls are enabled and may incur cost."
        if missing:
            msg += f" Missing credentials: {', '.join(missing)} (those features are unavailable)."
        st.success(msg, icon="🟢")


def disposition_badge(disposition: str | None, status: str = "success") -> str:
    if status != "success" or disposition is None:
        return f":gray-background[**{status.replace('_', ' ').upper()}**]"
    color = {"priority_review": "red", "standard_review": "orange", "no_flag": "green"}[disposition]
    return f":{color}-background[**{DISPOSITION_LABELS[disposition]}**]"


def fmt_pct(v: float | None, digits: int = 1) -> str:
    return "unavailable" if v is None or pd.isna(v) else f"{v * 100:.{digits}f}%"


def fmt_ci(ci: Any) -> str:
    if ci is None or (isinstance(ci, float) and pd.isna(ci)):
        return ""
    lo, hi = ci
    return f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"


def run_label(run) -> str:
    origin = (run.metadata_json or {}).get("data_origin", "?")
    tag = {"synthetic_fixture": " · FIXTURE", "live": "", "recorded_live": " · RECORDED"}.get(
        origin, ""
    )
    return f"#{run.id} {run.name} [{run.provider} · {run.cohort_name} · {run.status}{tag}]"


def origin_notice(run) -> None:
    meta = run.metadata_json or {}
    origin = meta.get("data_origin")
    if origin == "synthetic_fixture":
        st.warning(
            "This run replays **synthetic fixture values** authored for testing. They are not outputs of "
            "Jev or OpenAI Moderation, and its labels are synthetic author expectations, not human "
            "annotations. Metrics below demonstrate the pipeline only and are not model performance.",
            icon="🧪",
        )
    if meta.get("holdout_evaluation"):
        if meta.get("exploratory"):
            st.error(
                "Exploratory holdout evaluation: the holdout was previously consumed. "
                "These results are not an unbiased estimate.",
                icon="⚠️",
            )
        else:
            st.info(
                "Final holdout evaluation: the holdout is now consumed for this configuration.",
                icon="🔐",
            )
    if meta.get("derived_by") == "reroute":
        st.caption(
            f"Derived run: routing re-applied to stored predictions of run "
            f"#{meta.get('prediction_source_run_id')} (no new provider calls)."
        )


def proxy_disclaimer(run=None) -> None:
    if run is not None and (run.metadata_json or {}).get("data_origin") == "synthetic_fixture":
        st.caption(
            "Reference labels for this run are synthetic author expectations, not human annotations; "
            "these numbers only demonstrate the metric pipeline."
        )
        return
    st.caption(
        "Metrics measure agreement with published human annotations mapped to proxy labels "
        "(threat proxy, insult proxy). They are not confirmed X policy violations or platform "
        "enforcement accuracy."
    )


def require_write_ui(ctx: AppContext) -> bool:
    if not ctx.settings.writes_allowed:
        st.caption("🔒 Disabled in public demo mode.")
        return False
    return True

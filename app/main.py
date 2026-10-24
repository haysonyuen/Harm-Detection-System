"""Social Media Harm Detection Lab - Streamlit entry point.

Run:  uv run streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Social Media Harm Detection Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = ROOT / "app" / "pages"

nav = st.navigation(
    {
        "Lab": [
            st.Page(PAGES / "overview.py", title="Overview", icon="🏠", default=True),
            st.Page(PAGES / "playground.py", title="Detection Playground", icon="🔎"),
            st.Page(PAGES / "batch_evaluation.py", title="Batch Evaluation", icon="📦"),
        ],
        "Analysis": [
            st.Page(PAGES / "model_performance.py", title="Model Performance", icon="📈"),
            st.Page(PAGES / "case_explorer.py", title="Case Explorer", icon="🗂️"),
            st.Page(PAGES / "threshold_optimization.py", title="Threshold Optimization", icon="🎚️"),
            st.Page(PAGES / "experiment_history.py", title="Experiment History", icon="🕘"),
            st.Page(PAGES / "ai_improvement.py", title="AI Improvement Lab", icon="🧠"),
        ],
        "System": [
            st.Page(PAGES / "settings.py", title="Configuration & System Status", icon="⚙️"),
        ],
    }
)

with st.sidebar:
    st.markdown("### 🛡️ Harm Detection Lab")
    st.caption(
        "Independent T&S evaluation lab. Detection is not enforcement. Not affiliated with X, "
        "TypeSafe AI, or OpenAI."
    )

nav.run()

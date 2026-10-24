"""Streamlit UI tests (AppTest, headless). Offline + public-demo modes, no network."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from harm_detection.services.bootstrap_service import bootstrap
from harm_detection.services.context import create_context
from tests.conftest import make_settings

MAIN = str(Path(__file__).parents[2] / "app" / "main.py")
PAGES = [
    "overview",
    "playground",
    "batch_evaluation",
    "model_performance",
    "case_explorer",
    "threshold_optimization",
    "experiment_history",
    "ai_improvement",
    "settings",
]


@pytest.fixture
def offline_app(tmp_path, monkeypatch):
    db = tmp_path / "ui.sqlite"
    bootstrap(create_context(make_settings(tmp_path)))
    for k, v in {
        "APP_MODE": "local_offline",
        "DATABASE_URL": f"sqlite:///{db}".replace("ui.sqlite", "test.sqlite"),
        "TYPESAFE_API_KEY": "",
        "OPENAI_API_KEY": "",
    }.items():
        monkeypatch.setenv(k, v)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def run_page(page: str) -> AppTest:
    at = AppTest.from_file(MAIN, default_timeout=60)
    at.run()
    if page != "overview":
        at.switch_page(f"pages/{page}.py")
        at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_loads_offline(offline_app, page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert any("Offline mode" in w.value for w in at.warning)


def test_overview_labels_fixture_run(offline_app):
    at = run_page("overview")
    text = " ".join(m.value for m in at.markdown) + " ".join(w.value for w in at.warning)
    assert "synthetic fixture" in text
    assert "untuned placeholders" in " ".join(c.value for c in at.caption)


def test_playground_rejects_blank_and_offline_text(offline_app):
    at = run_page("playground")
    at.button[0].click().run()  # form submit with empty text
    assert any("required" in e.value for e in at.error)
    at.text_area[0].input("some arbitrary comment")
    at.button[0].click().run()
    assert any("unavailable in offline mode" in e.value for e in at.error)
    assert not any("Normalized Jev signals" in m.value for m in at.markdown)


def test_playground_fixture_playback(offline_app):
    at = run_page("playground")
    sb = at.selectbox[0]
    sb.select(next(o for o in sb.options if o.startswith("demo-002"))).run()
    next(b for b in at.button if b.label == "Play back recorded result").click().run()
    assert any("Synthetic fixture" in w.value for w in at.warning)
    assert any("Priority Review" in m.value for m in at.markdown)
    assert any("R-PRI-01" in m.value for m in at.markdown)


def test_threshold_changes_recompute(offline_app):
    at = run_page("threshold_optimization")
    before = next(c.value for c in at.caption if c.value.startswith("Routes:"))
    at.slider(key="wi_threat_priority").set_value(0.3).run()
    after = next(c.value for c in at.caption if c.value.startswith("Routes:"))
    assert before != after


def test_history_noncomparable_warning(offline_app):
    at = run_page("experiment_history")
    assert not at.exception
    # The two demo runs (Jev vs Moderation) share cohort/case set/input mode => comparable.
    assert any("share the same dataset" in s.value for s in at.success)


def test_ai_lab_unavailable_offline(offline_app):
    at = run_page("ai_improvement")
    assert any("AI analysis is unavailable (offline mode)" in w.value for w in at.warning)
    gen = [b for b in at.button if b.label == "Generate Recommendations"]
    assert all(b.disabled for b in gen)


def test_public_demo_mode_disables_live(tmp_path, monkeypatch):
    import importlib.util

    bootstrap(create_context(make_settings(tmp_path)))
    spec = importlib.util.spec_from_file_location(
        "exp", Path(MAIN).parents[1] / "scripts" / "export_public_demo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "pub.sqlite"
    mod.export(f"sqlite:///{tmp_path / 'test.sqlite'}", out)
    for k, v in {
        "APP_MODE": "public_demo",
        "PUBLIC_DEMO_DATABASE_URL": f"sqlite:///{out}",
        "TYPESAFE_API_KEY": "tk-ignored-123456",
        "OPENAI_API_KEY": "",
    }.items():
        monkeypatch.setenv(k, v)
    st.cache_resource.clear()
    try:
        at = run_page("playground")
        assert any("Public demo" in i.value for i in at.info)
        at.text_area[0].input("hello")
        at.button[0].click().run()
        assert any("disabled in public demo" in e.value for e in at.error)
        for page in ("batch_evaluation", "settings", "ai_improvement", "threshold_optimization"):
            at = run_page(page)
            assert not at.exception, page
        at = run_page("batch_evaluation")
        assert any("Disabled in public demo" in c.value for c in at.caption)
        assert not [b for b in at.button if b.label == "Create and start run"]
    finally:
        st.cache_resource.clear()

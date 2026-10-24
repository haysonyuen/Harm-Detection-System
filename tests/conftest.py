from __future__ import annotations

import json
from pathlib import Path

import pytest

from harm_detection.policy.loader import load_config_file
from harm_detection.services.bootstrap_service import bootstrap
from harm_detection.services.context import create_context
from harm_detection.settings import FIXTURES_DIR, load_settings

TEST_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch):
    """The default suite never uses real credentials or makes paid calls."""
    for k in ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "APP_MODE", "DATABASE_URL"):
        monkeypatch.delenv(k, raising=False)


def make_settings(tmp_path: Path, mode: str = "local_offline", **extra) -> object:
    env = {"APP_MODE": mode, "DATABASE_URL": f"sqlite:///{tmp_path / 'test.sqlite'}", **extra}
    return load_settings(env, use_dotenv=False)


@pytest.fixture
def ctx(tmp_path):
    return create_context(make_settings(tmp_path))


@pytest.fixture
def boot_ctx(tmp_path):
    c = create_context(make_settings(tmp_path))
    bootstrap(c)
    return c


@pytest.fixture
def questions():
    return load_config_file("jev_questions")[0]


@pytest.fixture
def routing_cfg():
    return load_config_file("routing")[0]


@pytest.fixture
def label_mapping():
    return load_config_file("label_mapping")[0]


@pytest.fixture
def demo_cases():
    with (FIXTURES_DIR / "demo_cases.jsonl").open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def jev_response(signals: dict, model: str = "jev-1.14.0") -> dict:
    return {
        "model": model,
        "answers": {k: {"type": "noul", "noul": v} for k, v in signals.items()},
        "usage": {"input_tokens": 512, "output_tokens": 4},
    }

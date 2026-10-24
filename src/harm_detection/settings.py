"""Application settings loaded from environment variables.

Secrets are only ever read from the environment (optionally via a local .env
file) and are never written to logs, the database, or exported artifacts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
FIXTURES_DIR = DATA_DIR / "fixtures"
MANIFESTS_DIR = DATA_DIR / "manifests"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
PUBLIC_DEMO_DIR = DATA_DIR / "public_demo"
DEFAULT_DB_PATH = DATA_DIR / "harm_detection.sqlite"
PUBLIC_DEMO_DB_PATH = PUBLIC_DEMO_DIR / "public_demo.sqlite"


class AppMode(StrEnum):
    LOCAL_LIVE = "local_live"
    LOCAL_OFFLINE = "local_offline"
    PUBLIC_DEMO = "public_demo"


@dataclass(frozen=True)
class Settings:
    app_mode: AppMode
    database_url: str
    typesafe_api_key: str | None = field(repr=False, default=None)
    openai_api_key: str | None = field(repr=False, default=None)
    jev_model: str = "jev-latest"
    jev_base_url: str = "https://api.typesafe.ai"
    openai_moderation_model: str = "omni-moderation-latest"
    openai_analyst_model: str = "gpt-5.4-mini"
    max_analyst_budget_usd: float = 5.00
    max_text_length: int = 5000

    # --- capability flags derived from mode + credentials -----------------
    @property
    def live_calls_allowed(self) -> bool:
        return self.app_mode == AppMode.LOCAL_LIVE

    @property
    def jev_available(self) -> bool:
        return self.live_calls_allowed and bool(self.typesafe_api_key)

    @property
    def openai_available(self) -> bool:
        return self.live_calls_allowed and bool(self.openai_api_key)

    @property
    def writes_allowed(self) -> bool:
        return self.app_mode != AppMode.PUBLIC_DEMO

    @property
    def is_public_demo(self) -> bool:
        return self.app_mode == AppMode.PUBLIC_DEMO

    def credential_status(self) -> dict[str, bool]:
        """Whether each credential is configured. Never exposes the value."""
        return {
            "TYPESAFE_API_KEY": bool(self.typesafe_api_key),
            "OPENAI_API_KEY": bool(self.openai_api_key),
        }


def _parse_mode(raw: str | None) -> AppMode:
    if not raw:
        return AppMode.LOCAL_OFFLINE
    try:
        return AppMode(raw.strip().lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in AppMode)
        raise ValueError(f"APP_MODE={raw!r} is invalid; expected one of: {valid}") from exc


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def load_settings(env: dict[str, str] | None = None, *, use_dotenv: bool = True) -> Settings:
    """Build settings from ``env`` (defaults to ``os.environ`` plus ``.env``)."""
    if env is None:
        if use_dotenv:
            load_dotenv(PROJECT_ROOT / ".env", override=False)
        env = dict(os.environ)

    mode = _parse_mode(env.get("APP_MODE"))
    if mode == AppMode.PUBLIC_DEMO:
        # Public demo never uses live credentials, whatever the environment holds.
        database_url = env.get("PUBLIC_DEMO_DATABASE_URL") or f"sqlite:///{PUBLIC_DEMO_DB_PATH}"
        typesafe_key = openai_key = None
    else:
        database_url = env.get("DATABASE_URL") or f"sqlite:///{DEFAULT_DB_PATH}"
        typesafe_key = _clean(env.get("TYPESAFE_API_KEY"))
        openai_key = _clean(env.get("OPENAI_API_KEY"))

    return Settings(
        app_mode=mode,
        database_url=database_url,
        typesafe_api_key=typesafe_key,
        openai_api_key=openai_key,
        jev_model=_clean(env.get("JEV_MODEL")) or "jev-latest",
        jev_base_url=_clean(env.get("JEV_BASE_URL")) or "https://api.typesafe.ai",
        openai_moderation_model=_clean(env.get("OPENAI_MODERATION_MODEL"))
        or "omni-moderation-latest",
        openai_analyst_model=_clean(env.get("OPENAI_ANALYST_MODEL")) or "gpt-5.4-mini",
        max_analyst_budget_usd=float(env.get("MAX_ANALYST_BUDGET_USD") or 5.0),
        max_text_length=int(env.get("MAX_TEXT_LENGTH") or 5000),
    )

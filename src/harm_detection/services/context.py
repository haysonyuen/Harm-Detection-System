"""Application context: settings + database session factory, and mode guards."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from harm_detection.persistence.db import make_engine, make_session_factory, run_migrations
from harm_detection.settings import Settings, load_settings


class ReadOnlyModeError(PermissionError):
    """Raised when a mutating action is attempted in public demo mode."""


class LiveCallsDisabledError(PermissionError):
    """Raised when a provider call is attempted outside local_live mode."""


@dataclass
class AppContext:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]

    def session(self) -> Session:
        return self.session_factory()

    def require_writable(self, action: str = "this action") -> None:
        if not self.settings.writes_allowed:
            raise ReadOnlyModeError(f"{action} is disabled in public demo mode (read-only).")

    def require_live(self, action: str = "provider inference") -> None:
        if not self.settings.live_calls_allowed:
            raise LiveCallsDisabledError(
                f"{action} requires APP_MODE=local_live (current mode: {self.settings.app_mode.value})."
            )


def create_context(settings: Settings | None = None, *, migrate: bool = True) -> AppContext:
    settings = settings or load_settings()
    if settings.is_public_demo:
        engine = make_engine(settings.database_url, read_only=True)
    else:
        if migrate:
            run_migrations(settings.database_url)
        engine = make_engine(settings.database_url)
    return AppContext(settings, engine, make_session_factory(engine))

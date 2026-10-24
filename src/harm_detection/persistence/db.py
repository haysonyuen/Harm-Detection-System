"""Engine/session management and Alembic migration helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.orm import Session, sessionmaker

from harm_detection.settings import PROJECT_ROOT


def _sqlite_path(url: str) -> Path | None:
    if url.startswith("sqlite:///") and ":memory:" not in url:
        return Path(url.removeprefix("sqlite:///"))
    return None


def make_engine(database_url: str, *, read_only: bool = False) -> Engine:
    connect_args: dict = {"check_same_thread": False}
    url = database_url
    if read_only:
        path = _sqlite_path(database_url)
        if path is None:
            raise ValueError("read-only mode requires a file-backed SQLite URL")
        url = f"sqlite:///file:{path}?mode=ro&uri=true"
    else:
        path = _sqlite_path(database_url)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args=connect_args, future=True)

    @event.listens_for(engine, "connect")
    def _fk_pragma(dbapi_conn, _):  # pragma: no cover - trivial
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def alembic_config(database_url: str):
    from alembic.config import Config

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


def run_migrations(database_url: str) -> None:
    from alembic import command

    path = _sqlite_path(database_url)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    command.upgrade(alembic_config(database_url), "head")


def migration_status(engine: Engine, database_url: str) -> dict[str, str | None | bool]:
    """Return current vs head revision without modifying the database."""
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(alembic_config(database_url))
    head = script.get_current_head()
    with engine.connect() as conn:
        current = MigrationContext.configure(conn).get_current_revision()
        tables = set(inspect(conn).get_table_names())
    return {
        "current": current,
        "head": head,
        "up_to_date": current == head,
        "has_tables": bool(tables),
    }

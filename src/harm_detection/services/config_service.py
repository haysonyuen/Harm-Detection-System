"""Versioned configuration management.

Rules:
* Configurations are stored as immutable snapshots (content + checksum).
* A configuration referenced by a completed evaluation run is frozen; it can
  never be modified, only superseded by a new version.
* Activating a configuration is an explicit, audited action.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from harm_detection.persistence.models import AuditEvent, Configuration, EvaluationRun
from harm_detection.policy.loader import (
    DEFAULT_CONFIG_FILES,
    checksum,
    load_config_file,
    parse_config,
)

RUN_CONFIG_COLUMNS = {
    "policy": "policy_config_id",
    "jev_questions": "question_config_id",
    "routing": "routing_config_id",
    "label_mapping": "label_mapping_config_id",
}


class ConfigImmutableError(Exception):
    pass


def audit(
    session: Session,
    event_type: str,
    entity_type: str,
    entity_id: Any,
    *,
    before: dict | None = None,
    after: dict | None = None,
    actor: str = "local_operator",
) -> None:
    session.add(
        AuditEvent(
            event_type=event_type,
            entity_type=entity_type,
            entity_id=str(entity_id),
            actor=actor,
            before_json=before,
            after_json=after,
        )
    )


def get_by_version(session: Session, config_type: str, version: str) -> Configuration | None:
    return session.scalar(
        select(Configuration).where(
            Configuration.config_type == config_type, Configuration.version == version
        )
    )


def save_config(
    session: Session,
    config_type: str,
    content: dict[str, Any],
    *,
    parent_config_id: int | None = None,
    change_note: str | None = None,
    activate: bool = False,
) -> Configuration:
    """Validate and store a configuration snapshot. Idempotent for identical content."""
    parse_config(config_type, content)  # validation
    version = content["version"]
    digest = checksum(content)
    existing = get_by_version(session, config_type, version)
    if existing is not None:
        if existing.checksum != digest:
            raise ConfigImmutableError(
                f"{config_type} version {version!r} already exists with different content; "
                "create a new version instead."
            )
        return existing
    cfg = Configuration(
        config_type=config_type,
        version=version,
        content_json=content,
        checksum=digest,
        parent_config_id=parent_config_id,
        change_note=change_note,
        is_active=False,
    )
    session.add(cfg)
    session.flush()
    audit(
        session,
        "config_created",
        "configuration",
        cfg.id,
        after={"type": config_type, "version": version},
    )
    if activate:
        activate_config(session, cfg.id)
    return cfg


def is_referenced_by_completed_run(session: Session, config: Configuration) -> bool:
    col = RUN_CONFIG_COLUMNS.get(config.config_type)
    if col is None:
        return False
    return (
        session.scalar(
            select(EvaluationRun.id).where(
                getattr(EvaluationRun, col) == config.id, EvaluationRun.status == "completed"
            )
        )
        is not None
    )


def update_config_content(
    session: Session, config_id: int, content: dict[str, Any]
) -> Configuration:
    """Only permitted for configurations not yet referenced by a completed run and not frozen."""
    cfg = session.get(Configuration, config_id)
    if cfg is None:
        raise KeyError(config_id)
    if cfg.is_frozen or is_referenced_by_completed_run(session, cfg):
        raise ConfigImmutableError(
            f"configuration {cfg.config_type}:{cfg.version} is referenced by a completed run or frozen"
        )
    parse_config(cfg.config_type, content)
    if content.get("version") != cfg.version:
        raise ConfigImmutableError("version cannot be changed in place; save a new configuration")
    before = {"checksum": cfg.checksum}
    cfg.content_json = content
    cfg.checksum = checksum(content)
    audit(
        session,
        "config_updated",
        "configuration",
        cfg.id,
        before=before,
        after={"checksum": cfg.checksum},
    )
    return cfg


def freeze_config(session: Session, config_id: int) -> None:
    cfg = session.get(Configuration, config_id)
    if cfg is not None and not cfg.is_frozen:
        cfg.is_frozen = True
        audit(session, "config_frozen", "configuration", cfg.id)


def activate_config(
    session: Session, config_id: int, *, rationale: str | None = None
) -> Configuration:
    cfg = session.get(Configuration, config_id)
    if cfg is None:
        raise KeyError(config_id)
    previous = session.scalars(
        select(Configuration).where(
            Configuration.config_type == cfg.config_type, Configuration.is_active.is_(True)
        )
    ).all()
    for p in previous:
        p.is_active = False
    cfg.is_active = True
    audit(
        session,
        "config_activated",
        "configuration",
        cfg.id,
        before={"active_ids": [p.id for p in previous]},
        after={"active_id": cfg.id, "version": cfg.version, "rationale": rationale},
    )
    return cfg


def active_config(session: Session, config_type: str) -> Configuration | None:
    return session.scalar(
        select(Configuration).where(
            Configuration.config_type == config_type, Configuration.is_active.is_(True)
        )
    )


def typed(cfg: Configuration) -> BaseModel:
    return parse_config(cfg.config_type, cfg.content_json)


def list_configs(session: Session, config_type: str | None = None) -> list[Configuration]:
    q = select(Configuration).order_by(Configuration.config_type, Configuration.id)
    if config_type:
        q = q.where(Configuration.config_type == config_type)
    return list(session.scalars(q))


def seed_default_configs(session: Session) -> dict[str, Configuration]:
    """Load YAML defaults into the database; activate them only if no config of that type is active."""
    out = {}
    for config_type in DEFAULT_CONFIG_FILES:
        _, content = load_config_file(config_type)
        cfg = save_config(session, config_type, content, change_note="seeded from repository YAML")
        if active_config(session, config_type) is None:
            activate_config(session, cfg.id, rationale="initial default")
        out[config_type] = cfg
    return out


def find_configs_by_ids(session: Session, ids: list[int]) -> dict[int, Configuration]:
    rows = (
        session.scalars(select(Configuration).where(or_(*[Configuration.id == i for i in ids])))
        if ids
        else []
    )
    return {r.id: r for r in rows}

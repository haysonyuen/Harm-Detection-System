"""System status checks for the Configuration & System Status page (no secrets exposed)."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select

from harm_detection.datasets.leakage import file_sha256
from harm_detection.persistence.db import migration_status
from harm_detection.persistence.models import IntegrationEvent
from harm_detection.services.context import AppContext
from harm_detection.settings import FIXTURES_DIR, MANIFESTS_DIR, PROCESSED_DIR


def manifest_status() -> dict[str, Any]:
    path = MANIFESTS_DIR / "civil_comments_v1.json"
    if not path.exists():
        return {"present": False, "valid": False, "detail": "not prepared"}
    try:
        manifest = json.loads(path.read_text())
    except ValueError as exc:
        return {"present": True, "valid": False, "detail": f"invalid JSON: {exc}"}
    problems = []
    for name, digest in (manifest.get("output_files") or {}).items():
        f = PROCESSED_DIR / name
        if not f.exists():
            problems.append(f"{name} missing")
        elif file_sha256(f) != digest:
            problems.append(f"{name} hash mismatch")
    return {
        "present": True,
        "valid": not problems,
        "detail": "; ".join(problems) or "all file hashes match",
        "revision": manifest.get("revision"),
        "counts": manifest.get("selected_record_count"),
    }


def fixture_status() -> dict[str, bool]:
    return {f: (FIXTURES_DIR / f).exists() for f in ("demo_cases.jsonl", "demo_predictions.jsonl")}


def system_status(ctx: AppContext) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mode": ctx.settings.app_mode.value,
        "credentials_configured": ctx.settings.credential_status(),
        "jev_available": ctx.settings.jev_available,
        "openai_available": ctx.settings.openai_available,
        "manifest": manifest_status(),
        "fixtures": fixture_status(),
    }
    try:
        out["migrations"] = migration_status(ctx.engine, ctx.settings.database_url)
    except Exception as exc:  # noqa: BLE001
        out["migrations"] = {"error": str(exc)}
    with ctx.session() as s:
        out["recent_integration_events"] = [
            {
                "time": str(e.timestamp),
                "provider": e.provider,
                "outcome": e.outcome,
                "code": e.error_code,
                "message": (e.message or "")[:200],
            }
            for e in s.scalars(
                select(IntegrationEvent).order_by(IntegrationEvent.id.desc()).limit(20)
            )
        ]
    return out

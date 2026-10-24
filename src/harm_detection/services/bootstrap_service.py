"""Idempotent initialization: configs, demo dataset, and fixture-replay demo runs."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from harm_detection.persistence.models import EvaluationRun
from harm_detection.providers.fixture import FixtureReplayProvider
from harm_detection.services import config_service as cs
from harm_detection.services.context import AppContext
from harm_detection.services.dataset_service import import_demo_dataset
from harm_detection.services.run_service import JEV, MODERATION, create_run, execute_run

DEMO_RUN_TAG = "synthetic_fixture_demo"


def active_config_ids(session) -> dict[str, int | None]:
    return {
        t: (c.id if (c := cs.active_config(session, t)) else None)
        for t in ("policy", "jev_questions", "routing", "label_mapping")
    }


def bootstrap(ctx: AppContext, *, with_demo_runs: bool = True) -> dict[str, Any]:
    ctx.require_writable("bootstrap")
    report: dict[str, Any] = {}
    with ctx.session() as s:
        cs.seed_default_configs(s)
        ds, imp = import_demo_dataset(s)
        s.commit()
        report["demo_dataset_id"] = ds.id
        report["demo_import"] = imp.to_dict()
        ids = active_config_ids(s)
        existing = s.scalars(select(EvaluationRun)).all()
        have = {(r.provider, (r.metadata_json or {}).get("demo_tag")) for r in existing}

    if not with_demo_runs:
        return report
    created = []
    for provider_name, model in ((JEV, "jev-latest"), (MODERATION, "omni-moderation-latest")):
        if (provider_name, DEMO_RUN_TAG) in have:
            continue
        with ctx.session() as s:
            run = create_run(
                s,
                name=f"Synthetic fixture demo - {provider_name}",
                provider=provider_name,
                dataset_id=report["demo_dataset_id"],
                cohort="development",
                requested_model=model,
                config_ids=ids,
                data_origin="synthetic_fixture",
                extra_metadata={"demo_tag": DEMO_RUN_TAG},
            )
            s.commit()
            run_id = run.id
        provider = FixtureReplayProvider(provider_name)
        summary = execute_run(ctx.session_factory, run_id, provider)
        created.append({"run_id": run_id, "provider": provider_name, "status": summary.status})
    report["demo_runs"] = created
    return report

"""Export a curated, read-only public demo database.

Copies only approved artifacts from the local database into data/public_demo/public_demo.sqlite:
  * completed/partial evaluation runs on synthetic or CC0 public datasets (never playground input),
  * their datasets' cases, configurations, predictions (normalized signals only; raw provider
    responses are dropped), routing decisions, metrics, and optimization experiments.
Analysis requests, recommendations, audit and integration events are NOT exported.

Usage:  uv run python scripts/export_public_demo.py [--runs 1 2 3] [--output PATH]
Then:   APP_MODE=public_demo uv run streamlit run app/main.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from harm_detection.persistence.db import make_engine, make_session_factory, run_migrations
from harm_detection.persistence.models import (
    Configuration,
    Dataset,
    DatasetCase,
    EvaluationMetric,
    EvaluationRun,
    OptimizationExperiment,
    Prediction,
    RoutingDecisionRow,
)
from harm_detection.security.redaction import redact
from harm_detection.settings import PUBLIC_DEMO_DB_PATH, load_settings

APPROVED_ORIGINS = {"synthetic", "public_dataset"}
APPROVED_LICENSE_MARKERS = ("CC0", "Synthetic")


def _clone(obj, **overrides):
    cols = {c.key: getattr(obj, c.key) for c in obj.__table__.columns}
    cols.update(overrides)
    return type(obj)(**cols)


def export(source_url: str, output: Path, run_ids: list[int] | None = None) -> dict:
    if output.exists():
        output.unlink()
    dest_url = f"sqlite:///{output}"
    run_migrations(dest_url)
    src = make_session_factory(make_engine(source_url))()
    dst = make_session_factory(make_engine(dest_url))()
    report = {"runs": [], "skipped_runs": []}
    try:
        runs = src.scalars(
            select(EvaluationRun).where(
                EvaluationRun.status.in_(("completed", "partial")),
                EvaluationRun.cohort_name != "playground",
            )
        ).all()
        if run_ids:
            runs = [r for r in runs if r.id in set(run_ids)]
        chosen = []
        for r in runs:
            ds = src.get(Dataset, r.dataset_id)
            if ds.data_origin not in APPROVED_ORIGINS or not any(
                m in (ds.license or "") for m in APPROVED_LICENSE_MARKERS
            ):
                report["skipped_runs"].append(
                    {"id": r.id, "reason": f"dataset {ds.name} not approved for export"}
                )
                continue
            chosen.append(r)
        # Derived runs need their prediction source.
        ids = {r.id for r in chosen}
        for r in list(chosen):
            src_id = (r.metadata_json or {}).get("prediction_source_run_id")
            if src_id and src_id not in ids:
                report["skipped_runs"].append(
                    {"id": r.id, "reason": "prediction source run not exported"}
                )
                chosen.remove(r)
        ids = {r.id for r in chosen}

        cfg_ids = {
            c
            for r in chosen
            for c in (
                r.policy_config_id,
                r.question_config_id,
                r.routing_config_id,
                r.label_mapping_config_id,
            )
            if c
        }
        exps = src.scalars(
            select(OptimizationExperiment).where(OptimizationExperiment.source_run_id.in_(ids))
        ).all()
        cfg_ids |= {e.selected_config_id for e in exps if e.selected_config_id}
        for t in ("pricing",):
            cfg_ids |= {
                c.id
                for c in src.scalars(
                    select(Configuration).where(
                        Configuration.config_type == t, Configuration.is_active.is_(True)
                    )
                )
            }
        # include parents so lineage resolves
        frontier = set(cfg_ids)
        while frontier:
            parents = {
                c.parent_config_id
                for c in src.scalars(select(Configuration).where(Configuration.id.in_(frontier)))
                if c.parent_config_id and c.parent_config_id not in cfg_ids
            }
            cfg_ids |= parents
            frontier = parents
        for c in src.scalars(
            select(Configuration).where(Configuration.id.in_(cfg_ids)).order_by(Configuration.id)
        ):
            dst.add(_clone(c))
        # Ensure an active config of each type exists in the demo for display.
        dst.flush()

        ds_ids = {r.dataset_id for r in chosen}
        for d in src.scalars(select(Dataset).where(Dataset.id.in_(ds_ids))):
            dst.add(_clone(d))
        dst.flush()
        for c in src.scalars(select(DatasetCase).where(DatasetCase.dataset_id.in_(ds_ids))):
            dst.add(_clone(c))
        for r in sorted(chosen, key=lambda x: x.id):
            dst.add(_clone(r, metadata_json=redact(r.metadata_json)))
            report["runs"].append(
                {"id": r.id, "name": r.name, "provider": r.provider, "cohort": r.cohort_name}
            )
        dst.flush()
        pred_ids = []
        for p in src.scalars(select(Prediction).where(Prediction.run_id.in_(ids))):
            dst.add(
                _clone(
                    p, raw_response_json=None, error_message=(p.error_message or "")[:300] or None
                )
            )
            pred_ids.append(p.id)
        dst.flush()
        for rd in src.scalars(
            select(RoutingDecisionRow).where(RoutingDecisionRow.prediction_id.in_(pred_ids))
        ):
            if rd.routing_config_id in cfg_ids:
                dst.add(_clone(rd))
        for m in src.scalars(select(EvaluationMetric).where(EvaluationMetric.run_id.in_(ids))):
            dst.add(_clone(m))
        for e in exps:
            dst.add(_clone(e))
        dst.commit()
    finally:
        src.close()
        dst.close()
    report["output"] = str(output)
    return report


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", type=int, nargs="*")
    p.add_argument("--output", type=Path, default=PUBLIC_DEMO_DB_PATH)
    args = p.parse_args()
    settings = load_settings()
    if settings.is_public_demo:
        print("Run the exporter from local mode, not public_demo.")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(export(settings.database_url, args.output, args.runs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

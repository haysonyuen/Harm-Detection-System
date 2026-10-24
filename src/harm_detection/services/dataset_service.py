"""Dataset registration, case import with duplicate/leakage prevention, cohort selection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from harm_detection.datasets.leakage import check_leakage, content_hash
from harm_detection.persistence.models import Dataset, DatasetCase
from harm_detection.policy.loader import checksum
from harm_detection.schemas.content import ContentInput, validate_content
from harm_detection.settings import FIXTURES_DIR, MANIFESTS_DIR, PROCESSED_DIR

COHORTS = ("development", "representative_dev", "holdout", "all")

DEMO_DATASET_NAME = "synthetic_demo"
DEMO_DATASET_VERSION = "v1"
CIVIL_DATASET_NAME = "civil_comments"
CIVIL_DATASET_VERSION = "v1"


@dataclass
class ImportReport:
    imported: int = 0
    skipped_existing: int = 0
    rejected_invalid: list[dict[str, str]] = field(default_factory=list)
    skipped_duplicates: list[dict[str, str]] = field(default_factory=list)
    excluded_leakage: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def get_dataset(session: Session, name: str, version: str) -> Dataset | None:
    return session.scalar(select(Dataset).where(Dataset.name == name, Dataset.version == version))


def import_cases(
    session: Session,
    *,
    name: str,
    version: str,
    records: list[dict[str, Any]],
    source_url: str | None = None,
    revision: str | None = None,
    license: str | None = None,
    manifest_path: str | None = None,
    data_origin: str = "public_dataset",
    max_text_length: int = 5000,
) -> tuple[Dataset, ImportReport]:
    """Import records into a dataset version, enforcing:

    * canonical input validation,
    * no duplicate case IDs, source record IDs, or normalized-content hashes per dataset,
    * no holdout case overlapping a development case (exact or near duplicate).
    Re-importing identical records is idempotent.
    """
    report = ImportReport()
    ds = get_dataset(session, name, version)
    if ds is None:
        ds = Dataset(
            name=name,
            version=version,
            source_url=source_url,
            revision=revision,
            license=license,
            manifest_path=manifest_path,
            data_origin=data_origin,
        )
        session.add(ds)
        session.flush()

    existing = session.scalars(select(DatasetCase).where(DatasetCase.dataset_id == ds.id)).all()
    seen_ids = {c.case_id for c in existing}
    seen_src = {c.source_record_id for c in existing if c.source_record_id}
    seen_hash = {c.content_hash: c.case_id for c in existing}

    valid: list[tuple[dict[str, Any], ContentInput, str]] = []
    for rec in records:
        payload = {
            k: rec.get(k)
            for k in (
                "case_id",
                "source",
                "text",
                "language",
                "parent_text",
                "target_context",
                "source_dataset",
                "source_record_id",
                "created_at",
            )
            if k in rec
        }
        payload["metadata"] = rec.get("metadata") or {}
        try:
            item = validate_content(payload, max_text_length)
        except ValueError as exc:
            report.rejected_invalid.append(
                {"case_id": str(rec.get("case_id")), "error": str(exc)[:300]}
            )
            continue
        if item.case_id in seen_ids:
            report.skipped_existing += 1
            continue
        h = content_hash(item.text)
        if h in seen_hash:
            report.skipped_duplicates.append(
                {"case_id": item.case_id, "duplicate_of": seen_hash[h]}
            )
            continue
        if item.source_record_id and item.source_record_id in seen_src:
            report.skipped_duplicates.append(
                {"case_id": item.case_id, "duplicate_source_record_id": item.source_record_id}
            )
            continue
        seen_ids.add(item.case_id)
        seen_hash[h] = item.case_id
        if item.source_record_id:
            seen_src.add(item.source_record_id)
        valid.append((rec, item, h))

    # Holdout leakage: compare incoming + existing holdout cases against development cases.
    dev = [
        (c.case_id, c.source_record_id or "", c.text) for c in existing if c.split == "development"
    ]
    dev += [
        (i.case_id, i.source_record_id or "", i.text)
        for r, i, _ in valid
        if r.get("split") == "development"
    ]
    hold = [
        (i.case_id, i.source_record_id or "", i.text)
        for r, i, _ in valid
        if r.get("split") == "holdout"
    ]
    leak = check_leakage(dev, hold)
    leaking: dict[str, dict[str, Any]] = {}
    src_to_case = {s: c for c, s, _ in hold}
    for s in leak.shared_source_ids:
        leaking[src_to_case[s]] = {"reason": "shared_source_record_id", "source_record_id": s}
    for h_id, d_id in leak.exact_overlaps:
        leaking[h_id] = {"reason": "exact_duplicate_of_development", "development_case_id": d_id}
    for h_id, d_id, sim in leak.near_overlaps:
        leaking.setdefault(
            h_id,
            {
                "reason": "near_duplicate_of_development",
                "development_case_id": d_id,
                "similarity": round(sim, 4),
            },
        )

    for rec, item, h in valid:
        if item.case_id in leaking:
            report.excluded_leakage.append({"case_id": item.case_id, **leaking[item.case_id]})
            continue
        session.add(
            DatasetCase(
                dataset_id=ds.id,
                case_id=item.case_id,
                source_record_id=item.source_record_id,
                split=rec.get("split", "development"),
                stratum=rec.get("stratum"),
                content_hash=h,
                text=item.text,
                parent_text=item.parent_text,
                target_context=item.target_context,
                reference_labels_json=rec.get("reference_labels") or {},
                metadata_json={
                    **item.metadata,
                    "source": item.source.value,
                    "source_dataset": item.source_dataset,
                },
            )
        )
        report.imported += 1
    session.flush()
    rows = session.scalars(select(DatasetCase).where(DatasetCase.dataset_id == ds.id)).all()
    ds.record_count = len(rows)
    ds.checksum = checksum(sorted((r.case_id, r.content_hash) for r in rows))
    return ds, report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def import_demo_dataset(session: Session) -> tuple[Dataset, ImportReport]:
    records = load_jsonl(FIXTURES_DIR / "demo_cases.jsonl")
    return import_cases(
        session,
        name=DEMO_DATASET_NAME,
        version=DEMO_DATASET_VERSION,
        records=records,
        source_url="data/fixtures/demo_cases.jsonl",
        revision="synthetic",
        license="Synthetic examples authored for this project",
        data_origin="synthetic",
    )


def import_civil_comments(session: Session) -> tuple[Dataset, ImportReport]:
    manifest_path = MANIFESTS_DIR / "civil_comments_v1.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run `uv run python scripts/prepare_dataset.py` first.")
    manifest = json.loads(manifest_path.read_text())
    records = load_jsonl(PROCESSED_DIR / "development_v1.jsonl") + load_jsonl(
        PROCESSED_DIR / "holdout_v1.jsonl"
    )
    ds, report = import_cases(
        session,
        name=CIVIL_DATASET_NAME,
        version=CIVIL_DATASET_VERSION,
        records=records,
        source_url=manifest["dataset_url"],
        revision=manifest["revision"],
        license=manifest["license"],
        manifest_path=str(manifest_path.relative_to(manifest_path.parents[2])),
        data_origin="public_dataset",
    )
    return ds, report


def cohort_cases(session: Session, dataset_id: int, cohort: str) -> list[DatasetCase]:
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}")
    q = select(DatasetCase).where(DatasetCase.dataset_id == dataset_id)
    if cohort == "development":
        q = q.where(DatasetCase.split == "development")
    elif cohort == "representative_dev":
        q = q.where(DatasetCase.split == "development", DatasetCase.stratum == "representative")
    elif cohort == "holdout":
        q = q.where(DatasetCase.split == "holdout")
    return list(session.scalars(q.order_by(DatasetCase.id)))


def case_to_input(case: DatasetCase, *, include_context: bool = True) -> ContentInput:
    meta = case.metadata_json or {}
    return ContentInput(
        case_id=case.case_id,
        source=meta.get("source", "public_dataset"),
        text=case.text,
        parent_text=case.parent_text if include_context else None,
        target_context=case.target_context if include_context else None,
        source_dataset=meta.get("source_dataset"),
        source_record_id=case.source_record_id,
        metadata={k: v for k, v in meta.items() if k not in ("source", "source_dataset")},
    )


def list_datasets(session: Session) -> list[Dataset]:
    return list(session.scalars(select(Dataset).order_by(Dataset.id)))


def delete_dataset(session: Session, dataset_id: int) -> None:
    """Delete a dataset and its cases (local data retention control)."""
    from harm_detection.persistence.models import EvaluationRun

    if session.scalar(select(EvaluationRun.id).where(EvaluationRun.dataset_id == dataset_id)):
        raise ValueError("delete the dataset's evaluation runs first")
    ds = session.get(Dataset, dataset_id)
    if ds is not None:
        for c in session.scalars(select(DatasetCase).where(DatasetCase.dataset_id == dataset_id)):
            session.delete(c)
        session.delete(ds)

"""Google/Jigsaw Civil Comments source adapter and reproducible sampler (PRD §9).

Source: https://huggingface.co/datasets/google/civil_comments (license: CC0-1.0).
Parquet files are downloaded at a pinned Hugging Face commit so the sample is
reproducible. The full download stays in ``data/raw`` (git-ignored).

Splits used:
  development  <- source ``validation`` split (stratified + representative sample)
  holdout      <- source ``test`` split (simple random sample, not stratified)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harm_detection.datasets.leakage import (
    NEAR_DUP_METHOD,
    check_leakage,
    content_hash,
    dedupe_exact,
    file_sha256,
    find_near_duplicates,
)

DATASET_NAME = "google/civil_comments"
DATASET_URL = "https://huggingface.co/datasets/google/civil_comments"
DATASET_LICENSE = "CC0-1.0 (per Hugging Face dataset card metadata)"
# Hugging Face commit verified 2026-10-23 (lastModified 2024-01-25).
DEFAULT_REVISION = "f2970eb3a55777454c94069077cc8d9b5866312d"
SPLIT_FILES = {
    "train": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
    "validation": ["data/validation-00000-of-00001.parquet"],
    "test": ["data/test-00000-of-00001.parquet"],
}
ANNOTATION_FIELDS = (
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
)
DEV_SOURCE_SPLIT = "validation"
HOLDOUT_SOURCE_SPLIT = "test"


@dataclass(frozen=True)
class SamplingPlan:
    seed: int = 20261023
    threat_enriched: int = 100
    insult_enriched: int = 100
    borderline: int = 50
    representative: int = 350
    holdout: int = 400
    positive_threshold: float = 0.5
    borderline_low: float = 0.3
    max_text_length: int = 5000

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


STRATUM_DEFINITIONS = {
    "threat_enriched": "threat >= positive_threshold",
    "insult_enriched": "insult >= positive_threshold and threat < positive_threshold",
    "borderline": (
        "not previously selected and (borderline_low <= threat < positive_threshold or "
        "borderline_low <= insult < positive_threshold or (threat >= positive_threshold and "
        "insult >= positive_threshold))"
    ),
    "representative": "uniform random sample of remaining eligible records",
}
FALLBACK_POLICY = (
    "If a stratum has fewer eligible records than requested, all eligible records are taken and "
    "the shortfall is NOT reallocated; actual counts are disclosed in the manifest."
)


def download_split(
    split: str, raw_dir: Path, revision: str = DEFAULT_REVISION, client=None
) -> list[Path]:
    import httpx

    raw_dir = raw_dir / revision[:12]
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    own = client is None
    client = client or httpx.Client(timeout=300.0, follow_redirects=True)
    try:
        for rel in SPLIT_FILES[split]:
            dest = raw_dir / Path(rel).name
            if not dest.exists():
                url = f"{DATASET_URL}/resolve/{revision}/{rel}"
                tmp = dest.with_suffix(".part")
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with tmp.open("wb") as fh:
                        for chunk in resp.iter_bytes():
                            fh.write(chunk)
                tmp.rename(dest)
            paths.append(dest)
    finally:
        if own:
            client.close()
    return paths


def load_split(paths: list[Path], split: str) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["row_index"] = np.arange(len(df))
    df["source_split"] = split
    return df


@dataclass
class EligibilityStats:
    total: int = 0
    empty_text: int = 0
    too_long: int = 0
    exact_duplicates: int = 0
    eligible: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def eligible_records(df: pd.DataFrame, max_len: int) -> tuple[pd.DataFrame, EligibilityStats]:
    stats = EligibilityStats(total=len(df))
    text = df["text"].fillna("").astype(str)
    nonempty = text.str.strip().str.len() > 0
    stats.empty_text = int((~nonempty).sum())
    short = text.str.strip().str.len() <= max_len
    stats.too_long = int((nonempty & ~short).sum())
    out = df[nonempty & short].copy()
    out["text"] = out["text"].astype(str).str.strip()
    dd = dedupe_exact(out["text"].tolist())
    stats.exact_duplicates = len(dd.duplicates)
    out = out.iloc[dd.kept].copy()
    out["content_hash"] = [content_hash(t) for t in out["text"]]
    stats.eligible = len(out)
    return out, stats


def _take(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(pool) <= n:
        return pool
    idx = rng.choice(len(pool), size=n, replace=False)
    return pool.iloc[np.sort(idx)]


def sample_development(
    df: pd.DataFrame, plan: SamplingPlan
) -> tuple[pd.DataFrame, dict[str, dict]]:
    rng = np.random.default_rng(plan.seed)
    hi, lo = plan.positive_threshold, plan.borderline_low
    selected: list[pd.DataFrame] = []
    taken = pd.Series(False, index=df.index)
    counts: dict[str, dict] = {}

    def pick(name: str, mask: pd.Series, n: int) -> None:
        pool = df[mask & ~taken]
        chosen = _take(pool, n, rng).copy()
        chosen["stratum"] = name
        taken.loc[chosen.index] = True
        selected.append(chosen)
        counts[name] = {
            "requested": n,
            "eligible_pool": int(len(pool)),
            "selected": int(len(chosen)),
        }

    threat, insult = df["threat"], df["insult"]
    pick("threat_enriched", threat >= hi, plan.threat_enriched)
    pick("insult_enriched", (insult >= hi) & (threat < hi), plan.insult_enriched)
    border = (
        ((threat >= lo) & (threat < hi))
        | ((insult >= lo) & (insult < hi))
        | ((threat >= hi) & (insult >= hi))
    )
    pick("borderline", border, plan.borderline)
    pick("representative", pd.Series(True, index=df.index), plan.representative)
    return pd.concat(selected), counts


def sample_holdout(
    test_df: pd.DataFrame, dev_df: pd.DataFrame, plan: SamplingPlan
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Random sample from the test split, skipping any record that overlaps development.

    Uses one seeded permutation and walks it in order, so the result is deterministic and
    never "resampled until attractive".
    """
    rng = np.random.default_rng(plan.seed + 1)
    order = rng.permutation(len(test_df))
    dev_hashes = set(dev_df["content_hash"])
    dev_texts = dev_df["text"].tolist()
    chosen_idx: list[int] = []
    removed_exact = removed_near = 0
    batch = max(plan.holdout * 2, 100)
    pos = 0
    while len(chosen_idx) < plan.holdout and pos < len(order):
        cand = order[pos : pos + batch]
        pos += batch
        sub = test_df.iloc[cand]
        near = {i for i, _, _ in find_near_duplicates(sub["text"].tolist(), dev_texts)}
        for k, (i, h) in enumerate(zip(cand, sub["content_hash"], strict=True)):
            if h in dev_hashes:
                removed_exact += 1
                continue
            if k in near:
                removed_near += 1
                continue
            chosen_idx.append(int(i))
            if len(chosen_idx) >= plan.holdout:
                break
    out = test_df.iloc[sorted(chosen_idx)].copy()
    out["stratum"] = "representative"
    return out, {
        "requested": plan.holdout,
        "selected": len(out),
        "removed_exact_overlap_with_dev": removed_exact,
        "removed_near_overlap_with_dev": removed_near,
    }


def to_case_records(df: pd.DataFrame, split_label: str) -> list[dict[str, Any]]:
    records = []
    for row in df.itertuples(index=False):
        src_split = row.source_split
        records.append(
            {
                "case_id": f"cc-{src_split}-{row.row_index}",
                "source": "public_dataset",
                "text": row.text,
                "language": "en",
                "parent_text": None,
                "target_context": None,
                "source_dataset": DATASET_NAME,
                "source_record_id": f"{src_split}:{row.row_index}",
                "created_at": None,
                "split": split_label,
                "stratum": row.stratum,
                "content_hash": row.content_hash,
                "reference_labels": {f: float(getattr(row, f)) for f in ANNOTATION_FIELDS},
                "metadata": {
                    "source_split": src_split,
                    "labels_origin": "published_human_annotation",
                },
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return file_sha256(path)


def prevalence(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    n = len(records)
    out: dict[str, Any] = {"n": n}
    for f in ("threat", "insult", "toxicity", "identity_attack"):
        pos = sum(1 for r in records if r["reference_labels"][f] >= threshold)
        out[f"{f}_positive"] = pos
        out[f"{f}_prevalence"] = round(pos / n, 6) if n else None
    return out


@dataclass
class PreparedDataset:
    development: list[dict[str, Any]]
    representative: list[dict[str, Any]]
    holdout: list[dict[str, Any]]
    manifest: dict[str, Any] = field(default_factory=dict)


def prepare(
    dev_source: pd.DataFrame,
    test_source: pd.DataFrame,
    plan: SamplingPlan,
    *,
    revision: str,
    label_mapping_version: str,
    source_file_hashes: dict[str, str] | None = None,
) -> PreparedDataset:
    dev_elig, dev_stats = eligible_records(dev_source, plan.max_text_length)
    test_elig, test_stats = eligible_records(test_source, plan.max_text_length)
    dev_df, strata_counts = sample_development(dev_elig, plan)
    hold_df, hold_counts = sample_holdout(test_elig, dev_df, plan)

    dev = to_case_records(dev_df, "development")
    rep = [r for r in dev if r["stratum"] == "representative"]
    hold = to_case_records(hold_df, "holdout")

    leak = check_leakage(
        [(r["case_id"], r["source_record_id"], r["text"]) for r in dev],
        [(r["case_id"], r["source_record_id"], r["text"]) for r in hold],
    )
    if not leak.clean:
        raise RuntimeError(f"leakage check failed after sampling: {leak.to_dict()}")

    manifest = {
        "manifest_version": "civil_comments_v1",
        "dataset_name": DATASET_NAME,
        "dataset_url": DATASET_URL,
        "revision": revision,
        "license": DATASET_LICENSE,
        "prepared_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_splits": {"development": DEV_SOURCE_SPLIT, "holdout": HOLDOUT_SOURCE_SPLIT},
        "source_file_sha256": source_file_hashes or {},
        "sampling": {
            "method": "stratified development sample + simple random holdout from a separate split",
            "plan": plan.to_dict(),
            "random_seed": plan.seed,
            "strata_definitions": STRATUM_DEFINITIONS,
            "fallback_policy": FALLBACK_POLICY,
            "development_strata_counts": strata_counts,
            "holdout_counts": hold_counts,
        },
        "eligibility": {
            "development_source": dev_stats.to_dict(),
            "holdout_source": test_stats.to_dict(),
            "rules": [
                "text non-empty after trimming",
                f"text length <= {plan.max_text_length} characters",
                "exact duplicates (normalized content hash) within a split dropped, first kept",
                "language assumed English per dataset card (no per-record language field)",
            ],
        },
        "label_mapping_version": label_mapping_version,
        "selected_record_count": {
            "development": len(dev),
            "representative_dev": len(rep),
            "holdout": len(hold),
        },
        "class_prevalence": {
            "development_all": prevalence(dev, plan.positive_threshold),
            "representative_dev": prevalence(rep, plan.positive_threshold),
            "holdout": prevalence(hold, plan.positive_threshold),
        },
        "duplicate_removal": {
            "within_split_exact": {
                "development_source": dev_stats.exact_duplicates,
                "holdout_source": test_stats.exact_duplicates,
            },
            "holdout_vs_development": hold_counts,
            "near_duplicate_method": NEAR_DUP_METHOD,
            "final_leakage_check": leak.to_dict(),
        },
        "reference_label_note": (
            "Reference labels are published crowd-annotation fractions. Proxy labels derived from them "
            "are not confirmed X policy violations."
        ),
    }
    return PreparedDataset(dev, rep, hold, manifest)

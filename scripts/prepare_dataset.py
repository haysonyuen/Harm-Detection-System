"""Prepare versioned Civil Comments development/holdout samples and manifest.

Downloads the pinned validation and test parquet files (~60 MB) into data/raw (git-ignored),
samples 600 development + 400 holdout records, checks leakage, writes:

  data/processed/development_v1.jsonl
  data/processed/representative_dev_v1.jsonl
  data/processed/holdout_v1.jsonl
  data/manifests/civil_comments_v1.json

and (unless --no-import) imports them into the local database.

Usage:  uv run python scripts/prepare_dataset.py [--seed N] [--no-import]
"""

from __future__ import annotations

import argparse
import json

from harm_detection.datasets import civil_comments as cc
from harm_detection.datasets.leakage import file_sha256
from harm_detection.policy.loader import load_config_file
from harm_detection.settings import MANIFESTS_DIR, PROCESSED_DIR, RAW_DIR, load_settings


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seed", type=int, default=cc.SamplingPlan.seed)
    p.add_argument("--revision", default=cc.DEFAULT_REVISION)
    p.add_argument("--no-import", action="store_true")
    args = p.parse_args()

    plan = cc.SamplingPlan(seed=args.seed)
    print(
        f"Downloading {cc.DATASET_NAME}@{args.revision[:12]} validation + test splits (cached in data/raw)…"
    )
    dev_paths = cc.download_split(cc.DEV_SOURCE_SPLIT, RAW_DIR, args.revision)
    test_paths = cc.download_split(cc.HOLDOUT_SOURCE_SPLIT, RAW_DIR, args.revision)
    hashes = {p.name: file_sha256(p) for p in dev_paths + test_paths}
    dev_src = cc.load_split(dev_paths, cc.DEV_SOURCE_SPLIT)
    test_src = cc.load_split(test_paths, cc.HOLDOUT_SOURCE_SPLIT)
    mapping, _ = load_config_file("label_mapping")

    prepared = cc.prepare(
        dev_src,
        test_src,
        plan,
        revision=args.revision,
        label_mapping_version=mapping.version,
        source_file_hashes=hashes,
    )
    out = {
        "development_v1.jsonl": cc.write_jsonl(
            PROCESSED_DIR / "development_v1.jsonl", prepared.development
        ),
        "representative_dev_v1.jsonl": cc.write_jsonl(
            PROCESSED_DIR / "representative_dev_v1.jsonl", prepared.representative
        ),
        "holdout_v1.jsonl": cc.write_jsonl(PROCESSED_DIR / "holdout_v1.jsonl", prepared.holdout),
    }
    manifest = {**prepared.manifest, "output_files": out}
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFESTS_DIR / "civil_comments_v1.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "selected": manifest["selected_record_count"],
                "strata": manifest["sampling"]["development_strata_counts"],
                "holdout": manifest["sampling"]["holdout_counts"],
                "prevalence": manifest["class_prevalence"],
            },
            indent=2,
        )
    )

    if not args.no_import:
        settings = load_settings()
        if settings.is_public_demo:
            print("public_demo mode: skipping import")
            return 0
        from harm_detection.services import config_service as cs
        from harm_detection.services.context import create_context
        from harm_detection.services.dataset_service import import_civil_comments

        ctx = create_context(settings)
        with ctx.session() as s:
            cs.seed_default_configs(s)
            ds, report = import_civil_comments(s)
            s.commit()
            print(
                f"Imported into dataset #{ds.id}: {report.imported} new, {report.skipped_existing} existing, "
                f"{len(report.skipped_duplicates)} duplicates skipped, {len(report.excluded_leakage)} leakage exclusions"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Initialize the local database: migrations, default configs, demo dataset and demo runs.

Usage:  uv run python scripts/bootstrap.py [--no-demo-runs]
"""

from __future__ import annotations

import argparse
import json
import sys

from harm_detection.services.bootstrap_service import bootstrap
from harm_detection.services.context import create_context
from harm_detection.settings import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-demo-runs", action="store_true", help="skip fixture-replay demo runs")
    args = parser.parse_args()
    settings = load_settings()
    if settings.is_public_demo:
        print(
            "APP_MODE=public_demo is read-only; use scripts/export_public_demo.py instead.",
            file=sys.stderr,
        )
        return 2
    ctx = create_context(settings)
    report = bootstrap(ctx, with_demo_runs=not args.no_demo_runs)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

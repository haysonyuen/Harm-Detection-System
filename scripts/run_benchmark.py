"""Create and execute an evaluation run from the command line.

Examples:
  # Offline: replay synthetic fixtures (no provider calls)
  uv run python scripts/run_benchmark.py --dataset synthetic_demo --provider jev

  # Live (APP_MODE=local_live, TYPESAFE_API_KEY set): paid Jev calls on the development sample
  uv run python scripts/run_benchmark.py --dataset civil_comments --provider jev --cohort development --yes

  # Resume an incomplete run
  uv run python scripts/run_benchmark.py --resume 7 --yes
"""

from __future__ import annotations

import argparse
import json
import sys

from harm_detection.persistence.models import Configuration, EvaluationRun
from harm_detection.providers.fixture import FixtureReplayProvider
from harm_detection.providers.jev import JevProvider
from harm_detection.providers.openai_moderation import OpenAIModerationProvider
from harm_detection.schemas.config import JevQuestionConfig
from harm_detection.services.bootstrap_service import active_config_ids
from harm_detection.services.context import create_context
from harm_detection.services.dataset_service import get_dataset
from harm_detection.services.evaluation_service import stored_metrics
from harm_detection.services.run_service import (
    JEV,
    MODERATION,
    create_run,
    execute_run,
    remaining_case_ids,
)

PROVIDERS = {"jev": JEV, "moderation": MODERATION}


def build_provider(ctx, run: EvaluationRun):
    if run.metadata_json.get("data_origin") == "synthetic_fixture":
        return FixtureReplayProvider(run.provider)
    ctx.require_live("benchmark inference")
    if run.provider == JEV:
        with ctx.session() as s:
            q = JevQuestionConfig.model_validate(
                s.get(Configuration, run.question_config_id).content_json
            )
        return JevProvider(
            ctx.settings.typesafe_api_key,
            q,
            model=run.requested_model,
            base_url=ctx.settings.jev_base_url,
        )
    return OpenAIModerationProvider(ctx.settings.openai_api_key, model=run.requested_model)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset", default="civil_comments", choices=["civil_comments", "synthetic_demo"]
    )
    p.add_argument("--provider", default="jev", choices=list(PROVIDERS))
    p.add_argument(
        "--cohort", default="development", choices=["development", "representative_dev", "holdout"]
    )
    p.add_argument("--max-cases", type=int)
    p.add_argument("--input-mode", default="text_only", choices=["text_only", "with_context"])
    p.add_argument("--routing-config-id", type=int)
    p.add_argument("--name")
    p.add_argument("--resume", type=int, help="resume an existing run id")
    p.add_argument("--acknowledge-holdout-reuse", action="store_true")
    p.add_argument("--yes", action="store_true", help="confirm paid provider inference")
    args = p.parse_args()

    ctx = create_context()
    ctx.require_writable("benchmark runs")
    with ctx.session() as s:
        if args.resume:
            run = s.get(EvaluationRun, args.resume)
            if run is None:
                print("run not found", file=sys.stderr)
                return 2
            n = len(remaining_case_ids(s, run))
        else:
            ds = get_dataset(s, args.dataset, "v1")
            if ds is None:
                print(
                    f"dataset {args.dataset} not found; run bootstrap/prepare_dataset first",
                    file=sys.stderr,
                )
                return 2
            provider = PROVIDERS[args.provider]
            ids = active_config_ids(s)
            if args.routing_config_id:
                ids["routing"] = args.routing_config_id
            origin = "synthetic_fixture" if ds.data_origin == "synthetic" else "live"
            if origin == "live" and not ctx.settings.live_calls_allowed:
                print(
                    f"Live inference is disabled (APP_MODE={ctx.settings.app_mode.value}). "
                    "Set APP_MODE=local_live and configure credentials. No run was created.",
                    file=sys.stderr,
                )
                return 2
            model = (
                ctx.settings.jev_model if provider == JEV else ctx.settings.openai_moderation_model
            )
            run = create_run(
                s,
                name=args.name or f"{args.provider} {args.dataset} {args.cohort}",
                provider=provider,
                dataset_id=ds.id,
                cohort=args.cohort,
                requested_model=model,
                config_ids={
                    **ids,
                    "jev_questions": ids["jev_questions"] if provider == JEV else None,
                },
                data_origin=origin,
                input_mode=args.input_mode,
                max_cases=args.max_cases,
                acknowledge_holdout_reuse=args.acknowledge_holdout_reuse,
            )
            n = run.total_cases
        origin = run.metadata_json.get("data_origin")
        if origin == "live" and not args.yes:
            s.rollback()
            print(
                f"This run makes up to {n} paid API calls to {run.requested_model}. Re-run with --yes to confirm."
            )
            return 1
        s.commit()
        run_id = run.id

    with ctx.session() as s:
        run = s.get(EvaluationRun, run_id)
    provider_obj = build_provider(ctx, run)

    def progress(i, total, summ):
        if i % 25 == 0 or i == total:
            print(
                f"  {i}/{total} ok={summ.succeeded} failed={summ.failed} errors={summ.error_counts}",
                flush=True,
            )

    summary = execute_run(ctx.session_factory, run_id, provider_obj, progress=progress)
    print(json.dumps(summary.__dict__, indent=2))
    with ctx.session() as s:
        m = stored_metrics(s, run_id)
    if not m.empty:
        key = m[
            (m.cohort == "all")
            & m.metric.isin(
                [
                    "precision",
                    "recall",
                    "priority_recall",
                    "under_prioritization_rate",
                    "total_review_rate",
                    "api_failure_rate",
                    "evaluated_cases",
                ]
            )
        ]
        print(
            key[["category", "metric", "value", "numerator", "denominator"]].to_string(index=False)
        )
    return 0 if summary.status in ("completed", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())

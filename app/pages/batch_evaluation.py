from __future__ import annotations

import json

import streamlit as st
from sqlalchemy import select

from app.components.common import get_ctx, mode_banner, origin_notice, require_write_ui, run_label
from harm_detection.persistence.models import Configuration, EvaluationRun
from harm_detection.providers.fixture import FixtureReplayProvider
from harm_detection.providers.jev import JevProvider
from harm_detection.providers.openai_moderation import OpenAIModerationProvider
from harm_detection.schemas.config import JevQuestionConfig, PricingConfig
from harm_detection.services import config_service as cs
from harm_detection.services.dataset_service import cohort_cases, list_datasets
from harm_detection.services.report_service import cases_csv, run_report
from harm_detection.services.run_service import (
    JEV,
    MODERATION,
    HoldoutReuseError,
    create_run,
    execute_run,
    remaining_case_ids,
)

ctx = get_ctx()
st.title("Batch Evaluation")
mode_banner(ctx)


def build_provider(run: EvaluationRun, origin: str):
    if origin == "synthetic_fixture":
        return FixtureReplayProvider(run.provider)
    ctx.require_live("batch inference")
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


def execute(run_id: int, origin: str, batch_size: int, max_new: int | None) -> None:
    with ctx.session() as s:
        run = s.get(EvaluationRun, run_id)
    try:
        provider = build_provider(run, origin)
    except PermissionError as exc:
        st.error(str(exc))
        return
    bar = st.progress(0.0, text="Starting…")
    status = st.empty()

    def progress(i, total, summ):
        bar.progress(i / max(total, 1), text=f"{i}/{total} processed this session")
        status.caption(
            f"succeeded {summ.succeeded} · failed {summ.failed} · errors {summ.error_counts}"
        )

    summary = execute_run(
        ctx.session_factory,
        run_id,
        provider,
        batch_size=batch_size,
        max_new_cases=max_new,
        progress=progress,
    )
    msg = (
        f"Run #{run_id}: status **{summary.status}** · attempted {summary.attempted} · "
        f"succeeded {summary.succeeded} · failed {summary.failed} · remaining {summary.remaining}"
    )
    if summary.stopped_reason:
        st.error(msg + f" · stopped: {summary.stopped_reason}")
    elif summary.failed:
        st.warning(msg)
    else:
        st.success(msg)


tab_new, tab_resume, tab_results = st.tabs(
    ["New run", "Resume incomplete run", "Run results & export"]
)

with tab_new:
    if require_write_ui(ctx):
        with ctx.session() as s:
            datasets = [d for d in list_datasets(s) if d.name != "playground"]
            q_cfgs = cs.list_configs(s, "jev_questions")
            r_cfgs = cs.list_configs(s, "routing")
            pricing_row = cs.active_config(s, "pricing")
            ids = {
                t: (c.id if (c := cs.active_config(s, t)) else None)
                for t in ("policy", "label_mapping")
            }
        if not datasets:
            st.info(
                "No datasets. Run `uv run python scripts/bootstrap.py` and/or "
                "`uv run python scripts/prepare_dataset.py`."
            )
            st.stop()
        ds = st.selectbox(
            "Dataset",
            datasets,
            format_func=lambda d: f"{d.name} {d.version} ({d.data_origin}, {d.record_count} cases)",
        )
        cohort = st.selectbox(
            "Cohort (source split)",
            ["development", "representative_dev", "holdout"],
            help="Development = validation split sample; holdout = untouched test split sample.",
        )
        provider = st.radio(
            "Model provider",
            [JEV, MODERATION],
            horizontal=True,
            format_func=lambda p: {
                "typesafe_jev": "TypeSafe Jev (primary)",
                "openai_moderation": "OpenAI Moderation (benchmark)",
            }[p],
        )
        is_synthetic = ds.data_origin == "synthetic"
        origin = "synthetic_fixture" if is_synthetic else "live"
        if is_synthetic:
            st.info(
                "Synthetic demo dataset: the run replays synthetic fixture outputs (no provider calls)."
            )
        c1, c2 = st.columns(2)
        q_cfg = c1.selectbox(
            "Question configuration",
            q_cfgs,
            format_func=lambda c: c.version,
            disabled=provider != JEV,
        )
        r_cfg = c2.selectbox(
            "Routing configuration",
            r_cfgs,
            format_func=lambda c: f"{c.version} ({c.content_json.get('tuning_status')})",
        )
        input_mode = "text_only"
        if provider == JEV:
            input_mode = st.radio(
                "Input representation",
                ["text_only", "with_context"],
                horizontal=True,
                help="Use text_only for comparisons with OpenAI Moderation (input parity).",
            )
        c3, c4, c5 = st.columns(3)
        with ctx.session() as s:
            n_cohort = len(cohort_cases(s, ds.id, cohort))
        max_cases = c3.number_input("Maximum case count", 1, max(n_cohort, 1), max(n_cohort, 1))
        batch_size = c4.number_input("Batch size (commit interval)", 1, 500, 25)
        label = c5.text_input("Evaluation run label", f"{provider} {ds.name} {cohort}")
        model = ctx.settings.jev_model if provider == JEV else ctx.settings.openai_moderation_model
        ack_holdout = False
        if cohort == "holdout":
            st.warning(
                "Holdout evaluation consumes the holdout for the selected configuration. Do not use "
                "holdout results to retune. The routing configuration will be frozen."
            )
            ack_holdout = st.checkbox(
                "I acknowledge that if this holdout was already consumed, results are exploratory."
            )
        confirmed = True
        if origin == "live":
            est = ""
            if pricing_row and provider == JEV:
                price = PricingConfig.model_validate(pricing_row.content_json).models.get(model)
                if price:
                    est = f" Estimated Jev cost ≈ \\${int(max_cases) * 900 * price.input_per_million_usd / 1e6:.4f} (≈900 input tokens/case; estimate only)."
            st.warning(
                f"This run will make up to {int(max_cases)} paid API calls to `{model}`.{est}"
            )
            confirmed = st.checkbox("I confirm paid provider inference for this run.")
            if not ctx.settings.live_calls_allowed:
                st.error("Live calls are disabled in this mode (set APP_MODE=local_live).")
        if st.button("Create and start run", type="primary", disabled=not confirmed):
            try:
                with ctx.session() as s:
                    run = create_run(
                        s,
                        name=label,
                        provider=provider,
                        dataset_id=ds.id,
                        cohort=cohort,
                        requested_model=model,
                        config_ids={
                            "policy": ids["policy"],
                            "jev_questions": q_cfg.id if provider == JEV else None,
                            "routing": r_cfg.id,
                            "label_mapping": ids["label_mapping"],
                        },
                        data_origin=origin,
                        input_mode=input_mode,
                        max_cases=int(max_cases),
                        acknowledge_holdout_reuse=ack_holdout,
                    )
                    s.commit()
                    run_id = run.id
            except (HoldoutReuseError, ValueError) as exc:
                st.error(str(exc))
            else:
                execute(run_id, origin, int(batch_size), None)

with tab_resume:
    with ctx.session() as s:
        incomplete = s.scalars(
            select(EvaluationRun).where(
                EvaluationRun.status.in_(("created", "running", "partial", "failed")),
                EvaluationRun.cohort_name != "playground",
            )
        ).all()
        remaining = {r.id: len(remaining_case_ids(s, r)) for r in incomplete}
    if not incomplete:
        st.info("No incomplete runs.")
    else:
        run = st.selectbox(
            "Run", incomplete, format_func=lambda r: f"{run_label(r)} — {remaining[r.id]} remaining"
        )
        st.caption(
            f"Completed {run.completed_cases} · failed {run.failed_cases} · remaining {remaining[run.id]} · "
            f"errors {run.error_summary_json}"
        )
        origin = run.metadata_json.get("data_origin", "live")
        if require_write_ui(ctx):
            ok = True
            if origin == "live":
                ok = st.checkbox(f"I confirm up to {remaining[run.id]} paid API calls.")
            max_new = st.number_input(
                "Process at most", 1, max(remaining[run.id], 1), max(remaining[run.id], 1)
            )
            if st.button("Resume run", disabled=not ok):
                execute(run.id, origin, 25, int(max_new))

with tab_results:
    with ctx.session() as s:
        runs = s.scalars(
            select(EvaluationRun)
            .where(EvaluationRun.cohort_name != "playground")
            .order_by(EvaluationRun.id.desc())
        ).all()
    if not runs:
        st.info("No runs yet.")
    else:
        run = st.selectbox("Run ", runs, format_func=run_label)
        origin_notice(run)
        c = st.columns(4)
        c[0].metric("Status", run.status)
        c[1].metric("Completed", f"{run.completed_cases}/{run.total_cases}")
        c[2].metric("Errors", run.failed_cases)
        with ctx.session() as s:
            from sqlalchemy import func

            from harm_detection.persistence.models import Prediction

            cost = s.scalar(
                select(func.sum(Prediction.estimated_cost_usd)).where(Prediction.run_id == run.id)
            )
            report = run_report(s, run.id)
            csv = cases_csv(s, run.id)
        c[3].metric(
            "Estimated provider cost", f"${cost:.4f}" if cost is not None else "unavailable"
        )
        if run.error_summary_json:
            st.caption(f"Error summary: {run.error_summary_json}")
        st.download_button(
            "Export report (JSON)",
            json.dumps(report, indent=2, default=str),
            file_name=f"run_{run.id}_report.json",
            mime="application/json",
        )
        st.download_button(
            "Export cases (CSV)", csv, file_name=f"run_{run.id}_cases.csv", mime="text/csv"
        )
        st.page_link(
            "pages/model_performance.py", label="Open metrics in Model Performance", icon="📈"
        )

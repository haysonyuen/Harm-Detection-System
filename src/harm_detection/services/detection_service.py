"""Single-case detection for the Playground (live inference or recorded fixture playback)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select

from harm_detection.persistence.models import Configuration, EvaluationRun, IntegrationEvent
from harm_detection.providers.fixture import FixtureReplayProvider
from harm_detection.providers.jev import JevProvider
from harm_detection.routing.engine import route
from harm_detection.schemas.config import JevQuestionConfig, PricingConfig, RoutingConfig
from harm_detection.schemas.content import ContentInput, ContentSource
from harm_detection.schemas.detection import DetectionResult, DetectionStatus, RoutingDecision
from harm_detection.services import config_service as cs
from harm_detection.services.context import AppContext
from harm_detection.services.dataset_service import get_dataset, import_cases
from harm_detection.services.run_service import JEV, record_prediction

PLAYGROUND_DATASET = ("playground", "v1")


@dataclass
class PlaygroundResult:
    item: ContentInput
    detection: DetectionResult
    routing: RoutingDecision
    saved_case_id: str | None
    saved_run_id: int | None


def make_jev_provider(ctx: AppContext, questions: JevQuestionConfig) -> JevProvider:
    ctx.require_live("Jev inference")
    return JevProvider(
        ctx.settings.typesafe_api_key,
        questions,
        model=ctx.settings.jev_model,
        base_url=ctx.settings.jev_base_url,
    )


def _playground_run(session, ds_id: int, ids: dict[str, int], model: str) -> EvaluationRun:
    for run in session.scalars(
        select(EvaluationRun).where(
            EvaluationRun.dataset_id == ds_id, EvaluationRun.cohort_name == "playground"
        )
    ):
        if (
            run.question_config_id == ids["jev_questions"]
            and run.routing_config_id == ids["routing"]
            and run.requested_model == model
        ):
            return run
    run = EvaluationRun(
        name=f"Playground ({model})",
        provider=JEV,
        requested_model=model,
        dataset_id=ds_id,
        cohort_name="playground",
        policy_config_id=ids["policy"],
        question_config_id=ids["jev_questions"],
        routing_config_id=ids["routing"],
        label_mapping_config_id=ids["label_mapping"],
        status="running",
        metadata_json={
            "planned_case_ids": [],
            "data_origin": "live",
            "input_mode": "with_context",
            "playground": True,
        },
    )
    session.add(run)
    session.flush()
    return run


def analyze_live(
    ctx: AppContext,
    text: str,
    parent_text: str | None,
    target_context: str | None,
    *,
    provider: JevProvider | None = None,
) -> PlaygroundResult:
    """Validate input, call Jev, route, and persist. Missing access yields a failure, never a fabricated result."""
    ctx.require_writable("Playground analysis")
    item = ContentInput(
        case_id=f"manual-{uuid.uuid4().hex[:12]}",
        source=ContentSource.MANUAL,
        text=text,
        parent_text=parent_text,
        target_context=target_context,
        source_dataset="playground",
    )
    item.check_length(ctx.settings.max_text_length)
    with ctx.session() as s:
        ids = {
            t: cs.active_config(s, t).id
            for t in ("policy", "jev_questions", "routing", "label_mapping")
        }
        questions = JevQuestionConfig.model_validate(
            s.get(Configuration, ids["jev_questions"]).content_json
        )
        routing_cfg = RoutingConfig.model_validate(
            s.get(Configuration, ids["routing"]).content_json
        )
        pricing = PricingConfig.model_validate(cs.active_config(s, "pricing").content_json)
    provider = provider or make_jev_provider(ctx, questions)
    detection = provider.detect(item)
    decision = route(
        item.case_id,
        detection.signals,
        routing_cfg,
        detection_succeeded=detection.status == DetectionStatus.SUCCESS,
    )

    with ctx.session() as s:
        s.add(
            IntegrationEvent(
                provider=JEV,
                outcome=detection.status.value,
                error_code=(detection.error or {}).get("code"),
                message=(detection.error or {}).get("message"),
            )
        )
        ds = get_dataset(s, *PLAYGROUND_DATASET)
        ds, _ = import_cases(
            s,
            name=PLAYGROUND_DATASET[0],
            version=PLAYGROUND_DATASET[1],
            records=[{**item.model_dump(mode="json"), "split": "playground"}],
            source_url="interactive",
            license="operator-entered text",
            data_origin="manual",
            max_text_length=ctx.settings.max_text_length,
        )
        run = _playground_run(s, ds.id, ids, provider.model)
        run.metadata_json = {
            **run.metadata_json,
            "planned_case_ids": [*run.metadata_json.get("planned_case_ids", []), item.case_id],
        }
        run.total_cases += 1
        record_prediction(s, run, detection, pricing=pricing, routing_config=routing_cfg)
        if detection.status == DetectionStatus.SUCCESS:
            run.completed_cases += 1
            run.resolved_model = detection.resolved_model
        else:
            run.failed_cases += 1
        s.commit()
        run_id = run.id
    return PlaygroundResult(item, detection, decision, item.case_id, run_id)


def playback_fixture(ctx: AppContext, case: dict, routing_cfg: RoutingConfig) -> PlaygroundResult:
    """Offline playback of a recorded fixture result (clearly labeled; nothing is invented)."""
    item = ContentInput.model_validate(
        {
            k: case.get(k)
            for k in (
                "case_id",
                "source",
                "text",
                "language",
                "parent_text",
                "target_context",
                "source_dataset",
                "source_record_id",
                "metadata",
            )
        }
    )
    detection = FixtureReplayProvider(JEV).detect(item)
    decision = route(
        item.case_id,
        detection.signals,
        routing_cfg,
        detection_succeeded=detection.status == DetectionStatus.SUCCESS,
    )
    return PlaygroundResult(item, detection, decision, None, None)

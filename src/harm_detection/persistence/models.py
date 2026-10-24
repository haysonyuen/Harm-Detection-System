"""SQLAlchemy ORM models (PRD §14)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_dataset_name_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(100))
    source_url: Mapped[str | None] = mapped_column(String(500))
    revision: Mapped[str | None] = mapped_column(String(200))
    license: Mapped[str | None] = mapped_column(String(200))
    manifest_path: Mapped[str | None] = mapped_column(String(500))
    checksum: Mapped[str | None] = mapped_column(String(64))
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    data_origin: Mapped[str] = mapped_column(String(50), default="public_dataset")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class DatasetCase(Base):
    __tablename__ = "dataset_cases"
    __table_args__ = (
        UniqueConstraint("dataset_id", "case_id", name="uq_case_per_dataset"),
        UniqueConstraint("dataset_id", "source_record_id", name="uq_source_record_per_dataset"),
        Index("ix_dataset_cases_hash", "dataset_id", "content_hash"),
        Index("ix_dataset_cases_split", "dataset_id", "split"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    case_id: Mapped[str] = mapped_column(String(200))
    source_record_id: Mapped[str | None] = mapped_column(String(200))
    split: Mapped[str] = mapped_column(String(50))
    stratum: Mapped[str | None] = mapped_column(String(50))
    content_hash: Mapped[str] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text)
    parent_text: Mapped[str | None] = mapped_column(Text)
    target_context: Mapped[str | None] = mapped_column(Text)
    reference_labels_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Configuration(Base):
    __tablename__ = "configurations"
    __table_args__ = (
        UniqueConstraint("config_type", "version", name="uq_config_type_version"),
        Index("ix_config_type_active", "config_type", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_type: Mapped[str] = mapped_column(String(50))
    version: Mapped[str] = mapped_column(String(100))
    content_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    checksum: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    parent_config_id: Mapped[int | None] = mapped_column(ForeignKey("configurations.id"))
    change_note: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_frozen: Mapped[bool] = mapped_column(Boolean, default=False)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(300))
    provider: Mapped[str] = mapped_column(String(50))
    requested_model: Mapped[str | None] = mapped_column(String(100))
    resolved_model: Mapped[str | None] = mapped_column(String(100))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    cohort_name: Mapped[str] = mapped_column(String(100))
    policy_config_id: Mapped[int | None] = mapped_column(ForeignKey("configurations.id"))
    question_config_id: Mapped[int | None] = mapped_column(ForeignKey("configurations.id"))
    routing_config_id: Mapped[int | None] = mapped_column(ForeignKey("configurations.id"))
    label_mapping_config_id: Mapped[int | None] = mapped_column(ForeignKey("configurations.id"))
    status: Mapped[str] = mapped_column(String(20), default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    total_cases: Mapped[int] = mapped_column(Integer, default=0)
    completed_cases: Mapped[int] = mapped_column(Integer, default=0)
    failed_cases: Mapped[int] = mapped_column(Integer, default=0)
    error_summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("run_id", "case_id", name="uq_prediction_run_case"),
        Index("ix_predictions_run_status", "run_id", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("evaluation_runs.id", ondelete="CASCADE"))
    case_id: Mapped[str] = mapped_column(String(200))
    provider: Mapped[str] = mapped_column(String(50))
    provider_response_id: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20))
    raw_response_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    normalized_signals_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    resolved_model: Mapped[str | None] = mapped_column(String(100))
    data_origin: Mapped[str] = mapped_column(String(50), default="live")
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float)
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RoutingDecisionRow(Base):
    __tablename__ = "routing_decisions"
    __table_args__ = (
        UniqueConstraint("prediction_id", "routing_config_id", name="uq_routing_prediction_config"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id", ondelete="CASCADE"))
    routing_config_id: Mapped[int] = mapped_column(ForeignKey("configurations.id"))
    disposition: Mapped[str | None] = mapped_column(String(30))
    triggered_rules_json: Mapped[list[Any]] = mapped_column(JSON, default=list)
    reason_codes_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    context_warning: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"
    __table_args__ = (Index("ix_metrics_run_routing", "run_id", "routing_config_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("evaluation_runs.id", ondelete="CASCADE"))
    routing_config_id: Mapped[int | None] = mapped_column(ForeignKey("configurations.id"))
    metric_name: Mapped[str] = mapped_column(String(100))
    category: Mapped[str] = mapped_column(String(100))
    cohort: Mapped[str] = mapped_column(String(100))
    value: Mapped[float | None] = mapped_column(Float)
    numerator: Mapped[float | None] = mapped_column(Float)
    denominator: Mapped[float | None] = mapped_column(Float)
    confidence_interval_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    calculation_version: Mapped[str] = mapped_column(String(50))


class OptimizationExperiment(Base):
    __tablename__ = "optimization_experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_run_id: Mapped[int] = mapped_column(ForeignKey("evaluation_runs.id"))
    constraints_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    search_space_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    candidate_results_json: Mapped[list[Any]] = mapped_column(JSON)
    selected_config_id: Mapped[int | None] = mapped_column(ForeignKey("configurations.id"))
    selection_rationale: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default="searched")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AnalysisRequest(Base):
    __tablename__ = "analysis_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model: Mapped[str] = mapped_column(String(100))
    source_run_id: Mapped[int | None] = mapped_column(ForeignKey("evaluation_runs.id"))
    case_ids_json: Mapped[list[Any]] = mapped_column(JSON)
    input_snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    structured_analysis_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(30))
    error_message: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ImprovementRecommendation(Base):
    __tablename__ = "improvement_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_request_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_requests.id", ondelete="CASCADE")
    )
    recommendation_key: Mapped[str] = mapped_column(String(100))
    change_type: Mapped[str] = mapped_column(String(50))
    affected_component: Mapped[str] = mapped_column(String(100))
    proposed_change_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(40), default="proposed")
    human_decision_rationale: Mapped[str | None] = mapped_column(Text)
    candidate_config_id: Mapped[int | None] = mapped_column(ForeignKey("configurations.id"))
    validation_run_id: Mapped[int | None] = mapped_column(ForeignKey("evaluation_runs.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(100))
    entity_type: Mapped[str] = mapped_column(String(100))
    entity_id: Mapped[str] = mapped_column(String(100))
    actor: Mapped[str] = mapped_column(String(100), default="local_operator")
    before_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    after_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class IntegrationEvent(Base):
    """Recent provider integration outcomes for the System Status page (no secrets)."""

    __tablename__ = "integration_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(50))
    outcome: Mapped[str] = mapped_column(String(30))
    error_code: Mapped[str | None] = mapped_column(String(100))
    message: Mapped[str | None] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

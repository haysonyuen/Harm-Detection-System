"""Canonical content input contract (PRD §7.1)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_MAX_TEXT_LENGTH = 5000


class ContentSource(StrEnum):
    MANUAL = "manual"
    PUBLIC_DATASET = "public_dataset"
    X_TIMELINE = "x_timeline"  # reserved for the Phase 2 connector; not ingested in MVP v1


class ContentInput(BaseModel):
    """A single item to classify. Missing context stays ``None``; it is never invented."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=200)
    source: ContentSource
    text: str
    language: str = "en"
    parent_text: str | None = None
    target_context: str | None = None
    source_dataset: str | None = None
    source_record_id: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("case_id")
    @classmethod
    def _case_id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("case_id must not be blank")
        return v.strip()

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must be non-empty after trimming")
        return v.strip()

    @field_validator("parent_text", "target_context")
    @classmethod
    def _blank_context_is_none(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @field_validator("language")
    @classmethod
    def _english_only(cls, v: str) -> str:
        if v.strip().lower() != "en":
            raise ValueError("only English-language ('en') content is supported in MVP v1")
        return "en"

    def check_length(self, max_length: int = DEFAULT_MAX_TEXT_LENGTH) -> None:
        """Raise ``ValueError`` if the text exceeds the configured maximum length."""
        if len(self.text) > max_length:
            raise ValueError(
                f"text length {len(self.text)} exceeds maximum of {max_length} characters"
            )

    @property
    def has_context(self) -> bool:
        return self.parent_text is not None or self.target_context is not None


def validate_content(
    data: dict[str, Any], max_length: int = DEFAULT_MAX_TEXT_LENGTH
) -> ContentInput:
    item = ContentInput.model_validate(data)
    item.check_length(max_length)
    return item

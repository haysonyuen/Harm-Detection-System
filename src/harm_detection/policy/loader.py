"""Load versioned YAML configuration files and compute content checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from harm_detection.schemas.config import (
    JevQuestionConfig,
    LabelMappingConfig,
    PolicyConfig,
    PricingConfig,
    RoutingConfig,
)
from harm_detection.settings import CONFIG_DIR

CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "policy": PolicyConfig,
    "jev_questions": JevQuestionConfig,
    "label_mapping": LabelMappingConfig,
    "routing": RoutingConfig,
    "pricing": PricingConfig,
}

DEFAULT_CONFIG_FILES: dict[str, Path] = {
    "policy": CONFIG_DIR / "policies" / "harassment_v1.yaml",
    "jev_questions": CONFIG_DIR / "models" / "jev_questions_v1.yaml",
    "label_mapping": CONFIG_DIR / "label_mappings" / "civil_comments_proxy_v1.yaml",
    "routing": CONFIG_DIR / "routing" / "routing_v1.yaml",
    "pricing": CONFIG_DIR / "pricing" / "pricing_v1.yaml",
}


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def checksum(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return data


def parse_config(config_type: str, content: dict[str, Any]) -> BaseModel:
    try:
        model = CONFIG_MODELS[config_type]
    except KeyError as exc:
        raise ValueError(f"unknown config type {config_type!r}") from exc
    return model.model_validate(content)


def load_config_file(
    config_type: str, path: Path | None = None
) -> tuple[BaseModel, dict[str, Any]]:
    """Return the validated config model and its raw content dict."""
    path = path or DEFAULT_CONFIG_FILES[config_type]
    content = load_yaml(path)
    return parse_config(config_type, content), content

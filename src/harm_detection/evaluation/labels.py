"""Versioned proxy label mapping (PRD §9.4) and proxy reference routing (§10.4)."""

from __future__ import annotations

from typing import Any

from harm_detection.schemas.config import LabelMappingConfig
from harm_detection.schemas.detection import Disposition


def proxy_labels(
    reference_labels: dict[str, Any], mapping: LabelMappingConfig
) -> dict[str, bool | None]:
    """Binarize each mapped annotation. Missing / non-numeric annotations map to ``None``.

    Proxies are multilabel: several may be positive for the same case.
    """
    out: dict[str, bool | None] = {}
    for proxy, field in mapping.proxies.items():
        v = reference_labels.get(field)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            out[proxy] = None
        else:
            out[proxy] = float(v) >= mapping.binarization_threshold
    return out


def reference_route(
    labels: dict[str, bool | None], mapping: LabelMappingConfig
) -> Disposition | None:
    """Threat proxy -> Priority; else insult proxy -> Standard; else No Flag.

    Benchmarking proxy only -- not an adjudicated review priority.
    """
    pri = labels.get(mapping.reference_routing["priority_review"])
    std = labels.get(mapping.reference_routing["standard_review"])
    if pri is None or std is None:
        return None
    if pri:
        return Disposition.PRIORITY_REVIEW
    if std:
        return Disposition.STANDARD_REVIEW
    return Disposition.NO_FLAG

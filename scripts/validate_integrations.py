"""Minimal, explicitly authorized live contract checks for external providers.

Each check makes at most one or two small requests and validates that the real response
maps onto the adapter's normalized contract. Results (no secrets) are written to
docs/integration_validation.json.

Usage:
  uv run python scripts/validate_integrations.py --jev --confirm
  uv run python scripts/validate_integrations.py --moderation --confirm
  uv run python scripts/validate_integrations.py --analyst --confirm   # ~1 small GPT call
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from harm_detection.persistence.models import IntegrationEvent
from harm_detection.policy.loader import load_config_file
from harm_detection.providers.base import ProviderError
from harm_detection.providers.jev import JevProvider
from harm_detection.providers.openai_analyst import OpenAIAnalyst
from harm_detection.providers.openai_moderation import OpenAIModerationProvider
from harm_detection.schemas.content import ContentInput
from harm_detection.services.context import create_context
from harm_detection.settings import PROJECT_ROOT

PROBE_TEXT = "I disagree with your take on the zoning vote, but I see where you're coming from."


def check_jev(settings) -> dict:
    questions, _ = load_config_file("jev_questions")
    out = {"provider": "typesafe_jev", "requested_model": settings.jev_model}
    if not settings.typesafe_api_key:
        return {**out, "status": "skipped", "reason": "TYPESAFE_API_KEY not configured"}
    jev = JevProvider(
        settings.typesafe_api_key,
        questions,
        model=settings.jev_model,
        base_url=settings.jev_base_url,
    )
    try:
        models = jev.list_models()
        out["models_endpoint"] = {"ok": True, "names": [m.get("name") for m in models]}
    except ProviderError as err:
        out["models_endpoint"] = {"ok": False, "code": err.code, "message": err.message}
    res = jev.detect(ContentInput(case_id="contract-probe-1", source="manual", text=PROBE_TEXT))
    out["systemone"] = {
        "status": res.status.value,
        "resolved_model": res.resolved_model,
        "signals": res.signals,
        "usage": res.provider_usage,
        "latency_ms": res.latency_ms,
        "error": res.error,
        "answer_keys": sorted((res.raw_response or {}).get("answers", {}).keys()),
        "answer_types": sorted(
            {a.get("type") for a in (res.raw_response or {}).get("answers", {}).values()}
        ),
    }
    # Unsupported model handling check (expected to fail cleanly, never fabricate a result).
    bad = JevProvider(
        settings.typesafe_api_key,
        questions,
        model="jev-does-not-exist",
        base_url=settings.jev_base_url,
    )
    bad_res = bad.detect(ContentInput(case_id="contract-probe-2", source="manual", text=PROBE_TEXT))
    out["unsupported_model_probe"] = {"status": bad_res.status.value, "error": bad_res.error}
    out["status"] = "verified" if res.status.value == "success" else "failed"
    return out


def check_moderation(settings) -> dict:
    out = {"provider": "openai_moderation", "requested_model": settings.openai_moderation_model}
    if not settings.openai_api_key:
        return {**out, "status": "skipped", "reason": "OPENAI_API_KEY not configured"}
    mod = OpenAIModerationProvider(settings.openai_api_key, model=settings.openai_moderation_model)
    res = mod.detect(ContentInput(case_id="contract-probe-3", source="manual", text=PROBE_TEXT))
    out.update(
        status="verified" if res.status.value == "success" else "failed",
        resolved_model=res.resolved_model,
        signals=res.signals,
        error=res.error,
    )
    return out


def check_analyst(settings) -> dict:
    out = {"provider": "openai_analyst", "requested_model": settings.openai_analyst_model}
    if not settings.openai_api_key:
        return {**out, "status": "skipped", "reason": "OPENAI_API_KEY not configured"}
    analyst = OpenAIAnalyst(settings.openai_api_key, model=settings.openai_analyst_model)
    data = json.dumps(
        {
            "cases": [
                {
                    "case_id": "probe-1",
                    "comment": PROBE_TEXT,
                    "proxy_labels": {"insult_proxy": False},
                    "predicted_disposition": "standard_review",
                    "triggered_rule_ids": ["R-CTX-01"],
                }
            ]
        }
    )
    try:
        r = analyst.analyze(
            f"<<<UNTRUSTED_DATA_START>>>\n{data}\n<<<UNTRUSTED_DATA_END>>>", max_output_tokens=1500
        )
        out.update(
            status="verified",
            resolved_model=r.model,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            n_recommendations=len(r.output.recommendations),
        )
    except ProviderError as err:
        out.update(status="failed", error={"code": err.code, "message": err.message})
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--jev", action="store_true")
    p.add_argument("--moderation", action="store_true")
    p.add_argument("--analyst", action="store_true")
    p.add_argument("--confirm", action="store_true", help="authorize the small live requests")
    args = p.parse_args()
    if not args.confirm:
        print("Live checks make real (small, possibly billed) API requests. Re-run with --confirm.")
        return 1
    ctx = create_context()
    ctx.require_live("integration validation")
    results = []
    if args.jev:
        results.append(check_jev(ctx.settings))
    if args.moderation:
        results.append(check_moderation(ctx.settings))
    if args.analyst:
        results.append(check_analyst(ctx.settings))
    report_path = PROJECT_ROOT / "docs" / "integration_validation.json"
    existing = json.loads(report_path.read_text()) if report_path.exists() else {}
    for r in results:
        existing[r["provider"]] = {
            **r,
            "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    report_path.write_text(json.dumps(existing, indent=2, default=str))
    with ctx.session() as s:
        for r in results:
            s.add(
                IntegrationEvent(
                    provider=r["provider"],
                    outcome=r["status"],
                    message=f"contract validation: {r['status']}",
                )
            )
        s.commit()
    print(json.dumps(results, indent=2, default=str))
    return 0 if all(r["status"] in ("verified", "skipped") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

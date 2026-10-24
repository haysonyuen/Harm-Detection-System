# Architecture

## Layers

| Layer | Location | Rules |
|---|---|---|
| UI | `app/main.py`, `app/pages/*`, `app/components/common.py` | Streamlit only; calls services; never parses provider JSON |
| Services | `src/harm_detection/services/` | Orchestration, persistence, mode guards (`AppContext.require_writable/require_live`) |
| Domain (pure) | `routing/engine.py`, `evaluation/metrics.py`, `evaluation/labels.py`, `optimization/*`, `improvement/*`, `datasets/leakage.py` | No I/O, no Streamlit, no network; unit-tested directly |
| Providers | `providers/jev.py`, `openai_moderation.py`, `openai_analyst.py`, `fixture.py` | Convert real provider responses to normalized contracts; never fabricate results |
| Persistence | `persistence/models.py`, `persistence/db.py`, `migrations/` | SQLAlchemy 2 + Alembic; SQLite |
| Config | `config/**.yaml`, `policy/loader.py`, `schemas/config.py` | Versioned YAML seeded into the `configurations` table |

## Key flows

**Single case (Playground, live):** `ContentInput` validation → `JevProvider.detect` (one System One request with
4 `noul` questions) → `normalize_response` (strict: missing/out-of-range answers ⇒ failure) →
`routing.engine.route` → persisted as a prediction + routing decision in a per-configuration playground run.
Offline, arbitrary text gets **no** prediction; only recorded fixture cases can be played back, labeled as such.

**Batch run:** `create_run` snapshots config IDs/checksums and the planned case list (cohort hash) →
`execute_run` processes cases sequentially, committing per batch, skipping cases with successful predictions on
resume, retrying failures, and stopping on fatal errors (auth, missing credentials, unsupported model) or a
failure rate above 50% → `compute_and_store_metrics`.

**Rerouting:** `reroute_run` creates a derived run that re-applies a routing config to stored predictions
(`prediction_source_run_id`). Used by threshold experiments, recommendation experiments, and holdout evaluation
of a selected configuration. No provider calls.

**Holdout integrity:** the first holdout evaluation per dataset freezes the routing config and is labeled final;
later ones raise `HoldoutReuseError` unless explicitly acknowledged and are labeled `exploratory`. Optimization
and AI analysis refuse holdout runs.

## Contracts

* `ContentInput` (PRD §7.1) — `schemas/content.py`
* `DetectionResult` (PRD §7.2) — `schemas/detection.py`; success ⇔ signals present; failure ⇔ error present
* `RoutingDecision` (PRD §7.3) — failure states (`detection_failed`, `routing_failed`, `not_evaluated`) are a
  separate `status` and can never carry a disposition.

## Jev integration (verified against documentation)

Contract source: `https://api.typesafe.ai/openapi.json` (API version 0.2.0), inspected 2026-10-23.

* `POST /v1/systemone` with `{"model", "state", "questions"}`; `Authorization: Bearer <key>`.
* `state` is sent as an object: `note` (untrusted-data preamble), `comment`, optional `parent_comment` /
  `target_context` (omitted when absent), and `context_available` when no context is supplied.
* Each question: `{"type": "noul", "instructions": ..., "criteria": {"true": ..., "false": ...}}`.
* Response: `{"model": <versioned id>, "answers": {name: {"type": "noul", "noul": p}}, "usage": {...}}`.
  The resolved versioned model ID is stored on each prediction.
* Errors 400/401/403/404/413/415/422/429/529/5xx mapped to an error taxonomy; 429/529/5xx/timeouts/network
  errors retried with bounded exponential backoff honoring `retry-after`.

Live verification status is recorded in `docs/integration_validation.json` after running
`scripts/validate_integrations.py`.

## Security

* Secrets come only from the environment / `.env` (git-ignored); `Settings.__repr__` hides them; public demo
  ignores them entirely.
* Stored raw responses pass through `security/redaction.py`; the public exporter drops raw responses.
* Comment text is treated as untrusted: Jev state carries a preamble telling the model not to follow embedded
  instructions; the analyst prompt isolates data between `UNTRUSTED_DATA` markers; nothing in content can change
  thresholds, trigger calls, or write files.

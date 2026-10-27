# Known limitations

## Measurement

* Proxy labels (Civil Comments threat/insult ≥ 0.5) are not X policy decisions, and *insult* is a different
  construct from the harassment rubric.
* Threat positives are rare. The representative development subset has 0 and the holdout 4, so threat recall
  outside the enriched sample is undefined or very uncertain.
* Review volumes come from Civil Comments, not platform traffic. Enriched rates are inflated by design.
* Only Civil Comments is implemented. The Wikipedia Toxicity Subtypes OOD benchmark is not.

## Detection

* Jev returns probabilities only. The lab never presents rationales or evidence spans as Jev's reasoning.
* `jev-latest` is an alias that can move to a new model version. Store and pin the resolved versioned ID before
  relying on tuned thresholds.
* Single comments lack conversation history, relationships, prior violations, and coordination signals.
  Sarcasm, coded threats and cultural nuance remain hard.
* `routing_v1` thresholds are untuned placeholders.
* A No Flag result does not guarantee content is safe. Priority Review is a recommendation for human inspection,
  not proof of a threat.

## Integrations (status at delivery)

* **TypeSafe Jev:** the adapter is built against the official OpenAPI contract and contract-tested with mocks.
  **Live contract verified on 2026-10-27** (`scripts/validate_integrations.py --jev --confirm`, result in
  `docs/integration_validation.json`): `/v1/models` listed `jev-latest` and `jev-preview`; one System One call
  with `jev-latest` resolved to `jev-1.14.0` and returned all 4 `noul` answers; an unknown model ID returned
  HTTP 400 "Unknown model", which mapped to `unsupported_model`. This verifies the API contract only. No live
  benchmark results exist yet, so detection quality is unmeasured.
* **OpenAI Moderation and GPT-5.4 mini:** built on the official SDK (v3.x) and tested against SDK response
  types with injected fakes. **Live verification is pending** (no `OPENAI_API_KEY` configured).
* Error mapping for an unsupported Jev model ID (404, or 400/422 mentioning `model`) is inferred from the
  documented error table. The 2026-10-27 live probe confirmed the 400 case. The 404 and 422 branches remain
  untested against the live API.
* GPT-5.4 mini pricing in `config/pricing/pricing_v1.yaml` came from third-party aggregators and has not been
  verified against OpenAI's pricing page. Cost controls are local estimates, not billing caps.

## Product

* Single local operator: there is no authentication or multi-user tenancy. The audit actor is `local_operator`.
* Batch execution is sequential and runs in the Streamlit request. Very large runs are better started from the CLI
  (`scripts/run_benchmark.py`).
* Screenshots are not included in the repository.

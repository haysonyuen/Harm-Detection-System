# Implementation decisions

Choices made where the PRD left room, with the reasoning for each.

| # | Decision | Why |
|---|---|---|
| D1 | Call Jev over plain HTTPX instead of `typesafe-sdk` | Full control of error mapping and retries; contract tests with `respx`; adapter targets the documented OpenAPI schema directly. |
| D2 | One System One request per case with 4 `noul` questions and `criteria.true/false` | Documented primitive for yes/no probabilities; one request keeps cost and latency low and avoids signal drift between calls. |
| D3 | Jev `state` sent as an object with an untrusted-data `note` | Lets the model distinguish comment from context and tells it not to follow embedded instructions. Absent context fields are omitted (never invented). |
| D4 | Category predictions derived from routing rule matches (R-PRI-01 → threat, R-STD-01 → harassment) | Ties category metrics to the actual routing configuration under test, so threshold changes move both consistently. |
| D5 | Under-prioritization = P(pred rank < proxy rank \| proxy flagged); priority false-alarm = P(pred priority \| proxy not priority) | Gives two measures distinct from priority recall. Definitions are shown in the UI and docs. |
| D6 | Derived "reroute" runs reference source predictions (`prediction_source_run_id`) instead of copying them | Threshold changes never trigger model calls or duplicate predictions, and routing decisions are unique per (prediction, routing config). |
| D7 | First holdout evaluation freezes the routing config; later holdout runs need acknowledgement and are labeled exploratory | Implements "don't retune on holdout" without making it impossible to look again. |
| D8 | Moderation runs use native-flag proxy routing (threatening → priority, harassment → standard) | Comparison without inventing thresholds on uncalibrated scores. |
| D9 | Synthetic demo dataset has no representative stratum, so optimization falls back to all cases with a loud "NOT representative" warning | Keeps the offline demo functional without implying representativeness. |
| D10 | Public demo = separate exported SQLite opened read-only (`mode=ro`) | Defense in depth: app-level guards plus DB-level read-only. Works with ephemeral storage. |
| D11 | AI analysis: one call per request, ≤ 10 cases, cost pre-check uses a conservative token estimate plus reserved output tokens | Implements the $5 local target with a pre-call block. Actual usage is recorded after each call. |
| D12 | Evidence validation rejects the whole analysis on any unknown case/rule ID or non-substring quote | Simple and strict: no partially trusted AI output. |
| D13 | Accepting a recommendation creates an inactive candidate config the operator edits as YAML | The model proposes text and the human authors the actual config change. |
| D14 | Pricing for gpt-5.4-mini taken from aggregators and marked unverified | No authoritative source was accessible during the build. The rates are configurable. |
| D15 | Wikipedia Toxicity Subtypes not implemented | PRD marks it optional after the primary pipeline. Documented as a limitation. |
| D16 | Bootstrap counts implemented with numpy instead of sklearn in hot loops | sklearn's confusion_matrix overhead made CI computation slow. sklearn is still used for the 3-class matrix. |

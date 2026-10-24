# Policy mapping

**Independent simplified harassment detection rubric inspired by published X policies. Not official X
enforcement guidance.**

Source policies (accessed 2026-10-23):

* Abuse and harassment — https://help.x.com/en/rules-and-policies/abusive-behavior
* Violent content — https://help.x.com/en/rules-and-policies/violent-content

The rubric (`config/policies/harassment_v1.yaml`) paraphrases these into:

| Category | Meaning | Jev question(s) | Routing |
|---|---|---|---|
| **H** Harassment-related behavior | Degrading, humiliating or intimidating conduct directed at a person | `abusive_behavior` + `targeted_individual` | R-STD-01 → Standard Review |
| **T** Threatening harassment | Threats / intent / encouragement of physical harm to a person | `physical_threat` | R-PRI-01 → Priority Review (targeting not required) |
| **U** Uncertain / context-limited | Not enough context to judge intent (a state, not a category) | `context_sufficient` | R-CTX-01 → at least Standard Review *only when a risk signal is also present* |

Exclusions encoded in question instructions: criticism of ideas/institutions/conduct, ordinary disagreement,
reporting or condemning abuse, non-endorsing quotation, consensual banter established by context, fictional or
metaphorical language.

## Routing rules (`config/routing/routing_v1.yaml`)

Evaluated in order:

1. **R-ERR-01** — detection failed or a required signal is missing/invalid → non-success status (never No Flag).
2. **R-PRI-01** — `physical_threat ≥ threat_priority` → Priority Review.
3. **R-STD-01** — `abusive_behavior ≥ harassment_standard` and `targeted_individual ≥ targeting` → Standard Review.
4. **R-CTX-01** — `max(abusive_behavior, physical_threat) ≥ uncertain_risk` and `context_sufficient < context_sufficiency` → Standard Review.
5. **R-NF-01** — otherwise No Flag.

`routing_v1` thresholds (0.5 / 0.5 / 0.5 / 0.5 / 0.3) are **untuned engineering placeholders**.

## What single-comment classification cannot see

Repeated targeting over time, relationships, prior violations, account coordination, whether content was
welcome, and many forms of sarcasm, coded threats, and cultural nuance. The UI surfaces these limitations
instead of inferring them.

## Benchmark label mapping (`config/label_mappings/civil_comments_proxy_v1.yaml`)

* Threat proxy = Civil Comments `threat ≥ 0.5`; insult proxy = `insult ≥ 0.5` (multilabel).
* Proxy reference route: threat → Priority; else insult → Standard; else No Flag.
* These are **proxies** — "insult" is not the same construct as rubric category H, and crowd fractions are not
  adjudicated decisions. Results are reported as *agreement with published human annotations*, never as X policy
  accuracy.

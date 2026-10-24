# Evaluation methodology

## Dataset selection

Primary benchmark: **Google/Jigsaw Civil Comments** (`google/civil_comments`, Hugging Face commit
`f2970eb3a55777454c94069077cc8d9b5866312d`, CC0-1.0 per the dataset card). Comments are from ~50 English-language
news sites (2015–2017) with crowd-annotation *fractions* for toxicity, severe toxicity, obscenity, threat,
insult, identity attack, and sexually explicit content. Original fractions are preserved unmodified in
`reference_labels`.

The secondary Wikipedia Toxicity Subtypes benchmark (PRD §9.3) is **not implemented** in v1.

## Sampling (seed 20261023)

| Cohort | Source split | Method | n |
|---|---|---|---|
| Threat-enriched | validation | random among `threat ≥ 0.5` | 100 |
| Insult-enriched | validation | random among `insult ≥ 0.5 ∧ threat < 0.5` | 100 |
| Borderline / overlap | validation | random among remaining with a score in [0.3, 0.5) or both ≥ 0.5 | 50 |
| **Representative dev** | validation | uniform random among remaining eligible | 350 |
| **Holdout** | test | uniform random, **not stratified**, single seeded permutation | 400 |

Eligibility: non-empty text, ≤ 5,000 characters, first occurrence of each normalized-text hash per split.
Shortfalls are never reallocated; actual counts are in the manifest.

## Class prevalence (from `data/manifests/civil_comments_v1.json`)

| Cohort | n | threat proxy + | insult proxy + |
|---|---|---|---|
| Development (all) | 600 | 100 (16.7%) | 141 (23.5%) |
| Representative dev | 350 | **0 (0.0%)** | 22 (6.3%) |
| Holdout | 400 | **4 (1.0%)** | 30 (7.5%) |

Threat positives are rare in the natural distribution. The threat-enriched stratum also removes threat-positive
records before the representative draw, which makes the representative subset's 0 threat positives more likely.
Consequences:

* Threat-proxy **recall** on the representative subset is *unavailable* (zero denominator) and on the holdout
  rests on 4 cases. Its confidence interval is very wide. Threat recall therefore has to come mostly from the
  enriched development data, which is not a natural distribution.
* Review-volume estimates use the representative subset. Threat-driven priority volume there reflects false
  alarms only.

## Leakage prevention

Distinct source splits; no shared `source_record_id`; exact overlap by normalized SHA-256; near overlap by
word-token-set Jaccard ≥ 0.9 (≥ 4 tokens); a final leakage check fails preparation if anything overlaps; the
import layer re-checks holdout vs development; fixed seeds; dataset revision and file hashes recorded.

## Metrics

Per proxy category: TP/FP/TN/FN, precision, recall, F1, FPR, FNR, positive/negative counts. Routing proxy: priority
recall, under-prioritization (pred rank below proxy rank among proxy-flagged cases), priority false-alarm rate,
standard-review detection, total/priority/no-flag rates, proxy routing accuracy, 3×3 confusion matrix. Coverage:
planned cases, evaluable share, API failure rate, routing failures.

* Zero denominators are reported as **unavailable** (never 0 or 1).
* Metrics use only successful, evaluable predictions; failures are counted separately and never treated as No Flag.
* Jev category predictions: threat = R-PRI-01 matched; harassment = R-STD-01 matched.
  Moderation: native `harassment/threatening` and `harassment` flags.

## Statistical uncertainty

95% percentile bootstrap (1,000 resamples, seed 0) for precision, recall, F1, priority recall,
under-prioritization and total review rate. Resamples where a metric is undefined are skipped, and no interval
is reported when fewer than half are defined.

## Model comparison validity

Jev and OpenAI Moderation are compared on the **same cohort and case set with text-only input**
(`input_mode=text_only`). Context-rich Jev runs are reported separately. Runs are flagged non-comparable if the
dataset, cohort, case set, label mapping, input mode, or data origin differ, and the UI then suppresses deltas.
Raw scores are never compared across providers. OpenAI applies its own policy definitions.

## Threshold tuning

Grid search on stored **development** predictions only. Default objectives: maximize threat-proxy recall,
minimize representative review volume (optionally threat precision / under-prioritization). Constraints are
provisional. No candidate is auto-selected. A human selects one with a rationale, it is frozen, and then
evaluated once on the holdout. Retuning after seeing holdout results makes later holdout numbers *exploratory*.

## Review-rate interpretation

Enriched development review rates are inflated by design. Representative development and holdout rates reflect
Civil Comments' distribution, not X traffic.

## Sources of bias

Crowd annotators' interpretations; news-comment domain vs social media; English only; 2015–2017 language; label
construct mismatch (insult ≠ harassment rubric); rare threat positives; text-only input without context; and
possible changes to the provider model behind the `jev-latest` alias. Pin a versioned ID for tuned configs.

## Why proxy accuracy is not platform policy accuracy

Proxy labels measure whether the detector agrees with crowd judgments of "insult"/"threat" on news comments.
Platform enforcement depends on a different policy, full context, account history, and human adjudication.
High proxy agreement does not establish enforcement accuracy, and low agreement may partly reflect label
mismatch rather than detector error.

## Verified vs pending

No live Jev or OpenAI benchmark results are included in this repository yet. All metrics currently visible in
the offline demo come from **synthetic fixtures** and demonstrate the pipeline only.

# Complete shared feature-space pilot

The full workflow replaces the hash-selected 150 columns with every accepted human and task discovery in one shared candidate bank. It reads the original experiments without changing their manifests, journals, model settings or collectors. The historical 150-feature studies remain baselines.

Implementation: `feature_space_pilot/full/`. Configuration: `configs/feature_space_full.json`. Outputs: `artifacts/feature_space_full/projects/full_<design>/`. The existing `artifacts/feature_space_pilot/budget_live.jsonl` remains the single cumulative ledger.

## Prepare and inspect

Use the project virtual environment from the repository root:

```bash
.venv/bin/python -m feature_space_pilot.full.run --dry-run
.venv/bin/python -m feature_space_pilot.full.run --prepare
```

These commands do not load an API key, call a provider, reserve money, or start another generation run. Preparation exports the entire available bank, all origins, the 75-item matrix with unknown cells marked `?`, a cell-level costed completion list, readable task tables, human-reference analyses, and PNG/SVG/PDF figures. No figure substitutes zero for unknown data. Complete geometry is withheld until all required binary cells exist.

`--prepare --watch` refreshes local analysis checkpoints as existing collectors save new data. It does not collect remote batches itself. Source generations must all succeed before semantic-assistance or applicability submission. A partially collected source is explicitly provisional, even when most provider requests have finished.

## Conservative consolidation

1. Collect **every** accepted generated feature, including uncapped P0 phrases beyond the separate human-comparison cap. Keep explicit exclusions from parseable failed attempts.
2. Normalize spelling/case/spacing and apply the narrow declared equivalent rewrites. Keep all raw phrases, source jobs, items, prompt families, repetitions, models and receipt hashes.
3. Propose nearby pairs using token retrieval across the entire available vocabulary. Neighbour count controls review proposals; it never caps the bank or excludes a feature. Retrieval is not exhaustive: unproposed pairs are not established distinct.
4. A separate Luna Batch assessor evaluates candidate-to-candidate bidirectional entailment and checks negation, modality, roles, conditionality, causal direction, conjunction and quantification. Similarity, shared topic and one-way entailment do not authorize merging. Equivalent decisions need both entailments and every preservation check, plus deterministic protected-meaning guards.
5. Merge reviewed groups only with complete-link support. An A–B–C chain does not establish A–C. Related, distinct and uncertain pairs remain separate. Conflicting/uncertain narrow-rule reviews split challenged groups.

These assessments are automated and provisional, never human approval. `semantic_review_queue.csv`, model receipts and blocked-chain records expose remaining uncertainty. This operation is separate from the existing **generated-to-human-label matcher**, which is used only for evaluation.

Optional named human equivalence reviews use `semantic_reviews_file` with JSON rows containing `left`, `right`, `decision` (`merge`, `related`, `distinct`, `uncertain`), `reviewer` and `reason`. A review cannot bypass protected-meaning checks. Changing reviews makes a new design; compatible applicability evidence may be reused under the strict cache policy.

## Count costs before paid execution

Once the original task generations have finished and been saved:

```bash
.venv/bin/python -m feature_space_pilot.full.run --count-inputs
.venv/bin/python -m feature_space_pilot.full.run --dry-run
```

The counter loads `OPENAI_API_KEY` from `.env`, uses the provider input-count endpoint and submits **no completion or embedding jobs**. It counts every actual item/rubric/schema template and, once source generations close, every full equivalence payload. Counts are hashed against serialized requests and model settings.

Atomic pair estimates use measured item-template counts plus an explicitly conservative UTF-8 allowance for the actual feature and identifiers, and a framing reserve. This is labeled a bound, not falsely reported as an exact count of each full pair. The report includes observed usage grouped by model/operation, Batch prices, the frozen 400-token atomic output ceiling (including reasoning), previous settled costs and held reservations. Equivalence uses its own 2,000-token output ceiling. Neither stage uses the former 30,000-token blanket input allowance.

There are two separately preflighted stages:

- **Equivalence assistance:** every compatible proposed pair, at most five per request, with up to two attempts and the original adjudication allocation.
- **Global applicability:** every final canonical feature × all 75 frozen items. The entire remaining first pass must fit the original judgment allocation and cumulative $100 cap. It never substitutes an affordable feature subset.

The explicit retry policy is `staged_primary_then_costed_invalid_retries`. Invalid retries cannot consume money still required for unattempted cells. Collect all primary requests, then preflight the complete set of known eligible invalid retries; submit none if that entire retry stage does not fit. Two attempts per cell remains the maximum. Ambiguous judgments stay unresolved for review rather than being retried until they produce a preferred numeric answer. The all-two-attempt cost envelope remains a diagnostic, not a guarantee that all possible retries are funded. No stage budgets are reallocated.

The final consolidated count is unavailable until task generation and equivalence assistance finish. Interim prices explicitly cover only the currently known vocabulary. Paid execution remains blocked by missing source data, counts, or whole-stage funding. This implements the user's staged execution request without promising scientific completion within an unknown final workload.

## Explicit execution and collection

After checking a fresh dry-run's `live_ready`, use its exact `preflight_hash`:

```bash
.venv/bin/python -m feature_space_pilot.full.run --execute-live \
  --preflight-hash HASH_FROM_FRESH_DRY_RUN --watch --auto-count-inputs
```

The full controller uses Luna, low reasoning and OpenAI Batch under the model configuration. It prints submission/collection/progress to stderr. It can queue up to eight 1,000-request batches subject to the frozen queued-token ceiling. All original collectors continue unchanged. Do not launch multiple full controllers for the same design.

The initial hash is mandatory. In watch mode the controller rechecks source identity, the complete bank, compatible reuse and fresh ledger liability under the shared lock before each submission. It advances from equivalence to the final matrix only when the next stage independently passes preflight. It stops at a budget block or review requirement. No expanded job should be submitted while current source gates are blocked.

Resume with a fresh dry run and the same command. Durable Batch acknowledgement state, custom IDs, raw outputs, attempt journals, archive hashes and ledger tickets prevent replay. Unknown provider acknowledgements remain held for reconciliation. Disconnecting does not cancel acknowledged provider work; a collector must eventually run with internet access to download it.

## Scientific outputs

`latest_report.json` points to an immutable report checkpoint. `full_manifest.json` records operational status separately from scientific completeness.

- `scientific_report.md`, `task_case_studies.md`, `completion_plan.md`: readable findings, actual proposition examples, source limits and next steps.
- `binary_matrix.csv`: emitted only when every cell is resolved. Otherwise `applicability_matrix_partial.csv` and `blocked_cells.csv` retain every unknown and its next-attempt bound.
- `task_applicability.csv`, `task_level_feature_audit.csv`, `task_shared_distinct_features.csv`: full proposition tables spanning domains, types, instances, standalone situations and combinations. Shared/distinguishing labels require resolved cells for both tasks.
- `compositionality.json`: full-sample and original equal-call allocations (three direct vs alternating 2+1 or 1+2 component generations per family). Distinguishes shared discoveries, direct-only discoveries, independently discovered but inapplicable features, constituent discovery gaps, and potential interactions.
- `interaction_review.csv`: candidates with combination=1, task=0 and situation=0. Missing generation alone does not establish emergence. Named relational interpretation is required to confirm an interaction. Optional `relational_reviews_file` is a JSON object keyed by combination item ID then feature ID, with `confirmed` boolean, `reviewer`, and `note`.
- `discovery_saturation.csv`: uncapped and protocol-cap-normalized discovery curves. Missing source generations and unjudged validity are explicit.
- `human_norm_comparison.json`, `automated_human_norm_matching.json`, `human_category_reference.csv`: human-reference correspondence and category data; automatic semantic scores are not human gold. Harpaintner provides category summaries, not individual feature norms. Pending translation/coding reviews still suppress affected comparisons.
- `feature_space_structure.json`, `legacy_150_coverage.csv`: applicability distributions and, once complete, global overlap/geometry and the same-global-judge comparison against mapped original columns. Historical Mini vs new Luna differences cannot be attributed solely to vocabulary size.
- `embedding_geometry.json`: aligned stimulus embeddings only when supported. Existing target-conditioned feature embeddings are not stimulus vectors.
- `cost_report.json`, `observed_usage.json`, `reuse_policy_and_audit.json`: financial and evidence audit. `current_cost_report.json` is refreshed even when scientific evidence is unchanged.

Candidate descriptive representations do not demonstrate features internally or causally used by a model to choose actions or express preferences. Unresolved semantic equivalences can inflate discovery and distance counts. A matrix can be operationally complete while scientific reviews remain outstanding.

## Validation

```bash
.venv/bin/python -m pytest -q tests/feature_space_pilot
```

Offline fixtures cover uncapped banks, protected meaning, provenance/stable IDs, strict model/prompt reuse, unknowns, global scope, equal-call composition, budget stages, interruption recovery and unchanged source bytes. Synthetic execution is explicitly labeled and cannot pass the empirical live gate. See `scratchpad_feature_space_pilot.md` for recorded RED → GREEN cycles and `FULL_PILOT_AUDIT.md` for the starting-state audit.

## Overnight amendment — 27 October 2026

The user authorized progress from the valid generations currently available, without waiting for perfect sampling, with **one shared cumulative $100 limit**. Current full configuration selects `generation_policy=frozen_available_checkpoint` and `stage_budget_policy=shared_total_only`.

`overnight_generation_checkpoint.json` freezes all accepted discoveries and their source receipts, all75items, observed coverage and every missing generation job. The checkpoint closes vocabulary discovery for this execution; it does **not** label missing generations successful. Late original outputs remain preserved for a subsequent version. Composition/saturation analyses retain their actual sampling gaps, and named scientific reviews remain required. The original source jobs and 150-feature experiments remain unchanged.

The shared ledger continues to count every previous cost and held liability. Separate stage partitions described earlier no longer restrict this amended full controller; each stage draws from the same$100 ceiling. The complete remaining primary pass and all known invalid retries are still preflighted as whole stages, and every request is durably reserved before submission. Maximum attempts remains2. No smaller arbitrary vocabulary or unknown-to-zero conversion is introduced.

The overnight controller uses2,000-request chunks, up to8active batches, and the existing8Mqueued-input bound. A separate `caffeinate -i -w PID` process prevents idle system sleep while the controller lives; keep the laptop lid open. The controller automatically submits/collects stages when their measured-cost and integrity checks pass. Review requirements and the shared$100cap still stop unsafe or unsupported progress.

# Scientific and $100 budget preflight

The current pilot uses **GPT-5 mini (`gpt-5-mini-2025-08-07`), low reasoning**, per the user's model choice. It retains **three repetitions**, uses a predeclared **150-feature bank**, and uses **English stimuli** as requested. German originals remain source metadata, not silently substituted model inputs.

## Validate this checkout

From the repository root with Python 3.11:

```bash
.venv/bin/python -m pip install -r requirements-feature-space-viz.txt
.venv/bin/python -m feature_space_pilot.scientific --config configs/feature_space_pilot.json --prepare
.venv/bin/python -m feature_space_pilot.preflight --config configs/feature_space_pilot.json --campaign
```

`--prepare` checks the pinned reference/workbook/crosswalk hashes before making those files read-only. Git does not preserve read-only permission bits, so run it once after checkout. It never edits data values, reads credentials or calls a model. Modified source bytes fail instead of being resealed.

The combined preflight should report `live_ready: true` and `blockers: []`. That means the configured scientific generation condition and conservative budgets pass; it does not approve every downstream human comparison or guarantee provider account access. The dated GPT-5-mini snapshot is currently marked deprecated by the provider; the isolated live diagnostic confirmed access for this account on 2026-10-27. No model is silently substituted if access changes.

## Run the human pilot

Get a fresh human-stage hash:

```bash
.venv/bin/python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage human --dry-run
```

Copy that result's `preflight_hash`, then explicitly execute:

```bash
.venv/bin/python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage human --execute-live --preflight-hash HASH_FROM_HUMAN_DRY_RUN
```

Use `--stage tasks` with its own fresh dry-run hash for the 15 case-study targets. Human and task stages keep separate frozen outputs and share one cumulative live ledger. A paid `--stage all` is not supported. The runner loads the project `.env` only after preflight and hash checks pass.

For the smaller eight-target source-stratified smoke, use `--stage human --smoke --dry-run`, then the same flags plus `--execute-live --preflight-hash HASH_FROM_SMOKE_DRY_RUN`. It plans 48 generation jobs and up to 80 judgment cells. A smoke produces partial engineering evidence, not the full scientific pilot.

## What was verified

**Vinson & Vigliocco:** the registered raw ZIP and derived CSV are both hashed and read-only. The raw adapter verifies the 456-item/1,029-feature axes; the scientific audit compares all 17,932 long-form category/feature rows with the derived CSV. [Methods p186 and Appendix p190](https://link.springer.com/content/pdf/10.3758/BRM.40.1.183.pdf) support separate short descriptive features, without pure associations or dictionary-style definitions. The source imposed no fixed feature count. The versioned single-target JSON adaptation declares its deviations, including omission of the original multi-item sheets and example demonstrations. Model-facing `the`/`to` markers distinguish noun and verb senses; uppercase reference labels remain unchanged in `source_text`.

Vinson P0 has `cap: null`: no false four- or ten-feature human instruction is imposed. A finite output-token ceiling still applies. Reports separately label the ten-feature family-comparison truncation as an analysis choice, not a source protocol cap. Harpaintner P0 independently adapts its approximately-four response method with an explicit four-entry machine envelope.

**Harpaintner:** the supplied `Data Sheet 1.xlsx` matches SHA-256 `a0d94ee950f5dcdb5bd753cb9050ddeda4a96d8493bc63772113bc31a39af293`. Its `Wortliste` sheet contains 296 concepts and explicit original-German/loose-English/category/rating columns. All 24 supplied selections match exact workbook rows on both labels, four category proportions and two ratings; `harpaintner_crosswalk.json` records zero discrepancies. [Published Methods](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.01748/full) were checked separately. This establishes the supplied workbook's internal derivation and schema, while preserving `publisher_download_verified: false`; it does not invent an independent publisher-download history.

**English condition:** `published_english_exploratory` is the primary generation condition, explicitly selected by the user. It permits generating/judging features for the published loose English labels. All 24 entries in `pilot_abstract_translation_review.csv` remain pending; none was signed on the user's behalf. Reports and the explorer keep translated human quantitative comparisons unavailable until named approval and exact word mapping exist. A rejected item blocks English generation until resolved. `german_primary` remains an explicitly selectable alternative, not the default.

The workbook contains category proportions, not individual human feature strings or participant records. Abstract lexical recovery, binary human feature RDMs and participant split-half estimates therefore remain unavailable. A separate supplementary **Euclidean category-proportion RDM** is supported only with at least four reviewed aligned items and complete independent coding of the compared generated profiles. It is never labeled binary-feature Jaccard geometry. Partial category profiles disclose coding coverage.

## Token and spending controls

| Scope | Generation jobs | Maximum judged cells |
|---|---:|---:|
| Human | 540 | 9,000 |
| Tasks | 135 | 2,250 |
| Combined | 675 | 11,250 |

The frozen role envelopes are 7,500 input tokens, 700 generation output tokens and 400 judgment output tokens, including reasoning/format tokens. Two attempts are budgeted for every job. Current verified standard GPT-5-mini rates are $0.25 input/$2 output per million tokens. The combined ceiling is **$64.60875**, plus prior cumulative liabilities. After the one-call diagnostic, total existing spend is **$0.003158**, so the current maximum combined liability is **$64.611908**, leaving about $35.39 of the $100 cap. These are conditional token-based bounds; invoices remain the billing authority.

Stage allocations are generation $10, judgment $65, adjudication $10, embeddings $5 and reserve $10; they sum to $100. Combined preflight adds human and task remaining work and counts prior liability once. Both stage entrypoints also check the combined budget. Reservations precede requests; unknown usage stays reserved; actual provider usage settles the same durable ledger. Never delete or replace that ledger to obtain a new allowance.

The single request builder is shared by transport, preflight and the controller. `tiktoken` tokenizes the complete canonical payload, including JSON schema, with the selected model's mapped `o200k_base` encoding. The table is pinned and vendored for offline use; unknown model mappings fail. An explicit **512-token framing reserve** is added. This is a conservative local bound, not an exact server count; byte lengths are reported separately. Both controller and client reject over-cap payloads before reservation/transport. Any provider usage beyond the envelope records actual cost and halts. [OpenAI token-counting guidance](https://developers.openai.com/api/docs/guides/token-counting) explains the distinction and that output ceilings include non-visible tokens.

Detailed reproducible audits:

```bash
.venv/bin/python -m feature_space_pilot.human_inputs --config configs/feature_space_pilot.json --output-dir artifacts/human_input_audit/v1
.venv/bin/python -m feature_space_pilot.scientific --config configs/feature_space_pilot.json --output artifacts/scientific_preflight.json
.venv/bin/python -m feature_space_pilot.token_accounting --config configs/feature_space_pilot.json --output artifacts/prompt_token_audit.json
```

The older `pilot_prompt_size_audit.json` is preserved as supplied historical byte measurements based on ten repetitions; it is not used to assert current token counts. The checked current audit covers all 225 generation templates and 75 representative judgment templates, with a conservative allowance for the maximum permitted feature text.

The 150-bank amendment keeps the existing deterministic phrase-hash selection before judging. All excluded proposals remain in provenance/coverage reports. It lowers vocabulary coverage relative to 200; no judgment cells are dropped midway to meet a desired result.

## Diagnostic versus study

The implemented calibration command defaults to one judgment call, no retries, <=$0.01 reservation, through the real schema/parser/token guard/shared ledger:

```bash
.venv/bin/python -m feature_space_pilot.calibrate --config configs/feature_space_pilot.json --role judge --execute-live
```

One such call passed on 2026-10-27: 304 input / 134 output tokens, local input bound 890, normal completion and estimated $0.000344 cost. It is an isolated diagnostic and never enters pilot matrices. It does not start the full pilot. Token ceilings can cause truncation on longer responses; failed attempts remain audited and are never converted to successful empty lists.

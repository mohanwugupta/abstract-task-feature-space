# Human-norm feature-space pilot

The standalone pilot is implemented in `feature_space_pilot/` and uses hosted, OpenAI-compatible frontier-model APIs. The offline workflow needs only Python, NumPy, SciPy, and pytest.

Future generator and judge runs default to user-selected **GPT-6 Luna (`gpt-6-luna`), low reasoning, using OpenAI Batch**. The current Mini pilot retains its scientific settings, with unfinished judgments handed off to Batch under a separate execution amendment. Three repetitions and the $100 cumulative cap remain. This original runner preserves the 150-feature baseline. The [expanded full workflow](FULL_FEATURE_SPACE_PILOT.md) builds an uncapped, shared 75-item vocabulary and writes separate outputs without changing these studies. Pending abstract translation/coding reviews still suppress affected comparisons. See [BATCH_AND_PROGRESS.md](BATCH_AND_PROGRESS.md) for live progress monitoring and future asynchronous runs, and [PREFLIGHT_WORKFLOW.md](PREFLIGHT_WORKFLOW.md) for source and budget checks. Offline demos are engineering fixtures, not scientific benchmark results.

## Quick start

From the repository root, using Python 3.11:

```bash
python -m pip install -r requirements-feature-space-pilot.txt
python -m feature_space_pilot.scientific --config configs/feature_space_pilot.json --prepare
python -m feature_space_pilot.human_inputs --config configs/feature_space_pilot.json --output-dir artifacts/human_input_audit/v1
python -m feature_space_pilot.preflight --config configs/feature_space_pilot.json
python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage human --dry-run
python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage tasks --dry-run
python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage all --demo
python -m feature_space_pilot.report --config configs/feature_space_pilot.json
pytest -q tests/feature_space_pilot
```

Dry runs read inputs and compute plans. They do not create output directories, write receipts, contact providers, or charge credits. `blockers` and `live_ready` communicate scientific and spending readiness; an inspectable blocked plan exits successfully. Invalid configuration/inputs exit with code 2.

The offline demo uses all **60 synthetic norm targets + 15 case targets**, three prompt families and three independent replicates: **675 generation requests**, followed by the complete selected binary-judgment scope. Fake requests have zero billing. Its read-only synthetic reference CSV is marked `kind=synthetic`, hashed, and rejected by the live source gate.

Artifacts are in `artifacts/feature_space_pilot/demo_all_<design-hash>/`. Repeating a run with unchanged inputs performs no completed model calls again. Source, prompts, model settings, seed, fixtures and code enter the design hash. Human and task live stages have separate outputs. `--stage all` is for offline demonstration.

## Current source gates

See `data/feature_space_pilot/source_registry.json` and `selection_rules.json`.

- **Vinson & Vigliocco (2008):** raw publisher ZIP and derived CSV hashes verify. The 456-item × 1,029-feature raw axes and all 17,932 derived rows are cross-checked. Published Methods p186 and Appendix p190 were inspected: 280 participants, 20 per 30–40-word list, separate short descriptive features, no fixed feature count. The versioned single-target JSON P0 is a documented adaptation, not a verbatim replication. Source feature cap is null; a ten-feature analysis truncation is separately labeled. Original noun/verb `the`/`to` markers disambiguate model input while preserving norm lookup keys.
- **Harpaintner et al. (2018):** `Data Sheet 1.xlsx` matches the supplied digest, contains 296 concepts, and all 24 selected label/category/rating rows match exactly. The source registry verifies this internal derivation and the published schema/method while retaining `publisher_download_verified=false`. The four averaged category proportions do not provide individual feature strings or participant records. Model-facing stimuli and P0 are English per user preference; this is a declared cross-language condition. Pending translation reviews do not falsely approve equivalence and suppress human comparisons until reviewed.
- **Wiemer-Hastings & Xu (2005):** conceptual reference only; excluded from quantitative data.

The CSV pair supplies all 60 human targets; IDs, splits and raw bytes remain frozen. Prior JSONL configurations remain supported. See [HUMAN_INPUTS.md](HUMAN_INPUTS.md) for mapping, selection and input-audit details. Source preparation checks hashes before making reference files read-only.

Primary sources: [Vinson & Vigliocco](https://link.springer.com/article/10.3758/BRM.40.1.183), [Harpaintner et al.](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.01748/full).

## Configure hosted frontier models

Edit `generator` and `judge` independently in `configs/feature_space_pilot.json`:

| Field | Required value |
|---|---|
| `provider` | `openai_compatible` |
| `base_url` | Provider's HTTPS API base ending in `/v1` |
| `api_key_env` | Environment variable **name** containing the provider key |
| `model_id`, `revision` | Verified, pinned frontier model snapshot and revision |
| `supports_structured_outputs` | Verified strict JSON-schema support; must be true |
| `supports_seed`, `supports_temperature` | Verified model capabilities; unsupported parameters are omitted and logged |
| `capabilities_verified_at`, `capabilities_source` | Verification date and official documentation URL |
| `max_input_tokens`, `max_output_tokens` | Explicit positive token ceilings, including reasoning-token allowance where relevant |
| `reasoning_effort` | Optional provider-supported setting |
| `prices` | Model-specific table described below |

`prices` and `batch_prices` must include `model_id`, `revision`, numeric `input_per_million`, numeric `output_per_million`, `verified_at`, and `source_url`. Model/revision must match the role. Luna standard rates are $0.10 input / $0.50 output per million tokens; Batch rates are $0.05 / $0.25, verified 2026-10-27 against official pricing. Mode normalization selects the matching table; individual-request clients reject Batch pricing. The current Luna identifier is undated; actual response model/fingerprint are logged. Reverify account access and pricing before paid work.

The default adapter uses HTTPS Responses, strict `text.format` JSON schema, `max_output_tokens` and `reasoning.effort`. The retained Mini config uses Chat Completions, strict `response_format.json_schema` and `max_completion_tokens`. Batch preserves each exact API body and selects the corresponding endpoint. Native Anthropic/Gemini request formats require a separate adapter. The runner loads only the named API-key variables from the project-root `.env` after preflight/hash checks pass; exported values take precedence. Values are literal and never executed, logged or stored in artifacts. Dry runs, reports and demos do not read credentials. Official [structured-output guidance](https://developers.openai.com/api/docs/guides/structured-outputs) describes the contracts.

P0 study-specific protocols are configured in `protocols[dataset_id]` with `text`, `cap` (positive integer or null for source-uncapped Vinson), `version`, and `verified`. Harpaintner's supplied German adaptation has cap 4 and is explicitly not a verbatim instruction replication. Task P0 is a predeclared general analogue. Family recovery comparisons downsample each response to the configured P0 cap (maximum 10). Prompts never receive human feature strings, category/stratum labels, other targets, previous generations, judge results, or norm ratings.

## API key setup

Copy `.env.example` to `.env` in the project root if you do not already have a local `.env`, then set:

```dotenv
OPENAI_API_KEY=your-openai-api-key-here
```

No manual `source` or `export` is required. Both generator and judge already use `api_key_env: "OPENAI_API_KEY"` in the default config. Previously exported values remain supported and take precedence over the file. The file may contain comments, quoted values, or `export KEY=value` syntax. Variable interpolation and shell commands are not evaluated. Missing/blank keys or malformed selected entries fail with a value-free error before execution. `.env` is ignored by Git; only its empty example template is committed.

## Live execution and cost accounting

```bash
python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage tasks --dry-run
# Copy preflight_hash from that successful, blocker-free plan:
python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage tasks --execute-live --preflight-hash HASH_FROM_DRY_RUN --watch
```

The human command uses `--stage human` and additionally requires all source/sample/protocol gates. The explicit preflight hash is an addition to the PRD CLI; it ensures that the reviewed dry run matches the current frozen design and remaining-spend bound. A changed prior balance requires a fresh dry run. The implementation never starts a paid run implicitly. The default task stage now plans 135 generation calls (15 targets × 3 families × 3 repetitions), 70% fewer than the original 450-call design. The binary-judgment upper bound is now 2,250 cells at the frozen 150-feature bank; fewer generation repetitions do not guarantee the same percentage reduction in total cost. See `data/feature_space_pilot/sampling_amendment.json` for the frozen amendment.

The current cumulative pilot cap is **$100**, with an enforced maximum of **$500**, with stage allocations of $10 generation, $65 judgments, $10 adjudication, $5 embeddings and $10 reserve, summing to $100. All live stages share `budget_live.jsonl` under the configured artifact root. Keep that root/ledger for the entire pilot; a new directory is not a new spending allowance. No automated paid merge/adjudication/embedding calls are currently scheduled; those stage allocations are retained for a separately preflighted extension. Reviews are manual; embeddings are imported as pinned artifacts.

Each attempt is reserved and fsynced **before** transmission. Failed parses are charged from returned usage and retained before a separately reserved retry. Unknown usage/timeouts retain the full reservation. Missing/invalid usage or an over-reservation provider response halts execution; an overage records the actual cost. Provider invoices remain the authority for billing; the ledger is a conservative token-based estimate. Provider errors are logged by class rather than potentially sensitive error bodies. There are no hidden SDK retries or response caches.

A process lock serializes spending across stages. Existing liabilities enter preflight; resume estimates only remaining attempts. Candidate selection uses a predeclared phrase-hash order and a predeclared default maximum of 150 (implementation hard maximum 200), independent of human matches/judge outcomes. If the upper bound is too high for a chosen frontier model, freeze a smaller cap before execution and document the coverage trade-off. All excluded candidate phrases remain auditable.

Use `--stage human --smoke --dry-run` for the predeclared eight-target smoke: one item per human class plus the poem task, standalone deadline, direct composition and hierarchy parents, two replicates per family, cap 10 (48 generation requests plus binary judgments). It retains source/protocol/model/price gates and requires all three source classes; the selected CSV inventory now includes all three classes, and scientific generation preflight now passes; affected human comparisons remain review-gated. Execute only with `--smoke --execute-live --preflight-hash HASH_FROM_SMOKE_DRY_RUN`. Responses and usage are captured in the ordinary journals and manifest (`smoke=true`). For an offline smoke, add `--demo`. No billed smoke was performed during implementation.

`--max-jobs N` separately limits pending generation jobs for an interruption checkpoint and writes a partial manifest.

## Reviews and evaluation-only inputs

A judgment is exactly integer 0 or 1. `ambiguous` requires `applies=null` and remains outside final matrices until resolved. A valid empty generation list is retained; an entirely empty candidate inventory blocks matrix production.

Add manual `adjudications` to config as records containing `item_id`, `feature_id`, `applies` (0/1), `status="ok"`, `evidence`, and `reviewer`. Primary judgments remain in `judgments.jsonl`; adjudication provenance is stored separately in evaluation inputs. A conflicting already-resolved cell requires a new rubric/design.

`relational_audit` maps situated item IDs to feature IDs to records such as:

```json
{
  "poem_S_deadline": {
    "FEATURE_ID_FROM_INTERACTION_REVIEW": {
      "confirmed": true,
      "reviewer": "reviewer-identifier",
      "note": "Reviewed relationship between the concrete task and the deadline."
    }
  }
}
```

Only named, documented manual confirmations can move an operational interaction candidate to confirmed. They do not demonstrate mathematical emergence or internal model representations.

For category profiles, set `category_coding_file` and `category_coding_sha256`. The reviewed CSV has `item_id,feature_id,category,reviewer`; only the corresponding source's real category vocabulary is accepted. Multi-label source categories are retained. Profiles report the portion of applicable generated features actually coded. Missing coding is unavailable, not guessed.

For embeddings, set `embedding_file` and `embedding_sha256` to an evaluation-only JSON artifact:

```json
{
  "model_id": "PINNED_EMBEDDING_MODEL",
  "revision": "PINNED_REVISION",
  "source_url": "OFFICIAL_MODEL_DOCUMENTATION_URL",
  "items": [
    {"item_id": "STIMULUS_ID", "text_hash": "SHA256_OF_CANONICAL_JSON_TEXT", "vector": [0.1, 0.2]}
  ]
}
```

Include exactly the run's item IDs; `text_hash` is `feature_space_pilot.io.digest(item["text"])`. Imported embeddings compare with both generated-feature geometry and available source-specific human geometry. The pipeline does not silently download embedding models or use embedding agreement to tune prompts.

Rerun the same stage with reviewed evaluation inputs to refresh reports without repeating completed calls. `manifest_history.jsonl` preserves preceding manifests. Frozen primary config/prompts/jobs/bank remain traceable; current review inputs have a separate fingerprint.

## Analysis and interpretation

- Final matrices contain only 0/1 for the entire explicitly declared stage × selected-feature scope. Missing, failed or ambiguous cells never become zero. Out-of-scope combinations are declared separately.
- Equivalence is exact after NFC/case/whitespace/terminal-period normalization. Similar wording enters review; negation, agent roles, modality and conditions are protected. Explicit reviewed merges have provenance. Atomicity checking flags clear syntactic bundles; full semantic clarity/nonredundancy still requires the provided manual audits.
- Human recovery is precision/recall relative to **listed productions**. Unlisted generated propositions are not deemed false. Matching uses exact rules or a frozen reviewed `semantic_equivalences` map, not an unverified semantic matcher.
- Human geometry uses binary Jaccard RDMs; embedding comparisons use cosine. Both-empty vectors have distance 0, one-empty vectors distance 1. Spearman intervals resample **items**, excluding copies of the same original item from pairs. Small-sample intervals are descriptive. Participant split-half estimates require actual participant records.
- The hierarchy is domain → task type → instance. Three standalone situations form an independent axis; six nonbaseline compositions complete the 15 targets. The synthetic 20-invoice input, exact matching policy, reference report and report schema are versioned fixtures; reference answers never enter generation/judge prompts.
- Composition separates discovery G from applicability A. Each family uses equal total call budgets for component and direct paths, retaining every direct replicate. At the default three repetitions, the component budget is 2 task + 1 situation or 1 task + 2 situation versus 3 direct calls. The extra component sample alternates by frozen composition index and prompt-family index; across all six compositions each role gets 27 component calls. Allocation is independent of outcomes and is recorded as `sampling_allocation` in each composition audit. Even repetition counts retain an equal task/situation split. The **same** T+S judge cells evaluate the component bank and its union with direct discoveries. Full-sample discoveries are also retained. Reports include tokens and gain per call/token for every composition.

## Artifacts and implementation map

`DATA_DICTIONARY.md` describes every artifact. `scratchpad_feature_space_pilot.md` contains incremental RED → GREEN → REFACTOR evidence, source decisions, failures, spending and resume instructions. No scientific run is considered complete merely because tests pass. The automatic decision memo keeps the synthetic demo at **revise before scale-up**.

`registry.py`/`fixtures.py`: validated stimuli and synthetic inputs. `references.py`/`source_adapters.py`: evaluation-only sources. `prompts.py`/`planner.py`: isolation and independent jobs. `clients.py`/`budget.py`/`engine.py`: hosted requests, reservations and checkpoints. `bank.py`/`judge.py`: canonical candidates and binary matrices. `metrics.py`/`composition.py`/`report.py`: analyses, audits and reports.

## Visualization

The separate read-only explorer and static exporter are documented in `VISUALIZATION_EXPLORER.md`. They operate on existing artifacts without reading `.env`, changing judgments or running the API pipeline. Install `requirements-feature-space-viz.txt` for the full test suite and explorer.

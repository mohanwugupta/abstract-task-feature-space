# Strict binary applicability and selective ambiguity recovery

Future applicability judgments use **binary_v2**. A successfully parsed judgment must contain integer `applies=0` or `applies=1` and `status=ok`. The API schema excludes null and the ambiguous status; the parser separately rejects nulls, booleans, floats and other nonbinary labels. This follows the supported [OpenAI Structured Outputs schema](https://developers.openai.com/api/docs/guides/structured-outputs).

## Decision rule

- **1:** The proposition reasonably characterizes the target using its description and ordinary semantic knowledge. Bare concepts may have ordinary characteristic properties without those properties being explicitly spelled out. A feature need not be universal unless its wording asserts universality.
- **0:** The proposition does not meet that standard: it is irrelevant, contradicted, unsupported as a reasonable characterization, or requires an invented context. A zero does not require proof that the proposition is impossible in every imaginable circumstance.

The judge preserves agent roles, negation, modality, quantifiers, numbers and conditions. It makes its best binary decision under that operational rule. Its evidence note explains the decision; no manual adjudication is required to turn a valid response into a binary value. Calibration or quality reviews can still assess whether automated judgments are useful scientifically.

The prompt uses generic controls, rather than benchmark targets or human norm labels. It does not estimate human production frequency, preference, desirability or internal model feature use. This change applies to **applicability**; semantic feature equivalence and human-label matching remain separate protocols with their own evidence categories.

Technical failures—timeouts, refusals, truncation or invalid JSON—remain request errors. They can receive at most two attempts per new v2 job. Unknown provider execution/usage is held for reconciliation, not assigned zero or blindly resubmitted.

## Versioning and preservation

`feature_space_pilot/prompts/binary_v1.txt` is unchanged. Explicit legacy parsing still accepts the historical abstention format. Frozen historical configuration dictionaries without a rubric flag resolve to v1, preserving original request hashes, payloads and receipt verification. Newly loaded configurations and demo defaults use v2; the supplied configuration explicitly sets `judge.rubric_version` to `binary_v2`.

Standard, Batch, continuation, calibration, immediate, token-profile, evidence-reuse and reporting paths carry the chosen version explicitly. A v1 receipt is not silently reused as if it were produced under v2. Cached v1 token profiles cannot substitute for the changed prompt/schema.

## Prepared human-only retry

Prepared run: `artifacts/feature_space_binary_retry/retry_4ce8e706b4d1455d/`.

The source is the sealed independent-human checkpoint `artifacts/feature_space_human_refresh/checkpoints/separated_99f0a371fc62c20a/`, not the historical mixed human/task matrix.

| Source human cells | Count | Retry selection |
|---|---:|---|
| Definitive 0/1 | 230,271 | Preserved; no repeat requests |
| Ambiguous | **81,161** | Selected for new binary-v2 judgments |
| Unknown usage | 1,973 | Excluded; retain reconciliation holds |
| Invalid | 284 | Excluded from this ambiguity-only stage |
| Missing compatible evidence | 291 | Excluded from this ambiguity-only stage |

The selected ambiguous cells include **51,443 for the 36 Vinson concepts**; the remainder concern the 24 abstract targets. No task-generated feature, task primary inventory, new generation or new consolidation enters this stage. Its source membership is fixed by the audited ambiguous-cell status and original source-record hashes, not by selecting favorable model outputs.

This is a **new rubric pass** over selected cells, with new job IDs and at most two v2 attempts each. Original v1 attempts and their accounting remain intact. New responses live in a separate append-only journal. New API failures never become zero.

`effective_cells.csv` and `effective_matrix_partial.csv` explicitly combine preserved v1 binary answers with v2 resolutions of selected abstentions. Every cell records the effective rubric and old/new evidence provenance. This is a **mixed-rubric dataset**, not an independent uniformly v2 replication. If every selected ambiguity resolves, 2,548 original technical/missing cases would still remain unknown. Updated scientific analyses must disclose the combined policy and continue preserving those unknowns.

## Cost preflight and execution status

Current status: **prepared and preflighted; no paid judgments submitted**.

The new input profile measures 60 target templates using the input-token-counting endpoint, then applies the existing conservative feature/ID byte allowance and boundary reserve. Template counts are 485–494 tokens; request input bounds are 666–799. This is a measured-template bound, not a claim of exact per-request counts. The collector checks actual usage against each bound.

The prepared run uses **GPT-6 Luna, low reasoning, OpenAI Batch, 400 maximum output tokens**, with a maximum of two attempts. Pricing was checked against [official OpenAI pricing](https://developers.openai.com/api/docs/pricing): Batch $0.05 input/$0.25 output per million tokens. Batch transport follows the [official Batch API](https://developers.openai.com/api/docs/guides/batch), with durable custom IDs, reservations and uncertain-acknowledgment reconciliation.

- Existing cumulative spending and reserved liability: **$47.51803943**.
- Full selected stage, including up to two v2 attempts per cell: **at most $22.03766430**.
- Combined completion ceiling at this snapshot: **$69.55570373**, below the unchanged **$100** cap.

Review [the full cost preflight](artifacts/feature_space_binary_retry/retry_4ce8e706b4d1455d/cost_preflight_20261030.json) and [the exact selection summary](artifacts/feature_space_binary_retry/retry_4ce8e706b4d1455d/selection_summary.json). Execution computes a fresh ledger-dependent hash; a changed ledger or plan invalidates the old authorization hash.

The user's existing requirement remains: “Do not make new paid API calls without an explicit cost preflight and authorization.” Preparing/measuring this run does not launch the paid stage. The launch command below is for an explicitly authorized submission.

## Commands

The completed preparation and token measurement can be reproduced with:

```bash
.venv/bin/python -m feature_space_pilot.binary_retry --prepare
.venv/bin/python -m feature_space_pilot.binary_retry --measure-inputs
.venv/bin/python -m feature_space_pilot.binary_retry --dry-run
```

After authorization, copy the fresh `preflight_hash` from the dry run:

```bash
.venv/bin/python -u -m feature_space_pilot.binary_retry \
  --run-dir artifacts/feature_space_binary_retry/retry_4ce8e706b4d1455d \
  --execute-live --preflight-hash HASH_FROM_FRESH_DRY_RUN --watch
```

Collection without submitting any new jobs:

```bash
.venv/bin/python -m feature_space_pilot.binary_retry \
  --run-dir artifacts/feature_space_binary_retry/retry_4ce8e706b4d1455d --collect --watch
```

Collection stops when no acknowledged batches remain. The execution watcher can collect and advance later chunks, rechecking full remaining cost and the shared cap each cycle. Uncertain acknowledgments, unknown usage and exhausted failures remain explicit. It never retries a resolved new job or another target's feature by expanding the candidate inventory.

Verify frozen source/job/code identity without API calls:

```bash
.venv/bin/python -m feature_space_pilot.binary_retry \
  --run-dir artifacts/feature_space_binary_retry/retry_4ce8e706b4d1455d --verify
```

`--run-dir` can be omitted to use `artifacts/feature_space_binary_retry/latest.json`. Original results, manifests and inference ledger entries remain preserved. The future execution appends reservations/settlements to the same cumulative ledger; it does not reset the budget.

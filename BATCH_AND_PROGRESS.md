# Progress and future Batch runs

The current human pilot continues with its frozen GPT-5-mini configuration and individual API requests. The user chose to leave that process running. Changing this checkout does not change its already-loaded settings. Do not restart it with the new default config: Luna is a separate iteration, with a separate design hash and output directory.

Monitor the current process from a second terminal:

```bash
.venv/bin/python -m feature_space_pilot.status --watch
```

The watcher reads journals without loading credentials, locking the ledger or modifying the run. It reports successful generation and judgment counts, resolved versus ambiguous cells, failed attempts, latest item, time since the last saved result, and cumulative accounted spending. A partially written last line is included on the next refresh. Counts are approximate live snapshots; an old timestamp alone cannot distinguish a slow request from an interrupted process. Use `--run-dir artifacts/feature_space_pilot/RUN_ID` to select a particular run, or `--json` for structured output. Ctrl+C stops only the watcher.

New runner processes also print elapsed time, stage, target, attempt, completion counts and spending to stderr. Their final machine-readable JSON stays on stdout. `--quiet` suppresses progress messages.

## Future iteration defaults

Both generator and judge now use **`gpt-6-luna`, low reasoning**, through the Responses API. Model IDs remain configurable. The current provider identifier is undated; actual response models and request IDs remain in usage receipts. No live Luna or Batch diagnostic was performed during implementation.

`execution_mode: batch` selects OpenAI Batch. Official [Batch guidance](https://developers.openai.com/api/docs/guides/batch) specifies a 50% discount and an asynchronous completion window of up to 24 hours. Verified [Luna pricing](https://developers.openai.com/api/docs/pricing) is $0.05 input / $0.25 output per million tokens for Batch, versus $0.10 / $0.50 for individual requests (2026-10-27). Provider account access and queued-token limits still apply.

The scope stays at three repetitions, 150 selected features, 675 generation jobs and at most 11,250 judgment cells across human and task stages. With two attempts and 16,000 input / 700 generation-output / 400 judgment-output ceilings, the new combined model-completion ceiling is **$21.56625**, plus existing cumulative liabilities. This is a bound for a new Luna iteration; it does not replace or refund the ongoing Mini run's spending. The same $100 cumulative ledger and stage allocations apply across iterations. The provider invoice remains the billing authority.

Luna has no verified mapping in the pinned local tokenizer. Offline audits therefore label payload bytes and a conservative byte-based upper bound separately; they never call it a measured token count. Explicit live execution obtains the provider's model-specific Responses input count, including schema/input, and adds the 512-token reserve before allocating any completion budget or submitting a request. Counts over the frozen ceiling fail before submission. The existing mapped Mini tokenizer remains supported.

## Run and collect a new iteration

Prepare sources and inspect a fresh plan as described in [PREFLIGHT_WORKFLOW.md](PREFLIGHT_WORKFLOW.md). For example, after the current process has finished:

```bash
.venv/bin/python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage tasks --dry-run
.venv/bin/python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage tasks --execute-live --preflight-hash HASH_FROM_DRY_RUN --watch
```

Use `--stage human` and its own hash for a new human iteration. The project-root `.env` still supplies `OPENAI_API_KEY`. The shared ledger prevents concurrent spending controllers; the read-only status watcher can run alongside either controller.

`--watch` polls every 30 seconds, collects completed Batch outputs, and advances remaining chunks and phases. Without it, a submission returns `status: batch_pending`; repeat the dry-run and explicit execution commands later to collect and advance. Ctrl+C detaches the local command and retains provider work, journals and reservations. Resume with a fresh dry-run hash and the same config. Do not delete Batch state or resubmit files manually after an uncertain acknowledgement.

Each phase has at most one outstanding chunk, with `batch_max_requests: 1000` by default. Each request has a durable pre-reservation and exact custom ID. Responses can arrive unordered; only the declared IDs are accepted, and collecting the same results twice does not charge or append them twice. An uncertain creation acknowledgement is reconciled against provider metadata rather than blindly resubmitted. Interrupted preparation recovers existing reservations. Unknown usage retains the full reservation; only explicit unexecuted expiry/validation failures can settle to zero. Failed/truncated responses remain audited and retry only within the two-attempt limit. Ambiguous results remain unresolved for review.

`--online` explicitly selects individual requests and standard prices; use it consistently for both the dry run and execution. It still uses the configured Luna model. `configs/feature_space_pilot_mini_online.json` preserves the earlier Mini settings for deliberate same-model work; it is not an instruction to restart the existing process. Neither option silently switches models.

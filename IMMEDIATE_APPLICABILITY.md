# Immediate applicability migration

The user requested faster completion on28October2026. The immediate runner retains the frozen75-item/5,825-feature vocabulary, Luna low reasoning,400maximum output tokens, exact literal single-cell binary prompts and all completed compatible judgments. Only transport/billing/concurrency change.

`feature_space_pilot/immediate/` is separate from the pinned Batch/full/source implementation. No generation or consolidation is repeated. The cumulative live ledger and shared **$100** cap cover every previous cost, pending liability and immediate attempt. Standard Luna pricing is$0.10input/$0.50output per1Mtokens, compared with Batch$0.05/$0.25; prices were checked against the official model documentation on28October2026.

The old full scheduler is stopped between locked cycles. Its active batches are canceled through the provider's cancellation lifecycle, preserving partial results. A collection-only backend cannot create Batch jobs. It reconciles terminal request counts with raw output/error IDs before recognizing canceled requests as unexecuted. Derived zero-inference usage is explicitly labeled and linked to hashed cancellation evidence; it never becomes a zero applicability label. Uncertain acknowledgements stay held and are not blindly resubmitted.

Input costing uses75provider-measured empty-feature item templates and5,825provider-measured full feature payloads on a fixed reference item. Localized feature offsets were validated on150cross-item shortest/longest-feature pairs, maximum residual0tokens, with an additional8tokenboundary reserve. The method is an additive measured bound, not an unsupported Luna tokenizer or an exact count claim for every pair. The frozen profile and payload/item/bank/model hashes are retained.

At migration preflight390,055unsubmitted primary cells had a **$93.14265250** standard-price upper bound, including the unchanged400-tokenoutput ceiling. With$6.61237823accounted prior liability, the combined ceiling was **$99.75503073**. Actual usage settles each reservation and frees unused funds. All remaining primary cells must pass preflight before execution; known invalid retries are funded afterward as a separate complete stage. No feature subset is substituted. Ambiguous model judgments remain unresolved for review, and every model cell retains the two-attempt maximum.

The runner supports up to128concurrent HTTP calls and a configured maximum50requests/second. It starts at5requests/second, ramps gradually, reads provider request/token limit headers, follows Retry-After for admission rejections and backs off when throttled. Rate-limit admission rejections have separately audited bounded retries; uncertain network outcomes are held. There are no hidden SDK retries. Actual throughput depends on the project's standard API limits and service latency; a completion time cannot be guaranteed from the Batch rate.

Monitor with the same stable path:

```bash
tail -F artifacts/feature_space_full/controller_latest.log
```

The adjacent `controller_latest.json` contains the current PID, project directory and command. It now points to the immediate controller; immutable Batch outputs remain in the source project.

For a read-only fresh plan:

```bash
.venv/bin/python -m feature_space_pilot.immediate.run \
  --source-full-run artifacts/feature_space_full/projects/full_4a33bf7950555173 \
  --dry-run
```

Resume only after confirming no controller owns the project, using a fresh preflight hash:

```bash
.venv/bin/python -m feature_space_pilot.immediate.run \
  --source-full-run artifacts/feature_space_full/projects/full_4a33bf7950555173 \
  --execute-live --preflight-hash HASH_FROM_FRESH_DRY_RUN
```

The process drains outstanding responses on SIGTERM/SIGINT. Durable `applicability/dispatches.jsonl`, `judgments.jsonl`, `transport_rejections.jsonl`, shared reservations and source receipts prevent replay. Any prior reservation without a located dispatch/receipt is held. New full-compatible manifests, reused-source evidence and scientific reports remain under `artifacts/feature_space_full/projects/full_immediate_<design>/`. Complete binary matrices are withheld until all cells resolve; finished execution does not assert that scientific reviews are complete.

A PID-bound `caffeinate -i -w PID` prevents idle system sleep. Keep the Mac connected and the lid open for local submission/collection. No permanent power setting is changed.

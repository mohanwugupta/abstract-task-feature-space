# Automated semantic matching

This separate, exploratory analysis asks whether the **existing generated propositions express the human norm features**, allowing paraphrases. For example, a pencil proposition saying “has a core made of graphite” can recover the human label “graphite.” Saying “contains no graphite” cannot. Saying “makes marks on paper” does not by itself establish a graphite core.

The original Mini generations and exact-match analysis remain unchanged. The grader uses the configured **GPT-6 Luna, low reasoning, OpenAI Batch** settings. No features are regenerated and no task study is started.

## Matching protocol

1. Verify a read-only empirical source snapshot and the aligned English Vinson reference. Exclude the abstract targets, whose available reference contains category proportions rather than feature strings.
2. Take the original comparison cap of the first ten features from each successful generation. Deduplicate normalized propositions within each target, retaining every family/repetition origin locally.
3. Send up to five propositions with **all human feature labels for that target**, including their source categories. The grader does not see the prompt family, repetition, generator model, split or production frequency. There is no embedding retrieval cutoff.
4. Require structured decisions: **clear match**, **related but different**, or **uncertain**, with a brief evidence note. Only a complete, valid response can classify the remaining labels as no match. Preserve negation, roles, conditions and direction; association alone is insufficient.
5. Reattach the frozen origins. Count each matched human label once per target/family, pooling one, two and three repetitions. Strict recall is the fraction of the supplied human labels recovered. Related labels do not count; uncertainty is a separate sensitivity estimate. An incomplete target has no final recall score, and an unlisted generated property is not automatically false.

The current inventory contains **36 targets, 1,240 unique target/label pairs and 2,944 distinct propositions**. It creates **605 study requests and one synthetic logical-control request**. The control checks positive paraphrases, negation and unsupported inference. Passing it is a smoke check, not an estimate of semantic-grader accuracy. Exported figures require that control to pass and compare identical completely scored targets across P0, P1 and P2.

## Budget and execution

The full two-attempt completion envelope is **$1.8786**, below the declared $2 matching allowance, using the pinned 16,000-input/3,000-output ceilings and Batch prices. All matching reservations share the original $100 ledger and its adjudication allocation. Provider input counts are obtained before submission; actual usage settles each ticket once. Unknown usage and uncertain creation acknowledgements retain reservations. The collector never blindly duplicates an uncertain submission.

These commands load `OPENAI_API_KEY` from the project `.env` only for live execution:

```bash
SOURCE=artifacts/feature_space_analysis_snapshots/live_human_74608725aae4e0dc/d76ce6eff476ac64
.venv/bin/python -m feature_space_pilot.matching.run --run-dir "$SOURCE" --dry-run
.venv/bin/python -m feature_space_pilot.matching.run --run-dir "$SOURCE" --execute-live --preflight-hash HASH_FROM_DRY_RUN --watch
```

The dry run is offline. Live progress goes to stderr. Watch mode polls every 120 seconds, collects unordered Batch outputs by exact custom ID, and submits only eligible unfinished requests. An interrupted local collector can resume the same command with a fresh dry-run hash; provider work continues independently of the local connection. Keep the computer awake and connected for automatic collection. A background local process is not remote hosting.

```bash
.venv/bin/python -m feature_space_pilot.matching.run --status --analysis-dir artifacts/feature_space_semantic_matching/ANALYSIS_ID
```

## Outputs and interpretation

Each immutable design gets its own folder under `artifacts/feature_space_semantic_matching/`. The manifest records the source snapshot, grader configuration, rubric, code hashes, input hashes, coverage, controls and spending. `inventory.json` preserves the human labels, propositions and blinded origins; `jobs.json` preserves the full planned comparison.

- `semantic_pairs.csv`: clear/related/uncertain pairs, proposition text, human label and evidence.
- `semantic_recovery.csv`: per-target/family/repetition recall, explicit scoring coverage, uncertainty and split labels.
- `summary.json` and `analysis_summary.md`: paired-cohort descriptive results and limitations.
- `figure_01_semantic_recovery.*`: target-level recall and family means with descriptive item-bootstrap intervals.
- `figure_02_semantic_saturation.*`: recovery as one, two and three repetitions are pooled.
- `judgments.jsonl`, `batch_state.json` and Batch input/output files: raw auditable attempts, IDs, usage and settlement provenance, separate from the original pilot journals.

Figures are exported as PNG, SVG and PDF. Missing coverage remains explicit. Bootstrap intervals describe variation across targets, not grader accuracy. Every decision remains `human_reviewed=false`; this workflow does not create human-approved semantic equivalences, validate pending translations, or establish scientific success. A subsequent blinded human audit can assess the automated grader if stronger claims are needed.

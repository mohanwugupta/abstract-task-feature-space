# Similarity baselines for human-feature matching

This separate analysis compares simple similarity rules with the existing Luna semantic-matching decisions. Every pair is a generated proposition and a human label **for the same target**, using the same first-ten-features comparison cap and within-target text deduplication as the matching study. It does not regenerate features, change Luna decisions, or modify the task applicability study.

| Method | How it scores a pair |
|---|---|
| Exact text | Equality after the existing Unicode/case/spacing/terminal-period normalization. |
| Token Jaccard | Number of shared unique word tokens divided by the size of their union. |
| Human-label token coverage | Fraction of the human label's unique word tokens present in the generated proposition. This asymmetric check accommodates short human codes. |
| Embedding cosine | Cosine similarity of OpenAI `text-embedding-3-small` vectors, with the identical target context supplied on each side. |

Lexical rules retain negation tokens, use no stopword removal or stemming, and make no semantic repair. A high overlap score for “contains no graphite” versus “graphite” is a real weakness of the baseline. Embedding similarity measures relatedness and can also confuse opposite claims or different roles. Neither method is assumed to establish equivalence.

## Calibration and evaluation

The existing frozen split contains **27 development and 9 held-out targets**. Threshold selection waits for complete Luna coverage of every development target, then maximizes development F1 against clear Luna matches. Ties favor fewer false positives and then a higher threshold. Exact equality uses its fixed threshold of one. Similarity thresholds come from the declared 0–1 grid in increments of 0.05, plus an all-negative option above one. Each method's threshold and development evidence are frozen before held-out evaluation.

Clear Luna matches are positive. Related-but-different and no-match pairs are negative for this strict same-property comparison. Uncertain decisions are excluded and counted explicitly. Uncollected judgments never become negatives; held-out evaluation includes only completely judged targets, with incomplete target IDs listed.

Outputs report **precision, recall, F1 and agreement against the automated judge**, alongside confusion counts and prevalence. They do not measure independent semantic accuracy. Most candidate pairs are nonmatches, so overall agreement alone can be misleading. Pair observations share targets and are dependent; the report does not claim pair-independent confidence intervals. Held-out threshold curves are descriptive and never select thresholds.

The primary purpose is to expose disagreements for inspection and determine whether a simpler method reproduces the judge's behavior. Establishing which method is more accurate requires an independent, preferably blinded human audit.

## Embedding execution and budget

The user selected OpenAI embeddings within the existing shared budget. Model: **`text-embedding-3-small`**, default **1,536 dimensions**, via **Batch**. Inputs use the literal format `Target: <target>\nFeature: <feature>` on both sides; family/repetition, production frequencies and Luna's decisions are absent. The current inventory needs **4,184 unique contextualized texts**, grouped into **66 requests** of up to 64 inputs.

The full two-attempt completion envelope is **$0.02112**, within the explicit **$0.05 baseline allowance**, existing **$5 embedding-stage allocation**, and shared **$100 cumulative cap**. Prices are pinned at $0.01 per million Batch input tokens, derived from the official model price and Batch discount. UTF-8 byte upper bounds plus a reserve are conservative planning bounds, explicitly not measured token counts. Provider usage settles each reservation once; unknown usage stays reserved. No tokenizer/model downloads or API calls occur during offline planning.

Official sources: [embedding model](https://developers.openai.com/api/docs/models/text-embedding-3-small), [embedding request schema and limits](https://developers.openai.com/api/reference/resources/embeddings/methods/create), and [Batch discount and lifecycle](https://developers.openai.com/api/docs/guides/batch).

```bash
MATCHING=artifacts/feature_space_semantic_matching/semantic_8f8897426b82de20
.venv/bin/python -m feature_space_pilot.baselines.run --matching-dir "$MATCHING" --dry-run
.venv/bin/python -m feature_space_pilot.baselines.run --matching-dir "$MATCHING" --offline
.venv/bin/python -m feature_space_pilot.baselines.run --matching-dir "$MATCHING" --execute-live --preflight-hash HASH_FROM_DRY_RUN --watch
```

The offline command computes lexical baselines and can reuse already collected vectors. It never loads credentials or calls the API. Explicit live execution loads the project `.env` through the existing credential loader and uses the existing budget ledger. The watcher polls every 120 seconds, tolerates shared-budget contention, reconciles uncertain Batch acknowledgements without blind resubmission, and updates reports as Luna results arrive. One collector owns each baseline design. A permanent incomplete Luna run produces a review-needed result rather than silently filling missing labels or waiting forever.

## Artifacts

Results live under `artifacts/feature_space_baselines/baseline_<design>/`, separate from all source studies. `baseline_manifest.json` and `latest_report.json` point to the latest report. Frozen inputs include the exact inventory, embedding texts/jobs, configuration, code/model identity and source-matching identity. `batch_state.json`, raw provider output, `embeddings.jsonl` and per-request vector files preserve submission and settlement provenance. Responses join by Batch custom ID and embedding index; missing, duplicate, nonfinite, zero or incorrectly dimensioned vectors are rejected.

Reports under `reports/<evidence_checkpoint>/` contain:

- `comparison.md` and `summary.json`: coverage, exclusions, threshold availability and agreement metrics.
- `pair_scores.csv`: all within-target pairs, baseline scores, and Luna clear/related/uncertain/no-match/unscored labels.
- `threshold_curves.csv`: descriptive development and held-out threshold sensitivity.
- `disagreements.csv`: pairs where a calibrated baseline and a non-uncertain Luna decision differ.
- `figure_01_score_distributions.*`: baseline-score distributions for collected clear matches and strict nonmatches.
- `figure_02_held_out_agreement.*`: fixed-threshold held-out precision/recall/F1, available after calibration and passing Luna smoke controls.
- `source_manifest.json`, `source_judge_records.json` and `report_manifest.json`: exact source checkpoint and artifact hashes.

Figures are PNG/SVG/PDF. Report snapshots are immutable and reused when only provider-status metadata changes. Threshold files record development evidence separately. Synthetic fixtures carry explicit nonempirical labels; all comparisons retain `human_reviewed=false`.

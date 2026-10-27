# Human pilot CSV inputs

The default configuration now loads `human_norm_pilot_60_stimuli.csv` directly, with `human_norm_pilot_abstract_24.csv` as its required cross-checked companion. Both uploaded files retain their original bytes. Their SHA-256 digests are pinned in `configs/feature_space_pilot.json` and enter the design/preflight provenance. Existing JSONL configurations remain supported.

## Validate and inspect without API calls

From the repository root:

```bash
.venv/bin/python -m feature_space_pilot.human_inputs \
  --config configs/feature_space_pilot.json \
  --output-dir artifacts/human_input_audit/v1

.venv/bin/python -m feature_space_pilot.run \
  --config configs/feature_space_pilot.json \
  --stage human --dry-run
```

The input audit writes:

| File | Contents |
|---|---|
| `stimuli.jsonl` | 60 validated, hashed registry items, original source metadata and frozen splits |
| `abstract_category_shares.csv` | 96 rows: 24 German source labels × four original category proportions |
| `input_manifest.json` | Input/output hashes, class counts, held-outs, translation status and scientific limits |

The live workflow reads the CSVs itself; the audit export is for inspection and does not need to be copied into the configuration. The audit loads no API key and does not run generation or judgments. These files are input audits, not completed pilot runs for the visualization explorer.

The dry run now includes **60 human targets and 540 generation jobs**: three prompt families × three repetitions × 60. The matrix upper bound is now 9,000 judgments at the predeclared 150-feature cap. Combined human and task planning uses 675 generation jobs and up to 11,250 cells; see [PREFLIGHT_WORKFLOW.md](PREFLIGHT_WORKFLOW.md).

## Mapping and frozen selection

- `item_text` and `stimulus_language` supply model-facing text/language. The primary abstract condition retains published English labels, with `english_text` metadata and German `source_text`/`original_german` provenance. Vinson model inputs add published `the`/`to` syntax markers while retaining exact original uppercase norm keys in `source_text`.
- `actionN` and `actionV` map to `event`; `object` maps to `object`; `abstract_concept` maps to `concept`. Counts are 24 concepts, 24 events and 12 objects.
- `harpaintner_2018_original_supplement` maps to the pipeline's `harpaintner_2018` source ID. The original identifier remains in `source_dataset_original`; publication row, reported supplement digest, URL and translation status remain available.
- All 36 Vinson development/held-out assignments are preserved. The 24 abstract `unassigned` slots receive deterministic splits by sorted item ID within each of the four six-item strata, using seed 42: five development and one held-out per stratum. Overall: **47 development, 13 held-out**. Source CSV split tags are not rewritten; `source_split` retains them. This is an assignment of missing splits, not a new selection from the full publication inventory.
- `selection_rules.json` records the exact held-out IDs, raw input hashes and amendment. It does not claim to reproduce the uploaded inventory's original selection process.

The adapter checks BOM-aware CSV structure, unique IDs and abstract source/translated labels, supported sources/classes/strata, exact class quotas, split tags, finite numeric values, original source-row metadata and agreement of the two files by ID and fields. Category proportions must be in 0–1 and sum to one within 0.0005 rounding tolerance. Reordering rows cannot change split assignments.

## Evaluation and source limits

`human_sm`, `human_is_e`, `human_sc`, and `human_va` are retained as **rounded category proportions**, labeled `SM`, `IS/E`, `SC`, and `VA`. The existing `category_counts` reference container is used with `measurement=rounded_category_proportions`; these values are not interpreted as integer production counts. Profile normalization may correct rounding. They support source-specific category composition after the scientific/alignment gates pass.

These CSVs do not contain individual abstract human feature strings or participant records. They therefore cannot supply abstract lexical feature recovery, binary human feature RDMs or participant split-half estimates. Generated category profiles still require independent reviewed coding and explicit coverage; ratings, human category values, source strata and split labels are excluded from model prompts.

The supplied original `Data Sheet 1.xlsx` is now materialized and hashed. Its 296-row schema and all 24 selected rows were verified against the CSV on original/English labels, category proportions and ratings. `harpaintner_crosswalk.json` records zero discrepancies. Both raw and derived sources are checked at preflight. The registry records verified internal derivation, while preserving that the user-supplied workbook's independent publisher-download chain was not established. Vinson raw/derived rows and the published participant instructions were verified separately. Run `python -m feature_space_pilot.scientific --prepare` after checkout to verify bytes and set reference files read-only.

All 24 English abstract translations remain marked unreviewed. The user-selected `published_english_exploratory` condition permits model generation and binary applicability judgments, with an explicitly adapted English P0. It does not approve semantic equivalence with German. Reports and the explorer still require named approval and exact mappings for translated human quantitative comparisons. Pending rows in `pilot_abstract_translation_review.csv` remain pending; rejected items block generation. German-primary is an optional, separately declared condition.

The generation/source and combined budget preflight now pass with the future Luna-low Batch defaults, three repetitions and the shared $100 cap (the current Mini run retains its frozen configuration). The 150-bank reduction is documented before judging. Supplementary Euclidean category-proportion geometry requires at least four reviewed aligned items and complete independent generated-category coding; binary feature geometry and lexical recovery remain unsupported by category-only data. See [PREFLIGHT_WORKFLOW.md](PREFLIGHT_WORKFLOW.md) for exact source, tokenizer, budget and execution details.

## Updating input data

Treat a CSV edit as a new data revision: keep both inventories consistent, review the revised provenance and update the corresponding pinned SHA-256 values in configuration. The reference registry's abstract `sha256` must also match the abstract file before it can be loaded. An ordinary dry run refuses changed bytes instead of silently accepting them. Preserve previous CSV versions and audits for reproducibility; do not change verification flags merely to get a run past preflight.

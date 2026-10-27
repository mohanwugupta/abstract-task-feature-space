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

The dry run now includes **60 human targets and 540 generation jobs**: three prompt families × three repetitions × 60. The matrix upper bound is still 12,000 judgments at the unchanged 200-feature cap.

## Mapping and frozen selection

- `item_text` and `stimulus_language` supply model-facing text/language. Existing English labels are preserved; German originals are retained as `source_text` with `source_language=de`.
- `actionN` and `actionV` map to `event`; `object` maps to `object`; `abstract_concept` maps to `concept`. Counts are 24 concepts, 24 events and 12 objects.
- `harpaintner_2018_original_supplement` maps to the pipeline's `harpaintner_2018` source ID. The original identifier remains in `source_dataset_original`; publication row, reported supplement digest, URL and translation status remain available.
- All 36 Vinson development/held-out assignments are preserved. The 24 abstract `unassigned` slots receive deterministic splits by sorted item ID within each of the four six-item strata, using seed 42: five development and one held-out per stratum. Overall: **47 development, 13 held-out**. Source CSV split tags are not rewritten; `source_split` retains them. This is an assignment of missing splits, not a new selection from the full publication inventory.
- `selection_rules.json` records the exact held-out IDs, raw input hashes and amendment. It does not claim to reproduce the uploaded inventory's original selection process.

The adapter checks BOM-aware CSV structure, unique IDs and abstract source/translated labels, supported sources/classes/strata, exact class quotas, split tags, finite numeric values, original source-row metadata and agreement of the two files by ID and fields. Category proportions must be in 0–1 and sum to one within 0.0005 rounding tolerance. Reordering rows cannot change split assignments.

## Evaluation and source limits

`human_sm`, `human_is_e`, `human_sc`, and `human_va` are retained as **rounded category proportions**, labeled `SM`, `IS/E`, `SC`, and `VA`. The existing `category_counts` reference container is used with `measurement=rounded_category_proportions`; these values are not interpreted as integer production counts. Profile normalization may correct rounding. They support source-specific category composition after the scientific/alignment gates pass.

These CSVs do not contain individual abstract human feature strings or participant records. They therefore cannot supply abstract lexical feature recovery, binary human feature RDMs or participant split-half estimates. Generated category profiles still require independent reviewed coding and explicit coverage; ratings, human category values, source strata and split labels are excluded from model prompts.

The reference registry now says the supplied abstract summary is available, while retaining `verified=false`. The reported supplement SHA-256 is provenance metadata; it does not prove the original supplement bytes were obtained and inspected. Publication/codebook/terms verification remains separate from structural CSV validation. References must be materialized, hashed and read-only before passing the existing live source gate.

All 24 English abstract translations remain marked `published_loose_english_translation_not_independently_reviewed`. No translation was silently approved or replaced with German. Preflight, reports and the explorer require a reviewed item translation and its exact reviewed source mapping before a cross-language comparison can appear. A nonlive P0 placeholder allows inspection of the complete plan; it is explicitly labeled `awaiting_reviewed_harpaintner_translated_instructions`, and cannot execute live without reviewed translated instructions.

The full live pilot remains blocked by source/protocol/translation verification and the conservative budget bound above the $100 cap. This integration does not raise that cap, reduce the bank, change three repetitions, alter the configured Sol-low models or authorize new paid calls.

## Updating input data

Treat a CSV edit as a new data revision: keep both inventories consistent, review the revised provenance and update the corresponding pinned SHA-256 values in configuration. The reference registry's abstract `sha256` must also match the abstract file before it can be loaded. An ordinary dry run refuses changed bytes instead of silently accepting them. Preserve previous CSV versions and audits for reproducibility; do not change verification flags merely to get a run past preflight.

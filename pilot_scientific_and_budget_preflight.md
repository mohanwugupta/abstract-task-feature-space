# Human-norm pilot: scientific verification and $100 cost preflight

**Scope:** audit of supplied standalone ZIP and user-provided `Data Sheet 1.xlsx`, 27 October 2026. No live model run was initiated. The $0.002814 spend is **user-reported**, not independently visible in the uploaded ZIP. The ZIP appears to predate the latest local $100-budget changes: its config still specifies a $500 cap and 30,000 input tokens for generator and judge.

## Verified primary-source material

**Vinson & Vigliocco (2008)**: Primary paper: https://link.springer.com/content/pdf/10.3758/BRM.40.1.183 ; methods p.186 (PDF page 4), full instruction Appendix p.190 (PDF page 8). The original task asked participants to describe each word completely via short, separate features, not dictionary definitions or pure associations, with examples that included an object noun and an action verb. No fixed count of four or ten features was imposed. Each 30–40-word list was completed by 20 participants; the entire study reports 280 participants. The bundled original source archive and derived CSV exist in `data/feature_space_pilot/references/`; verify the registered sha256 values during preflight, then update `verified` and provide a versioned, source-faithful P0 protocol. Do not apply Harpaintner's four-response cap to Vinson.

**Harpaintner et al. (2018)**: https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.01748/full (Materials and Methods). Subjects were asked for about four spontaneously produced *properties, associations or situations*, without time limits, avoiding synonyms. The supplied workbook explicitly labels its English words **loose English translations** and includes 296 German concepts and aggregate category frequencies (SM, IS/E, SC, VA). Uploaded workbook SHA-256: `a0d94ee950f5dcdb5bd753cb9050ddeda4a96d8493bc63772113bc31a39af293`. **All 24 selected rows** in `human_norm_pilot_abstract_24.csv` were matched back to original workbook rows on original German, English label, and four human category proportions (0 discrepancies). This verifies internal derivation, not an independent publisher-download chain of custody. Add reference schema and origin URL as provenance and retain raw workbook read-only.

The abstract reference supports category-composition and category-RDM comparisons; it does **not** provide participant-level properties or an item-by-free-text-feature reference for feature recall. Treat those analyses as unavailable, not as zeros. Original categories derive from German responses: using loose English translations as model stimuli changes the language and sometimes the concept sense. **Recommended primary comparison:** German stimuli for Harpaintner P0; optionally test English as a separately documented cross-language robustness condition. Use `pilot_abstract_translation_review.csv` for a simple human sign-off before claims about translated items.

## Important standalone repository integration fixes

- `stimuli_human.jsonl` in the supplied ZIP currently contains only 36 Vinson items. Convert the supplied 24 abstract rows to the registry item schema, preserve original German (`text` for primary P0) and loose English (`english_text` metadata), seal item hashes, preserve intended development/held-out splits, and assert the exact frozen 24/24/12 stratum counts.
- `source_registry.json` still marks Harpaintner unavailable; its expected dataset_id is `harpaintner_2018`, whereas the CSV says `harpaintner_2018_original_supplement`. Map these deliberately. Add a verified source adapter for the Excel source or a deterministic derived CSV with a hash and workbook traceability.
- `generation_prompt()` blocks human P0 on Vinson unless a source-specific `protocols['vinson_2008']` entry is supplied and `verified=true`. Supply a versioned prompt derived from Appendix p.190 with no false assertion of being verbatim where its format is modified. Keep Harpaintner's P0 separate.
- Registry and source comparisons must preserve German words/diacritics and not equate an English translation with the original German without declaring the language condition.
- Preserve TDD: write failing tests for missing abstract source, source hash mismatches, German P0 routing, Vinson P0 routing, ambiguous translation flag, full 60 quota, and downstream metric availability; record RED, GREEN and REFACTOR in `scratchpad_feature_space_pilot.md`.

## Prompt-size audit (from checked-in prompt builders, not billable API logs)

75 targets = 60 human + 15 case-study targets; 3 generation templates per item, 10 independent replicates each = 2,250 generation calls. Includes actual prompt text, serialized user JSON, and response schema, not chat transport overhead.

| Stage | Measured templates | Median request bytes | 95th percentile | Largest |
|---|---:|---:|---:|---:|
| Generation | 225 | 955 | 1402 | 6701 |
| Binary judgments (one representative feature) | 75 | 1351 | 1680 | 6979 |

The maximum is the finance+deadline+review task, which includes serialized invoice/payment data. **Bytes are not tokens.** A model-specific tokenizer or provider token-counting endpoint must be used for final limits, and API output ceilings must account for JSON/response_schema overhead and model reasoning tokens where applicable. Use a one-call live sample to calibrate token counts (only with budget authorization), then choose limits at or above the measured maximum and add explicit validation that requests stay within budgeted limits. A 30,000-token assumed input for every atomic judgment is not supported by these actual prompt sizes.

## Conditional budget scenarios (not promises of a bill)

Illustrative published GPT-5 mini **standard** rates: $0.25 per million input tokens, $2 per million output tokens (verify selected provider/snapshot and current rates immediately before live run): https://openai.com/index/introducing-gpt-5-for-developers/ . For planning only, suppose each attempt reserves **7,500 input tokens**, up to **700 generation output tokens** or **400 binary-judgment output tokens**, with **two attempts reserved for every job** (actual usage likely much lower). These values are proposed engineering envelopes, not a measured-token guarantee.

- Generation (2,250 jobs): **$14.74 ceiling** at the above rates.
- 12,000 binary judgments (human stage only, 60 × 200): **$64.20 ceiling**, combined **$78.94** before merge/adjudication/embedding.
- The FULL pilot at bank cap 200 is **15,000** judgment cells (75 × 200), **$80.25** judgment ceiling, combined **$94.99**, leaving little for any other stages.
- At bank cap **150** for all 75 targets, 11,250 judgments: **$60.19** judgment ceiling, combined **$74.92**, leaving **$25.08** for audit, merge, embeddings and buffer.

These counts count two attempts on every job as a conservative reservation. The smaller 150-candidate bank must be selected/frozen by documented label-blind criteria **before** judging; no skipping cells mid-run. Source/category-similarity coverage tradeoffs must be reported.

Do not automatically enforce a 7,500-token envelope by setting only `max_input_tokens` in config: the current HostedClient does not transmit or verify an input-token limit. Have the controller count tokens and reject over-cap payloads before reserving; settle all actual usage from the provider ledger. Set output limits separately for generation/judgment and verify schema compliance. Apply $100 cumulative cap across human **and** tasks with stage caps totaling no more than $100, including the user's existing spent amount if it occurred under the same ledger. Do not overwrite the actual spent amount with the old ZIP's $0 fixture balance.

Suggested test-first implementation slices: source loader and crosswalk → stimulus registry 60 → P0 instructions separation → exact prompt-size counting → budget simulations with 75×150 scope and cumulative cap → one limited paid smoke with ledger verification. No full pilot has run.

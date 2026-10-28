# Human validation and target-specific task discovery

The pilot asks whether a reproducible feature-generation procedure yields meaningful, interpretable properties before studying LLM task preferences. Experiment 1 evaluates correspondence with human semantic norms. Experiment 2 evaluates elicitation for realistic tasks. Their feature provenance and analyses remain separate. These findings reuse completed GPT-5 Mini generations and original automated Luna judgments; they are not a comparison across generator models. Neither experiment establishes internal feature use, a preference value, or a causal decision mechanism.

## Current outputs

The immutable empirical export is `artifacts/feature_space_collaborator/followup_23476f15019ef1c1/`. `artifacts/feature_space_collaborator/latest.json` points to the latest version.

- [Human scientific report](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/human/scientific_report.md)
- [Human concept-similarity matrices](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/human/figure_H1_similarity_matrices.svg)
- [Secondary feature-aligned cosines](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/human/figure_H2_feature_alignment.svg)
- [Task scientific report](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/task/scientific_report.md)
- [Complete side-by-side P0/P1/P2 table](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/task/prompt_comparison.html)
- [Own-target validity figure](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/task/figure_T1_target_local_validity.svg)
- [Empirical prompt profiles](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/task/figure_T2_prompt_profiles.svg)
- [Discovery-only compositionality](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/task/figure_T3_discovery_composition.svg)
- [Collaborator motivation](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/motivation.md)

PNG copies, every underlying matrix/table, full proposition/repetition lists, provenance audits and a 19-case manual-review table accompany these files. Generated empirical exports remain local and are excluded from Git.

## Human benchmark: independent representational similarity

The primary analysis compares 36 selected Vinson concepts in the exact same order. Human vectors retain all **1,029 original coordinates and published frequencies**. Raw word plus action-noun/action-verb/object type resolves eleven homonymous selected words; word text alone would select the wrong representation. Original repeated coordinate labels are retained under their published indices.

Each LLM prompt uses the independently generated human-experiment vocabulary, with weights counting how many of three repetitions generated a canonical feature. A feature counts at most once per repetition. Primary first-ten-per-call normalization preserves comparable generation budgets; binary, content-only, surface-form and all-output sensitivities are exported. No task feature or human-label matching defines these primary coordinates.

Spearman correlates the 630 upper-triangle cosine similarities. Each of 9,999 permutations shuffles concept labels consistently along both axes, preserving dependence among pairwise observations. Two-sided p-values use the +1 estimator; Holm adjusts the three primary tests. Undefined cosine/rank correlations remain unavailable.

| Prompt | Primary rho | Permutation p | Holm p | Zero off-diagonal similarities | Content-only rho |
|---|---:|---:|---:|---:|---:|
| P0 | 0.117 | 0.0004 | 0.0008 | 99.4% | 0.117 |
| P1 | 0.189 | 0.0002 | 0.0006 | 95.1% | 0.141 |
| P2 | 0.114 | 0.0006 | 0.0008 | 99.4% | 0.114 |

Human similarities are zero for 23.5% of pairs. The LLM spaces have only 4, 31 and 4 nonzero pairs, respectively. They recover meaningful relations such as cow–goat, but much of the concept geometry is unmeasured by exact shared canonical features. A low overall correlation under this sparsity cannot establish poor semantic organization. It also does not demonstrate broad recovery.

Binary-weighted RSA is similar (P0 0.117, P1 0.188, P2 0.114). Before semantic consolidation, raw-surface frequency RSA is lower (0.097, 0.166, 0.092). These sensitivities are exported, not used to select a favorable primary result.

P1's apparent advantage partly reflects grammar/metalinguistic properties. After their removal only nine P1 pairs remain nonzero. Exploratory within-type label permutations give content-only p-values of 0.0008, 0.0038 and 0.0020, respectively. These sensitivities and shared-pair original propositions expose the limited sources of similarity; they do not establish a universally superior prompt or a human-noise-normalized result.

## Human benchmark: secondary supervised alignment

Existing automated semantic judgments project LLM propositions onto the original human coordinates. One canonical proposition/repetition contributes at most one unit of mass, split equally across clear matches. Within-call paraphrase matches must agree; per-label contributions are capped before averaging. Related/uncertain labels do not count. Binary and single-label-only sensitivities are exported.

| Prompt | Mean aligned cosine on the same 35 concepts | Matched generation units | Unmatched generation units |
|---|---:|---:|---:|
| P0 | 0.663 | 787/1,063 | 276/1,063 |
| P1 | 0.589 | 548/968 | 420/968 |
| P2 | 0.536 | 548/1,035 | 487/1,035 |

P2 has an empty aligned projection for “the clash”; its cosine is undefined, not zero. All 36 per-concept results remain exported, with paired averages over the same 35 defined concepts. Unmatched properties are not automatically invalid.

The audit found eleven matcher inventories containing labels from other word types. Those labels are filtered against the exact typed human representation, with every decision retained. The original grader still saw the larger inventories, so filtering cannot remove all resulting grading bias. Two published text labels occur twice in the coordinate list; all coordinates are preserved and ambiguous coordinate assignments are withheld rather than guessed.

A stricter single-label-only check changes the picture substantially. On the same 34 concepts with defined scores under both policies for all three prompts, mass-split versus single-label-only mean cosines are P0 **0.663 vs 0.491**, P1 **0.591 vs 0.391**, and P2 **0.534 vs 0.375**. These means are computed from `feature_aligned_cosine.csv` after requiring all six scores to be defined. Thus correspondence depends appreciably on the grader’s multi-label mappings, even with mass limits.

This is **supervised, target-conditioned correspondence**, not independent feature-space RSA. It supports some human-property recovery but does not independently validate global representational structure or grader accuracy. Historical human-list recall predates this sense audit and should not be treated as the new primary validation result.

## Reliability and abstract concepts

[The source-paper audit](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/human/human_reliability.md) records exact metrics, samples, procedures, values, pages and citations. Neither reviewed publication supplies a comparable numerical participant-level feature-space ceiling. Researcher coding agreement and consensus requirements cannot serve as that ceiling. No participant LOOCV or ceiling-normalized similarity was computed.

Harpaintner's 24 abstract concepts remain unavailable for quantitative profile comparison: they have aggregate category proportions rather than individual human feature strings, and reviewed translations/generated category coding are incomplete. Original German, English translations, SM/IS-E/SC/VA summaries and provenance remain preserved.

## Tasks: primary judging is target-specific

Each description has its own independently consolidated inventory from its P0/P1/P2 outputs and repetitions. Shared identifiers support discovery comparison; they never import another target's candidate into its judging set.

Local reconstruction yields **713 primary cells** rather than the historical 8,880-cell cross-product. Completed evidence resolves **648 cells: 635 yes and 13 no**. The other **65 are unknown: 62 ambiguous, three invalid**. No valid native judgment was rerun, and no new inference was submitted. These are automated judgments, not human-gold precision estimates.

The full table preserves all **875 accepted raw phrases**, 131/135 accepted jobs and every repetition. Per-target primary inventories, original prompts/models/hashes, verdicts, incomplete jobs and a manageable review sample are exported. The historical 15 × 592 matrix and all original API responses remain untouched.

Native judging still needs validation. Two review cases have concrete counterevidence: the frozen finance description contains 19 `payment_id` records, yet the corresponding P1 statement was judged no; the poem retains ABAB CDCD, yet P2’s statement that lines 6 and 8 must rhyme was judged no. The original reconciliation fixture matches its source-manifest hash. These indicate questionable judge negatives, not verified generation failures. No verdict was changed.

## How do prompts differ empirically?

Rates below use the same thirteen fully sampled targets, averaging within calls/targets before averaging targets. The two incompletely sampled finance targets remain in the comprehensive export. Categories are transparent lexical screening, not validated semantic annotations or statistical claims about all tasks.

| Prompt | Mean canonical discoveries / target | Resource-demand cues | Evaluation/feedback cues | Instruction-restatement cues |
|---|---:|---:|---:|---:|
| P0 | 19.9 | 10.2% | 17.1% | 39.5% |
| P1 | 13.0 | 13.1% | 19.0% | 65.6% |
| P2 | 16.9 | 18.5% | 20.5% | 58.1% |

P0 often supplies familiar broad properties, such as time/money constraining a creative project. P1 can provide operational detail but often reads the description literally; for a broad label it can simply state that the target is labeled “Creative production.” P2 elicits more resource/evaluation/social language at broad levels, but concrete descriptions still elicit line, theme, rhyme, invoice and amount details. P2 has the highest specific-detail cue rate here, so it is not automatically the most generalizable procedure.

No explicit incentive, uncertainty or conflict/trade-off cues were detected in this complete cohort. Consequence cues were rare. This is a limitation of the current elicitation evidence and the screening method, not proof that no implicit meanings occur. Full per-level tables, common/prompt-specific discoveries, near-duplicate candidates and actual propositions support review.

## Compositionality and next questions

G(T), G(S), G(T+S) are compared descriptively under equal-planned-call and full-sample allocations. **Not generated does not mean inapplicable.** No 0/0/1 interaction is inferred from discovery absence, and no interaction is confirmed. Shared original discoveries survive local reconsolidation in the comparison identities.

The separate targeted-validation design contains 575 logical cells, with 450 already having resolved historical responses potentially reusable in a separately reviewed analysis. These are not 575 new API requests. The design does not execute; unresolved or new paid work requires a cost preflight and authorization under the unchanged cumulative $100 cap.

Before preference experiments, review native judgment failures and unknowns, distinguish grounded facts from conditional hypotheses, and refine elicitation of functional demands and genuinely evaluative consequences. Retain P0 as a broad-property baseline and P1 as an explicit-constraint comparator; P2 is promising for resource/evaluation demands but has not established reliable decision-feature discovery. Additional task families and behavioral manipulations are needed before drawing preference or causal conclusions.

## Reproduce offline

```bash
MPLCONFIGDIR=artifacts/feature_space_full/matplotlib_cache .venv/bin/python -m feature_space_pilot.collaborator.run
```

Use `--source-export PATH`, `--permutations 9999`, `--seed 20261028`, or `--no-figures` as needed. No API key or provider connection is required. Existing exports are immutable and new source/code/library identities receive new output IDs.

```bash
.venv/bin/python -m feature_space_pilot.collaborator.run --verify artifacts/feature_space_collaborator/followup_23476f15019ef1c1
```

Original empirical files, source manifests, provider jobs and ledgers remain unchanged. Code and source hashes, library versions and permutation policy are recorded. See [the data dictionary](DATA_DICTIONARY.md#current-human-validation-and-target-local-task-analyses) and [the TDD record](scratchpad_feature_space_pilot.md).

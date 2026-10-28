# Task and situation dataset — corrected primary analysis

Use the independent **Experiment 2** outputs. The previous task-only view of the pooled 5,825-feature vocabulary is retained as secondary exploration; it is no longer the primary task representation. [The experimental separation guide](EXPERIMENT_SEPARATION.md) explains the correction and the separate human benchmark.

## What has been processed?

- **131/135 generation jobs** were accepted across the 15 task/situation items, P0/P1/P2 and three repetitions.
- The **875 task-origin phrases** were independently reconstructed and conservatively consolidated into **592 canonical task features**. The count coincides with the previously reported task-origin subset, but reconstruction uses original provenance and local equivalence evidence, not a text filter on the pooled bank.
- Every one of the **15 × 592 = 8,880 cells** has compatible prior attempt evidence. **6,976 are definitively 0/1; 1,879 ambiguous and 25 invalid remain `?`.**
- Shared/distinctive features, abstraction transitions, situation inheritance, discovery coverage and component/direct comparisons have been recomputed using only this bank.
- Semantic families and specificity tags are provisional offline coding. They do not merge propositions or replace binary columns.

## Figures to open

1. [Figure A — Task feature-space overview](artifacts/feature_space_experiments/separated_9c9d7f47f2e8514d/task/figure_A_task_feature_space.svg): actual task-generated propositions organized by overlapping semantic families, with all 15 applicability profiles.
2. [Figure B — Abstraction and generalization](artifacts/feature_space_experiments/separated_9c9d7f47f2e8514d/task/figure_B_abstraction.svg): broad and specific properties across domain, type, instance and situated variants, separately for writing and reconciliation. Red outlines mark inherited-text disagreements requiring review.
3. [Figure C — Compositionality](artifacts/feature_space_experiments/separated_9c9d7f47f2e8514d/task/figure_C_compositionality.svg): shared discoveries, component-only discoveries, direct additions attributable to component omissions, provisional 0/0/1 candidates, inapplicable additions and unresolved evidence. Each deadline/review/both condition and prompt family is separate.

PNG versions accompany every SVG. Display examples are exported explicitly; the full 592-feature bank and all matrix cells remain available. The figure selection is not a feature cap.

Read [the full task scientific report](artifacts/feature_space_experiments/separated_9c9d7f47f2e8514d/task/scientific_report.md) for actual propositions, per-item coverage, transition counts, sampling gaps, quality flags and candidate properties for later behavioral validation.

## How to interpret the results

The automated family comparison finds 119 shared, 164 writing-distinctive, 138 reconciliation-distinctive and 171 unresolved features. Family presence requires at least one yes; family absence requires every item in that family to be no. These unreviewed labels include questionable positives, which are flagged rather than automatically corrected.

Task abstraction and feature abstraction are different: domain → task type → instance is the existing item hierarchy; a feature may describe a broad functional demand or an exact instance constraint. Situations are another axis. Semantic families, specificity and identical applicability profiles are separate interpretation aids. None establishes semantic equivalence or an internal model mechanism.

Direct generation often finds an existing component property omitted during independent generation. Six features exhibit the 0/0/1 candidate pattern; no genuine relational emergence is confirmed. Missing generation roles and ambiguous labels limit those comparisons. The diagnostic set includes unchanged requirements with inconsistent labels, cross-domain positives and candidate interactions with all three descriptions for inspection.

The next scientific step is targeted grounding/consistency review and later behavioral tests of promising properties. A feature's presence does not assign it a preference value or demonstrate that the model uses it in decisions. No new paid judgments were launched by this correction.

## Reproduce offline

```bash
MPLCONFIGDIR=artifacts/feature_space_full/matplotlib_cache .venv/bin/python -m feature_space_pilot.experiments.run
```

`artifacts/feature_space_experiments/latest.json` points to the newest immutable export. Original experiments, provider jobs, frozen manifests and the cumulative $100 ledger remain unchanged. [The human benchmark report](artifacts/feature_space_experiments/separated_9c9d7f47f2e8514d/human/scientific_report.md) is separate and supplies no vocabulary to the primary task study.

# Task and situation study: current primary analysis

The primary judging scope is now **target-specific**. Each of the 15 descriptions is evaluated only against features independently generated for that same description by P0/P1/P2 across repetitions. The historical 15 × 592 cross-task matrix remains exploratory and is not the primary task representation.

## Start here

- [Full side-by-side P0/P1/P2 feature table](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/task/prompt_comparison.html): every accepted feature and repetition, target descriptions, summaries, native validity badges and original generation provenance. Opens offline without a server.
- [Task scientific report](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/task/scientific_report.md)
- [Own-target validity counts](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/task/figure_T1_target_local_validity.svg)
- [Empirical prompt profiles](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/task/figure_T2_prompt_profiles.svg)
- [Discovery-only compositionality](artifacts/feature_space_collaborator/followup_23476f15019ef1c1/task/figure_T3_discovery_composition.svg)
- [Collaborator findings and independent human validation](HUMAN_VALIDATION_AND_LOCAL_TASKS.md)

PNG versions accompany the SVG figures. Full local inventories and judgments are under `task/target_inventories/<target>/`; `generation_features.csv` preserves all 875 raw task phrases, prompt families, three repetitions, model/prompt hashes and local verdicts. No examples-only feature cap is imposed.

## Current evidence

Local reconsolidation yields **713 primary cells**. Existing evidence resolves **648: 635 yes and 13 no**. **65 remain unknown: 62 ambiguous and three invalid.** All previous responses are preserved; no native judgments were rerun. The 131/135 accepted generation jobs leave missing samples visible.

The complete paired comparison uses thirteen fully sampled targets; incomplete finance targets remain in full tables. P0 tends to produce broad/familiar properties; P1 is more literal and can produce label metadata; P2 has more resource/evaluation/social cues but still restates many explicit instructions and specific details. The declared cue screen detected no explicit incentive, uncertainty or conflict/trade-off language in this complete cohort. Cue coding is provisional, not semantic gold or a claim of universal prompt superiority.

## How to interpret comparisons

Shared property identities support comparing independently generated sets. They never add a candidate to another target's primary inventory. If a feature was not generated for the base task, its applicability there is unknown under the primary design; it is not assigned zero.

Compositionality therefore describes overlap among G(T), G(S), and G(T+S), with equal-planned-call and full-sample allocations plus explicit sampling gaps. New direct discoveries do not establish interaction-dependent applicability. A separate non-executing targeted validation design identifies where compatible historical evidence might be reused; new paid judgments require a fresh cost preflight and authorization.

Read `manual_review.csv` for a manageable 19-case provenance-preserving review set. Native judge negatives and unknowns, near duplicates, instruction restatements, specific details and possible evaluation features are candidates for inspection. No human approvals or automatic corrections are invented. A feature's presence does not establish preference direction or internal/causal feature use.

## Reproduce

```bash
MPLCONFIGDIR=artifacts/feature_space_full/matplotlib_cache .venv/bin/python -m feature_space_pilot.collaborator.run
```

`artifacts/feature_space_collaborator/latest.json` points to the latest immutable export. The workflow is offline, makes no model API calls and preserves original experiments and the cumulative $100 ledger.

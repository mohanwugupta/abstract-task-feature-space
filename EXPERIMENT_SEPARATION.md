# Independent human benchmark and task feature discovery

The scientific correction follows [the separation specification](PRD_Experimental_Separation.md). Experiment 1 validates the feature-generation method against human semantic norms. Experiment 2 applies that method to construct the task representations for future preferences experiments. Human-only discoveries supply no vocabulary to Experiment 2.

## Current primary outputs

The corrected empirical export is `artifacts/feature_space_experiments/separated_9c9d7f47f2e8514d/`. `artifacts/feature_space_experiments/latest.json` points to the most recently reconstructed checkpoint.

- [Experiment 1: human-norm validation report](artifacts/feature_space_experiments/separated_9c9d7f47f2e8514d/human/scientific_report.md)
- [Experiment 2: task feature-space scientific report](artifacts/feature_space_experiments/separated_9c9d7f47f2e8514d/task/scientific_report.md)
- [Figure A: task feature-space overview](artifacts/feature_space_experiments/separated_9c9d7f47f2e8514d/task/figure_A_task_feature_space.svg)
- [Figure B: abstraction and generalization](artifacts/feature_space_experiments/separated_9c9d7f47f2e8514d/task/figure_B_abstraction.svg)
- [Figure C: task–situation compositionality](artifacts/feature_space_experiments/separated_9c9d7f47f2e8514d/task/figure_C_compositionality.svg)

Generated empirical artifacts stay local and are excluded from Git. The code, specification, summaries and data dictionary are versioned.

| Primary experiment | Accepted generation jobs | Source phrases | Canonical features | Matrix | Reused definitive cells | Unresolved |
|---|---:|---:|---:|---|---:|---:|
| Human methodology benchmark | 540/540 | 5,653 | 5,233 | 60 × 5,233 | 69,365 | 244,615 |
| Task/situation discovery | 131/135 | 875 | 592 | 15 × 592 | 6,976 | 1,904 |

Every task cell has compatible prior attempt evidence. The unresolved task cells comprise 1,879 ambiguous responses and 25 invalid responses, rather than 1,904 new primary requests waiting to be sent. The human applicability matrix remains substantially incomplete; the separate human-label correspondence analysis is fully scored for its available 36-target reference scope.

## How well does feature generation recover human norms?

The preserved automated matcher recovers the following fractions of supplied human feature labels, averaged over the same 36 targets, with the original first-ten-features-per-call normalization and three repetitions:

| Prompt | All 36 targets | 12 concrete objects | 24 events/actions |
|---|---:|---:|---:|
| P0 | 39.3% | 46.5% | 35.7% |
| P1 | 32.0% | 42.6% | 26.7% |
| P2 | 32.5% | 41.0% | 28.3% |

This is partial human-list recovery judged by Luna, not precision against truth or validated grader accuracy. P0 has the highest point estimate; that alone does not establish a significant prompt-family advantage. Generated properties absent from a human list are not automatically false.

The 24 abstract concepts have aggregate Harpaintner category proportions, not individual feature lists. English translation review and generated category coding remain pending, so abstract feature-list recall and a validated category-profile comparison cannot be supplied. The report preserves reference availability, alignment diagnostics, original model identities, and discovery/saturation tables by stimulus group. Three repetitions do not establish asymptotic saturation.

## What did we discover about tasks?

The task-only bank contains constraints on outputs, operations and information, quality assessment, resources, audiences and task–situation relations. The export keeps actual propositions, original discovery origins and the independent applicability values.

Across each family's six domain/type/instance/situated items, automated labels classify **119 features as shared**, **164 as writing-distinctive**, **138 as reconciliation-distinctive**, and **171 as unresolved**. Presence means at least one observed yes; distinctiveness additionally requires every item in the other family to be independently no. These are descriptive model-label results, with cross-domain and inheritance disagreements explicitly audited.

Grounded examples include the poem's eight-line ABAB CDCD requirement and reconciliation by exact invoice ID and currency. Writing features concern imagery, theme, vivid language and rhyme; reconciliation features concern arithmetic, payment/invoice matching, discrepancy status and traceability. More general candidates concern deadlines, checking for errors, audiences and independent evaluation. Specific counts and imagery are distinguished from functional properties. These categories do not imply that any property causes an LLM preference.

Figure A adds provisional overlapping semantic families. Figure B follows actual propositions through domain → task type → instance and across deadline, review and combined situations. Figure C compares component and direct discoveries within each prompt family. Full binary columns remain intact; the offline category/specificity tags are supplementary, transparent rule coding rather than human-reviewed semantic annotations.

## What changes with abstraction and situations?

Many general features carry across levels; explicit instance constraints become applicable at the concrete task. Situations add timing and evaluation properties. Every individual transition is exported, including unknowns and observed losses. Raw positive counts are lower bounds when cells remain unknown.

There are **86 parent-positive/situated-negative screening flags** where the parent text is unchanged. For example, “lines 1 and 3 must rhyme” is positive for the base poem and negative for its deadline version despite both retaining ABAB CDCD. These flags are not all verified errors: context-sensitive properties can legitimately change. No labels were automatically corrected.

## Does direct generation add genuinely new relational information?

Direct task–situation generation discovers many phrases not independently generated for the components. Most definitively classified applicable additions are component-discovery omissions: the feature already applies to the task or situation. Novel wording or discovery membership is not emergence.

Six distinct features have the operational T=0, S=0, TS=1 pattern in the primary equal-planned-call comparisons. All occur in P2 reconciliation comparisons with incomplete generation coverage. They include binding a finished discrepancy report to a five-minute deadline and specifying independent review of matching accuracy and traceability. Such features may restate explicit instruction conjunctions or reflect component false negatives. **No genuinely new relational interaction is confirmed.** The audit supplies all three descriptions for targeted grounding review.

## What matters before preferences research?

Use the task feature space as a descriptive candidate inventory. The diagnostic set screens irrelevant cross-domain positives, unchanged requirements with inconsistent labels, instruction restatements/instance details, near duplicates, word/dataset-label properties, uncertain evaluation relevance, and interaction-candidate grounding. Suspected issues, unresolved cells and verified corrections remain separate; verified corrections are currently empty.

The behavioral-validation shortlist excludes features with cross-domain, inherited-disagreement or linguistic-property flags. It identifies candidate functional/evaluation properties without assigning positive/negative preference values. Testing whether models actually use those properties requires later behavioral interventions. Two case-study families do not establish population-wide generalization.

## Offline workflow and preservation

```bash
MPLCONFIGDIR=artifacts/feature_space_full/matplotlib_cache .venv/bin/python -m feature_space_pilot.experiments.run
```

Use `--pooled-run PATH` to select another compatible source project or `--no-figures` for tables/reports only. There is no paid-execution option. Repeated evidence/code checkpoints get distinct output identities; existing sealed exports are immutable.

```bash
.venv/bin/python -m feature_space_pilot.experiments.run --verify artifacts/feature_space_experiments/separated_9c9d7f47f2e8514d
```

The source checkpoint records immutable byte hashes and fixed complete-line prefixes of append-only judgments and the budget ledger. Original frozen manifests, generation responses, provider jobs, active processes and ledgers are untouched. The pooled 5,825-feature experiment remains available as secondary exploration. No API credentials are read and no additional paid calls are made. Remaining paid judgments require a new cost preflight and authorization under the existing cumulative $100 cap.

See [the artifact dictionary](DATA_DICTIONARY.md#independent-primary-experiments), [the task figure guide](TASK_DATASET_BRIEFING.md), and [the append-only TDD record](scratchpad_feature_space_pilot.md).

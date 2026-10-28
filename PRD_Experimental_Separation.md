# Fix the Experimental Separation: Human-Norm Benchmark vs. Task Feature Discovery

We identified an important conceptual error in the current feature-space pilot. The human-norm benchmark and task feature-discovery study were intended as **two distinct experiments**, but the implementation pools their generated features into a single 5,825-feature vocabulary and uses this shared vocabulary as the primary basis for both analyses.

**Please correct this separation without discarding, regenerating, or overwriting existing data.**

## 1. Understand the intended experimental design

### Experiment 1: Human-Norm Benchmark

**Purpose:** Validate our LLM-based feature-generation methodology against existing human semantic feature norms.

- Stimuli: 60 concepts (abstract concepts, actions/events, concrete objects).
- Generate features using P0, P1, and P2.
- Compare LLM-generated features against the available human reference data.
- Evaluate semantic correspondence, feature coverage, differences between prompt families, and discovery saturation.
- Harpaintner supplies aggregate feature-category proportions, not individual human feature lists.
- The main scientific question is whether our approach can generate coherent, meaningful properties of concepts, including abstract concepts.

**This is a methodological benchmark, not the source of the task feature vocabulary.**

### Experiment 2: Task Feature-Space Discovery

**Purpose:** Construct a generalizable descriptive feature space for real-world tasks that we can subsequently use to investigate LLM preferences.

- Stimuli: 15 task/situation items.
- Case studies: creative writing and financial reconciliation.
- Representations: domain, task type, concrete instance, standalone situation, and situated task.
- Generate features using P0, P1, and P2.
- Semantically consolidate the features generated from these 15 items.
- Independently judge whether every resulting task feature applies to each of the 15 items.
- Analyze abstraction, generalization, and task–situation compositionality.

**Only features discovered from the task/situation stimuli should define Experiment 2's primary feature vocabulary.**

The current repository reports 875 task-generated phrases mapping to 592 canonical feature IDs in the pooled vocabulary. These are the starting point for reconstructing the task-only bank, not necessarily its final canonical count.

## 2. Separate the feature banks

Create two independent, provenance-preserving feature inventories:

- `human_feature_bank`: features generated from the 60 human-norm stimuli.
- `task_feature_bank`: features generated from the 15 task/situation stimuli.

The banks must have separate identities, hashes, consolidation decisions, and analysis outputs.

Important implementation details:

- Use original generation provenance to identify each feature's source.
- Do not simply filter by text if a feature was generated independently in both experiments.
- A feature generated from a task remains eligible for the task bank even if the same proposition was also generated from a human-norm concept.
- Reconstruct consolidation within each experiment, using existing semantic-equivalence evidence where valid.
- Preserve original source phrases, prompt families, repetitions, model identities, and canonical mappings.
- Preserve meaningful differences involving negation, conditions, modality, roles, and relational direction.
- Do not introduce an arbitrary feature cap.

If existing pooled-bank equivalence decisions cannot be safely transferred, flag them for review rather than assuming equivalence.

The pooled 5,825-feature bank can remain available as a **secondary exploratory analysis**, but it must no longer be the primary representation for either experiment.

## 3. Reconstruct the applicability matrices

Create separate matrices:

**Experiment 1:** 60 human-norm concepts × human-generated canonical features.

**Experiment 2:** 15 task/situation items × task-generated canonical features.

Use existing applicability judgments whenever the source item, proposition meaning, rubric, model configuration, and prompt identity are compatible.

Do not rerun judgments just because feature IDs or bank organization changed. Maintain an explicit mapping from old IDs to new IDs.

Preserve unknown or ambiguous judgments as `?`. Never convert missing values into zeros.

Do not alter any active provider jobs, running inference processes, original frozen manifests, or existing budget ledgers. The ongoing pooled experiment may remain intact; the corrected experiments should be constructed from existing evidence.

## 4. Rebuild the scientific analyses separately

### Experiment 1 report

Provide a focused methodological validation report:

- Human-feature correspondence for P0, P1, and P2.
- Comparisons across abstract concepts, events/actions, and concrete objects.
- Discovery coverage and saturation.
- Semantic feature validity and precision diagnostics where feasible.
- Human-reference availability and limitations.
- Conclusions about how well feature generation works as a methodological approach.

Do not mix task-discovered features into these analyses.

### Experiment 2 report

This should be the **primary output for the preferences research program**.

Using only the task-generated feature vocabulary:

1. Show which features characterize creative writing versus financial reconciliation.
2. Identify features shared between the task families and features that distinguish them.
3. Show how features change from domain → task type → concrete instance.
4. Show which features are inherited when situations are added.
5. Analyze separately the effects of deadlines, external evaluation, and both situations.
6. Compare direct feature generation for task + situation against independent generation for task and situation.
7. Distinguish additional feature discovery from genuinely interaction-dependent properties.
8. Report incomplete generations, unresolved applicability judgments, and inconsistent labels explicitly.

Recompute coverage, feature overlap, and co-applicability statistics using the task-only vocabulary.

Do not treat identical binary applicability patterns as evidence that propositions are semantically equivalent.

## 5. Make the task results interpretable

The goal is not just to create another large matrix. We need to understand whether the discovered properties could be useful in preference experiments.

Produce three primary visualizations:

**Figure A — Task feature-space overview**

Organize task-generated features into interpretable, potentially overlapping semantic families, such as:

- Goals and outputs
- Operations and information requirements
- Constraints and resources
- Quality and evaluation criteria
- Social context and beneficiaries
- Relationships and trade-offs

These are provisional analytical labels, not predetermined feature dimensions or human-norm categories.

**Figure B — Abstraction and generalization**

Show how features carry across domain, task type, instance, and situated task. Distinguish broad functional features from highly specific details of an individual task.

**Figure C — Compositionality**

For each situated task, show:

- Features recovered through both component and direct generation.
- Additional features discovered directly from the combined description.
- Features that were merely omitted during independent generation.
- Candidate relational/conjunctive features.
- Unresolved or contradictory evidence.

Use actual feature propositions in the figures and accompanying tables, not only IDs or counts.

Keep the full binary feature representation intact. Semantic groupings and abstraction labels are supplementary metadata, not replacements for the original features.

## 6. Address obvious quality issues

The prior analysis identified several problems requiring targeted auditing:

- Clearly irrelevant cross-domain applicability judgments.
- Unchanged task requirements classified differently after adding situations.
- Features that merely restate explicit instructions.
- Near-duplicate propositions that remain separate.
- Features describing linguistic properties of words rather than properties of tasks.
- Features that are descriptive but not obviously useful for explaining task evaluations.

Build a modest, manually auditable diagnostic set.

Do not automatically alter all labels based on heuristics. Separate suspected errors, verified corrections, and unresolved cases.

We want the task feature space to contain valid properties of tasks—not merely propositions that are technically true or imaginable.

## 7. Preserve the preferences research objective

The downstream objective is to investigate which features LLMs evaluate when choosing or performing tasks.

The feature-space experiment does **not** yet establish that any discovered feature is used internally or determines preferences.

Accordingly, the final report should distinguish:

- Descriptive features directly grounded in tasks.
- Generalizable functional features.
- Candidate evaluatively relevant features.
- Features that are overly specific or potentially irrelevant.

Do not assign positive/negative preference values or invent a preference model at this stage.

Instead, identify which task features seem most promising for subsequent behavioral validation, especially those that generalize across task categories or interact with situations.

## 8. Strict implementation requirements

Follow **test-driven development: RED → GREEN → REFACTOR** for every substantial change.

Tests must verify:

- Complete separation of human and task feature provenance.
- No human-only features entering the primary task vocabulary.
- Correct handling of features independently discovered in both experiments.
- Conservative semantic consolidation.
- Valid reuse of previously completed judgments.
- Correct experiment-specific matrix dimensions.
- Preservation of unknown judgments.
- Correct compositionality calculations.
- Unchanged original empirical artifacts.
- No additional paid API calls during offline reconstruction.

Continue updating `scratchpad_feature_space_pilot.md` with the scientific rationale, implementation decisions, failing and passing tests, blockers, and final results.

Respect the existing cumulative $100 spending cap. This correction should initially be **entirely offline**. Do not launch additional paid judgments without a new cost preflight and authorization.

## 9. Definition of done

Deliver:

1. Separate human and task feature banks.
2. Separate experiment-specific applicability matrices.
3. Corrected human-norm validation report.
4. Corrected task feature-space scientific report.
5. Updated task-only visualizations and compositionality analysis.
6. An audit of reused versus unresolved judgments.
7. Tests demonstrating that the experiments are scientifically and computationally separated.
8. Updated README, data dictionary, and scratchpad.

In the final report, explicitly answer:

- How well does our feature-generation method perform against human norms?
- What generalizable task properties did we discover?
- Which task features are shared versus distinctive?
- What happens to feature representations as tasks become more specific or situations are introduced?
- Does direct task–situation generation add genuinely new relational information?
- Which unresolved issues matter before using these representations to study preferences?

**The most important correction is conceptual: Experiment 1 validates the methodology; Experiment 2 applies that methodology to construct the task representations we care about. They should not be treated as one pooled scientific experiment.**

Proceed with the offline correction, tests, and reports using existing data. Preserve active experiments and report any work that cannot be completed without new API calls.
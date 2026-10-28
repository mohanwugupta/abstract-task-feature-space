"""Scientific reports keep benchmark validation distinct from task discovery."""

from ..io import write_csv
from .analysis import behavioral_candidates


def csv_rows(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["status"]
    write_csv(path, rows, fields)


def _examples(rows, *, limit=4):
    return (
        "\n".join("- " + r["proposition"].replace("\n", " ") for r in rows[:limit])
        or "- None established in this checkpoint."
    )


def write_task_report(
    out, items, bank, reconstruction, analysis, diagnostics, coverage
):
    audit = reconstruction["audit_summary"]
    rows = analysis["family_features"]
    general = [r for r in rows if r["generalizable_candidate"]]
    promising = behavioral_candidates(
        rows, analysis["annotations"], diagnostics["all_flags"]
    )
    csv_rows(
        out / "behavioral_validation_candidates.csv",
        [
            dict(
                r,
                preference_value=None,
                behavioral_validation_required=True,
                rationale="Provisional functional/evaluation cue plus observed task applicability; no causal or preference claim",
            )
            for r in promising
        ],
    )
    equal = [r for r in analysis["compositions"] if r["sampling_mode"] == "equal_calls"]
    item_coverage = "| Item | Yes observed | No | Unresolved |\n|---|---:|---:|---:|\n"
    for n, item in enumerate(items):
        values = reconstruction["state"]["matrix"][n]
        item_coverage += f"| {item['item_id']} | {int((values==1).sum())} | {int((values==0).sum())} | {int((values<0).sum())} |\n"
    text = f"""# Experiment 2 — Task feature-space discovery

This is the primary descriptive feature space for subsequent preferences research. It contains only propositions independently generated from the task/situation stimuli. The human benchmark validates the method and supplies no feature vocabulary here. The preserved pooled experiment is secondary exploration.

## Evidence and coverage

- {len(items)} task/situation items; {coverage['completed']}/{coverage['planned']} accepted generation jobs.
- {len(bank['origins'])} accepted source phrases; **{len(bank['features'])} task-only canonical features**, with no arbitrary cap.
- Primary matrix: **{audit['n_items']} × {audit['n_features']} = {audit['planned_cells']:,} cells**.
- {audit['reused_binary_cells']:,} compatible definitive judgments reused; {audit['unresolved_cells']:,} unresolved ({100*audit['unresolved_cells']/max(1,audit['planned_cells']):.1f}%). Unknown is `?`, never zero.
- Cell states: {audit['cell_statuses']}. Generation gaps are explicit in generation_coverage.csv and missing_generation_jobs.csv.
- Bank identity: `{bank['bank_id']}`; hash `{bank['inventory_hash']}`.

{item_coverage}

Positive counts are observed lower bounds while unknowns remain. Unequal uncertainty prevents interpreting raw counts as richer or poorer representations.

## What characterizes the two task families?

Across each family's domain, type, instance and three situated variants, observed presence means at least one yes. Family absence requires all of its items to be independently no. Counts are {analysis['family_counts']}. These are automated applicability results, not independently verified task properties. The quality audit flags questionable positives explicitly. Family-level comparisons differ from the concrete-instance contrast, which is exported separately. Unknowns do not establish distinctiveness.

Shared properties:

{_examples([r for r in rows if r['comparison']=='shared'])}

Writing-family distinctive properties:

{_examples([r for r in rows if r['comparison']=='poem_distinctive'])}

Reconciliation-family distinctive properties:

{_examples([r for r in rows if r['comparison']=='finance_distinctive'])}

Generalizable functional candidates observed in both families: **{len(general)}**. These are provisional descriptive candidates, not a validated universal task ontology.

{_examples(general)}

## Abstraction, inheritance and the separate situations

Domain → task type → instance is an item hierarchy. Deadline, external evaluation and their combination form a separate situation axis. Figure B follows actual propositions across these items; transitions.csv retains every retained, gained, lost/disagreement, absent and unresolved cell. A general statement can be absent from a particular generation and still independently apply: generation frequency and applicability are different measurements.

| Parent → child | Retained | Gained | Lost / disagreement | Unresolved |
|---|---:|---:|---:|---:|
"""
    for r in analysis["transition_counts"]:
        c = r["counts"]
        text += f"| {r['parent_id']} → {r['child_id']} | {c.get('retained',0)} | {c.get('gained',0)} | {c.get('lost_or_inconsistent',0)} | {c.get('unresolved',0)} |\n"
    text += f"""
There are **{len(analysis['inheritance_flags'])}** parent-positive/situated-negative screening flags for unchanged parent text. These are not all verified errors: context can change some properties. Inspect explicitly unchanged requirements first. No labels were automatically corrected.

## Does direct task–situation generation add new relational information?

We compare G(T)∪G(S) with G(T+S) within P0, P1 and P2. Primary comparisons allocate equal planned generation-call counts, retaining full-sample sensitivity analyses separately. Equal calls do not mean equal tokens; actual tokens and missing roles are exported. Canonical IDs come exclusively from this task bank.

| Situated task | Prompt | Component/direct calls | Shared discoveries | Direct-only discoveries | Constituent omissions | 0/0/1 candidates | Unresolved features |
|---|---|---:|---:|---:|---:|---:|---:|
"""
    for r in equal:
        counts = r["counts"]
        text += f"| {r['item_id']}{'' if r['generation_coverage_complete'] else ' †'} | {r['prompt_family']} | {r['component_calls']}/{r['direct_calls']} | {counts['recovered_both']} | {counts['direct_only_discovered']} | {counts['constituent_discovery_gaps']} | {counts['interaction_candidates']} | {counts['unresolved_features']} |\n"
    text += (
        """
† Incomplete generation samples. Shared and additional discovery counts are set membership, not validated truths. A direct-only feature with T=1 or S=1 is an omission in component generation. A 0/0/1 pattern (T=0, S=0, TS=1) is an operational interaction candidate. Conditional or relational wording is separately tagged, and does not confirm an interaction. Missing/ambiguous cells prevent classification. Candidates are repeated across prompt-family analyses, so counts must not be added as independent features. **No genuinely new relational interaction has been confirmed.**

## Semantic families, feature scope and preference relevance

Figure A uses provisional, overlapping families: goals/outputs; operations/information; constraints/resources; quality/evaluation; social context/beneficiaries; relationships/trade-offs. The coding is transparent offline lexical screening, not an additional LLM judgment or a human-validated taxonomy. Unclassified features remain visible in the full table. Family memberships can overlap and counts are not additive. The full binary representation remains intact.

Specificity and semantic theme are separate metadata. The tables distinguish functional/general candidates, instance details, linguistic/label metadata, and cases requiring scope review. Generality suggested by wording is distinct from observed cross-family applicability. Identical binary profiles in coapplicability_patterns.json are not semantic equivalences; complete-case selection is reported explicitly.

For these case studies, distinguish four claims:

- Directly grounded descriptive requirements: the poem's eight-line ABAB CDCD format; reconciliation by exact invoice ID and currency. These are stated in the task descriptions, although individual applicability labels can still be inconsistent.
- Generalizable functional candidates: producing an artifact, checking constraints, working within available time, and delivering work for an audience or reviewer. General wording and applicability across these cases do not establish universality.
- Candidate evaluative properties: matching accuracy, traceability, vivid language, delivery time and independent quality assessment. Their importance to preferences remains untested.
- Specific or potentially irrelevant properties: the exact invoice count, particular imagery, or dataset/word labels. An instance detail can be valid while offering little cross-task explanatory value.

Promising subsequent behavioral-validation candidates:

"""
        + _examples(promising, limit=8)
        + f"""

These features have functional/evaluation cues and observed applicability; the shortlist excludes features with cross-domain, inheritance-disagreement or linguistic-property flags. Exclusion from a shortlist does not change a matrix label. They have no assigned positive/negative preference values. A descriptive feature, a generalizable demand, and a feature used internally to choose or perform a task are different claims. Behavioral interventions are required for the latter.

## Quality audit and unresolved limits

Diagnostic flags by category: {diagnostics['counts']}. The modest diagnostic_set.csv is a balanced, manually auditable screening set. diagnostic_flags.csv retains all flags, source texts and actual propositions. Near duplicates are suggestions, not merges. Explicit instruction restatements can be valid but may be overly specific; unmatched relevance cues do not prove irrelevance. Verified corrections: none; original labels are preserved.

Before preferences experiments, inspect inconsistent inherited requirements, inappropriate cross-domain positives, near-duplicate distinctions, the provisional category/scope annotations, unresolved applicability, and missing generations. The two task families support descriptive case studies, not population-wide task-generalization claims.

Figures: [A: task space](figure_A_task_feature_space.svg), [B: abstraction](figure_B_abstraction.svg), [C: compositionality](figure_C_compositionality.svg). Complete propositions and T/S/TS evidence are in composition_features.csv; each condition is reported separately. The reconstruction made no API calls and changed no original artifacts, provider jobs or ledgers.
"""
    )
    (out / "scientific_report.md").write_text(text)


def write_human_report(out, items, bank, reconstruction, analysis, coverage):
    audit = reconstruction["audit_summary"]
    norm_conclusion = (
        "The available automated evidence demonstrates partial correspondence with existing human property productions for the scored event/action and object targets."
        if any(
            r["mean_recall"] is not None for r in analysis["family_summary"].values()
        )
        else "Complete semantic scoring is unavailable; no evidence-supported norm-recovery conclusion can yet be drawn."
    )
    saturation_rows = {}
    for r in analysis["saturation_summary"]:
        saturation_rows.setdefault((r["stratum"], r["prompt_family"]), {})[
            r["replicates_included"]
        ] = r["mean_canonical_discoveries"]
    saturation_table = "| Stimulus group | Prompt | 1 call | 2 calls | 3 calls |\n|---|---|---:|---:|---:|\n"
    for (group, family), points in sorted(saturation_rows.items()):
        values = [
            f"{points[n]:.2f}" if points.get(n) is not None else "unavailable"
            for n in (1, 2, 3)
        ]
        saturation_table += f"| {group} | {family} | " + " | ".join(values) + " |\n"
    text = f"""# Experiment 1 — Human-norm methodological benchmark

This experiment evaluates the feature-generation method. Its vocabulary comes only from the 60 human-norm stimuli; task-discovered features are excluded. It is not the source vocabulary for the task study.

## Scope and independently reconstructed representation

{len(items)} benchmark stimuli; {coverage['completed']}/{coverage['planned']} accepted generation jobs; {len(bank['origins'])} accepted phrases and {len(bank['features'])} human-generated canonical features. Independent matrix: {len(items)} × {len(bank['features'])} = {audit['planned_cells']:,} cells, of which {audit['reused_binary_cells']:,} are resolved and {audit['unresolved_cells']:,} unresolved in this checkpoint. The applicability matrix is distinct from the feature-to-human-label semantic matching analysis below.

Bank identity: `{bank['bank_id']}`; hash `{bank['inventory_hash']}`. Benchmark source groups: {analysis['reference_scope']['target_groups']}.

Original generation models: {analysis['generation_models']}. Semantic matching uses its separately frozen grader configuration, preserved in human_validation.json; no model has been substituted during reconstruction.

## How well does feature generation recover human norms?

"""
    if analysis["semantic_available"]:
        text += f"Automated semantic correspondence is available for **{analysis['reference_scope']['individual_feature_norm_targets']} targets** with individual feature lists. It reuses all {analysis['scored_propositions']}/{analysis['planned_propositions']} already-scored proposition judgments, provenance-checked against human generation records. The comparison retains the original first-{analysis['comparison_cap']}-features-per-call protocol, pooling up to three repetitions. This is a comparison normalization, not a cap on the reconstructed bank.\n\n| Prompt | Mean supplied-human-feature recall | Item bootstrap interval | Complete paired targets |\n|---|---:|---|---:|\n"
        for family, r in sorted(analysis["family_summary"].items()):
            ci = r.get("item_bootstrap_ci")
            interval = (
                "unavailable" if ci is None else f"{100*ci[0]:.1f}–{100*ci[1]:.1f}%"
            )
            recall = (
                f"{100*r['mean_recall']:.1f}%"
                if r["mean_recall"] is not None
                else "unavailable"
            )
            text += f"| {family} | {recall} | {interval} | {r['n_complete_items']} |\n"
        text += "\n| Stimulus group | Prompt | Mean recall | Complete paired targets |\n|---|---|---:|---:|\n"
        for r in analysis["stratified_recovery"]:
            value = (
                f"{100*r['mean_recall']:.1f}%"
                if r["mean_recall"] is not None
                else "unavailable"
            )
            text += f"| {r['group']} | {r['prompt_family']} | {value} | {r['n_complete_items']} |\n"
        text += "\nRecall measures overlap with supplied human feature lists where scoring is complete. It does not establish comprehensive recovery, human-level performance, semantic-grader accuracy, or correctness of every generated feature. Intervals describe between-item variability, not uncertainty in the grader. Point-estimate ordering does not alone establish a significant prompt-family difference.\n"
    else:
        text += "Compatible individual-feature matching evidence was not supplied for this checkpoint; no recovery score is invented.\n"
    text += f"""
## Abstract concepts and reference limitations

The **{analysis['reference_scope']['abstract_category_only_targets']} abstract concepts** have Harpaintner aggregate feature-category proportions, not individual human feature lists. Abstract feature-list recall is unavailable. English uses published loose translations; pending translation review and generated category coding prevent a defensible quantitative category-profile comparison. Supplied category summaries are preserved separately where available, without fabricating human feature strings or importing task categories into the benchmark.

Vinson event/action and concrete-object inventories are finite, source-specific human productions. A proposition missing from those lists is not automatically invalid. Word-overlap, embedding alignment, and semantic-grader matches are correspondence measures, not human-gold validity labels.

## Discovery coverage and saturation

discovery_curves.csv retains both all-generated candidates and protocol-normalized comparisons with missing generation calls explicit. saturation_summary.csv compares the same completely sampled items across all three prompt families and all three repetition points, separately for abstract concepts, events/actions, and objects. Three repetitions measure marginal discovery yield; they do not establish an asymptotic vocabulary or complete saturation.

Mean canonical discoveries per item in the fixed, protocol-normalized cohort:

{saturation_table}

## Semantic validity and precision diagnostics

alignment_diagnostics.csv distinguishes clear label alignment, only related/uncertain alignment, and no supplied-list alignment. None is a precision-against-truth estimate. Self-origin applicability diagnostics are exported by item/prompt family; automated positives are supporting diagnostics, not independent human validation. No precision-against-truth estimate or semantic human approval is invented.

## Methodological conclusion

{norm_conclusion} Evidence for abstract-concept validity is less direct because the reference is category-level and cross-language review is incomplete. The benchmark supports further methodological testing; a general task ontology and model preferences have not been validated.

All primary analyses exclude task-origin-only features. Original matching responses, generation provenance, unresolved applicability, and source limitations are preserved. This offline reconstruction made zero model calls and changed no empirical inputs.
"""
    (out / "scientific_report.md").write_text(text)

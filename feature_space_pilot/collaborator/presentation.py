"""Readable scientific reports, complete prompt tables, and static figures."""

import html
import json
import textwrap
from collections import Counter
import numpy as np
from ..experiments.reports import csv_rows
from ..task_focus.run import NAMES, ORDER


def fmt(value, digits=3):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def prompt_html(out, items, qual, jobs, *, demo=False):
    sections = []
    md = [
        "# Full P0/P1/P2 feature comparison\n\nEvery accepted proposition and repetition is retained. Missing jobs are explicit. Local judgments are automated; ? is unresolved.\n"
    ]
    grouped = {}
    for row in qual["raw_features"]:
        grouped.setdefault(
            (row["item_id"], row["prompt_family"], row["replicate"]), []
        ).append(row)
    for item in items:
        iid = item["item_id"]
        columns = []
        mdcells = []
        summary_rows = [r for r in qual["target_prompt_metrics"] if r["item_id"] == iid]
        summary = "; ".join(
            f"{r['prompt_family']}: {r['accepted_calls']}/{r['planned_calls']} calls, {r['observed_canonical_features']} distinct local discoveries"
            for r in summary_rows
        )
        common = [
            r["proposition"]
            for r in qual["prompt_feature_patterns"]
            if r["item_id"] == iid and r["all_three_observed"]
        ]
        shared_text = (
            "Shared canonical examples: " + "; ".join(common[:4])
            if common
            else "No canonical proposition was observed in all three prompts; related paraphrases may remain distinct."
        )
        for family in ("P0", "P1", "P2"):
            reps = []
            mdrep = []
            for rep in range(3):
                rows = grouped.get((iid, family, rep), [])
                job = next(
                    (
                        j
                        for j in jobs
                        if j["item_id"] == iid
                        and j["prompt_family"] == family
                        and j["replicate"] == rep
                    ),
                    {},
                )
                if not rows:
                    accepted = next(
                        (
                            r["accepted"]
                            for r in qual["call_metrics"]
                            if r["generation_job_id"] == job.get("job_id")
                        ),
                        False,
                    )
                    content = (
                        "Accepted empty feature list"
                        if accepted
                        else "Missing accepted generation; no features inferred"
                    )
                    reps.append(f"<h4>Repetition {rep+1}</h4><p>{content}</p>")
                    mdrep.append(f"**R{rep+1}:** {content}")
                    continue
                features = []
                for r in rows:
                    status = (
                        "?" if r["local_applies"] is None else str(r["local_applies"])
                    )
                    features.append(
                        f'<li>{html.escape(r["proposition"])} <span class="badge">local judge: {status}</span></li>'
                    )
                provenance = dict(
                    job_id=job.get("job_id"),
                    model=rows[0]["generation_model"],
                    prompt_hash=rows[0]["generation_prompt_hash"],
                    source_record_hash=rows[0]["source_record_hash"],
                    system=job.get("system"),
                    user=job.get("user"),
                )
                reps.append(
                    f"<h4>Repetition {rep+1} · {len(rows)} propositions</h4><ol>"
                    + "".join(features)
                    + "</ol><details><summary>Generation provenance and original prompt</summary><pre>"
                    + html.escape(json.dumps(provenance, indent=2, ensure_ascii=False))
                    + "</pre></details>"
                )
                mdrep.append(
                    f"**R{rep+1}:** "
                    + "<br>".join(
                        f"{n+1}. {r['proposition'].replace('|','/')} [judge: {'?' if r['local_applies'] is None else r['local_applies']}]"
                        for n, r in enumerate(rows)
                    )
                )
            columns.append(f"<section><h3>{family}</h3>" + "".join(reps) + "</section>")
            mdcells.append("<br><br>".join(mdrep))
        sections.append(
            f'<article id="{iid}"><h2>{html.escape(NAMES.get(iid,iid))}</h2><p>{html.escape(summary)}</p><p>{html.escape(shared_text)}</p><details><summary>Full target description</summary><pre>{html.escape(item["text"])}</pre></details><div class="columns">'
            + "".join(columns)
            + "</div></article>"
        )
        md.append(
            f"\n## {NAMES.get(iid,iid)}\n\n{summary}\n\n{shared_text}\n\n{item['text']}\n\n| P0 | P1 | P2 |\n|---|---|---|\n| "
            + " | ".join(mdcells)
            + " |\n"
        )
    banner = "SYNTHETIC FIXTURE — " if demo else ""
    navigation = " · ".join(
        f'<a href="#{i["item_id"]}">{html.escape(NAMES.get(i["item_id"],i["item_id"]))}</a>'
        for i in items
    )
    document = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task prompt comparison</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#172b3a;background:#f5f7fa;margin:0;padding:28px;max-width:1800px}h1,h2,h3{line-height:1.2}h2{margin-top:0}h3{color:#167d8d}article{margin:28px 0;padding:24px;background:white;border:1px solid #ccd6df;border-radius:8px}.columns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:28px}section{min-width:0}li{margin:0 0 10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 ui-monospace,monospace;background:#edf1f5;padding:12px}.badge{font-size:11px;color:#526675;white-space:nowrap}summary{cursor:pointer;color:#356d81}nav{font-size:13px;line-height:2}a{color:#126d82}@media(max-width:1000px){.columns{grid-template-columns:1fr}body{padding:14px}}@media print{details{display:block}article{break-before:page}.badge{color:black}}
</style><h1>"""
        + banner
        + """P0 / P1 / P2: actual task features</h1><p>All three repetitions, full proposition lists, original target text and provenance. The local judge evaluated only features independently generated for this target. A feature can be descriptive without being useful for preferences; labels are automated and unreviewed.</p><nav>"""
        + navigation
        + "</nav>"
        + "".join(sections)
        + "</html>"
    )
    (out / "prompt_comparison.html").write_text(document)
    (out / "prompt_comparison_full.md").write_text(
        (banner + "\n\n" if demo else "") + "\n".join(md)
    )


def write_reports(out, bundle, human, local, qual, compositions, literature):
    demo = (
        "SYNTHETIC FIXTURE — engineering demonstration, not empirical findings.\n\n"
        if bundle.get("demo")
        else ""
    )
    n = len(human["vectors"]["item_ids"])
    htext = (
        demo
        + f"""# Human benchmark: independent representation geometry and supervised alignment

## Motivation and primary estimand

The methodological question is whether independently elicited properties recover the relational organization of human concepts. The primary comparison uses cosine similarity within each feature vocabulary, then correlates corresponding concept-pair similarities. It does not match LLM propositions to human labels. The separate secondary analysis asks which supplied human properties are expressed and is explicitly supervised.

The same {n} Vinson concepts appear in the same declared order in every matrix. Published human weights and all {len(human['vectors']['feature_labels'])} original coordinates are retained, including duplicate text labels as distinct indexed coordinates. Word-plus-actionN/actionV/object identity resolves homonyms. LLM vectors use only the independent human-experiment bank. Each family has three generation calls per target; primary first-ten-per-call normalization is declared and the uncapped-output sensitivity is retained. Actual token use is exported; equal calls do not imply equal tokens.

## Primary RSA

LLM weights count repetitions containing a canonical feature, once per repetition. Human weights are published production frequencies. Cosine is scale-invariant to dividing these by three or twenty. Binary LLM presence is a sensitivity. Spearman uses average ranks across the upper triangle; concept-label permutations apply the same shuffle to rows and columns. Two-sided Monte Carlo p-values use the +1 estimator; Holm adjustment covers the three primary family tests. Pair entries are never treated as independent observations. Empty-vector cosine and constant-rank RSA are unavailable, not zero.

| Family | Spearman rho | Permutation p | Holm p | Zero off-diagonal LLM similarities |
|---|---:|---:|---:|---:|
"""
    )
    for r in human["rsa_primary"]:
        htext += f"| {r['prompt_family']} | {fmt(r['rho'])} | {fmt(r['permutation_p_two_sided'],4)} | {fmt(r.get('holm_p'),4)} | {fmt(100*r['zero_offdiagonal_proportion'],1) if r['zero_offdiagonal_proportion'] is not None else 'unavailable'}% |\n"
    htext += f"\nHuman zero off-diagonal proportion: {fmt(100*human['human_similarity']['zero_offdiagonal_proportion'],1)}%. Full matrices, all pairwise values, permutation nulls and effect sizes are exported.\n\n"
    htext += """These are small positive associations where available, not evidence of complete semantic recovery. The LLM spaces are extremely sparse. With very few nonzero concept pairs, a low overall correlation cannot by itself establish poor semantic organization; vocabulary alignment/consolidation and sampled discovery constrain this estimator. Shared-pair propositions are exported to inspect what drives the signal. The cow–goat relation is particularly prominent in this pilot.

P1 also shares grammatical or metalinguistic statements. A declared, exploratory content-only sensitivity removes such columns without changing any concept rows or consulting human labels. Within-word-type label permutations assess whether results exceed coarse noun/verb/object organization; raw-surface-form and binary-weighting sensitivities are separate. These analyses do not turn grammatical similarity into semantic validity or rank a prompt as universally best.

| Family | Content-only rho | Content-only p | Content-only / within-type p |
|---|---:|---:|---:|
"""
    for f in ("P0", "P1", "P2"):
        c = next(
            r
            for r in human["rsa_sensitivities"]
            if r["prompt_family"] == f and r["variant"] == "content_frequency"
        )
        w = next(
            r
            for r in human["rsa_sensitivities"]
            if r["prompt_family"] == f
            and r["variant"] == "content_frequency_within_type_null"
        )
        htext += f"| {f} | {fmt(c['rho'])} | {fmt(c['permutation_p_two_sided'],4)} | {fmt(w['permutation_p_two_sided'],4)} |\n"
    alignment = human["alignment"]
    htext += f"""
## Secondary feature-aligned cosine

This projects generated propositions onto the human coordinates using existing automated semantic judgments. The grader saw target-specific human labels; this is supervised, target-conditioned alignment and is **not independent RSA**. {alignment['policy']}

| Prompt | Mean aligned cosine, paired concepts | Paired concepts | Unmatched generation units | Unscored units |
|---|---:|---:|---:|---:|
"""
    for f, r in alignment["family_summary"].items():
        htext += f"| {f} | {fmt(r['mean_cosine'])} | {r['n_complete_paired_concepts']} | {r['unmatched_units']}/{r['generation_units']} | {r['unscored_units']} |\n"
    mismatches = [
        r
        for r in alignment["inventory_audit"]
        if r["missing_target_labels"] or r["unexpected_target_labels"]
    ]
    undefined = [r for r in alignment["concept_scores"] if r["cosine"] is None]
    htext += f"""
Audit found **{len(mismatches)} target inventories** with word-type label mismatches. Existing clear matches to labels outside the exact selected sense are excluded from the primary aligned vector and retained in the audit; their propositions are not declared invalid. The original archive also has repeated text labels; their indexed coordinates are preserved. A label with multiple positive candidate coordinates would be withheld rather than assigned using human weight magnitudes.

The original grader saw the larger inventories for those targets. Filtering the results cannot remove any influence those extra labels had on grading; the reused alignment remains exploratory.

Undefined aligned scores: {[(r['item_id'],r['prompt_family']) for r in undefined]}. An empty projection means cosine is undefined, not zero semantic quality. Paired averages use the same defined concepts; per-family available counts and per-concept averages are exported separately.

Many-to-many mappings split at most one unit of mass per canonical proposition/repetition across clear labels. Within-call paraphrase mappings must agree; related and uncertain labels do not count. Repetition-level label contributions are capped before averaging. Single-label-only and binary sensitivities expose dependence on multi-label alignment. Unmatched features remain in full proposition tables; norms are incomplete productions, not truth labels. Neither the grader nor the alignments are human-verified.

## Published reliability and abstract concepts

No comparable participant-level feature-space ceiling was identified in either source paper. Vinson's researcher consensus procedures and Harpaintner's coder agreement are distinct from participant generation consistency. See human_reliability.md for exact units, values, procedures, pages and citations; no participant LOOCV was performed and no ceiling-normalized score is computed.

Harpaintner remains unavailable for quantitative category-profile comparison: the 24 selected concepts have aggregate SM, IS/E, SC and VA summaries, but translations and generated-feature category coding are not reviewed. Original German, English translations and source/category provenance are retained. No human feature-string vectors are invented for these concepts.

## Interpretation

The pilot supports some meaningful correspondence, such as animal properties, while broad independent geometric recovery remains uncertain because overlap is sparse and some similarity is grammatical. Supervised alignment finds additional correspondence but cannot independently validate representational organization. No result demonstrates features used internally or features causing preferences.
"""
    (out / "human/scientific_report.md").write_text(htext)
    ltext = "# Published human reliability review\n\n| Paper | Unit and metric | Sample / procedure | Value | Locator |\n|---|---|---|---|---|\n"
    for r in literature["records"]:
        value = "not numerically reported" if r["value"] is None else str(r["value"])
        ltext += f"| [{r['paper']}]({r['citation_url']}) | {r['unit']}: {r['metric']} | {r['sample']}; {r['procedure']} | {value} | {r['locator']} |\n"
    ltext += "\n" + literature["conclusion"] + " No LOOCV was performed.\n"
    (out / "human/human_reliability.md").write_text(ltext)
    counts = Counter(j["applies"] for j in local["judgments"])
    ttext = (
        demo
        + f"""# Task experiment: target-specific discovery and validity

## Scope and scientific objective

The aim is an elicitation procedure for useful, generalizable task descriptions before behavioral preferences experiments. Descriptive truth, generalizability and evaluative relevance are separate. The primary inventory of each target contains only its own accepted P0/P1/P2 generations across repetitions. Shared canonical identifiers never import a feature into another target.

Local reconsolidation produces **{local['primary_cell_count']} primary cells**, compared with the preserved historical {local['historical_cross_product_cells']} cells. Existing compatible evidence resolves {local['reused_binary_cells']} ({counts[1]} yes, {counts[0]} no), with {counts[None]} unknown. All originals and historical cross-task judgments remain intact. Existing unknowns are not rerun. Target inventories, provenance and complete judgment reuse chains are exported separately.

## What do P0, P1 and P2 actually elicit?

The full HTML comparison presents every target description, all three repetitions, every accepted proposition and original prompts/model provenance side by side. Missing generations are explicit. The fixed complete cohort has {len(qual['fixed_complete_target_ids'])} targets; incomplete finance targets remain in the full export and all-target descriptive tables. Cue screening is provisional and manually reviewable; it is not a validated semantic coding scheme. Means below first average within calls and targets, then across the same complete target cohort.

| Prompt | Mean canonical discoveries / target | Resource cues | Operational / relational cues | Instruction-restatement cues | Specific-detail cues | Evaluation / feedback cues |
|---|---:|---:|---:|---:|---:|---:|
"""
    )
    for r in qual["prompt_summary"]:
        ttext += f"| {r['prompt_family']} | {fmt(r['mean_canonical_features'],1)} | {fmt(100*r['resource_demands'],1)}% | {fmt(100*r['operational_or_relational'],1)}% | {fmt(100*r['instruction_restatement_candidate'],1)}% | {fmt(100*r['instance_detail'],1)}% | {fmt(100*r['evaluation_feedback'],1)}% |\n"
    ttext += """
P0 often supplies familiar, broad properties for minimally specified domains, but restates explicit constraints for concrete tasks. P1 can produce operational details, yet also literal description-reading and target-label properties. In this pilot P2 has more resource/evaluation/social cues, especially at broad levels, but it also returns concrete instruction details. More features are not automatically more useful.

Declared cue checks detect no explicit incentive, uncertainty or conflict/trade-off language in the completely sampled comparison cohort; P2 has a small amount of consequence language. This does not prove the absence of implicit meanings. These are empirical descriptive patterns, not confirmation of prompt intent or statistically established universal differences.

Examples from the empirical case studies include P0's “Limited time or money can constrain a project,” P1's “The target is labeled Creative production,” and P2's medium/length/review requirements for creative production. For the concrete poem all prompts largely reproduce line, theme and rhyme requirements; P2 splits several rhyme relations into individual line-pair statements. P1 finance outputs include exact IDs, amounts and records. Read the full provenance tables rather than treating these selected examples as the full evidence.

prompt_feature_patterns.csv distinguishes observed common and prompt-specific canonical discoveries. Near-duplicate candidates are lexical screening, with protected numeric/role/modality differences flagged; they are never automatic merges. Repetition consistency and within-call redundancy are separate. Local judge rejection is a suspected validity problem, not independently verified error. manual_review.csv supplies a manageable balanced sample; all raw propositions remain in generation_features.csv.

## Abstraction and situations

level_prompt_summary.csv and every target's table separate domain, task type, concrete instance, standalone situation and situated task. Broad domain labels can elicit general hypotheses or mere label descriptions. Concrete descriptions anchor outputs to explicit constraints, sometimes excessively specific details. Deadlines and external review add time and quality language. The two families and overlapping variants are not independent population samples, so these summaries do not establish general prompt superiority.

## Discovery-only compositionality

G(T), G(S), and G(T+S) are discovery sets. The exported equal-planned-call comparison balances component and direct generation, preserves missing roles, and includes full-sample sensitivity. A shared identifier describes a property independently generated for each corresponding target; it never expands a judging inventory. The comparison registry follows original phrase provenance so local reconsolidation cannot manufacture a loss of overlap.

**Absence from G(T) is not T=0.** No interaction is confirmed or classified from generation absence. Feature tables show native applicability only where the feature belongs to that target's own inventory; otherwise the status is “not generated / not in primary scope.” Additional direct discoveries may be paraphrases, omitted component properties, instruction bindings or new relations, and descriptive discovery alone cannot distinguish them.

targeted_validation_design.json is a separate, non-executing design for selectively testing combined-only properties against components. Compatible historical exploratory evidence should be reused first; unresolved or new paid work requires explicit cost preflight and authorization. No pooled judging workload is submitted.

## Implications for preferences research

P2 is a plausible starting point for resource/evaluation demands, but current outputs do not establish reliable elicitation of incentives, uncertainty, conflicts or consequences. P0's broader hypotheses and P1's operational detail remain useful comparators. A next elicitation revision should demand grounded functional properties and distinguish explicit constraints from plausible but unstated assumptions; test it on additional task families before choosing a primary procedure. No preference values, utility model, internal feature-use mechanism or causal claim is assigned here.
"""
    (out / "task/scientific_report.md").write_text(ttext)
    motivation = """# Motivation and collaborator framing

We want a reproducible way to describe tasks with interpretable properties before investigating which properties influence an LLM's choices or performance. Experiment 1 tests methodological correspondence with human semantic norms: independent geometry asks whether conceptual relations are recovered, while supervised feature alignment asks which human properties are expressed. Experiment 2 evaluates elicitation on realistic tasks, restricting validity judgments to each target's own discoveries. A useful procedure must yield grounded, reusable functional properties, not simply a large vocabulary. Preference relevance and causal feature use require subsequent behavioral tests.
"""
    (out / "motivation.md").write_text(demo + motivation)
    root = (
        demo
        + "# Feature-space pilot: collaborator results\n\n"
        + motivation.split("\n\n", 1)[1]
        + "\n\n[Human RSA and aligned comparisons](human/scientific_report.md) · [Target-specific task results](task/scientific_report.md) · [Full P0/P1/P2 comparison](task/prompt_comparison.html) · [Published reliability review](human/human_reliability.md)\n\nThe two experiments remain separate. Primary RSA is sparse and partially grammatical; aligned correspondence is supervised. Task validity is target-local, and compositionality is discovery-only. Unknowns, missing calls and provisional qualitative coding remain explicit. No paid inference, preference model, or empirical overwrite occurred.\n"
    )
    (out / "scientific_report.md").write_text(root)


def render_figures(out, bundle, human, local, qual, compositions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    banner = "SYNTHETIC FIXTURE | " if bundle.get("demo") else ""

    def save(fig, stage, name, foot):
        fig.text(0.015, 0.012, banner + foot, fontsize=9, color="#405465", va="bottom")
        for ext in ("png", "svg"):
            fig.savefig(out / stage / (name + "." + ext), dpi=180, facecolor="white")
        plt.close(fig)

    labels = [
        i["text"]
        for i in bundle["human_items"]
        if i["item_id"] in human["vectors"]["item_ids"]
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 17))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#e7eaee")
    for ax, key in zip(axes.flat, ("Human", "P0", "P1", "P2")):
        matrix = human["similarities"][key].copy()
        np.fill_diagonal(matrix, np.nan)
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        title = (
            "Human production frequencies"
            if key == "Human"
            else key
            + " · ρ="
            + fmt(
                next(r for r in human["rsa_primary"] if r["prompt_family"] == key)[
                    "rho"
                ]
            )
        )
        ax.set_title(title, fontsize=13, pad=12)
    fig.suptitle(
        banner + "Independent concept geometry · cosine similarity",
        fontsize=18,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.11, right=0.88, top=0.96, bottom=0.11, wspace=0.34, hspace=0.42
    )
    color_axis = fig.add_axes([0.93, 0.30, 0.015, 0.38])
    fig.colorbar(image, cax=color_axis, label="Cosine similarity (same 0–1 scale)")
    save(
        fig,
        "human",
        "figure_H1_similarity_matrices",
        "Same concept order in all panels; diagonal hidden. LLM similarity is mostly zero. P1 includes grammatical properties; content-only sensitivity is reported separately.",
    )
    scores = human["alignment"]["concept_scores"]
    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.25 + 2)))
    for f, color, offset in zip(
        ("P0", "P1", "P2"), ("#167d8d", "#be7d3e", "#7867a3"), (-0.2, 0, 0.2)
    ):
        values = [
            next(r for r in scores if r["item_id"] == iid and r["prompt_family"] == f)[
                "cosine"
            ]
            for iid in human["vectors"]["item_ids"]
        ]
        ax.scatter(
            [np.nan if x is None else x for x in values],
            np.arange(len(labels)) + offset,
            color=color,
            label=f,
            s=25,
        )
    ax.set(
        yticks=np.arange(len(labels)),
        yticklabels=labels,
        xlim=(0, 1),
        xlabel="Feature-aligned cosine similarity",
    )
    ax.invert_yaxis()
    ax.legend(frameon=False)
    ax.set_title(banner + "Secondary supervised feature alignment", fontsize=16, pad=15)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.92, bottom=0.12)
    save(
        fig,
        "human",
        "figure_H2_feature_alignment",
        "Automated, target-conditioned mapping; mass split across clear labels and capped per repetition. Undefined empty projections are omitted; unmatched properties remain exported.",
    )
    items = bundle["task_items"]
    fig, ax = plt.subplots(figsize=(13, max(6, len(items) * 0.37 + 2)))
    y = np.arange(len(items))
    left = np.zeros(len(items))
    for value, label, color in (
        (1, "Own-target yes", "#167d8d"),
        (0, "Own-target no", "#ba6c61"),
        (None, "Unresolved", "#d9a256"),
    ):
        counts = np.array(
            [
                sum(
                    j["applies"] == value
                    for j in local["targets"][i["item_id"]]["judgments"]
                )
                for i in items
            ]
        )
        ax.barh(y, counts, left=left, label=label, color=color)
        left += counts
    ax.set(
        yticks=y,
        yticklabels=[NAMES.get(i["item_id"], i["item_id"]) for i in items],
        xlabel="Only features independently generated for this target",
    )
    ax.invert_yaxis()
    fig.legend(
        *ax.get_legend_handles_labels(),
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.61, 0.955),
    )
    fig.suptitle(
        banner + f"Primary task scope · {local['primary_cell_count']} local cells",
        fontsize=16,
        y=0.995,
    )
    fig.subplots_adjust(left=0.25, right=0.98, top=0.86, bottom=0.12)
    save(
        fig,
        "task",
        "figure_T1_target_local_validity",
        "Each bar has its own candidate inventory. No cross-target feature import; no generation absence is converted into a negative judgment.",
    )
    metrics = [
        "operational_or_relational",
        "resource_demands",
        "incentives",
        "uncertainty",
        "conflict_or_tradeoff",
        "consequences",
        "evaluation_feedback",
        "social_context",
        "linguistic_or_label_metadata",
        "instance_detail",
        "instruction_restatement_candidate",
    ]
    mat = np.array(
        [
            [r[m] if r[m] is not None else np.nan for r in qual["prompt_summary"]]
            for m in metrics
        ]
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), ("P0", "P1", "P2"))
    ax.set_yticks(range(len(metrics)), [m.replace("_", " ") for m in metrics])
    for n in range(len(metrics)):
        for k in range(3):
            ax.text(
                k,
                n,
                "?" if not np.isfinite(mat[n, k]) else f"{100*mat[n,k]:.1f}%",
                ha="center",
                va="center",
                color="white" if mat[n, k] > 0.5 else "#193548",
            )
    ax.set_title(
        banner
        + f"Observed prompt profiles · {len(qual['fixed_complete_target_ids'])} fully sampled targets",
        pad=20,
        fontsize=15,
    )
    fig.colorbar(image, ax=ax, label="Mean within-call cue proportion")
    fig.subplots_adjust(left=0.35, right=0.95, top=0.89, bottom=0.10)
    save(
        fig,
        "task",
        "figure_T2_prompt_profiles",
        "Transparent lexical screening, not semantic gold. Same paired target cohort; missing generations excluded from this comparison and retained in full tables.",
    )
    rs = [r for r in compositions["comparisons"] if r["sampling_mode"] == "equal_calls"]
    fig, ax = plt.subplots(figsize=(15, max(7, len(rs) * 0.36 + 2)))
    left = np.zeros(len(rs))
    y = np.arange(len(rs))
    for key, label, color in (
        ("shared_count", "Shared discoveries", "#167d8d"),
        ("component_only_count", "Component only", "#bdc9d3"),
        ("direct_only_count", "Direct only", "#b78a4e"),
    ):
        vals = np.array([r[key] for r in rs])
        ax.barh(y, vals, left=left, label=label, color=color)
        left += vals
    ax.set(
        yticks=y,
        yticklabels=[
            NAMES.get(r["item_id"], r["item_id"])
            + " · "
            + r["prompt_family"]
            + (" †" if not r["generation_coverage_complete"] else "")
            for r in rs
        ],
        xlabel="Observed discovered properties · shared task-only semantic comparison IDs",
    )
    ax.invert_yaxis()
    fig.legend(
        *ax.get_legend_handles_labels(),
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.61, 0.955),
    )
    fig.suptitle(
        banner + "Task–situation discovery, without applicability inference",
        fontsize=16,
        y=0.995,
    )
    fig.subplots_adjust(left=0.24, right=0.98, top=0.87, bottom=0.12)
    save(
        fig,
        "task",
        "figure_T3_discovery_composition",
        "Equal planned calls; † incomplete samples. Direct-only discovery is not T=0/S=0/TS=1. Full propositions, membership and native-only evidence are exported.",
    )

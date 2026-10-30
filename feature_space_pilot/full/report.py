"""Readable full-vocabulary reports, with incomplete evidence gated explicitly."""

from collections import defaultdict, Counter
from itertools import groupby
import json
from pathlib import Path
import numpy as np
from ..budget import estimate_cost
from ..io import canonical, digest, file_hash, write_csv, write_json
from ..prompts import normalize_phrase
from ..judge import rubric_for
from ..report import _human_analysis, _embedding_analysis, _references
from .analysis import (
    structure_summary,
    composition_report,
    discovery_report,
    task_contrast,
    observed_usage,
    saturation_summary,
)
from .budget import template_payload


def _csv(path, rows, fields=None):
    if fields is None:
        rows = list(rows)
        fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["status"]
    write_csv(path, rows, fields)


def _legacy(source, bank):
    mapping = {o["normalized"]: o["canonical_id"] for o in bank["origins"]}
    subsets = {}
    coverage = []
    for s in source["snapshots"]:
        ids = {
            mapping[normalize_phrase(f["text"])]
            for f in s["baseline_features"]
            if normalize_phrase(f["text"]) in mapping
        }
        rid = s["manifest"]["run_id"]
        subsets[rid] = sorted(ids)
        coverage.append(
            dict(
                source_run_id=rid,
                original_feature_columns=len(s["baseline_features"]),
                mapped_canonical_columns=len(ids),
                complete_global_columns=len(bank["features"]),
                additional_canonical_columns=len(bank["features"]) - len(ids),
                original_generation_complete=s["generations_complete"],
                original_judge_model=s["config"]["judge"]["model_id"],
                canonical_subset_is_compared_under_the_global_judge=True,
            )
        )
    return subsets, coverage


def write_full_report(
    out,
    source,
    bank,
    state,
    c,
    cost,
    reuse,
    *,
    profile=None,
    proposals=(),
    render=True,
    relational_reviews=None,
    semantic_matching_dir=None,
    primary_records=(),
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seal = out / "report_manifest.json"
    if seal.exists():
        old = json.loads(seal.read_text())
        for name, h in old["artifacts"].items():
            if not (out / name).exists() or file_hash(out / name) != h:
                raise ValueError("Frozen full report artifact changed: " + name)
        return json.loads((out / "summary.json").read_text())
    items = source["items"]
    features = bank["features"]
    ii = {i["item_id"]: n for n, i in enumerate(items)}
    ff = {f["feature_id"]: n for n, f in enumerate(features)}
    mapping = {o["normalized"]: o["canonical_id"] for o in bank["origins"]}
    byid = {f["feature_id"]: f for f in features}
    subsets, legacy = _legacy(source, bank)
    structure = structure_summary(items, features, state, legacy_subsets=subsets)
    jobs = [j for s in source["snapshots"] for j in s["jobs"]]
    curves = discovery_report(items, bank, source["generations"], jobs, state)
    compositions = composition_report(
        items,
        bank,
        source["generations"],
        state,
        replicates=c["replicates"],
        relational_reviews=relational_reviews,
    )
    write_json(out / "feature_space_structure.json", structure)
    write_json(out / "compositionality.json", compositions)
    _csv(
        out / "semantic_review_queue.csv",
        proposals,
        [
            "left_id",
            "right_id",
            "left",
            "right",
            "retrieval_score",
            "protected_compatible",
            "decision",
            "reviewer",
            "automatic_merge",
            "reason",
            "note",
        ],
    )
    write_json(out / "blocked_equivalence_chains.json", bank["blocked_chains"])
    contrasts = task_contrast(items, features, state)
    _csv(
        out / "task_shared_distinct_features.csv",
        contrasts,
        [
            "left_item",
            "right_item",
            "feature_id",
            "proposition",
            "left_applies",
            "right_applies",
            "comparison",
        ],
    )
    write_json(
        out / "observed_usage.json", observed_usage(source, primary_records, c["judge"])
    )
    _csv(out / "item_feature_counts.csv", structure["items"])
    _csv(out / "feature_frequency_applicability.csv", structure["features"])
    _csv(out / "discovery_saturation.csv", curves)
    _csv(
        out / "discovery_saturation_fixed_cohort.csv",
        saturation_summary(curves, replicates=c["replicates"]),
    )
    _csv(out / "legacy_150_coverage.csv", legacy)
    _csv(
        out / "generation_coverage.csv",
        [dict(item_id=i, **v) for i, v in source["coverage"].items()],
    )
    _csv(out / "missing_generation_jobs.csv", source.get("missing_generation_jobs", []))
    write_json(out / "cost_report.json", cost)
    write_json(out / "reuse_policy_and_audit.json", reuse)
    # Only a completely resolved scope receives a binary-matrix filename.
    matrix_name = (
        "binary_matrix.csv" if state["complete"] else "applicability_matrix_partial.csv"
    )
    fields = ["item_id"] + [f["feature_id"] for f in features]
    _csv(
        out / matrix_name,
        (
            dict(
                item_id=i["item_id"],
                **{
                    f["feature_id"]: (
                        int(state["matrix"][n, k])
                        if state["matrix"][n, k] >= 0
                        else "?"
                    )
                    for k, f in enumerate(features)
                },
            )
            for n, i in enumerate(items)
        ),
        fields,
    )
    bases = {
        i["item_id"]: (
            profile["items"][i["item_id"]]["input_tokens"]
            if profile
            else len(canonical(template_payload(i, c["judge"])).encode())
        )
        for i in items
    }
    additions = {
        f["feature_id"]: len(canonical(f["text"]).encode())
        + 2 * len(f["feature_id"].encode())
        + 128
        for f in features
    }
    priors = {
        (r["item_id"], r["feature_id"]): r["reason"] for r in reuse["unresolved_prior"]
    }
    pending = {
        (r.get("item_id"), r.get("feature_id"))
        for r in reuse["external_pending"].values()
    }

    def blocked():
        for row in state["unresolved"]:
            iid, fid = row["item_id"], row["feature_id"]
            reason = row["reason"]
            if reason == "missing":
                reason = (
                    "external_pending"
                    if (iid, fid) in pending
                    else priors.get((iid, fid), reason)
                )
            yield dict(
                item_id=iid,
                feature_id=fid,
                proposition=byid[fid]["text"],
                reason=reason,
                next_attempt_upper_bound_usd=estimate_cost(
                    c["judge"]["prices"],
                    bases[iid] + additions[fid],
                    c["judge"]["max_output_tokens"],
                ),
                judge_model=c["judge"]["model_id"],
                rubric_version=rubric_for(c["judge"]),
                missing_is_zero=False,
            )

    _csv(
        out / "blocked_cells.csv",
        blocked(),
        [
            "item_id",
            "feature_id",
            "proposition",
            "reason",
            "next_attempt_upper_bound_usd",
            "judge_model",
            "rubric_version",
            "missing_is_zero",
        ],
    )
    discovered = defaultdict(set)
    for o in bank["origins"]:
        discovered[o["item_id"]].add(o["canonical_id"])
    task_rows = []
    task_ids = {
        i["item_id"]
        for i in items
        if i.get("target_type") not in {"concept", "event", "object"}
    }
    itemmap = {i["item_id"]: i for i in items}
    for o in bank["origins"]:
        iid = o["item_id"]
        if iid not in task_ids:
            continue
        item = itemmap[iid]
        fid = o["canonical_id"]
        v = state["matrix"][ii[iid], ff[fid]]
        parent = item.get("parent_id")
        pv = state["matrix"][ii[parent], ff[fid]] if parent in ii else -1
        task_rows.append(
            dict(
                item_id=iid,
                task_id=item.get("task_id"),
                level=item.get("level"),
                parent_id=parent,
                situation_ids=item.get("situation_ids", []),
                phrase=o["phrase"],
                canonical_id=fid,
                canonical_proposition=byid[fid]["text"],
                prompt_family=o["prompt_family"],
                replicate=o["replicate"],
                source_run_id=o["source_run_id"],
                generation_model=o["generation_model"],
                applies=int(v) if v >= 0 else "?",
                parent_applies=int(pv) if pv >= 0 else "?",
                inherited_discovery=fid in discovered[parent] if parent else None,
                parent_generation_complete=(
                    (
                        source["coverage"].get(parent, {}).get("completed")
                        == source["coverage"].get(parent, {}).get("planned")
                    )
                    if parent
                    else None
                ),
                newly_applicable=(v == 1 and pv == 0) if v >= 0 and pv >= 0 else None,
                consolidation_method=o["consolidation_method"],
            )
        )
    _csv(out / "task_level_feature_audit.csv", task_rows)
    _csv(
        out / "task_applicability.csv",
        (
            dict(
                item_id=i["item_id"],
                task_id=i.get("task_id"),
                level=i.get("level"),
                feature_id=f["feature_id"],
                proposition=f["text"],
                applies=(
                    int(state["matrix"][ii[i["item_id"]], ff[f["feature_id"]]])
                    if state["matrix"][ii[i["item_id"]], ff[f["feature_id"]]] >= 0
                    else "?"
                ),
                discovered_here=f["feature_id"] in discovered[i["item_id"]],
            )
            for i in items
            if i["item_id"] in task_ids
            for f in features
        ),
        [
            "item_id",
            "task_id",
            "level",
            "feature_id",
            "proposition",
            "applies",
            "discovered_here",
        ],
    )
    interaction_rows = []
    for row in compositions:
        if row["sampling_mode"] != "equal_calls":
            continue
        for fid in row["interaction_candidates"]:
            interaction_rows.append(
                dict(
                    item_id=row["item_id"],
                    prompt_family=row["prompt_family"],
                    feature_id=fid,
                    proposition=byid[fid]["text"],
                    confirmed="",
                    reviewer="",
                    note="Candidate only; named relational review required",
                )
            )
    _csv(out / "interaction_review.csv", interaction_rows)
    human = _human_analysis(
        c,
        items,
        [r for r in source["generations"] if r.get("parse_status") == "ok"],
        features,
        [
            dict(
                item_id=i["item_id"],
                feature_id=f["feature_id"],
                applies=int(state["matrix"][n, k]),
                status="ok",
            )
            for n, i in enumerate(items)
            for k, f in enumerate(features)
            if state["matrix"][n, k] >= 0
        ],
    )
    write_json(out / "human_norm_comparison.json", human)
    embedding = _embedding_analysis(
        c,
        items,
        features,
        [
            dict(
                item_id=i["item_id"],
                feature_id=f["feature_id"],
                applies=int(state["matrix"][n, k]),
                status="ok",
            )
            for n, i in enumerate(items)
            for k, f in enumerate(features)
            if state["matrix"][n, k] >= 0
        ],
    )
    write_json(out / "embedding_geometry.json", embedding)
    category_rows = []
    for dataset, reference in _references(c, items).items():
        if reference.get("data_level") != "category_counts":
            continue
        for r in reference["rows"]:
            category_rows.append(
                dict(
                    dataset=dataset,
                    source_item=r["item"],
                    category=r["category"],
                    source_value=r.get("count"),
                    measurement="source_category_summary_not_individual_features",
                    cross_language_comparison_requires_review=True,
                )
            )
    _csv(out / "human_category_reference.csv", category_rows)
    semantic = dict(
        available=False,
        reason="No verified human-norm semantic-matching checkpoint supplied",
    )
    if semantic_matching_dir:
        from ..baselines.run import load_snapshot
        from ..matching.analysis import aggregate, paired_summary

        snapshot = load_snapshot(semantic_matching_dir)
        result = aggregate(snapshot["inventory"], snapshot["records"])
        final = [
            r for r in result["recovery"] if r["replicates_included"] == c["replicates"]
        ]
        families, paired = paired_summary(final)
        semantic = dict(
            available=True,
            source_analysis=snapshot["manifest"]["analysis_id"],
            source_manifest_sha256=snapshot["manifest_hash"],
            automated_not_human_reviewed=True,
            bank_consolidation_source=False,
            comparison_cap=snapshot["inventory"]["comparison_cap"],
            scored_propositions=result["n_scored_propositions"],
            planned_propositions=result["n_propositions"],
            families=families,
            paired_items=paired,
        )
    write_json(out / "automated_human_norm_matching.json", semantic)
    summary = dict(
        synthetic=bool(c.get("demo")),
        matrix_complete=state["complete"],
        source_generations_complete=source["generations_complete"],
        generation_scope_closed=bool(
            source.get("generation_scope_closed", source["generations_complete"])
        ),
        missing_generation_jobs=len(source.get("missing_generation_jobs", [])),
        generation_checkpoint_hash=source.get("generation_checkpoint_hash"),
        n_items=len(items),
        source_generated_features=bank["source_feature_count"],
        normalized_unique=bank["normalized_unique"],
        consolidated_features=len(features),
        feature_cap=None,
        explicit_rejected_attempt_features=len(bank["exclusions"]),
        pending_equivalence_proposals=sum(
            r.get("decision") == "pending" for r in proposals
        ),
        blocked_equivalence_chains=len(bank["blocked_chains"]),
        resolved_cells=state["resolved_cells"],
        planned_cells=state["planned_cells"],
        unresolved_cells=len(state["unresolved"]),
        features_with_observed_applicability=structure[
            "features_with_observed_applicability"
        ],
        reused_cells=len(reuse["reused"]),
        judge_model=c["judge"]["model_id"],
        generation_models=sorted(
            {
                r["generation_model"]
                for r in source["generations"]
                if r["parse_status"] == "ok"
            }
        ),
        legacy_coverage=legacy,
        human_reviewed=False,
        cost_blockers=cost["blockers"],
        inference="Candidate descriptive representations, not causal evidence of features used by models in decisions",
    )
    write_json(out / "summary.json", summary)
    case_text = (
        ("# SYNTHETIC ENGINEERING FIXTURE\n\n" if c.get("demo") else "")
        + "# Task case studies\n\nAll underlying propositions are in the complete task CSVs. Domain → type → instance is a hierarchy; situations are an independent axis.\n\n"
    )
    for task in dict.fromkeys(i.get("task_id") for i in items if i.get("task_id")):
        case_text += (
            "## "
            + (
                "Creative writing"
                if task == "poem"
                else "Financial reconciliation" if task == "finance" else task
            )
            + "\n\n"
        )
        for item in [i for i in items if i.get("task_id") == task]:
            iid = item["item_id"]
            case_text += f"### {item.get('level')} — {iid}\n\n{item['text'][:350]}{' … (full frozen input in stimuli.jsonl)' if len(item['text'])>350 else ''}\n\n"
            examples = []
            seen = set()
            for r in task_rows:
                if r["item_id"] == iid and r["canonical_id"] not in seen:
                    examples.append(r)
                    seen.add(r["canonical_id"])
            if not examples:
                case_text += "No accepted task-generation outputs are available at this checkpoint; this is not evidence of no properties.\n\n"
                continue
            case_text += "| Canonical proposition | Origin | Applies |\n|---|---|---|\n"
            for r in examples[:12]:
                case_text += f"| {r['canonical_proposition'].replace('|','/')} | {r['prompt_family']} / rep {r['replicate']+1} | {r['applies']} |\n"
            case_text += f"\nShowing up to 12 examples for readability; all {len(examples)} distinct discoveries and all global applicability columns remain in the CSV exports.\n\n"
    case_text += "## Shared and distinguishing task properties\n\nThe complete task_shared_distinct_features.csv compares the two concrete task instances using the same global vocabulary. Each distinction requires independent binary judgments on both sides.\n\n"
    for category, rows in groupby(
        sorted(contrasts, key=lambda r: r["comparison"]), key=lambda r: r["comparison"]
    ):
        rows = list(rows)
        case_text += f"**{category}**: {len(rows):,} features.\n\n"
        if category != "unresolved":
            case_text += "".join(f"- {r['proposition']}\n" for r in rows[:12]) + "\n"
    case_text += "## Each task–situation composition\n\n| Combination | Prompt | Component / direct calls | Shared discoveries | Direct-only applicable | Constituent discovery gaps | Interaction candidates |\n|---|---|---|---|---|---|---|\n"
    for r in compositions:
        if r["sampling_mode"] != "equal_calls":
            continue
        values = [
            len(r[k]) if r["available"] else "pending"
            for k in [
                "recovered_both",
                "direct_only_applicable",
                "constituent_discovery_gaps",
                "interaction_candidates",
            ]
        ]
        case_text += (
            f"| {r['item_id']} | {r['prompt_family']} | {r['component_calls']} / {r['direct_calls']} | "
            + " | ".join(map(str, values))
            + " |\n"
        )
    case_text += "\nAn interaction candidate applies to the combination and is independently judged false for both constituents. Absence from generation lists alone establishes only discovery novelty. Candidates remain unconfirmed until named relational review. Full-sample and equal-call results are exported separately.\n"
    (out / "task_case_studies.md").write_text(case_text)
    text = (
        "# SYNTHETIC ENGINEERING FIXTURE — NOT SCIENTIFIC RESULTS\n\n"
        if c.get("demo")
        else ""
    ) + "# Expanded feature-space pilot\n\n"
    text += f"{len(items)} items; {bank['source_feature_count']:,} accepted generated phrases; {bank['normalized_unique']:,} normalized strings; {len(features):,} canonical candidate features. No feature-frequency or bank-size cap. All raw origins and explicit failed-attempt exclusions are retained.\n\n"
    text += f"Binary applicability: **{state['resolved_cells']:,} / {state['planned_cells']:,} resolved**. Complete matrix: **{state['complete']}**. Source generations complete: **{source['generations_complete']}**. Global judge: `{c['judge']['model_id']}`; generation models: {summary['generation_models']}.\n\n"
    if source.get("generation_checkpoint_hash"):
        text += f"**Available-generation sampling amendment:** this vocabulary is frozen from all valid generations available at the user-authorized checkpoint. {summary['missing_generation_jobs']} planned generations were absent; they are listed in missing_generation_jobs.csv and remain absent in sampling-normalized analyses. Pending/late outputs remain in the original experiments and require a new vocabulary version. This closes the execution scope without asserting complete generation coverage, approved semantic equivalence, or scientific completion.\n\n"
    if c.get("full_options", {}).get("stage_budget_policy") == "shared_total_only":
        text += "**Budget amendment:** one shared $100 cumulative ceiling includes previous costs and all held reservations. Separate stage partitions are removed for this controller at the user's request. Whole-stage estimates and durable per-request reservations still enforce the cumulative cap.\n\n"
    text += f"Semantic consolidation uses narrow declared rules plus supported complete-link equivalence decisions. {summary['pending_equivalence_proposals']:,} proposed comparisons remain pending; other unproposed pairs are not proven distinct. Automated judgments and merges are not human approvals. Remaining semantic uncertainty can inflate discovery and distance counts. A complete candidate vocabulary does not establish complete semantic resolution.\n\n"
    text += f"Costed known scope: {cost['new_primary_calls']:,} new primary calls; {cost['remaining_attempts_upper_bound']:,} maximum remaining attempts. First-attempt bound ${cost['first_attempt_upper_bound_usd']:.4f}; full configured-attempt bound ${cost['remaining_completion_upper_bound_usd']:.4f}, plus ${cost['cumulative_accounted_usd']:.4f} existing accounted liability. Stage cap ${cost['stage_cap_usd']}; cumulative cap ${cost['budget_cap_usd']}. Cost blockers: {cost['blockers']}. Item/feature cell-level prices and reasons are in blocked_cells.csv. Final task-derived additions are not guessed while generation is incomplete.\n\n"
    text += f"Retry funding policy: `{cost['retry_policy']}`. Current authorized stage: `{cost['authorized_stage']}`, entire-stage bound ${cost['authorized_stage_upper_bound_usd']:.4f}. All unattempted cells are funded before invalid-response retries. Pending requests must finish before the full set of known invalid retries is costed; if that set does not fit, all new retries stop. Maximum two attempts per cell remains fixed. The all-configured-attempt bound above is diagnostic and is not a guarantee that every possible retry is funded. Observed usage is exported by model and operation in observed_usage.json; unrelated semantic-matcher output usage does not calibrate atomic judging.\n\n"
    text += "## Original 150-feature baselines\n\n| Source | Original columns | Canonical columns retained | Additional full-bank columns | Original judge |\n|---|---|---|---|---|\n"
    for row in legacy:
        text += f"| {row['source_run_id']} | {row['original_feature_columns']} | {row['mapped_canonical_columns']} | {row['additional_canonical_columns']} | {row['original_judge_model']} |\n"
    text += "\nThe original experiments are preserved. Historical Mini and new Luna scores are not pooled or attributed solely to bank size. Where complete, the fair vocabulary comparison uses the same global judge/rows with all columns versus the mapped original subset. Geometry and applicable-count comparisons remain unavailable until their required cells are resolved.\n\n"
    if semantic.get("available"):
        text += f"## Automated correspondence to human labels\n\nThe separate norm matcher has scored {semantic['scored_propositions']:,} / {semantic['planned_propositions']:,} unique propositions. These results use the original cap of {semantic['comparison_cap']} features per response and the original human generations, rather than the expanded shared vocabulary. Scores are relative to human production lists and are not independently human-validated precision.\n\n| Prompt | Mean listed-feature recall | Item bootstrap interval | Complete paired items |\n|---|---|---|---|\n"
        for family, row in semantic["families"].items():
            text += f"| {family} | {row.get('mean_recall')} | {row.get('item_bootstrap_ci')} | {row.get('n_complete_items')} |\n"
        text += "\nRemaining category and translation reviews are independent of this automated Vinson feature-label comparison.\n\n"
    text += "## Scientific analyses and limits\n\nHuman correspondence is relative to published human productions, not truth. The separate automated norm matcher uses the original cap-normalized comparison and is not a bank-consolidation source. Harpaintner source category values are retained without inventing individual feature norms; pending English translation/category reviews suppress affected quantitative comparisons. Discovery curves include all generated candidates and separate protocol-cap-normalized curves. Applicable yield, overlap, geometry and compositional gains require independent binary cells. Novel phrases absent from human lists are not automatically false.\n\nRead task_case_studies.md and the full proposition tables for the two workflows. Missing values are ?, not zero; binary_matrix.csv is produced only for a fully resolved scope. Interaction candidates are distinct from confirmed relational interpretations and are not proof of emergence. Existing feature-conditioned embeddings are not silently repurposed as stimulus embeddings. No result demonstrates features internally or causally used by models, preferences or decision policies.\n"
    (out / "scientific_report.md").write_text(text)
    (out / "completion_plan.md").write_text(
        "# Costed completion plan\n\n"
        + "1. Let already-submitted source batches finish; preserve their receipts.\n2. Freeze all accepted generations and review/assist candidate equivalences under its separate preflight. Keep unresolved pairs distinct and auditable.\n3. Recompute final global dimensions, compatible reuse, measured prompt costs and full retry liability. Fund every remaining primary cell under the original caps; if the entire first pass fails, submit none.\n4. Schedule every eligible cell, holding existing external jobs and reusing exact compatible evidence. Collect all primary results, then preflight all known eligible invalid retries as a single stage; submit none if the entire retry stage does not fit. Maximum two attempts per cell; ambiguous results require review.\n5. Resolve the explicitly exported blocked cells and named relational/translation/category reviews, then regenerate the complete binary matrix and supported scientific analyses.\n\n"
        + f"Current budget/source blockers: {cost['blockers']}. No smaller arbitrary vocabulary is substituted.\n"
    )
    if render:
        _figures(out, source, bank, state, structure, curves, legacy, c)
    write_json(
        seal,
        dict(
            summary_hash=digest(summary),
            inventory_hash=bank["inventory_hash"],
            artifacts={
                p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p != seal
            },
        ),
    )
    return summary


def _figures(out, source, bank, state, structure, curves, legacy, c):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def finish(fig, name):
        fig.text(
            0.02,
            0.015,
            ("SYNTHETIC FIXTURE | " if c.get("demo") else "")
            + "Candidate representations; unresolved data retained; no causal/internal-use claim",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(out / (name + "." + ext), dpi=220, facecolor="white")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ["Generated phrases", "Normalized strings", "Consolidated candidates"]
    values = [
        bank["source_feature_count"],
        bank["normalized_unique"],
        len(bank["features"]),
    ]
    for row in legacy:
        if row["original_feature_columns"]:
            labels.append(
                "Original "
                + ("human" if "human" in row["source_run_id"] else "task")
                + " bank"
            )
            values.append(row["original_feature_columns"])
    ax.barh(
        labels,
        values,
        color=["#587D9C", "#35898A", "#C1813C"] + ["#97A7B3"] * len(legacy),
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="Number of phrases / feature columns",
        title="Complete available vocabulary versus preserved subsets\nSource generations may still be pending",
    )
    finish(fig, "figure_01_bank_coverage")
    fixed = saturation_summary(curves, replicates=c["replicates"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, stratum in zip(axes, ["human", "task"]):
        for family, color in [("P0", "#3267A2"), ("P1", "#147F77"), ("P2", "#D76B31")]:
            selected = [
                r
                for r in fixed
                if r["stratum"] == stratum
                and r["prompt_family"] == family
                and r["n_items"] > 0
            ]
            ns = sorted({r["replicates_included"] for r in selected})
            if ns:
                ax.plot(
                    ns,
                    [
                        next(
                            r["mean_canonical_discoveries"]
                            for r in selected
                            if r["replicates_included"] == n
                        )
                        for n in ns
                    ],
                    marker="o",
                    label=family,
                    color=color,
                )
        ax.set(
            title=stratum
            + f" generation (n={next((r['n_items'] for r in fixed if r['stratum']==stratum),0)} common targets)",
            xlabel="Independent repetitions pooled",
            ylabel="Mean canonical candidates discovered",
            xticks=[1, 2, 3],
        )
        if ax.lines:
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "Source responses not yet collected",
                ha="center",
                transform=ax.transAxes,
            )
    fig.suptitle(
        "Protocol-cap-normalized candidate discovery; applicability validated separately"
    )
    finish(fig, "figure_02_discovery_saturation")
    if state["complete"]:
        fig, ax = plt.subplots(figsize=(12, 11))
        im = ax.imshow(
            np.asarray(structure["geometry"]["rdm"]), vmin=0, vmax=1, cmap="viridis"
        )
        ids = state["item_ids"]
        ax.set(
            xticks=range(len(ids)),
            yticks=range(len(ids)),
            xticklabels=ids,
            yticklabels=ids,
            title="Global binary-feature Jaccard distances",
        )
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        fig.colorbar(im, ax=ax, label="Jaccard distance")
        finish(fig, "figure_03_global_geometry")
    else:
        (out / "figure_03_global_geometry_WITHHELD.md").write_text(
            "Geometry withheld: the global item × feature scope has unresolved cells; no zero filling or pair-dependent feature subsets.\n"
        )

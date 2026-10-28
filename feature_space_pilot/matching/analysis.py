"""Item-level automated recall with explicit scoring coverage and uncertainty."""

from collections import defaultdict
from pathlib import Path
import textwrap
import numpy as np
from ..io import write_json, write_csv


def aggregate(inventory, records):
    decisions = {}
    for record in records:
        if record.get("parse_status") != "ok" or record.get("is_control"):
            continue
        for row in record["results"]:
            key = (record["item_id"], row["proposition_id"])
            if key in decisions:
                raise ValueError("Duplicate successful semantic score")
            decisions[key] = row
    features = defaultdict(set)
    for h in inventory["norm_features"]:
        features[h["item_id"]].add(h["human_id"])
    recovery = []
    for item in inventory["items"]:
        iid = item["item_id"]
        for family in ["P0", "P1", "P2"]:
            all_origins = [
                o
                for o in inventory["origins"]
                if o["item_id"] == iid and o["prompt_family"] == family
            ]
            if not all_origins:
                continue
            for n in range(1, max(o["replicate"] for o in all_origins) + 2):
                origins = [o for o in all_origins if o["replicate"] < n]
                expected = {o["proposition_id"] for o in origins}
                scored = [
                    decisions[(iid, p)]
                    for p in sorted(expected)
                    if (iid, p) in decisions
                ]
                matched = (
                    set().union(*(set(r["matches"]) for r in scored))
                    if scored
                    else set()
                )
                uncertain = (
                    set().union(*(set(r["uncertain"]) for r in scored))
                    if scored
                    else set()
                )
                complete = len(scored) == len(expected)
                denominator = len(features[iid])
                recovery.append(
                    dict(
                        item_id=iid,
                        target=item["text"],
                        split=item["split"],
                        prompt_family=family,
                        replicates_included=n,
                        n_human_features=denominator,
                        n_expected_propositions=len(expected),
                        n_scored_propositions=len(scored),
                        scoring_complete=complete,
                        n_generation_calls=len(
                            {o["generation_job_id"] for o in origins}
                        ),
                        n_matched_human_features=len(matched),
                        n_uncertain_additional_human_features=len(uncertain - matched),
                        recall=len(matched) / denominator if complete else None,
                        observed_recall_lower_bound=len(matched) / denominator,
                        recall_including_uncertain=(
                            len(matched | uncertain) / denominator if complete else None
                        ),
                        matched_human_ids=sorted(matched),
                        uncertain_additional_human_ids=sorted(uncertain - matched),
                        human_reviewed=False,
                    )
                )
    norm = {(h["item_id"], h["human_id"]): h for h in inventory["norm_features"]}
    prop = {(p["item_id"], p["proposition_id"]): p for p in inventory["propositions"]}
    pairs = []
    for (iid, pid), row in sorted(decisions.items()):
        for relation in ["matches", "related", "uncertain"]:
            for hid in row[relation]:
                pairs.append(
                    dict(
                        item_id=iid,
                        proposition_id=pid,
                        proposition=prop[iid, pid]["text"],
                        human_id=hid,
                        human_label=norm[iid, hid]["label"],
                        relation=relation,
                        evidence=row["evidence"],
                        human_reviewed=False,
                    )
                )
    return dict(
        human_reviewed=False,
        evaluation_kind="automated_exploratory_semantic_alignment",
        n_propositions=len(inventory["propositions"]),
        n_scored_propositions=len(decisions),
        recovery=recovery,
        pairs=pairs,
    )


def control_status(jobs, records):
    scored = {r["job_id"]: r for r in records if r.get("parse_status") == "ok"}
    checks = []
    for job in jobs:
        if not job.get("is_control"):
            continue
        record = scored.get(job["job_id"])
        actual = (
            {r["proposition_id"]: set(r["matches"]) for r in record["results"]}
            if record
            else {}
        )
        checks.append(
            dict(
                job_id=job["job_id"],
                scored=record is not None,
                passed=record is not None
                and all(
                    actual.get(pid) == set(labels)
                    for pid, labels in job["expected_control_matches"].items()
                ),
            )
        )
    return dict(
        available=bool(checks) and all(c["scored"] for c in checks),
        passed=bool(checks) and all(c["passed"] for c in checks),
        checks=checks,
        caveat="Synthetic logical smoke checks do not establish grader validity",
    )


def _csv(path, rows):
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["status"]
    write_csv(path, [{k: r.get(k) for k in fields} for r in rows], fields)


def paired_summary(final):
    families = ["P0", "P1", "P2"]
    sets = [
        {
            r["item_id"]
            for r in final
            if r["prompt_family"] == family and r["scoring_complete"]
        }
        for family in families
    ]
    paired = sorted(set.intersection(*sets))
    summary = {}
    for family in families:
        values = np.asarray(
            [
                r["recall"]
                for r in final
                if r["prompt_family"] == family and r["item_id"] in paired
            ]
        )
        rng = np.random.default_rng(42)
        samples = (
            rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
            if len(values)
            else None
        )
        summary[family] = dict(
            n_complete_items=len(values),
            mean_recall=float(values.mean()) if len(values) else None,
            item_bootstrap_ci=(
                np.quantile(samples, [0.025, 0.975]).tolist()
                if samples is not None
                else None
            ),
        )
    return summary, paired


def write_analysis(
    out, inventory, jobs, records, *, source_id, model_id, synthetic=False
):
    out = Path(out)
    result = aggregate(inventory, records)
    result["demo"] = synthetic
    result["control_checks"] = control_status(jobs, records)
    write_json(out / "semantic_recovery.json", result)
    _csv(out / "semantic_recovery.csv", result["recovery"])
    _csv(out / "semantic_pairs.csv", result["pairs"])
    final = [
        r
        for r in result["recovery"]
        if r["replicates_included"]
        == max(
            x["replicates_included"]
            for x in result["recovery"]
            if x["item_id"] == r["item_id"] and x["prompt_family"] == r["prompt_family"]
        )
    ]
    summary, paired = paired_summary(final)
    write_json(
        out / "summary.json",
        dict(
            source_run_id=source_id,
            grader_model=model_id,
            human_reviewed=False,
            demo=synthetic,
            scored_propositions=result["n_scored_propositions"],
            planned_propositions=result["n_propositions"],
            control_checks=result["control_checks"],
            families=summary,
            paired_item_ids=paired,
        ),
    )
    text = f"# Automated semantic matching — exploratory\n\nSource: `{source_id}`. Grader: `{model_id}`. These are model-scored correspondences, not human-reviewed validation. Original exact-match scores and primary pilot data remain unchanged.\n\nScored propositions: {result['n_scored_propositions']} / {result['n_propositions']}. Synthetic logical checks passed: {result['control_checks']['passed']}. Missing scores never become no-match labels.\n\n"
    if synthetic:
        text = "SYNTHETIC TEST FIXTURE — NOT EMPIRICAL\n\n" + text
    for family, s in summary.items():
        text += f"- {family}: {s['n_complete_items']} completely scored targets; mean strict automated recall = {s['mean_recall']}.\n"
    text += "\nRecall counts unique supplied human feature labels recovered per target at a fixed first-ten-features-per-generation comparison cap, pooling up to three replicates. Categories and label inventories are source-specific. Related labels do not count as clear matches; uncertain labels are reported separately as a sensitivity analysis. Unlisted generated features are not automatically false. All target labels were offered to the grader; there is no candidate-retrieval cutoff. Grader inputs omit prompt family, repetition, generator model, split and production frequency. Development/held-out splits remain in the exported tables.\n\nItem-bootstrap intervals describe variability across completely scored targets and do not measure semantic-grader accuracy. Incomplete coverage and missing targets must be considered before comparing families. Named human translation/category/semantic approvals were not supplied or inferred. The 24 abstract category-only targets are excluded from lexical/semantic feature recovery.\n"
    (out / "analysis_summary.md").write_text(text)
    if not result["control_checks"]["passed"] or not paired:
        return result
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def finish(fig, name):
        fig.text(
            0.04,
            0.025,
            f"{'SYNTHETIC FIXTURE' if synthetic else 'AUTOMATED EXPLORATORY — NO HUMAN VALIDATION'} | {source_id} | {model_id}",
            fontsize=8,
        )
        for extension in ["png", "svg", "pdf"]:
            fig.savefig(out / (name + "." + extension), dpi=220, facecolor="white")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = {"P0": "#3267A2", "P1": "#147F77", "P2": "#D76B31"}
    for iid in paired:
        rows = [
            next(
                (
                    r
                    for r in final
                    if r["item_id"] == iid
                    and r["prompt_family"] == f
                    and r["scoring_complete"]
                ),
                None,
            )
            for f in colors
        ]
        if all(r is not None for r in rows):
            ax.plot(
                range(3),
                [r["recall"] for r in rows],
                color="#B8C2C8",
                alpha=0.45,
                lw=0.7,
            )
    for n, (family, color) in enumerate(colors.items()):
        values = [
            r["recall"]
            for r in final
            if r["prompt_family"] == family and r["item_id"] in paired
        ]
        ax.scatter([n] * len(values), values, color=color, alpha=0.55, s=20)
        if summary[family]["mean_recall"] is not None:
            mean = summary[family]["mean_recall"]
            low, high = summary[family]["item_bootstrap_ci"]
            ax.errorbar(
                n,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="D",
                color="#172F3A",
                capsize=6,
                markersize=8,
            )
    ax.set(
        xticks=range(3),
        xticklabels=list(colors),
        ylim=(-0.02, 1.02),
        ylabel="Fraction of listed human labels recovered",
        title="Automated clear-match recall by prompt family\nDots = targets; diamonds = means; bars = descriptive item-bootstrap intervals",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    finish(fig, "figure_01_semantic_recovery")
    fig, ax = plt.subplots(figsize=(11, 7))
    for family, color in colors.items():
        xs = []
        means = []
        for n in sorted({r["replicates_included"] for r in result["recovery"]}):
            values = [
                r["recall"]
                for r in result["recovery"]
                if r["prompt_family"] == family
                and r["replicates_included"] == n
                and r["scoring_complete"]
                and r["item_id"] in paired
            ]
            if values:
                xs.append(n)
                means.append(float(np.mean(values)))
        ax.plot(xs, means, marker="o", label=family, color=color)
    ax.set(
        xlabel="Independent repetitions pooled per target",
        ylabel="Mean fraction of listed human labels recovered",
        ylim=(-0.02, 1.02),
        xticks=[1, 2, 3],
        title="Automated recovery versus repetition budget\nFirst ten propositions per original call; human labels counted once per target",
    )
    ax.legend()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    finish(fig, "figure_02_semantic_saturation")
    return result

"""Export read-only task figures from an immutable scientific report checkpoint."""

import argparse
import csv
import json
import shutil
import textwrap
from pathlib import Path
from collections import defaultdict
import numpy as np
from ..io import digest, file_hash, write_csv, write_json
from .analysis import task_audit, pattern_groups, select_examples, inheritance_flags

ORDER = [
    "poem_domain",
    "poem_type",
    "poem_T",
    "poem_S_deadline",
    "poem_S_review",
    "poem_S_both",
    "finance_domain",
    "finance_type",
    "finance_T",
    "finance_S_deadline",
    "finance_S_review",
    "finance_S_both",
    "S_deadline",
    "S_review",
    "S_both",
]
NAMES = dict(
    poem_domain="Writing: domain",
    poem_type="Writing: task type",
    poem_T="Writing: instance",
    poem_S_deadline="Writing + deadline",
    poem_S_review="Writing + review",
    poem_S_both="Writing + both",
    finance_domain="Finance: domain",
    finance_type="Finance: task type",
    finance_T="Finance: instance",
    finance_S_deadline="Finance + deadline",
    finance_S_review="Finance + review",
    finance_S_both="Finance + both",
    S_deadline="Deadline alone",
    S_review="Review alone",
    S_both="Both situations alone",
)


def load_snapshot(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "full_manifest.json").read_text())
    report = Path(manifest["report_dir"])
    for name in ["stimuli.json", "frozen_global_bank.json"]:
        if file_hash(root / name) != manifest["frozen_hashes"][name]:
            raise ValueError("Frozen task input hash mismatch: " + name)
    seal = json.loads((report / "report_manifest.json").read_text())
    for name in [
        "task_applicability.csv",
        "compositionality.json",
        "generation_coverage.csv",
    ]:
        if file_hash(report / name) != seal["artifacts"][name]:
            raise ValueError("Task report hash mismatch: " + name)
    items = [
        i
        for i in json.loads((root / "stimuli.json").read_text())
        if i.get("target_type") not in {"concept", "object", "event"}
    ]
    items.sort(
        key=lambda i: ORDER.index(i["item_id"]) if i["item_id"] in ORDER else len(ORDER)
    )
    bank = json.loads((root / "frozen_global_bank.json").read_text())
    features = bank["features"]
    ii = {i["item_id"]: n for n, i in enumerate(items)}
    ff = {f["feature_id"]: n for n, f in enumerate(features)}
    if len(ii) != len(items) or len(ff) != len(features):
        raise ValueError("Duplicate task scope identities")
    a = np.full((len(items), len(features)), -1, dtype=np.int8)
    seen = set()
    for row in csv.DictReader((report / "task_applicability.csv").open()):
        key = row["item_id"], row["feature_id"]
        if (
            key in seen
            or key[0] not in ii
            or key[1] not in ff
            or row["applies"] not in {"0", "1", "?"}
        ):
            raise ValueError("Invalid task cell or duplicate")
        seen.add(key)
        if row["applies"] != "?":
            a[ii[key[0]], ff[key[1]]] = int(row["applies"])
    if len(seen) != a.size:
        raise ValueError("Task report omits declared cells")
    coverage = {
        r["item_id"]: r
        for r in csv.DictReader((report / "generation_coverage.csv").open())
        if r["item_id"] in ii
    }
    provenance = dict(
        source_run=manifest["analysis_id"],
        source_report=str(report),
        report_manifest_sha256=file_hash(report / "report_manifest.json"),
        bank_inventory_hash=bank["inventory_hash"],
        demo=bool(manifest.get("demo")),
    )
    return dict(
        items=items,
        features=features,
        bank=bank,
        matrix=a,
        coverage=coverage,
        compositions=json.loads((report / "compositionality.json").read_text()),
        provenance=provenance,
        report_dir=report,
    )


def render(snapshot, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    items = snapshot["items"]
    features = snapshot["features"]
    a = snapshot["matrix"]
    bank = snapshot["bank"]
    audit = task_audit(items, features, a)
    patterns = pattern_groups(a, features)
    examples = select_examples(items, features, a, bank["origins"])
    labels = [NAMES.get(i["item_id"], i["item_id"]) for i in items]
    colors = ["#D9A256", "#EDF1F5", "#167D8D"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
    banner = "SYNTHETIC FIXTURE | " if snapshot["provenance"]["demo"] else ""

    def finish(fig, name, footnote):
        fig.text(
            0.02, 0.012, banner + footnote, fontsize=9, color="#334155", va="bottom"
        )
        for ext in ["png", "svg"]:
            fig.savefig(out / (name + "." + ext), dpi=200, facecolor="white")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 8.5))
    y = np.arange(len(items))
    left = np.zeros(len(items))
    for key, label, color in [
        ("applicable_observed", "Applies (1)", colors[2]),
        ("inapplicable", "Does not apply (0)", "#CBD5E1"),
        ("unresolved", "Unresolved (?)", colors[0]),
    ]:
        values = np.array([r[key] for r in audit["items"]])
        ax.barh(y, values, left=left, color=color, label=label, height=0.72)
        left += values
    ax.set(
        yticks=y,
        yticklabels=labels,
        xlim=(0, len(features)),
        xlabel="Number of features (the complete shared vocabulary)",
        title=banner + "Task/situation judgments: what is resolved?",
    )
    ax.invert_yaxis()
    ax.set_title(
        banner + "Task/situation judgments: what is resolved?", pad=52, fontsize=16
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    for n, r in enumerate(audit["items"]):
        ax.text(
            len(features) + 45,
            n,
            f"{100*r['unresolved']/len(features):.1f}% ?",
            va="center",
            fontsize=9,
        )
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.89, top=0.82, bottom=0.11)
    finish(
        fig,
        "figure_01_task_coverage",
        "Every shared feature is included. Unresolved means ambiguous, invalid or missing; it never means 0.",
    )
    ff = {f["feature_id"]: n for n, f in enumerate(features)}
    if examples:
        ks = [ff[r["feature_id"]] for r in examples]
        fig, ax = plt.subplots(figsize=(18, max(9, len(ks) * 0.42 + 3)))
        ax.imshow(
            a[:, ks].T, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest"
        )
        ax.set(
            xticks=range(len(items)),
            xticklabels=labels,
            yticks=range(len(ks)),
            yticklabels=["\n".join(textwrap.wrap(features[k]["text"], 65)) for k in ks],
        )
        ax.tick_params(axis="x", labelrotation=55, labelsize=9)
        ax.tick_params(axis="y", labelsize=8)
        for v in [5.5, 11.5]:
            if v < len(items):
                ax.axvline(v, color="#334155", linewidth=2)
        ax.set_xticks(np.arange(-0.5, len(items), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ks), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title(
            banner
            + "Actual propositions across tasks, abstraction levels and situations",
            pad=22,
            fontsize=15,
        )
        fig.legend(
            handles=[
                Patch(color=c, label=l)
                for c, l in zip(
                    colors, ["Unresolved (?)", "Does not apply (0)", "Applies (1)"]
                )
            ],
            loc="upper center",
            bbox_to_anchor=(0.72, 0.95),
            ncol=3,
            frameon=False,
        )
        fig.subplots_adjust(left=0.49, right=0.98, top=0.89, bottom=0.19)
        finish(
            fig,
            "figure_02_task_feature_map",
            f"Display examples: {len(ks)} features, up to 2 positive discoveries per item; the full {len(features):,}-feature matrix remains in task_applicability.csv.",
        )
    groups = patterns["groups"][:12]
    if groups:
        fig, ax = plt.subplots(figsize=(18, max(7, len(groups) * 0.43 + 3)))
        ax.imshow(
            np.asarray([r["pattern"] for r in groups]),
            cmap=ListedColormap([colors[1], colors[2]]),
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set(
            xticks=range(len(items)),
            xticklabels=labels,
            yticks=range(len(groups)),
            yticklabels=[
                f"{r['n_features']} features | "
                + "\n".join(textwrap.wrap(r["texts"][0], 65))
                for r in groups
            ],
        )
        ax.tick_params(axis="x", labelrotation=55, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(
            banner + "Exploratory feature groups with identical applicability patterns",
            fontsize=15,
            pad=22,
        )
        fig.subplots_adjust(left=0.49, right=0.98, top=0.86, bottom=0.24)
        foot = f"Restricted view: {patterns['positive_complete_columns']:,} / {len(features):,} features have fully resolved, nonzero profiles; {patterns['unknown_columns']:,} have unresolved cells.\nSame pattern does not imply same meaning. Showing {len(groups)} of {len(patterns['groups'])} groups; all memberships and full propositions are exported."
        finish(fig, "figure_03_coapplicability_patterns", foot)
    return audit, patterns, examples


def export(snapshot, out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        m = json.loads((out / "manifest.json").read_text())
        if any(file_hash(out / n) != h for n, h in m["artifacts"].items()):
            raise ValueError("Existing task figure artifact changed")
        return json.loads((out / "summary.json").read_text())
    audit, patterns, examples = render(snapshot, out)
    tids = {i["item_id"] for i in snapshot["items"]}
    origins = [o for o in snapshot["bank"]["origins"] if o["item_id"] in tids]
    task_ids = {o["canonical_id"] for o in origins}
    flags = inheritance_flags(
        snapshot["items"], snapshot["features"], snapshot["matrix"], task_ids
    )
    write_csv(
        out / "inheritance_review_flags.csv",
        flags,
        list(flags[0]) if flags else ["status"],
    )
    coverage = list(snapshot["coverage"].values())
    generation_completed = sum(int(r["completed"]) for r in coverage)
    generation_planned = sum(int(r["planned"]) for r in coverage)
    summary = dict(
        audit,
        task_generated_phrases=len(origins),
        task_discovered_canonical_features=len(task_ids),
        generation_completed=generation_completed,
        generation_planned=generation_planned,
        pattern_analysis={k: v for k, v in patterns.items() if k != "groups"},
        source=snapshot["provenance"],
        human_reviewed=False,
        api_calls=0,
        inheritance_review_flags=len(flags),
    )
    write_json(out / "summary.json", summary)
    write_json(out / "pattern_groups.json", patterns)
    write_json(out / "source_checkpoint.json", snapshot["provenance"])
    write_csv(out / "task_item_coverage.csv", audit["items"], list(audit["items"][0]))
    write_csv(
        out / "displayed_features.csv",
        examples,
        ["feature_id", "proposition", "display_origin"],
    )
    write_csv(
        out / "pattern_groups.csv",
        patterns["groups"],
        ["pattern", "n_features", "feature_ids", "texts"],
    )
    for name in [
        "task_applicability.csv",
        "task_level_feature_audit.csv",
        "compositionality.json",
        "interaction_review.csv",
        "generation_coverage.csv",
        "task_case_studies.md",
    ]:
        if (snapshot["report_dir"] / name).exists():
            shutil.copyfile(snapshot["report_dir"] / name, out / name)
    feature_rows = []
    for n, f in enumerate(snapshot["features"]):
        if f["feature_id"] in task_ids:
            os = [o for o in origins if o["canonical_id"] == f["feature_id"]]
            feature_rows.append(
                dict(
                    feature_id=f["feature_id"],
                    proposition=f["text"],
                    source_items=sorted({o["item_id"] for o in os}),
                    prompt_families=sorted({o["prompt_family"] for o in os}),
                    n_task_origins=len(os),
                    known_applicable_items=int((snapshot["matrix"][:, n] == 1).sum()),
                    unresolved_items=int((snapshot["matrix"][:, n] < 0).sum()),
                )
            )
    write_csv(
        out / "task_discovered_features.csv",
        feature_rows,
        list(feature_rows[0]) if feature_rows else ["status"],
    )
    (out / "briefing.md").write_text(
        f"# Task and situation feature-space briefing\n\n{generation_completed}/{generation_planned} task generation jobs accepted; {len(origins)} accepted phrases map to {len(task_ids)} canonical features in the shared {len(snapshot['features']):,}-feature bank. All 15 items are independently judged against every shared feature.\n\n{audit['resolved_cells']:,}/{audit['planned_cells']:,} cells resolved ({100*audit['resolved_cells']/audit['planned_cells']:.1f}%); {audit['unresolved_cells']:,} unresolved. This is beyond feature creation, but incomplete judgments limit final geometry and composition claims. Missing generation samples remain visible.\n\nStart with figure_01_task_coverage.png, then figure_02_task_feature_map.png. Figure_03_coapplicability_patterns.png groups identical fully resolved profiles; it is an exploratory complete-case subset, not a semantic merge or learned abstraction hierarchy. Only {patterns['positive_complete_columns']} positive fully resolved feature columns support that panel.\n\nDomain → task type → instance is the existing task hierarchy. Situations are an independent axis. Proposed additional feature tags: goals/outputs, operations, constraints/resources, quality/evaluation, social context, interactions/trade-offs. These are illustrative analytical categories, not independently reviewed labels. Multiple tags may apply to one proposition. A cluster of related features does not replace the original binary columns.\n\nCompositionality compares independently discovered G(T)∪G(S) with directly discovered G(T+S). Novel discovery alone does not establish an interaction; candidate interactions require T=0, S=0, TS=1 and further interpretation. The copied compositionality.json retains incomplete-generation and unresolved-cell flags.\n\nMethod precedents: [Harpaintner et al.,2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6156367/) use category profiles, dendrograms and cluster profiles; [Kriegeskorte et al.,2008](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full) use representational dissimilarity matrices and multidimensional scaling; [Lex et al.,2014](https://vdl.sci.utah.edu/publications/2014_infovis_upset/) use set-intersection matrices. Our initial figures adapt the matrix/profile approach. Full geometry is not inferred by replacing unresolved cells with zero. No output demonstrates causal/internal model feature use.\n\nSource checkpoint: {snapshot['provenance']['source_report']}\n\nOffline export only: no API requests; active execution and source artifacts are untouched.\n"
    )
    with (out / "briefing.md").open("a") as briefing:
        briefing.write(
            f"\n## Immediate quality review\n\n{len(flags)} parent-positive / situated-child-negative task-origin cells are flagged in inheritance_review_flags.csv. These are review candidates, not confirmed errors: situations can legitimately change some properties. Start with unchanged explicit requirements.\n\n"
        )
        for row in flags:
            if row["proposition"] == "lines 1 and 3 must rhyme":
                briefing.write(
                    f"Observed example: **{row['proposition']}** is 1 for {row['parent_id']} and 0 for {row['situated_id']}, although the situated description retains the same ABAB CDCD requirement. Inspect this judgment before interpreting the difference as a situation effect. No value was changed.\n\n"
                )
        ii = {i["item_id"]: n for n, i in enumerate(snapshot["items"])}
        for k, f in enumerate(snapshot["features"]):
            if (
                f["text"] == "may result in civilian casualties"
                and "poem_T" in ii
                and snapshot["matrix"][ii["poem_T"], k] == 1
            ):
                briefing.write(
                    "The complete-profile grouping also exposes an unexpected cross-domain label: a civilian-casualty proposition is marked applicable to the poem instance in this checkpoint. This merits review. Clustering describes these provisional labels and does not validate them.\n\n"
                )
        briefing.write(
            "Feature-family tags and a feature-abstraction ontology have not yet been assigned.\n"
        )
    write_json(
        out / "manifest.json",
        dict(
            source=snapshot["provenance"],
            artifacts={p.name: file_hash(p) for p in out.iterdir() if p.is_file()},
            code={p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
        ),
    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run")
    p.add_argument("--output-dir")
    a = p.parse_args()
    root = (
        Path(a.source_run)
        if a.source_run
        else Path(
            json.loads(
                Path("artifacts/feature_space_full/controller_latest.json").read_text()
            )["project_dir"]
        )
    )
    snap = load_snapshot(root)
    out = (
        Path(a.output_dir)
        if a.output_dir
        else Path("artifacts/feature_space_task_focus")
        / digest(
            [
                snap["provenance"],
                {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
            ]
        )[:16]
    )
    summary = export(snap, out)
    write_json(
        Path("artifacts/feature_space_task_focus/latest.json"),
        dict(output_dir=str(out.resolve()), source=snap["provenance"]),
    )
    print(json.dumps(dict(output_dir=str(out.resolve()), summary=summary), indent=2))


if __name__ == "__main__":
    main()

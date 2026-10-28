"""Three proposition-level task figures; interpretation overlays never replace columns."""

import textwrap
import numpy as np
from ..task_focus.run import NAMES, ORDER
from .analysis import FAMILY_RULES
from .reports import csv_rows


def render_task_figures(out, items, bank, state, analysis, *, demo=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch, Rectangle

    a = state["matrix"]
    features = bank["features"]
    fm = {f["feature_id"]: k for k, f in enumerate(features)}
    im = {i["item_id"]: i for i in items}
    discovered_by = {f["feature_id"]: set() for f in features}
    for o in bank["origins"]:
        discovered_by[o["canonical_id"]].add(im[o["item_id"]].get("task_id"))
    am = {r["feature_id"]: r for r in analysis["annotations"]}
    colors = ["#D9A256", "#EDF1F5", "#167D8D"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
    banner = "SYNTHETIC FIXTURE | " if demo else ""
    legend = [
        Patch(facecolor=c, label=s)
        for c, s in zip(colors, ["Unresolved (?)", "Does not apply (0)", "Applies (1)"])
    ]

    def finish(fig, stem, foot):
        fig.text(
            0.015,
            0.012,
            banner + foot,
            fontsize=9,
            color="#334155",
            va="bottom",
            wrap=True,
        )
        for ext in ("png", "svg"):
            fig.savefig(out / (stem + "." + ext), dpi=190, facecolor="white")
        plt.close(fig)

    def rank(f, ns):
        k = fm[f["feature_id"]]
        p = am[f["feature_id"]]
        return (
            p["potential_irrelevance"],
            not bool((a[ns, k] == 1).any()),
            not p["generalizable_functional_candidate"],
            -int((a[ns, k] >= 0).sum()),
            len(f["text"]),
            f["text"],
        )

    # A: overlapping semantic coding, with the actual distinct propositions retained.
    selected = []
    seen = set()
    for family in FAMILY_RULES:
        candidates = [
            f for f in features if family in am[f["feature_id"]]["semantic_families"]
        ]
        for f in sorted(candidates, key=lambda f: rank(f, list(range(len(items))))):
            if f["feature_id"] in seen:
                continue
            selected.append((family, f))
            seen.add(f["feature_id"])
            if sum(label == family for label, _ in selected) >= 4:
                break
    if not selected:
        selected = [("Unclassified — requires review", f) for f in features[:12]]
    fig, ax = plt.subplots(figsize=(20, max(9, len(selected) * 0.48 + 4)))
    column_order = sorted(
        range(len(items)),
        key=lambda n: (
            ORDER.index(items[n]["item_id"])
            if items[n]["item_id"] in ORDER
            else len(ORDER) + n
        ),
    )
    if selected:
        ax.imshow(
            a[np.ix_(column_order, [fm[f["feature_id"]] for _, f in selected])].T,
            cmap=cmap,
            norm=norm,
            aspect="auto",
        )
        ax.set_yticks(
            range(len(selected)),
            [textwrap.fill(label + " · " + f["text"], 76) for label, f in selected],
            fontsize=9,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No accepted task discoveries",
            ha="center",
            transform=ax.transAxes,
        )
    ax.set_xticks(
        range(len(items)),
        [NAMES.get(items[n]["item_id"], items[n]["item_id"]) for n in column_order],
        rotation=50,
        ha="right",
        fontsize=9,
    )
    ax.set_title(
        banner
        + "Figure A · Task-generated feature space\nProvisional overlapping semantic families; actual applicability judgments",
        pad=24,
        fontsize=16,
    )
    fig.legend(
        handles=legend,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.985),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.54, right=0.99, top=0.90, bottom=0.18)
    finish(
        fig,
        "figure_A_task_feature_space",
        f"Readable examples only; all {len(features)} task-discovered features remain in the bank and matrix. Family labels use transparent offline cues and require review; unknown is never zero.",
    )
    csv_rows(
        out / "figure_A_display_examples.csv",
        [
            dict(family=family, feature_id=f["feature_id"], proposition=f["text"])
            for family, f in selected
        ],
    )

    # B: same feature statements followed across each task's hierarchy and situations.
    fig, axes = plt.subplots(2, 1, figsize=(19, 13))
    flagpairs = {
        (r["situated_id"], r["feature_id"]) for r in analysis["inheritance_flags"]
    }
    displayed = []
    for ax, task, title in zip(
        axes, ("poem", "finance"), ("Creative writing", "Financial reconciliation")
    ):
        ns = [n for n, i in enumerate(items) if i.get("task_id") == task]
        candidates = (
            sorted(
                [f for f in features if task in discovered_by[f["feature_id"]]],
                key=lambda f: rank(f, ns),
            )
            if ns
            else []
        )
        broad = [
            f
            for f in candidates
            if am[f["feature_id"]]["specificity"] == "functional_or_general_candidate"
        ][:4]
        detail = [
            f
            for f in candidates
            if am[f["feature_id"]]["specificity"] == "instance_detail_candidate"
        ][:4]
        flagged = sorted(
            [
                f
                for f in candidates
                if any((items[n]["item_id"], f["feature_id"]) in flagpairs for n in ns)
            ],
            key=lambda f: (
                "lines 1 and 3 must rhyme" not in f["text"],
                am[f["feature_id"]]["specificity"] != "instance_detail_candidate",
                len(f["text"]),
            ),
        )
        chosen = []
        for f in broad + flagged[:1] + detail:
            if f not in chosen:
                chosen.append(f)
        displayed.extend(
            dict(
                task_id=task,
                feature_id=f["feature_id"],
                proposition=f["text"],
                specificity=am[f["feature_id"]]["specificity"],
            )
            for f in chosen
        )
        if ns and chosen:
            ax.imshow(
                a[np.ix_(ns, [fm[f["feature_id"]] for f in chosen])].T,
                cmap=cmap,
                norm=norm,
                aspect="auto",
            )
            ax.set_yticks(
                range(len(chosen)),
                [
                    textwrap.fill(
                        (
                            "General candidate · "
                            if am[f["feature_id"]]["generalizable_functional_candidate"]
                            else "Detail / scope review · "
                        )
                        + f["text"],
                        76,
                    )
                    for f in chosen
                ],
                fontsize=9,
            )
            ax.set_xticks(
                range(len(ns)),
                [NAMES.get(items[n]["item_id"], items[n]["item_id"]) for n in ns],
                fontsize=9,
                rotation=15,
            )
            for row, f in enumerate(chosen):
                for col, n in enumerate(ns):
                    if (items[n]["item_id"], f["feature_id"]) in flagpairs:
                        ax.add_patch(
                            Rectangle(
                                (col - 0.5, row - 0.5),
                                1,
                                1,
                                fill=False,
                                edgecolor="#C4514A",
                                lw=2,
                            )
                        )
        else:
            ax.text(
                0.5,
                0.5,
                "This fixture contains no " + title.lower() + " hierarchy",
                ha="center",
                transform=ax.transAxes,
            )
        ax.set_title(
            title + ": domain → task type → instance; situations are a separate axis",
            fontsize=12,
            pad=16,
        )
    fig.suptitle(
        banner + "Figure B · Abstraction, generalization and inheritance",
        fontsize=16,
        y=0.98,
    )
    fig.legend(
        handles=legend,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.955),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.54, right=0.99, top=0.88, bottom=0.09, hspace=0.43)
    finish(
        fig,
        "figure_B_abstraction",
        "Red outlines flag parent-positive / situated-negative disagreements, not confirmed errors. Broad/detail labels are provisional. Full transitions and proposition tables accompany the figure.",
    )
    csv_rows(out / "figure_B_display_examples.csv", displayed)

    # C: mutually exclusive discovery categories, separate from evidence of emergence.
    categories = [
        ("shared_discovery", "Shared discovery", "#167D8D"),
        ("component_only_discovery", "Component only", "#BBC7D2"),
        ("constituent_omission", "Direct addition: constituent omission", "#4F83B8"),
        (
            "interaction_pattern_candidate",
            "Direct addition: 0/0/1 candidate",
            "#8B6CA8",
        ),
        ("direct_only_unresolved", "Direct addition: unresolved", "#D9A256"),
        ("direct_only_inapplicable", "Direct addition: judged inapplicable", "#C97C70"),
    ]
    rs = [r for r in analysis["compositions"] if r["sampling_mode"] == "equal_calls"]
    fig, (ax, examples) = plt.subplots(
        1,
        2,
        figsize=(23, max(9, len(rs) * 0.52 + 3)),
        gridspec_kw={"width_ratios": [1.1, 1]},
    )
    handles = [Patch(facecolor=c, label=label) for _, label, c in categories]
    displayed = []
    if rs:
        y = np.arange(len(rs))
        left = np.zeros(len(rs))
        for category, label, color in categories:
            values = []
            for r in rs:
                rows = [
                    x
                    for x in analysis["composition_features"]
                    if x["item_id"] == r["item_id"]
                    and x["prompt_family"] == r["prompt_family"]
                    and x["sampling_mode"] == "equal_calls"
                ]
                values.append(sum(x["category"] == category for x in rows))
            ax.barh(y, values, left=left, color=color, height=0.72)
            for pos, v in enumerate(values):
                if v >= 2:
                    ax.text(
                        left[pos] + v / 2,
                        pos,
                        str(v),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            left += values
        labels = [
            NAMES.get(r["item_id"], r["item_id"])
            + " · "
            + r["prompt_family"]
            + (" †" if not r["generation_coverage_complete"] else "")
            for r in rs
        ]
        ax.set_yticks(y, labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Observed canonical discoveries in component ∪ direct sets")
        examples.set_ylim(ax.get_ylim())
        examples.set_xlim(0, 1)
        examples.axis("off")
        priority = {
            name: n
            for n, name in enumerate(
                (
                    "interaction_pattern_candidate",
                    "constituent_omission",
                    "direct_only_unresolved",
                    "direct_only_inapplicable",
                    "shared_discovery",
                    "component_only_discovery",
                )
            )
        }
        for n, r in enumerate(rs):
            candidates = [
                x
                for x in analysis["composition_features"]
                if x["item_id"] == r["item_id"]
                and x["prompt_family"] == r["prompt_family"]
                and x["sampling_mode"] == "equal_calls"
            ]
            if candidates:
                f = min(
                    candidates,
                    key=lambda x: (
                        priority[x["category"]],
                        len(x["proposition"]),
                        x["proposition"],
                    ),
                )
                displayed.append(f)
                examples.text(
                    0,
                    n,
                    textwrap.fill(
                        f"T/S/TS = {f['T']}/{f['S']}/{f['TS']} · " + f["proposition"],
                        76,
                    ),
                    va="center",
                    fontsize=8.5,
                )
        examples.set_title(
            "Example propositions and constituent judgments", fontsize=11, pad=14
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No situated task in this fixture",
            ha="center",
            transform=ax.transAxes,
        )
        examples.axis("off")
    fig.suptitle(
        banner + "Figure C · Component versus direct task–situation discovery",
        fontsize=16,
        y=0.99,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.955),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.14, right=0.99, top=0.86, bottom=0.09, wspace=0.07)
    finish(
        fig,
        "figure_C_compositionality",
        "Equal planned generation calls per prompt family; † marks missing samples. Shared discovery is not validated applicability. 0/0/1 patterns are unconfirmed candidates; novelty alone is not relational emergence.",
    )
    csv_rows(out / "figure_C_display_examples.csv", displayed)

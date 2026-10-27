"""Inspectable Plotly views and deterministic Matplotlib research figures."""

from __future__ import annotations
import html
import math
import textwrap
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go

from .load import SYNTHETIC_BANNER
from .derive import (
    atlas,
    matrix_data,
    composition_rows,
    discovery_rows,
    human_sources,
    human_geometry,
    geometry_data,
    embedding_data,
    compare_rdms,
    item_label,
)

COLORS = dict(
    poem="#3267A2",
    finance="#147F77",
    shared="#52616A",
    increment="#D76B31",
    candidate="#8061A0",
    P0="#3267A2",
    P1="#147F77",
    P2="#D76B31",
)
STATE_COLORS = ["#CCD4DB", "#F8DC99", "#F2F3F4", "#147F77"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "feature-space-v1",
        "pdf.fonttype": 42,
    }
)


def _layout(fig, title):
    return fig.update_layout(
        title=title,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", color="#1F303A"),
        margin=dict(l=20, r=20, t=58, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
    )


def plotly_matrix(run, data):
    custom = [
        [
            [
                i,
                f,
                run.features[f]["text"],
                state,
                run.judgments.get((i, f), {}).get("evidence", "No resolved judgment"),
            ]
            for f, state in zip(data["feature_ids"], states)
        ]
        for i, states in zip(data["item_ids"], data["states"])
    ]
    scale = []
    for index, color in enumerate(STATE_COLORS):
        scale.extend([(index / 4, color), ((index + 1) / 4, color)])
    fig = go.Figure(
        go.Heatmap(
            z=data["values"],
            zmin=-2.5,
            zmax=1.5,
            x=data["feature_ids"],
            y=data["item_labels"],
            customdata=custom,
            text=data["states"],
            texttemplate="%{text}",
            colorscale=scale,
            hovertemplate="<b>%{customdata[2]}</b><br>Item: %{customdata[0]}<br>Feature: %{customdata[1]}<br>Judgment: %{customdata[3]}<br>%{customdata[4]}<extra></extra>",
            colorbar=dict(
                tickvals=[-2, -1, 0, 1],
                ticktext=[
                    "N/A: outside scope",
                    "?: unresolved",
                    "0: inapplicable",
                    "1: applicable",
                ],
            ),
        )
    )
    fig.update_xaxes(
        tickvals=data["feature_ids"],
        ticktext=[
            "<br>".join(textwrap.wrap(html.escape(run.features[f]["text"]), 24))
            for f in data["feature_ids"]
        ],
        tickangle=-45,
    )
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Which judged properties co-occur?").update_layout(
        height=max(580, 24 * len(data["item_ids"]) + 280)
    )


def plotly_composition(rows):
    labels = [
        ("Poem" if r["task_id"] == "poem" else "Reconciliation")
        + " · "
        + r["situation_id"].replace("S_", "").replace("both", "deadline + review")
        for r in rows
    ]
    fig = go.Figure()
    for label, key, color in [
        ("Component-applicable", "component_applicable", COLORS["shared"]),
        ("Incremental applicable", "incremental_applicable", COLORS["increment"]),
        (
            "Interaction candidates (subset)",
            "interaction_candidates",
            COLORS["candidate"],
        ),
    ]:
        fig.add_trace(
            go.Bar(
                name=label,
                y=labels,
                x=[len(r[key]) for r in rows],
                orientation="h",
                marker_color=color,
                text=[len(r[key]) for r in rows],
                textposition="outside",
                customdata=[
                    [r["item_id"], r["component_calls"], r["direct_calls"]]
                    for r in rows
                ],
                hovertemplate="%{y}<br>Count: %{x}<br>Item: %{customdata[0]}<br>Component/direct calls: %{customdata[1]}/%{customdata[2]}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(barmode="group", height=560)
    fig.update_yaxes(autorange="reversed")
    return _layout(
        fig, "What does direct composition add? Counts overlap; bars are not additive."
    )


def plotly_discovery(rows):
    fig = go.Figure()
    for group in sorted({r["group"] for r in rows}):
        for family in ["P0", "P1", "P2"]:
            selected = [
                r for r in rows if r["group"] == group and r["prompt_family"] == family
            ]
            if not selected:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[r["call_count"] for r in selected],
                    y=[r["unique"] for r in selected],
                    name=family + " · " + group,
                    mode="lines+markers",
                    line=dict(
                        color=COLORS[family],
                        dash={"P0": "solid", "P1": "dash", "P2": "dot"}[family],
                    ),
                    customdata=[
                        [
                            r["job_id"],
                            r["item_id"],
                            r["replicate"],
                            r["tokens"],
                            r["excluded_proposals"],
                        ]
                        for r in selected
                    ],
                    hovertemplate="Calls: %{x}<br>Unique: %{y}<br>Item: %{customdata[1]}<br>Replicate: %{customdata[2]}<br>Tokens: %{customdata[3]}<br>Excluded proposals: %{customdata[4]}<extra>%{fullData.name}</extra>",
                )
            )
    fig.update_xaxes(title="Independent generation calls")
    fig.update_yaxes(title="Cumulative unique features", rangemode="tozero")
    return _layout(fig, "How quickly does the feature vocabulary accumulate?")


def plotly_rdm(data, labels, title):
    fig = go.Figure(
        go.Heatmap(
            z=data["rdm"],
            x=labels,
            y=labels,
            zmin=data["range"][0],
            zmax=data["range"][1],
            colorscale="Blues",
            colorbar=dict(title=data["metric"]),
            hovertemplate="%{x}<br>%{y}<br>Distance: %{z:.3f}<extra></extra>",
        )
    )
    return _layout(fig, title).update_layout(height=620)


def _frame(run, title, subtitle, *, size=(18, 10.125)):
    fig = plt.figure(figsize=size, facecolor="white")
    fig.text(
        0.035,
        0.975,
        SYNTHETIC_BANNER if run.demo else "FROZEN RUN · CANDIDATE SEMANTIC PROPERTIES",
        fontsize=9,
        color="#A54027" if run.demo else COLORS["shared"],
        weight="bold",
        va="top",
    )
    fig.text(0.035, 0.927, title, fontsize=22, weight="bold", color="#1F303A", va="top")
    fig.text(0.035, 0.875, subtitle, fontsize=10, color="#52616A", va="top")
    fig.text(
        0.035,
        0.025,
        f"Source: {run.manifest['run_id']} | manifest SHA-256 {run.manifest_hash[:20]} | 1/0 are independent judgments; no claim about internal model use",
        fontsize=7.2,
        color="#52616A",
    )
    return fig


def _ranked_atlas(data, limit):
    groups = ["shared", "left_only", "right_only"]
    selected = []
    positions = {g: 0 for g in groups}
    ordered = {g: sorted(data[g], key=lambda r: r["feature_id"]) for g in groups}
    while len(selected) < limit:
        added = False
        for group in groups:
            if positions[group] < len(ordered[group]) and len(selected) < limit:
                selected.append((group, ordered[group][positions[group]]))
                positions[group] += 1
                added = True
        if not added:
            break
    return selected


def static_atlas(run, left="poem_T", right="finance_T", *, max_features=24):
    if left not in run.items or right not in run.items:
        return (
            None,
            [],
            "Task comparison unavailable: these task targets are not present in this human-only run.",
        )
    data = atlas(run, left, right)
    selected = _ranked_atlas(data, max_features)
    fig = _frame(
        run,
        "What characterizes these two tasks?",
        f"{item_label(run.items[left])} versus {item_label(run.items[right])} · {len(selected)} displayed applicable propositions / {len(run.features)} frozen bank features",
    )
    for n, (group, title, color) in enumerate(
        [
            ("left_only", "Creative writing only", COLORS["poem"]),
            ("shared", "Applies to both", COLORS["shared"]),
            ("right_only", "Reconciliation only", COLORS["finance"]),
        ]
    ):
        ax = fig.add_axes([0.035 + n * 0.322, 0.12, 0.295, 0.69])
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        rows = [r for g, r in selected if g == group]
        ax.text(
            0,
            0.98,
            f"{title} ({len(rows)}/{len(data[group])})",
            color=color,
            weight="bold",
            fontsize=14,
            va="top",
        )
        ax.plot([0, 1], [0.91, 0.91], color=color, lw=2)
        y = 0.85
        for r in rows:
            text = textwrap.fill(r["text"], 42)
            ax.text(
                0,
                y,
                "• " + text,
                va="top",
                fontsize=11.2,
                color="#1F303A",
                linespacing=1.5,
            )
            y -= max(0.10, (text.count("\n") + 1) * 0.043 + 0.045)
    caption = "Applicable features are selected round-robin from shared, poem-only and reconciliation-only groups, then by canonical ID, up to the stated limit. No padding or outcome-based manual cherry-picking. Full 0/1/?/N/A cells and IDs are in the data export."
    fig.text(0.035, 0.065, textwrap.fill(caption, 170), fontsize=8, color="#52616A")
    rows = [
        dict(group=g, **r)
        for g in [
            "shared",
            "left_only",
            "right_only",
            "neither",
            "unresolved",
            "out_of_scope",
        ]
        for r in data[g]
    ]
    return fig, rows, caption


def static_composition(run):
    rows = composition_rows(run)
    if not rows:
        return None, [], "No situated task composition records in this run."
    fig = _frame(
        run,
        "What changes when task and situation are combined?",
        "Six matched-call comparisons. Interaction candidates are a subset of incremental applicable features; do not add these bars.",
    )
    grid = fig.add_gridspec(
        2, 3, left=0.13, right=0.96, top=0.79, bottom=0.24, wspace=0.56, hspace=0.85
    )
    max_count = max([len(r["component_applicable"]) for r in rows] + [1]) + 2
    for row in rows:
        col = {"S_deadline": 0, "S_review": 1, "S_both": 2}[row["situation_id"]]
        line = 0 if row["task_id"] == "poem" else 1
        ax = fig.add_subplot(grid[line, col])
        keys = [
            "component_applicable",
            "incremental_applicable",
            "interaction_candidates",
        ]
        counts = [len(row[k]) for k in keys]
        ax.barh(
            [2, 1, 0],
            counts,
            color=[COLORS["shared"], COLORS["increment"], COLORS["candidate"]],
            height=0.58,
        )
        ax.set_yticks(
            [2, 1, 0],
            [
                "Component applicable",
                "Incremental applicable",
                "Interaction candidates",
            ],
            fontsize=8,
        )
        ax.set_xlim(0, max_count)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="x", alpha=0.14)
        ax.set_axisbelow(True)
        for y, v in zip([2, 1, 0], counts):
            ax.text(v + 0.09, y, str(v), va="center", fontsize=10, weight="bold")
        label = (
            ("Creative writing" if row["task_id"] == "poem" else "Reconciliation")
            + " · "
            + {
                "S_deadline": "Deadline",
                "S_review": "Review",
                "S_both": "Deadline + review",
            }[row["situation_id"]]
        )
        ax.set_title(label, fontsize=11, loc="left", weight="bold", pad=15)
        states = row["interaction_status"]
        confirmed = sum(v == "confirmed" for v in states.values())
        rejected = sum(v == "rejected" for v in states.values())
        pending = sum(v == "unreviewed" for v in states.values())
        note = f"Calls {row['component_calls']} component / {row['direct_calls']} direct\nReview: {confirmed} confirmed · {rejected} rejected · {pending} unreviewed"
        if row["examples"]:
            note += "\nExample: " + textwrap.fill(row["examples"][0]["text"], 46)
        ax.text(
            0,
            -0.29,
            note,
            transform=ax.transAxes,
            fontsize=7.2,
            color="#52616A",
            va="top",
            linespacing=1.35,
        )
    caption = "B = component-discovered features applicable to TS; D = directly discovered features applicable to TS; gain = D minus B. Candidates additionally require T=0, S=0, TS=1, and named review for confirmation. Call budgets match; tokens may differ. Deadline + review is one combined situation, not a factorial effect."
    fig.text(0.035, 0.065, textwrap.fill(caption, 178), fontsize=7.8, color="#52616A")
    return fig, rows, caption


def static_matrix(run, *, max_features=24, cluster=False):
    data = matrix_data(run, cluster=cluster, present_only=True)
    if len(data["feature_ids"]) > max_features:
        data = matrix_data(
            run, data["item_ids"], data["feature_ids"][:max_features], cluster=cluster
        )
    if not data["item_ids"] or not data["feature_ids"]:
        return None, [], "No resolved applicable features in the selected target scope."
    human_only = not run.task_item_ids
    fig = _frame(
        run,
        (
            "Which properties are shared across human-norm targets?"
            if human_only
            else "Which properties are shared across the task hierarchy?"
        ),
        f"{data['n_items']} displayed targets · {data['n_features']} selected features / {data['n_bank']} bank · {data['n_resolved']}/{data['n_items']*data['n_features']} resolved cells · {data['ordering']}",
        size=(20, max(11.25, 4 + 0.32 * data["n_items"])),
    )
    ax = fig.add_axes([0.205, 0.26, 0.75, 0.53])
    ax.imshow(
        data["values"],
        cmap=ListedColormap(STATE_COLORS),
        vmin=-2.5,
        vmax=1.5,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_yticks(range(len(data["item_ids"])), data["item_labels"], fontsize=9)
    ax.set_xticks(
        range(len(data["feature_ids"])),
        [textwrap.fill(x, 40) for x in data["feature_labels"]],
        fontsize=7.6,
        rotation=90,
        ha="right",
        va="center",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="both", length=0, pad=8)
    for i, row in enumerate(data["states"]):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value == "1" else "#263641",
            )
    for i in range(1, len(data["item_ids"])):
        prev = run.items[data["item_ids"][i - 1]].get("task_id")
        now = run.items[data["item_ids"][i]].get("task_id")
        if prev != now:
            ax.axhline(i - 0.5, color="white", lw=3)
    caption = "1 = judged applicable; 0 = judged inapplicable; ? = unresolved/missing; N/A = out of scope. Default columns have at least one applicable displayed cell and are ordered by canonical text (or declared Jaccard/average clustering), capped deterministically. Blank discovery is not judged absence. All full feature texts and IDs are exported."
    fig.text(0.035, 0.065, textwrap.fill(caption, 195), fontsize=8, color="#52616A")
    rows = [
        dict(
            item_id=i,
            feature_id=f,
            text=run.features[f]["text"],
            state=data["states"][n][j],
        )
        for n, i in enumerate(data["item_ids"])
        for j, f in enumerate(data["feature_ids"])
    ]
    return fig, rows, caption


def _static_coverage(run, rows, *, title, subtitle, limit=None):
    selected = rows[:limit] if limit else rows
    fig = _frame(run, title, subtitle, size=(20, max(11.25, 5 + 0.32 * len(selected))))
    ax = fig.add_axes([0.39, 0.19, 0.56, 0.61])
    y = list(range(len(selected)))
    left = [0] * len(selected)
    for key, label, color in [
        ("applies", "Applicable", "#167D7F"),
        ("inapplicable", "Inapplicable", "#CBD9DB"),
        ("ambiguous", "Ambiguous", "#E7AF53"),
        ("missing", "Missing/failed", "#CA706A"),
    ]:
        values = [row[key] for row in selected]
        ax.barh(y, values, left=left, label=label, color=color, height=0.78)
        left = [a + b for a, b in zip(left, values)]
    ax.set_yticks(y, [textwrap.fill(row["text"], 60) for row in selected], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Targets" if "feature_id" in selected[0] else "Frozen bank features")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False)
    ax.grid(axis="x", alpha=0.15)
    caption = "Independent judgments from the frozen bank. Applicable and inapplicable are resolved 1/0 cells; ambiguous and missing/failed remain unknown. Counts include the unresolved denominator and do not estimate human production frequency. Full counts, IDs and labels are exported."
    fig.text(0.035, 0.065, textwrap.fill(caption, 190), fontsize=8, color="#52616A")
    return fig, rows, caption


def static_judgment_coverage(run):
    from .derive import coverage_rows

    rows = coverage_rows(run)
    return _static_coverage(
        run,
        rows,
        title="Which targets still need judgment review?",
        subtitle=f"{len(rows)} human targets · {len(run.features)} frozen features · ambiguous and missing judgments remain separate",
    )


def static_feature_applicability(run, *, max_features=24):
    from .derive import coverage_rows

    rows = sorted(
        coverage_rows(run, by="feature"),
        key=lambda row: (-row["applies"], row["text"], row["feature_id"]),
    )
    return _static_coverage(
        run,
        rows,
        title="Which selected properties apply across the most targets?",
        subtitle=f"Top {min(max_features,len(rows))} / {len(rows)} frozen features, ranked for exploratory inspection by known applicable count; all unresolved target judgments remain visible",
        limit=max_features,
    )


def static_discovery(run):
    rows = discovery_rows(run, representation="canonical", facet="level")
    groups = sorted({r["group"] for r in rows})
    if not rows:
        return (
            None,
            [],
            "No successful generation provenance is available for discovery curves.",
        )
    fig = _frame(
        run,
        "Where does the feature vocabulary come from?",
        "Cumulative selected canonical features, pooled within target level. Overlapping curves indicate identical yields; excluded proposals are retained separately.",
    )
    columns = min(4, max(1, len(groups)))
    grid = fig.add_gridspec(
        math.ceil(len(groups) / columns),
        columns,
        left=0.065,
        right=0.96,
        top=0.78,
        bottom=0.16,
        hspace=0.65,
        wspace=0.38,
    )
    for n, group in enumerate(groups):
        ax = fig.add_subplot(grid[n // columns, n % columns])
        for family in ["P0", "P1", "P2"]:
            rs = [
                r for r in rows if r["group"] == group and r["prompt_family"] == family
            ]
            if rs:
                ax.plot(
                    [r["call_count"] for r in rs],
                    [r["unique"] for r in rs],
                    label=family,
                    color=COLORS[family],
                    linestyle={"P0": "-", "P1": "--", "P2": ":"}[family],
                    marker={"P0": "o", "P1": "s", "P2": "^"}[family],
                    markersize=3,
                    markevery=max(1, len(rs) // 5),
                    lw=1.8,
                )
        subset = [r for r in rows if r["group"] == group]
        count = len({r["item_id"] for r in subset})
        ax.set_title(
            group.replace("_", " ").title() + f" · {count} items",
            fontsize=10,
            loc="left",
            weight="bold",
        )
        ax.set_xlabel("Independent calls", fontsize=8)
        ax.set_ylabel("Unique selected features", fontsize=8)
        ax.set_ylim(0, max([r["unique"] for r in subset] + [1]) + 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7, loc="lower right")
    caption = "Each line pools the union of actual normalized/canonical proposals in deterministic replicate, frozen-item and job-ID order; per-item unique counts are never summed. P0/P1/P2 cap and call denominators, tokens, and selected/excluded status are in the data export. No unseen proposal is labeled invalid."
    fig.text(0.035, 0.065, textwrap.fill(caption, 178), fontsize=8, color="#52616A")
    return fig, rows, caption


def static_human(run):
    sources = human_sources(run)
    ready = [
        s
        for s in sources
        if s["available"]
        and (s.get("recovery") or (s["categories_available"] and s.get("profiles")))
    ]
    if not ready:
        return (
            None,
            sources,
            "Awaiting source-specific verified human feature data and aligned language; synthetic reference values are not scientific results.",
        )
    fig = _frame(
        run,
        "How do generated features compare with human production?",
        "Sources remain separate. Recall uses feature lists; category shares require independently reviewed coding.",
        size=(18, max(10.125, 5 * len(ready))),
    )
    axes = fig.subplots(len(ready), 2, squeeze=False)
    fig.subplots_adjust(
        left=0.09, right=0.96, top=0.77, bottom=0.22, wspace=0.4, hspace=0.65
    )
    data = []
    for (ax, category_ax), source in zip(axes, ready):
        for n, family in enumerate(["P0", "P1", "P2"]):
            values = [
                r["recall_relative_to_lists"]
                for r in source["recovery"]
                if r["prompt_family"] == family
                and r.get("recall_relative_to_lists") is not None
            ]
            ax.scatter(
                [n] * len(values), values, color=COLORS[family], alpha=0.65, s=20
            )
        ax.set_xticks([0, 1, 2], ["P0", "P1", "P2"])
        ax.set_ylim(0, 1.05)
        ax.set_title(source["source_dataset"])
        ax.set_ylabel("Lexical / frozen-mapping recall relative to human lists")
        if not source["recovery"]:
            ax.text(
                0.5,
                0.5,
                "Feature-list recovery unavailable",
                transform=ax.transAxes,
                ha="center",
            )
        profiles = source.get("profiles", []) if source["categories_available"] else []
        if profiles:
            profile = sorted(profiles, key=lambda p: p["item_id"])[0]
            categories = sorted(profile["human_profile"])
            series = [("Human", profile["human_profile"], COLORS["shared"])] + [
                (family, values, COLORS[family])
                for family, values in sorted(
                    profile.get("generated_profiles", {}).items()
                )
                if values.get("available") is not False
            ]
            width = 0.8 / len(series)
            for n, (label, values, color) in enumerate(series):
                category_ax.bar(
                    [
                        k + (n - (len(series) - 1) / 2) * width
                        for k in range(len(categories))
                    ],
                    [values.get(k, 0) for k in categories],
                    width=width,
                    color=color,
                    label=label,
                )
            category_ax.set_xticks(
                range(len(categories)),
                [textwrap.fill(k, 16) for k in categories],
                fontsize=8,
            )
            category_ax.set_ylim(0, 1.05)
            category_ax.set_title(f"Reviewed category profile: {profile['item_id']}")
            category_ax.set_ylabel("Share among coded memberships")
            category_ax.legend(fontsize=8)
            coverage = "; ".join(
                f"{family}: {v['coded_applicable_features']}/{v['applicable_features']} coded"
                for family, v in sorted(profile.get("coding_coverage", {}).items())
            )
            category_ax.text(
                0,
                -0.32,
                textwrap.fill(coverage, 75),
                transform=category_ax.transAxes,
                fontsize=8,
            )
        else:
            category_ax.axis("off")
            category_ax.text(
                0.5,
                0.5,
                "Reviewed category profiles unavailable",
                ha="center",
                transform=category_ax.transAxes,
            )
        data.append(
            dict(
                source_dataset=source["source_dataset"],
                recovery=source["recovery"],
                profiles=profiles,
            )
        )
    fig.text(
        0.035,
        0.09,
        textwrap.fill(
            "Matching uses frozen reviewed equivalences where supplied, otherwise exact normalized phrase overlap. Zero lexical overlap does not rule out semantic correspondence. No new semantic mappings or category reviews are inferred.",
            175,
        ),
        fontsize=8,
        color="#52616A",
    )
    return (
        fig,
        {"sources": data},
        "Verified real sources and aligned language only. Recall uses frozen reviewed equivalences where supplied, otherwise exact normalized phrase matching; zero lexical overlap does not rule out semantic correspondence. Category profiles require independently coded and reviewed features. Full per-item profiles remain in exported data. Unavailable panels are not zero results.",
    )


def static_geometry(run, *, seed=42):
    sources = [
        s for s in human_sources(run) if s["available"] and len(s.get("items", [])) >= 4
    ]
    sources = [s for s in sources if human_geometry(run, s).get("available")]
    if not sources:
        return (
            None,
            [],
            "No verified aligned human feature subset with sufficient items. Raw embedding vectors are independently required for an embedding RDM; summary correlations are insufficient.",
        )
    groups = []
    for source in sources:
        human = human_geometry(run, source)
        generated = geometry_data(run, human["item_ids"])
        if not generated.get("available"):
            continue
        embedding = embedding_data(run, human["item_ids"])
        panels = [
            ("Human feature Jaccard", human),
            ("Generated feature Jaccard", generated),
        ] + (
            [("Pinned embedding cosine", embedding)]
            if embedding.get("available")
            else []
        )
        comparison = compare_rdms(
            human["rdm"],
            generated["rdm"],
            item_ids=human["item_ids"],
            other_item_ids=generated["item_ids"],
            seed=seed,
        )
        groups.append(
            dict(
                source_dataset=source["source_dataset"],
                panels=[dict(title=t, **d) for t, d in panels],
                comparison=comparison,
            )
        )
    if not groups:
        return (
            None,
            [],
            "Aligned human data exist but generated binary scope is incomplete.",
        )
    height = max(10.125, 7.5 * len(groups))
    fig = _frame(
        run,
        "Do aligned representations organize items similarly?",
        "Separate source-specific rows; one declared item order within each row. Jaccard 0-1; dense embedding cosine 0-2.",
        size=(18, height),
    )
    columns = max(len(g["panels"]) for g in groups)
    grid = fig.add_gridspec(
        len(groups),
        columns,
        left=0.10,
        right=0.95,
        top=0.78,
        bottom=0.19,
        wspace=0.45,
        hspace=0.6,
    )
    for n, group in enumerate(groups):
        for j, data in enumerate(group["panels"]):
            ax = fig.add_subplot(grid[n, j])
            image = ax.imshow(
                data["rdm"], vmin=data["range"][0], vmax=data["range"][1], cmap="Blues"
            )
            ax.set_title(group["source_dataset"] + "\n" + data["title"], fontsize=10)
            fig.colorbar(image, ax=ax, fraction=0.045)
            labels = [item_label(run.items[i]) for i in data["item_ids"]]
            ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
            ax.set_yticks(range(len(labels)), labels, fontsize=6)
    caption = "Source-specific aligned RDMs; no cross-dataset pooling. Upper-triangle Spearman/item bootstrap is descriptive. All source comparisons, exact item orders and metrics are exported. Agreement does not establish internal encoding."
    fig.text(0.035, 0.08, textwrap.fill(caption, 175), fontsize=8, color="#52616A")
    return fig, dict(sources=groups), caption

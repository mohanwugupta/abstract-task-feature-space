"""Local research explorer. Read frozen artifacts; never import or run API execution."""

from __future__ import annotations
import argparse
import csv
import html
import io
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from feature_space_pilot.viz.load import load_run, SYNTHETIC_BANNER
from feature_space_pilot.viz.derive import (
    select_features,
    atlas,
    matrix_data,
    composition_rows,
    discovery_rows,
    contribution_rows,
    ordered_task_ids,
    ordered_matrix_ids,
    coverage_rows,
    item_label,
    human_sources,
    human_geometry,
    geometry_data,
    embedding_data,
    compare_rdms,
)
from feature_space_pilot.viz.figures import (
    plotly_matrix,
    plotly_composition,
    plotly_discovery,
    plotly_rdm,
    COLORS,
)
from feature_space_pilot.io import canonical

PAGES = [
    "Task Atlas",
    "Composition",
    "Matrix",
    "Coverage",
    "Feature Discovery",
    "Human Baseline",
    "Geometry",
    "Feature Inspector",
    "Overview",
]


def _plain_rows(rows):
    return [
        {
            k: canonical(v) if isinstance(v, (dict, list, tuple)) else v
            for k, v in r.items()
        }
        for r in rows
    ]


def _table(rows, label, key):
    with st.expander(label):
        if not rows:
            st.info("No rows in the selected scope.")
            return
        frame = pd.DataFrame(_plain_rows(rows))
        st.dataframe(frame, hide_index=True, width="stretch")
        a, b = st.columns(2)
        a.download_button(
            "Download CSV",
            frame.to_csv(index=False).encode(),
            file_name=key + ".csv",
            mime="text/csv",
            key=key + "_csv",
        )
        b.download_button(
            "Download JSON",
            canonical(rows).encode(),
            file_name=key + ".json",
            mime="application/json",
            key=key + "_json",
        )


def _inspector(run, feature_ids, page):
    ids = list(feature_ids)
    if not ids:
        st.info("No selected bank features to inspect.")
        return
    chosen = st.session_state.get("chosen_feature")
    index = ids.index(chosen) if chosen in ids else 0
    fid = st.selectbox(
        "Inspect a feature and its evidence",
        ids,
        index=index,
        format_func=lambda f: run.features[f]["text"] + " [" + f + "]",
        key="inspector_" + page,
    )
    st.session_state["chosen_feature"] = fid
    detail = run.inspect(fid)
    st.subheader("The proposition")
    st.text(detail["text"])
    st.code(fid, language=None)
    tabs = st.tabs(
        [
            "Applicability & evidence",
            "Discovery provenance",
            "Target descriptions",
            "Merge & manual review",
        ]
    )
    with tabs[0]:
        _table(
            detail["judgments"],
            "View every 1 / 0 / ? / N/A cell and its evidence",
            "judgments_" + page,
        )
    with tabs[1]:
        st.caption(
            "These rows show where wording was generated. They do not determine whether a feature applies elsewhere."
        )
        _table(
            detail["origins"],
            "View source prompt / replicate / raw wording",
            "origins_" + page,
        )
        if detail["origins"]:
            jobs = sorted({r["job_id"] for r in detail["origins"]})
            job = st.selectbox(
                "Inspect the exact generation request", jobs, key="job_" + page
            )
            record = run.jobs[job]
            st.code(record["system"], language=None)
            st.code(record["user"], language="json")
    with tabs[2]:
        targets = sorted({r["item_id"] for r in detail["judgments"]})
        selected = st.selectbox(
            "Full target text",
            targets,
            format_func=lambda i: item_label(run.items[i]),
            key="target_" + page,
        )
        st.text(run.items[selected]["text"])
    with tabs[3]:
        st.json(
            dict(
                merge_decisions=detail["merge_decisions"],
                manual_reviews=detail["manual_reviews"],
            )
        )
        st.caption(
            "Unreviewed candidates are hypotheses. A named independent reviewer is required for confirmation."
        )


def _feature_column(run, rows, title, color, key):
    st.markdown(
        f'<div class="column-heading" style="border-color:{color}">{html.escape(title)} <span>{len(rows)}</span></div>',
        unsafe_allow_html=True,
    )
    if not rows:
        st.caption("No resolved applicable feature in this group.")
    for row in rows:
        with st.container(border=True):
            st.text(row["text"])
            if st.button(
                "Inspect evidence ↗",
                key=key + "_" + row["feature_id"],
                help=row["feature_id"],
            ):
                st.session_state["chosen_feature"] = row["feature_id"]
                st.session_state["inspector_atlas"] = row["feature_id"]
            st.caption("Judged " + row["left_state"] + " / " + row["right_state"])


def _atlas_page(run, features):
    st.subheader("What characterizes these two tasks?")
    options = {
        "Domain": ("_domain", "_domain"),
        "Task type": ("_type", "_type"),
        "Concrete instance": ("_T", "_T"),
        "Under deadline": ("_S_deadline", "_S_deadline"),
        "Under review": ("_S_review", "_S_review"),
        "Deadline + review": ("_S_both", "_S_both"),
    }
    level = st.selectbox(
        "Abstraction / situation", list(options), index=2, key="atlas_level"
    )
    left = "poem" + options[level][0]
    right = "finance" + options[level][1]
    if left not in run.items or right not in run.items:
        st.info("The selected pair is not present in this run.")
        return
    data = atlas(run, left, right, features)
    st.caption(
        f"{data['n_selected']} selected / {data['n_bank']} frozen features. Groups use independent applicability judgments at the same level. Origin filters change the vocabulary shown, not cell meanings."
    )
    if data["unresolved"] or data["out_of_scope"]:
        st.warning(
            "Selected pair has unresolved or out-of-scope cells. Those features are listed separately and never assigned zero."
        )
    a, b, c = st.columns([1, 0.95, 1], gap="large")
    with a:
        _feature_column(
            run, data["left_only"], "Creative writing only", COLORS["poem"], "left"
        )
    with b:
        _feature_column(
            run, data["shared"], "Applies to both", COLORS["shared"], "shared"
        )
    with c:
        _feature_column(
            run, data["right_only"], "Reconciliation only", COLORS["finance"], "right"
        )
    with st.expander("Read the full paired targets"):
        a, b = st.columns(2)
        with a:
            st.text(run.items[left]["text"])
        with b:
            st.text(run.items[right]["text"])
    with st.expander("Explore a standalone situation (independent axis)"):
        situations = [i for i in ["S_deadline", "S_review", "S_both"] if i in run.items]
        if situations:
            sid = st.selectbox(
                "Situation",
                situations,
                format_func=lambda i: item_label(run.items[i]),
                key="standalone",
            )
            st.text(run.items[sid]["text"])
            applicable = [
                dict(feature_id=f, text=run.features[f]["text"], state="1")
                for f in features
                if run.cell(sid, f) == 1
            ]
            _table(
                applicable,
                "View independently applicable situation features",
                "standalone_features",
            )
            if sid == "S_both":
                st.caption(
                    "One combined deadline + review situation. This is not a factorial statistical interaction."
                )
    rows = [
        dict(category=g, **r)
        for g in [
            "left_only",
            "shared",
            "right_only",
            "neither",
            "unresolved",
            "out_of_scope",
        ]
        for r in data[g]
    ]
    _table(
        rows,
        "View underlying atlas rows, including neither / unresolved / out of scope",
        "atlas_rows",
    )
    _inspector(run, features, "atlas")


def _composition_page(run, features):
    st.subheader("What does direct composition add?")
    try:
        rows = composition_rows(run)
    except ValueError as e:
        st.error(f"Composition plot blocked: {e}")
        return
    if not rows:
        st.info("No situated-task composition audit in this run.")
        return
    st.caption(
        "All six comparisons use the original frozen audit and independently recomputed identities. Bars overlap: interaction candidates are a subset of applicable gain. Call counts match; token counts may differ. Shared vocabulary filters apply to the inspector, not the frozen primary comparison counts."
    )
    st.plotly_chart(plotly_composition(rows), width="stretch", key="composition_plot")
    iid = st.selectbox(
        "Inspect a task + situation",
        [r["item_id"] for r in rows],
        format_func=lambda i: item_label(run.items[i]),
        key="composition_target",
    )
    record = next(r for r in rows if r["item_id"] == iid)
    st.caption(
        f"{record['component_calls']} component calls / {record['direct_calls']} direct calls · {record.get('component_tokens')} / {record.get('direct_tokens')} tokens · gain per 1,000 direct tokens: {record.get('gain_per_1000_direct_tokens')}"
    )
    st.info(record["situation_interpretation"])
    _table(
        record["examples"],
        "Inspect exact incremental propositions, review state and T / S / TS evidence",
        "composition_evidence",
    )
    _table(
        rows,
        "View every frozen count, set, allocation and denominator",
        "composition_data",
    )
    st.markdown(
        "**Definitions:** B = component-discovered, TS-applicable features. D = directly discovered, TS-applicable features. Gain = D minus B. Interaction candidates also require T=0 and S=0; independent named review is required for confirmation."
    )
    _inspector(run, features, "composition")


def _matrix_page(run, features):
    st.subheader("Which judged properties co-occur?")
    human_only = not run.task_item_ids
    family = st.selectbox(
        "Displayed target family",
        (
            ["All human targets", "Concepts", "Events", "Objects"]
            if human_only
            else [
                "All task + situation targets",
                "Creative writing",
                "Reconciliation",
                "Standalone situations",
            ]
        ),
        key="matrix_family",
    )
    items = ordered_matrix_ids(run)
    task = {
        "Creative writing": "poem",
        "Reconciliation": "finance",
        "Standalone situations": None,
    }.get(family, "all")
    if human_only:
        target_type = {
            "Concepts": "concept",
            "Events": "event",
            "Objects": "object",
        }.get(family)
        if target_type:
            items = [i for i in items if run.items[i]["target_type"] == target_type]
    elif task != "all":
        items = [i for i in items if run.items[i].get("task_id") == task]
    present = st.checkbox(
        "Only features applicable to at least one displayed target",
        value=True,
        key="matrix_present",
    )
    cluster = st.checkbox(
        "Cluster feature columns (binary Jaccard / average linkage)", key="cluster"
    )
    data = matrix_data(run, items, features, cluster=cluster, present_only=present)
    if not data["clustering_available"]:
        st.warning(
            "Incomplete cells: clustering is disabled. ? and N/A remain distinct from 0."
        )
    if not data["feature_ids"] or not items:
        st.info("No cells match the selected vocabulary and target filters.")
        return
    st.caption(
        f"{data['n_items']} items × {data['n_features']} shown features / {data['n_bank']} bank · {data['n_resolved']} resolved selected cells · {data['ordering']}. 1 applicable, 0 inapplicable, ? unresolved, N/A outside scope."
    )
    st.plotly_chart(plotly_matrix(run, data), width="stretch", key="matrix_plot")
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
    _table(rows, "View underlying matrix cells", "matrix_data")
    _inspector(run, data["feature_ids"], "matrix")


def _coverage_page(run):
    st.subheader("Which judgments still need review?")
    rows = coverage_rows(run)
    totals = {
        key: sum(row[key] for row in rows)
        for key in ["applies", "inapplicable", "ambiguous", "missing"]
    }
    columns = st.columns(4)
    for column, label, value in zip(
        columns,
        ["Applicable", "Inapplicable", "Ambiguous", "Missing/failed"],
        totals.values(),
    ):
        column.metric(label, value)
    fig = go.Figure()
    for key, label, color in [
        ("applies", "Applicable", "#167D7F"),
        ("inapplicable", "Inapplicable", "#CBD9DB"),
        ("ambiguous", "Ambiguous", "#E7AF53"),
        ("missing", "Missing/failed", "#CA706A"),
    ]:
        fig.add_trace(
            go.Bar(
                name=label,
                x=[row[key] for row in rows],
                y=[row["text"] for row in rows],
                orientation="h",
                marker_color=color,
                customdata=[row["item_id"] for row in rows],
                hovertemplate="%{y}: %{x}<br>%{customdata}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=max(500, 24 * len(rows)),
        xaxis_title="Frozen bank features",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch", key="coverage_plot")
    st.caption(
        "Ambiguous and missing judgments remain unresolved; none are filled with zero. Coverage is across the full frozen bank, independent of discovery filters."
    )
    _table(rows, "Per-target coverage and unresolved counts", "coverage_rows")


def _discovery_page(run, features):
    st.subheader("Where does the feature vocabulary come from?")
    labels = {
        "Raw generated phrases": "raw",
        "Selected canonical features": "canonical",
        "Independently judged-applicable selected features": "applicable",
    }
    label = st.selectbox(
        "Representation", list(labels), index=1, key="discovery_representation"
    )
    level = st.selectbox(
        "Target level",
        ["All levels"] + sorted({i["level"] for i in run.items.values()}),
        key="discovery_level",
    )
    items = (
        list(run.items)
        if level == "All levels"
        else [i for i, x in run.items.items() if x["level"] == level]
    )
    rows = discovery_rows(
        run, representation=labels[label], item_ids=items, facet="level"
    )
    st.caption(
        "Curves pool actual records in replicate / frozen item / job order. They do not sum overlapping per-item unique counts. Excluded bank candidates remain proposals, never judged zero. Missing applicability is explicitly unavailable. Global feature search does not alter these full-vocabulary denominator curves; use the target-level control."
    )
    st.plotly_chart(plotly_discovery(rows), width="stretch", key="discovery_plot")
    _table(
        rows,
        "View pooled calls, caps, tokens, counts and availability",
        "discovery_data",
    )
    by = st.radio(
        "Contribution grouping",
        ["Prompt family", "Abstraction level"],
        horizontal=True,
        key="contribution_group",
    )
    contribution = contribution_rows(
        run, by="prompt_family" if by == "Prompt family" else "level", item_ids=items
    )
    _table(
        contribution,
        "Compare unique-to-group, shared, selected/excluded vocabulary and cross-item applicability",
        "contribution_data",
    )
    raw = [
        dict(
            feature_id=r["feature_id"],
            canonical_text=run.features.get(r["feature_id"], {}).get("text"),
            raw_phrase=r["phrase"],
            selected=r["selected"],
            item_id=r["origin_item"],
            prompt_family=r["prompt_family"],
            replicate=r["replicate"],
            job_id=r["job_id"],
        )
        for r in run.candidates
        if r["origin_item"] in items
    ]
    _table(
        raw,
        "Inspect all raw candidate origins, including excluded proposals",
        "raw_candidates",
    )
    _inspector(run, features, "discovery")


def _human_page(run):
    st.subheader("What correspondence to human production is supported?")
    st.caption(
        "Human production is a reference observation, not truth. Recovery uses frozen reviewed equivalences if supplied, otherwise exact normalized phrase overlap. Zero lexical overlap does not rule out semantic correspondence. Sources, languages and coding vocabularies remain separate."
    )
    for source in human_sources(run):
        st.markdown("**" + source["source_dataset"] + "**")
        if not source["available"]:
            st.info(source["reason"])
            continue
        st.caption(
            "Verified source and aligned items/language. Source feature/participant availability is respected."
        )
        recovery = source["recovery"]
        fig = go.Figure()
        for family in ["P0", "P1", "P2"]:
            rows = [
                r
                for r in recovery
                if r["prompt_family"] == family
                and r.get("recall_relative_to_lists") is not None
            ]
            fig.add_trace(
                go.Box(
                    y=[r["recall_relative_to_lists"] for r in rows],
                    name=family,
                    boxpoints="all",
                    marker_color=COLORS[family],
                )
            )
        fig.update_layout(
            title="Lexical / mapped recovery relative to human lists",
            template="plotly_white",
            yaxis_title="List-relative lexical / mapped recall",
        )
        if any(r.get("recall_relative_to_lists") is not None for r in recovery):
            st.plotly_chart(
                fig, width="stretch", key="human_" + source["source_dataset"]
            )
        else:
            st.info(
                "Feature-list recovery unavailable for this source level; no feature strings are reconstructed from category summaries."
            )
        _table(
            recovery,
            "View item-level recovery and matching caps",
            "recovery_" + source["source_dataset"],
        )
        if not source["categories_available"]:
            st.info(
                "Generated category profiles unavailable: independently reviewed source-category coding is required."
            )
        else:
            profiles = source["profiles"]
            if profiles:
                chosen = st.selectbox(
                    "Reviewed category profile item",
                    [p["item_id"] for p in profiles],
                    key="category_item_" + source["source_dataset"],
                )
                profile = next(p for p in profiles if p["item_id"] == chosen)
                categories = sorted(profile["human_profile"])
                category_fig = go.Figure(
                    go.Bar(
                        name="Human production",
                        x=categories,
                        y=[profile["human_profile"].get(k, 0) for k in categories],
                        marker_color=COLORS["shared"],
                    )
                )
                for family, values in profile.get("generated_profiles", {}).items():
                    if values.get("available") is False:
                        continue
                    category_fig.add_trace(
                        go.Bar(
                            name=family + " independently coded subset",
                            x=categories,
                            y=[values.get(k, 0) for k in categories],
                            marker_color=COLORS[family],
                        )
                    )
                category_fig.update_layout(
                    barmode="group",
                    template="plotly_white",
                    title="Source-specific reviewed category composition",
                    yaxis_title="Relative share among coded memberships",
                )
                st.plotly_chart(
                    category_fig,
                    width="stretch",
                    key="categories_" + source["source_dataset"],
                )
                st.json(profile.get("coding_coverage", {}))
            _table(
                profiles,
                "View reviewed source-specific category profiles and coding coverage",
                "profiles_" + source["source_dataset"],
            )
        if not source["participant_available"]:
            st.caption("Human split-half unavailable: no participant-level records.")


def _geometry_page(run, features):
    st.subheader("How are aligned items organized?")
    st.caption(
        "Generated binary features use Jaccard 0–1. Dense embeddings use cosine 0–2. Agreement is descriptive and does not demonstrate internal feature use."
    )
    ids = ordered_matrix_ids(run)
    if ids and features:
        data = geometry_data(run, ids, features)
        if data.get("available"):
            st.plotly_chart(
                plotly_rdm(
                    data,
                    [item_label(run.items[i]) for i in ids],
                    "Exploratory generated-feature geometry",
                ),
                width="stretch",
                key="generated_rdm",
            )
            _table(
                [
                    dict(
                        item_i=i, item_j=j, distance=data["rdm"][n][m], metric="jaccard"
                    )
                    for n, i in enumerate(ids)
                    for m, j in enumerate(ids)
                    if n < m
                ],
                "View exact upper-triangle distances",
                "generated_geometry",
            )
        else:
            st.info(data["reason"])
    sources = human_sources(run)
    for source in sources:
        h = human_geometry(run, source)
        if not h.get("available"):
            st.info(source["source_dataset"] + ": " + h["reason"])
            continue
        g = geometry_data(run, h["item_ids"])
        if not g.get("available"):
            st.info(g["reason"])
            continue
        labels = [item_label(run.items[i]) for i in h["item_ids"]]
        a, b = st.columns(2)
        with a:
            st.plotly_chart(
                plotly_rdm(h, labels, "Verified human Jaccard"),
                width="stretch",
                key="human_rdm_" + source["source_dataset"],
            )
        with b:
            st.plotly_chart(
                plotly_rdm(g, labels, "Generated Jaccard, same item order"),
                width="stretch",
                key="aligned_rdm_" + source["source_dataset"],
            )
        result = compare_rdms(
            h["rdm"], g["rdm"], item_ids=h["item_ids"], other_item_ids=g["item_ids"]
        )
        st.json(result)
    try:
        embedding = embedding_data(run, ids or list(run.items))
    except (ValueError, OSError) as e:
        st.error(f"Embedding panel blocked: {e}")
        return
    if not embedding.get("available"):
        st.info(embedding["reason"])
    else:
        st.caption(
            f"Pinned model: {embedding['model_id']} · revision {embedding['revision']}"
        )
        st.plotly_chart(
            plotly_rdm(
                embedding,
                [item_label(run.items[i]) for i in embedding["item_ids"]],
                "Pinned raw embedding geometry (cosine)",
            ),
            width="stretch",
            key="embedding_rdm",
        )


def _publication_export(run):
    with st.expander("Publication figures and offline gallery"):
        st.caption(
            "Exports the frozen full comparison, independent of sidebar vocabulary filters."
        )
        if st.button("Export publication figures", key="export_publication"):
            from feature_space_pilot.viz.exports import export_run

            try:
                with st.spinner(
                    "Rendering PNG, SVG, PDF, evidence tables and offline gallery…"
                ):
                    manifest = export_run(run.path)
                    output = (
                        run.path.parent.parent
                        / "feature_space_visualizations"
                        / run.manifest["run_id"]
                        / "v1"
                    )
                    names = {
                        "index.html",
                        "feature_inspector.html",
                        "figure_manifest.json",
                        "figures_readme.md",
                        "data_exports/feature_inspector.json",
                        "data_exports/prompt_contributions.json",
                    }
                    for figure in manifest["figures"]:
                        names.add(figure["data_file"])
                        csv_name = str(Path(figure["data_file"]).with_suffix(".csv"))
                        if (output / csv_name).is_file():
                            names.add(csv_name)
                        names.update(
                            figure["basename"] + "." + ext for ext in figure["formats"]
                        )
                    bundle = io.BytesIO()
                    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
                        for name in sorted(names):
                            archive.write(output / name, arcname=name)
                    st.session_state["publication_bundle"] = (
                        run.fingerprint,
                        bundle.getvalue(),
                        str(output),
                    )
            except (ValueError, OSError, KeyError) as error:
                st.error(f"Publication export blocked: {error}")
        saved = st.session_state.get("publication_bundle")
        if saved and saved[0] == run.fingerprint:
            st.success("Publication gallery exported: " + saved[2])
            st.download_button(
                "Download publication bundle (ZIP)",
                saved[1],
                file_name=run.manifest["run_id"] + "_figures.zip",
                mime="application/zip",
                key="publication_download",
            )


def main():
    st.set_page_config(
        page_title="Feature Space · Research Atlas",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """<style>.block-container{padding-top:2.1rem;max-width:1550px}h1{letter-spacing:-1.8px;font-weight:750}.column-heading{font-size:20px;font-weight:700;border-top:3px solid;padding:15px 0;margin-bottom:12px}.column-heading span{font-size:13px;background:#edf0ed;padding:4px 9px;border-radius:20px;margin-left:8px}div[data-testid="stSidebar"]{border-right:1px solid #dce4dc}div[data-testid="stText"]{font-size:15px;line-height:1.5}div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:7px;background:#fff}button[kind="secondary"]{font-size:12px}div[data-testid="stAlert"]{border-radius:6px}</style>""",
        unsafe_allow_html=True,
    )
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir")
    parser.add_argument("--page", choices=PAGES, default="Task Atlas")
    args, _ = parser.parse_known_args()
    initial = args.run_dir or os.environ.get("FEATURE_SPACE_RUN_DIR", "")
    with st.sidebar:
        st.markdown("### FEATURE SPACE")
        st.caption("RESEARCH EXPLORER · V1")
        path = st.text_input("Frozen run directory", value=initial, key="run_path")
        page = st.selectbox("Explore", PAGES, index=PAGES.index(args.page), key="page")
    st.markdown(
        '<p style="font-size:12px;letter-spacing:2px;color:#66796e;font-weight:700">FEATURE SPACE / FROZEN RUN ATLAS</p>',
        unsafe_allow_html=True,
    )
    st.title("Inspect the representation.")
    st.caption(
        "Follow each pattern to an actual proposition, its discovery origin, and the independent evidence."
    )
    if not path:
        st.info(
            "Select one frozen run directory. The explorer never generates or judges features."
        )
        return
    try:
        run = load_run(path)
    except (ValueError, OSError, KeyError) as error:
        st.error(f"Run could not be loaded: {error}")
        return
    if run.demo:
        st.warning(SYNTHETIC_BANNER, icon="⚗️")
    else:
        st.info(
            "Candidate semantic properties — correspondence does not establish internal model use."
        )
    st.caption(
        f"{run.manifest['run_id']} · {run.manifest['stage']} · {run.manifest['status']} · {len(run.items)} items · {len(run.features)} bank features · {len(run.judgments)} / {run.n_in_scope} resolved in-scope cells"
    )
    frozen_models = run.manifest.get("model_snapshots", {})
    st.caption(
        "Frozen generator: "
        + str(frozen_models.get("generator", {}).get("model_id", "unknown"))
        + " / "
        + str(frozen_models.get("generator", {}).get("revision", "unknown"))
        + " · Judge: "
        + str(frozen_models.get("judge", {}).get("model_id", "unknown"))
        + " · Bank hash: "
        + str(run.manifest.get("candidate_inventory_hash", "unknown"))[:20]
    )
    with st.sidebar:
        st.divider()
        st.caption("VOCABULARY FILTERS · DISCOVERY ORIGIN")
        prompts = st.multiselect(
            "Prompt family",
            ["P0", "P1", "P2"],
            default=["P0", "P1", "P2"],
            key="prompts",
        )
        levels = st.multiselect(
            "Origin level",
            sorted({i["level"] for i in run.items.values()}),
            key="origin_levels",
        )
        origins = st.multiselect(
            "Source items",
            list(run.items),
            format_func=lambda i: item_label(run.items[i]),
            key="origin_items",
        )
        minimum = st.number_input(
            "Minimum independent discoveries",
            min_value=0,
            value=0,
            step=1,
            key="minimum_discoveries",
        )
        search = st.text_input("Feature text or ID", key="feature_search")
        include = st.checkbox(
            "Include unreviewed interaction candidates",
            value=True,
            key="include_candidates",
        )
        st.caption(
            "Filters select discovered vocabulary. Applicability always uses the frozen judge cells."
        )
    try:
        features = (
            select_features(
                run,
                prompts=prompts,
                origins=origins,
                levels=levels,
                min_count=minimum,
                search=search,
                include_unreviewed=include,
            )
            if prompts
            else []
        )
    except ValueError as error:
        st.warning(f"Interaction filter unavailable: {error}")
        features = select_features(run, prompts=prompts, search=search)
    st.caption(
        f"Vocabulary selection: {len(features)} / {len(run.features)} bank features. Source filtering does not relabel unmentioned features as zero."
    )
    if page == "Task Atlas":
        _atlas_page(run, features)
    elif page == "Composition":
        _composition_page(run, features)
    elif page == "Matrix":
        _matrix_page(run, features)
    elif page == "Coverage":
        _coverage_page(run)
    elif page == "Feature Discovery":
        _discovery_page(run, features)
    elif page == "Human Baseline":
        _human_page(run)
    elif page == "Geometry":
        _geometry_page(run, features)
    elif page == "Feature Inspector":
        _inspector(run, features, "feature")
    else:
        st.subheader("One run. Every claim traceable.")
        st.json(
            dict(
                run_id=run.manifest["run_id"],
                status=run.manifest["status"],
                demo=run.demo,
                stage=run.manifest["stage"],
                manifest_sha256=run.manifest_hash,
                candidate_bank_hash=run.manifest.get("candidate_inventory_hash"),
                model_snapshots=run.manifest.get("model_snapshots"),
                source_hashes=run.manifest.get("source_hashes"),
            )
        )
        st.caption(
            "The explorer validates the manifest on every interaction. It does not cache stale run data, reseal manifests, read API credentials, or contact models."
        )
        _table(
            [
                dict(
                    item_id=i,
                    display_label=item_label(x),
                    **{
                        k: x.get(k)
                        for k in [
                            "target_type",
                            "parent_id",
                            "task_id",
                            "situation_ids",
                            "source_dataset",
                            "language",
                        ]
                    },
                )
                for i, x in run.items.items()
            ],
            "View frozen target inventory",
            "inventory",
        )
    _publication_export(run)
    st.divider()
    st.caption(
        "Read-only researcher tool. No API calls, analytics, external assets or causal claims. Full proposition text and stable IDs are available in the inspector and data exports."
    )


if __name__ == "__main__":
    main()

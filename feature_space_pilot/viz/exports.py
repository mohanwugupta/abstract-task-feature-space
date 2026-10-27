"""Offline, read-only PNG/SVG/PDF exports and a self-contained figure gallery."""

from __future__ import annotations
from dataclasses import dataclass, asdict
import importlib.metadata
import csv
import html
import json
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt

from . import VIZ_VERSION
from .load import load_run, SYNTHETIC_BANNER
from . import figures
from .derive import composition_rows, contribution_rows, matrix_data
from ..io import canonical, digest, file_hash, write_json, write_csv


@dataclass(frozen=True)
class ExportConfig:
    max_features: int = 24
    dpi: int = 220
    seed: int = 42
    cluster: bool = False


def _tidy(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


def export_run(run_dir, output_dir=None, *, config=None):
    config = config or ExportConfig()
    if not 200 <= config.dpi <= 600 or config.max_features < 1:
        raise ValueError("Exports need DPI 200–600 and positive feature cap")
    run = load_run(run_dir)
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else run.path.parent.parent
        / "feature_space_visualizations"
        / run.manifest["run_id"]
        / VIZ_VERSION
    )
    if (
        output == run.path
        or output.is_relative_to(run.path)
        or run.path.is_relative_to(output)
    ):
        raise ValueError(
            "Visualization output must be outside the read-only source run"
        )
    source_snapshot = {p.name: file_hash(p) for p in run.path.iterdir() if p.is_file()}
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data_exports"
    data_dir.mkdir(exist_ok=True)
    definitions = [
        (
            "01_task_atlas",
            "What characterizes the two concrete tasks?",
            lambda: figures.static_atlas(run, max_features=config.max_features),
        ),
        (
            "01_task_atlas_deadline",
            "What changes under a deadline?",
            lambda: figures.static_atlas(
                run,
                "poem_S_deadline",
                "finance_S_deadline",
                max_features=config.max_features,
            ),
        ),
        (
            "02_compositionality",
            "What does direct composition discover?",
            lambda: figures.static_composition(run),
        ),
        (
            "03_binary_heatmap",
            "Which judged properties co-occur?",
            lambda: figures.static_matrix(
                run, max_features=config.max_features, cluster=config.cluster
            ),
        ),
        (
            "04_discovery",
            "Where does the vocabulary come from?",
            lambda: figures.static_discovery(run),
        ),
        (
            "05_human_baseline",
            "How do features correspond to human production?",
            lambda: figures.static_human(run),
        ),
        (
            "06_geometry",
            "How do aligned representation geometries compare?",
            lambda: figures.static_geometry(run, seed=config.seed),
        ),
    ]
    if not run.task_item_ids:
        definitions += [
            (
                "07_judgment_coverage",
                "Which human targets still need review?",
                lambda: figures.static_judgment_coverage(run),
            ),
            (
                "08_feature_applicability",
                "Which selected features apply across human targets?",
                lambda: figures.static_feature_applicability(
                    run, max_features=config.max_features
                ),
            ),
        ]
    records = []
    prefix = "synthetic_" if run.demo else ""
    for figure_id, title, builder in definitions:
        basename = prefix + "figure_" + figure_id
        try:
            figure, payload, caption = builder()
            reason = None
        except ValueError as error:
            figure, payload, caption = None, [], f"Blocked: {error}"
            reason = str(error)
        data_file = "data_exports/" + basename + ".json"
        write_json(output / data_file, payload)
        rows = _tidy(payload)
        if rows and all(isinstance(r, dict) for r in rows):
            fields = sorted(set().union(*(r.keys() for r in rows)))
            write_csv(
                output / "data_exports" / (basename + ".csv"),
                [{k: r.get(k) for k in fields} for r in rows],
                fields,
            )
        if figure is None:
            status = "blocked" if reason else "unavailable"
            (output / (basename + ".html")).write_text(
                '<!doctype html><meta charset="utf-8"><title>'
                + html.escape(title)
                + "</title><h1>"
                + html.escape(title)
                + "</h1><p>"
                + html.escape(
                    SYNTHETIC_BANNER if run.demo else "Scientific availability check"
                )
                + "</p><p>"
                + html.escape(caption)
                + "</p>"
            )
        else:
            status = "rendered"
            try:
                for extension in ["png", "svg", "pdf"]:
                    metadata = (
                        {"Title": title, "Description": caption}
                        if extension == "svg"
                        else (
                            {
                                "Title": title,
                                "Subject": caption,
                                "CreationDate": None,
                                "ModDate": None,
                            }
                            if extension == "pdf"
                            else {"Description": caption}
                        )
                    )
                    if extension == "svg":
                        metadata["Date"] = None
                    figure.savefig(
                        output / (basename + "." + extension),
                        dpi=config.dpi,
                        metadata=metadata,
                        facecolor="white",
                    )
            finally:
                plt.close(figure)
        records.append(
            dict(
                figure_id=figure_id,
                title=title,
                basename=basename,
                status=status,
                caption=caption,
                data_file=data_file,
                data_sha256=digest(payload),
                example_features=(
                    [
                        dict(item_id=r["item_id"], **e)
                        for r in payload
                        for e in r.get("examples", [])
                    ]
                    if figure_id == "02_compositionality" and isinstance(payload, list)
                    else []
                ),
                n_items=len(run.items),
                n_bank_features=len(run.features),
                n_resolved=len(run.judgments),
                formats=["png", "svg", "pdf"] if status == "rendered" else ["html"],
                reason=reason,
            )
        )
    # Every plotted mark can be followed to canonical text, generation provenance and evidence.
    provenance = [run.inspect(fid) for fid in sorted(run.features)]
    write_json(data_dir / "feature_inspector.json", provenance)
    contributions = contribution_rows(run)
    write_json(data_dir / "prompt_contributions.json", contributions)
    matrix = matrix_data(run, run.scope["item_ids"], run.scope["feature_ids"])
    full_cells = [
        dict(
            item_id=i,
            target_text=run.items[i]["text"],
            feature_id=f,
            feature_text=run.features[f]["text"],
            state=matrix["states"][n][j],
        )
        for n, i in enumerate(matrix["item_ids"])
        for j, f in enumerate(matrix["feature_ids"])
    ]
    write_json(data_dir / "full_matrix_cells.json", full_cells)
    write_csv(
        data_dir / "full_matrix_cells.csv",
        full_cells,
        ["item_id", "target_text", "feature_id", "feature_text", "state"],
    )
    (output / "feature_inspector.html").write_text(_inspector_html(run, provenance))
    manifest = dict(
        viz_version=VIZ_VERSION,
        source_run_id=run.manifest["run_id"],
        source_manifest_sha256=run.manifest_hash,
        evaluation_fingerprint=run.manifest.get("evaluation_inputs_hash"),
        source_fingerprint=run.fingerprint,
        visualization_code_hashes={
            p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        renderer_versions={
            name: importlib.metadata.version(name)
            for name in ["matplotlib", "plotly", "streamlit"]
        },
        demo=run.demo,
        stage=run.manifest["stage"],
        status=run.manifest["status"],
        n_in_scope=run.n_in_scope,
        analysis_config=asdict(config),
        figures=records,
        caveats=[
            "Candidate properties do not establish internal model use",
            "Discovery and applicability remain separate",
            "Overlapping composition categories are not additive",
            "Call budgets match; token budgets may differ",
            "No unlisted human production is deemed false",
            "Missing/unresolved/out-of-scope cells are never zero-filled",
        ],
    )
    write_json(output / "figure_manifest.json", manifest)
    (output / "index.html").write_text(_gallery(run, records, config))
    (output / "figures_readme.md").write_text(
        "# How to read these figures\n\n"
        + (SYNTHETIC_BANNER + "\n\n" if run.demo else "")
        + "\n\n".join(
            f"## {r['title']}\n\nStatus: {r['status']}.\n\n{r['caption']}\n\nData: `{r['data_file']}`. All canonical feature provenance: `feature_inspector.html`."
            for r in records
        )
        + "\n"
    )
    if source_snapshot != {
        p.name: file_hash(p) for p in run.path.iterdir() if p.is_file()
    }:
        raise ValueError("Read-only source integrity changed during export")
    return manifest


def _inspector_html(run, features):
    content = []
    for f in features:
        origins = "".join(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(r.get(k, ""))) + "</td>"
                for k in [
                    "item_id",
                    "prompt_family",
                    "replicate",
                    "raw_phrase",
                    "job_id",
                ]
            )
            + "</tr>"
            for r in f["origins"]
        )
        judgments = "".join(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(r.get(k, ""))) + "</td>"
                for k in ["item_id", "state", "evidence", "judge_model"]
            )
            + "</tr>"
            for r in f["judgments"]
        )
        target_details = "".join(
            "<details><summary>"
            + html.escape(r["item_id"])
            + ' — full target text</summary><pre style="white-space:pre-wrap">'
            + html.escape(r["text"])
            + "</pre></details>"
            for r in f["judgments"]
        )
        content.append(
            f'<section id="{html.escape(f["feature_id"])}"><h2>{html.escape(f["text"])}</h2><code>{html.escape(f["feature_id"])}</code><h3>Original generation provenance</h3><table><tr><th>Item</th><th>Prompt</th><th>Replicate</th><th>Raw phrase</th><th>Job ID</th></tr>{origins}</table><h3>Independent applicability and evidence</h3><table><tr><th>Item</th><th>State</th><th>Evidence</th><th>Judge</th></tr>{judgments}</table><h3>Complete target descriptions</h3>{target_details}<p>Merge decisions: {html.escape(canonical(f["merge_decisions"]))}</p></section>'
        )
    return (
        '<!doctype html><meta charset="utf-8"><title>Feature evidence inspector</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#21323c}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:9px;border:1px solid #ddd;text-align:left;overflow-wrap:anywhere}section{padding:25px 0;border-bottom:2px solid #ddd}</style><a href="index.html">← Figure gallery</a><h1>From a feature to its evidence</h1><p>'
        + html.escape(SYNTHETIC_BANNER if run.demo else run.manifest["run_id"])
        + "</p>"
        + "".join(content)
    )


def _gallery(run, records, config):
    cards = []
    for row in records:
        links = " · ".join(
            f'<a href="{html.escape(row["basename"])}.{ext}">{ext.upper()}</a>'
            for ext in row["formats"]
        )
        image = (
            f'<a href="{row["basename"]}.svg"><img src="{row["basename"]}.png" alt="{html.escape(row["caption"])}"></a>'
            if row["status"] == "rendered"
            else '<div class="unavailable">' + html.escape(row["caption"]) + "</div>"
        )
        examples = "".join(
            "<tr><td>"
            + html.escape(e["item_id"])
            + '</td><td><a href="feature_inspector.html#'
            + html.escape(e["feature_id"])
            + '">'
            + html.escape(e["text"])
            + "</a></td><td>"
            + html.escape(e["status"])
            + "</td></tr>"
            for e in row.get("example_features", [])
        )
        evidence = (
            "<h3>Exact incremental propositions and review state</h3><table><tr><th>Target</th><th>Proposition · click for evidence</th><th>Review status</th></tr>"
            + examples
            + "</table>"
            if examples
            else ""
        )
        cards.append(
            f'<section id="figure-{row["figure_id"]}"><p class="eyebrow">FIGURE {row["figure_id"].split("_")[0]} · {row["status"].upper()}</p><h2>{html.escape(row["title"])}</h2>{image}<p>{html.escape(row["caption"])}</p>{evidence}<p>{links} · <a href="{row["data_file"]}">Underlying rows (JSON)</a> · <a href="feature_inspector.html">Feature evidence</a></p></section>'
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Feature Space · Research Atlas</title><style>body{background:#f7f7f3;color:#20323d;font:16px/1.65 system-ui;margin:0}main{max-width:1320px;margin:auto;padding:44px 32px}h1{font-size:52px;letter-spacing:-2px;line-height:1.1;margin:18px 0}h2{font-size:29px;line-height:1.25}.eyebrow{font-size:12px;font-weight:750;letter-spacing:2px;color:#61716e}header{padding:20px 0 42px;border-bottom:1px solid #d6ded9}.banner{background:#fff0df;color:#8c3e1e;padding:14px 20px;border-left:4px solid #d76b31;font-weight:700}section{background:white;border:1px solid #e0e5df;border-radius:8px;padding:28px;margin:28px 0}img{width:100%;height:auto}a{color:#147f77}code{overflow-wrap:anywhere}.unavailable{padding:35px;background:#f3f4f0;border-left:3px solid #87958f}footer{font-size:12px;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;border-bottom:1px solid #e0e5df;text-align:left}</style></head><body><main><header><p class="eyebrow">FEATURE SPACE / FROZEN RUN ATLAS</p><h1>Inspect the representation.<br>Follow the evidence.</h1><p>'
        + (
            "Human-norm targets, independent feature judgments, and supported human production comparisons."
            if not run.task_item_ids
            else "Two realistic tasks, independent situations, and the exact propositions connecting them."
        )
        + '</p><div class="banner">'
        + html.escape(
            SYNTHETIC_BANNER
            if run.demo
            else "CANDIDATE FEATURES — NOT EVIDENCE OF INTERNAL MODEL USE"
        )
        + "</div><p><code>"
        + html.escape(run.manifest["run_id"])
        + "</code> · "
        + str(len(run.items))
        + " items · "
        + str(len(run.features))
        + " bank features · "
        + str(len(run.judgments))
        + ' resolved cells</p><p><a href="feature_inspector.html">Open the feature evidence inspector →</a> · <a href="figure_manifest.json">Figure provenance manifest</a></p></header>'
        + "".join(cards)
        + "<footer>Source manifest SHA-256: "
        + run.manifest_hash
        + "<br>Offline gallery. No scripts, analytics, remote fonts, or external asset requests. Source files remain read-only.</footer></main></body></html>"
    )

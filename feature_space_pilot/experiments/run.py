"""Offline-only correction. There is deliberately no live execution option."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from ..io import canonical, digest, file_hash, write_json, write_csv
from .banks import VERSION, separate_banks, experiment_of
from .evidence import reconstruct_matrix
from .sources import load_inputs, verify_preserved
from .analysis import FAMILY_RULES, analyze_tasks, diagnostic_audit
from .human import analyze_human
from .reports import csv_rows, write_task_report, write_human_report


def verify_export(out):
    out = Path(out).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    actual = {
        str(p.relative_to(out))
        for p in out.rglob("*")
        if p.is_file() and p != out / "manifest.json"
    }
    if actual != set(manifest["artifacts"]):
        raise ValueError("Corrected artifact inventory changed")
    for name, h in manifest["artifacts"].items():
        if file_hash(out / name) != h:
            raise ValueError("Corrected artifact hash mismatch: " + name)
    preservation = verify_preserved(
        json.loads((out / "source_checkpoint.json").read_text())
    )
    return dict(
        verified=True,
        artifacts=len(actual),
        source_preservation=preservation,
        api_calls=0,
    )


def _jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(canonical(row) + "\n")


def _common(out, experiment, items, bank, reconstruction, inputs):
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / (experiment + "_feature_bank.json"), bank)
    write_json(out / "stimuli.json", items)
    ids = {i["item_id"] for i in items}
    gens = [g for g in inputs["generations"] if g["item_id"] in ids]
    jobs = [j for j in inputs["generation_jobs"] if j["item_id"] in ids]
    _jsonl(out / "generation_provenance.jsonl", gens)
    _jsonl(out / "generation_jobs.jsonl", jobs)
    csv_rows(out / "feature_origins.csv", bank["origins"])
    csv_rows(out / "feature_inventory.csv", bank["features"])
    csv_rows(out / "old_to_new_feature_ids.csv", bank["old_to_new"])
    csv_rows(
        out / "consolidation_transfer_audit.csv", bank["consolidation_transfer_audit"]
    )
    write_json(
        out / "consolidation_decisions.json",
        dict(
            review_decisions=bank["review_decisions"],
            blocked_chains=bank["blocked_chains"],
            separate_identity=bank["inventory_hash"],
            human_reviewed=False,
        ),
    )
    scope = dict(
        experiment=experiment,
        bank_id=bank["bank_id"],
        inventory_hash=bank["inventory_hash"],
        item_ids=[i["item_id"] for i in items],
        feature_ids=[f["feature_id"] for f in bank["features"]],
        n_items=len(items),
        n_features=len(bank["features"]),
        planned_cells=len(items) * len(bank["features"]),
        primary=True,
        pooled_representation_is_secondary=True,
        unknown="?",
        unknown_is_zero=False,
        judge_configuration=inputs["config"]["judge"],
    )
    write_json(out / "matrix_scope.json", scope)
    rows = []
    for n, i in enumerate(items):
        rows.append(
            dict(
                item_id=i["item_id"],
                **{
                    f["feature_id"]: (
                        int(reconstruction["state"]["matrix"][n, k])
                        if reconstruction["state"]["matrix"][n, k] >= 0
                        else "?"
                    )
                    for k, f in enumerate(bank["features"])
                },
            )
        )
    fields = ["item_id"] + [f["feature_id"] for f in bank["features"]]
    write_csv(out / "applicability_matrix.csv", rows, fields)
    if reconstruction["state"]["complete"]:
        write_csv(out / "binary_matrix.csv", rows, fields)
    csv_rows(out / "applicability_cells.csv", reconstruction["cell_audit"])
    _jsonl(out / "reused_judgment_receipts.jsonl", reconstruction["reused_receipts"])
    csv_rows(
        out / "rejected_judgment_receipts.csv", reconstruction["rejected_receipts"]
    )
    write_json(out / "judgment_reuse_audit.json", reconstruction["audit_summary"])
    coverage = [
        dict(
            item_id=i["item_id"],
            **inputs["generation_checkpoint"]["coverage"][i["item_id"]],
        )
        for i in items
    ]
    csv_rows(out / "generation_coverage.csv", coverage)
    csv_rows(
        out / "missing_generation_jobs.csv",
        [
            j
            for j in inputs["generation_checkpoint"]["missing_generation_jobs"]
            if j["item_id"] in ids
        ],
    )
    return (
        gens,
        jobs,
        dict(
            planned=sum(r["planned"] for r in coverage),
            completed=sum(r["completed"] for r in coverage),
        ),
    )


def reconstruct(inputs, output_root, *, render=True, progress=None):
    output_root = Path(output_root).resolve()
    protected = {
        Path(inputs["provenance"][k]).resolve()
        for k in ("pooled_run", "generation_run")
    }
    protected |= {
        Path(r["path"]).resolve() for r in inputs["provenance"]["original_sources"]
    }
    if any(output_root == p or output_root.is_relative_to(p) for p in protected):
        raise ValueError(
            "Corrected output must be separate from original empirical runs"
        )

    def emit(message):
        if progress:
            progress(message)

    dependencies = [
        "full/bank.py",
        "full/equivalence.py",
        "full/reuse.py",
        "full/matrix.py",
        "full/analysis.py",
        "matching/analysis.py",
        "matching/protocol.py",
        "judge.py",
        "composition.py",
        "io.py",
        "clients.py",
        "prompts.py",
        "prompts/binary_v1.txt",
        "metrics.py",
        "task_focus/analysis.py",
        "task_focus/run.py",
    ]
    code = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    code.update(
        {
            "support/" + name: file_hash(Path(__file__).parents[1] / name)
            for name in dependencies
        }
    )
    identity = dict(
        version=VERSION,
        source=inputs["provenance"],
        code_hashes=code,
        render=render,
        demo=bool(inputs["config"].get("demo")),
    )
    out = output_root / ("separated_" + digest(identity)[:16])
    if (out / "manifest.json").exists():
        verify_export(out)
        return out
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Incomplete corrected export already exists; preserve it and use another output root"
        )
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "source_checkpoint.json", inputs["provenance"])
    emit("Reconstructing independently sourced feature banks")
    banks = separate_banks(
        inputs["items"], inputs["generations"], inputs["pooled_bank"]
    )
    summaries = {}
    # The task study is the user's deadline priority; export it first.
    for experiment in ("task", "human"):
        items = [i for i in inputs["items"] if experiment_of(i) == experiment]
        bank = banks[experiment]
        emit(
            f"{experiment}: {len(items)} items × {len(bank['features'])} features; validating compatible existing receipts"
        )
        rec = reconstruct_matrix(
            items, bank, inputs["evidence_sources"], inputs["config"]["judge"]
        )
        stage = out / experiment
        gens, jobs, coverage = _common(stage, experiment, items, bank, rec, inputs)
        reps = inputs["config"].get("replicates", 3)
        if experiment == "task":
            analysis = analyze_tasks(
                items, bank, gens, jobs, rec["state"], replicates=reps
            )
            diagnostics = diagnostic_audit(items, bank, rec["state"], analysis)
            for name, key in (
                ("semantic_annotations.csv", "annotations"),
                ("family_shared_distinctive.csv", "family_features"),
                ("instance_shared_distinctive.csv", "instance_contrast"),
                ("transitions.csv", "transitions"),
                ("composition_features.csv", "composition_features"),
                ("inheritance_review_flags.csv", "inheritance_flags"),
                ("discovery_curves.csv", "discovery_curves"),
            ):
                csv_rows(stage / name, analysis[key])
            for name, key in (
                ("compositionality.json", "compositions"),
                ("coapplicability_patterns.json", "pattern_groups"),
                ("transition_summary.json", "transition_counts"),
            ):
                write_json(stage / name, analysis[key])
            csv_rows(stage / "diagnostic_set.csv", diagnostics["diagnostic_set"])
            csv_rows(stage / "diagnostic_flags.csv", diagnostics["all_flags"])
            write_json(
                stage / "quality_audit.json",
                {
                    k: v
                    for k, v in diagnostics.items()
                    if k not in {"all_flags", "diagnostic_set"}
                },
            )
            counts = [
                dict(
                    family=family,
                    n_features=sum(
                        family in r["semantic_families"]
                        for r in analysis["annotations"]
                    ),
                    overlapping_provisional=True,
                )
                for family in FAMILY_RULES
            ]
            csv_rows(stage / "semantic_family_counts.csv", counts)
            write_task_report(stage, items, bank, rec, analysis, diagnostics, coverage)
            if render:
                emit("Rendering task Figures A, B and C with actual propositions")
                from .figures import render_task_figures

                render_task_figures(
                    stage,
                    items,
                    bank,
                    rec["state"],
                    analysis,
                    demo=bool(inputs["config"].get("demo")),
                )
            extra = dict(
                family_counts=analysis["family_counts"],
                inheritance_review_flags=len(analysis["inheritance_flags"]),
                semantic_family_counts=counts,
                unclassified_features=sum(
                    not r["semantic_families"] for r in analysis["annotations"]
                ),
                diagnostic_flags=diagnostics["counts"],
                diagnostic_sample_size=len(diagnostics["diagnostic_set"]),
                confirmed_interactions=0,
                unique_equal_call_interaction_candidates=len(
                    set().union(
                        *(
                            set(r["interaction_candidates"])
                            for r in analysis["compositions"]
                            if r["sampling_mode"] == "equal_calls"
                        )
                    )
                ),
                composition_memberships=sum(
                    len(r["interaction_candidates"])
                    for r in analysis["compositions"]
                    if r["sampling_mode"] == "equal_calls"
                ),
                complete_comparisons=sum(
                    r["available"] for r in analysis["compositions"]
                ),
            )
        else:
            analysis = analyze_human(
                items,
                bank,
                gens,
                jobs,
                rec["state"],
                inputs["semantic_snapshot"],
                replicates=reps,
            )
            write_json(stage / "human_validation.json", analysis)
            csv_rows(
                stage / "aggregate_category_reference.csv",
                inputs.get("category_reference", []),
            )
            for name, key in (
                ("semantic_recovery.csv", "semantic_recovery"),
                ("semantic_pairs.csv", "semantic_pairs"),
                ("stratified_recovery.csv", "stratified_recovery"),
                ("discovery_curves.csv", "discovery_curves"),
                ("saturation_summary.csv", "saturation_summary"),
                ("alignment_diagnostics.csv", "alignment_diagnostics"),
            ):
                csv_rows(stage / name, analysis.get(key, []))
            origin_groups = defaultdict(set)
            for o in bank["origins"]:
                origin_groups[o["item_id"], o["prompt_family"]].add(o["canonical_id"])
            im = {i["item_id"]: n for n, i in enumerate(items)}
            fm = {f["feature_id"]: n for n, f in enumerate(bank["features"])}
            yield_rows = []
            for (iid, family), fids in sorted(origin_groups.items()):
                values = [int(rec["state"]["matrix"][im[iid], fm[fid]]) for fid in fids]
                yield_rows.append(
                    dict(
                        item_id=iid,
                        prompt_family=family,
                        discovered_features=len(fids),
                        yes=sum(v == 1 for v in values),
                        no=sum(v == 0 for v in values),
                        unresolved=sum(v < 0 for v in values),
                        interpretation="Automated self-origin applicability diagnostic, not human validity precision",
                    )
                )
            csv_rows(stage / "self_origin_applicability_diagnostics.csv", yield_rows)
            write_human_report(stage, items, bank, rec, analysis, coverage)
            extra = dict(
                semantic_available=analysis["semantic_available"],
                reference_scope=analysis["reference_scope"],
                family_recovery=analysis["family_summary"],
            )
        if inputs["config"].get("demo"):
            report = stage / "scientific_report.md"
            report.write_text(
                "SYNTHETIC FIXTURE — engineering validation only; not empirical findings.\n\n"
                + report.read_text()
            )
        summaries[experiment] = dict(
            **rec["audit_summary"],
            bank_id=bank["bank_id"],
            bank_hash=bank["inventory_hash"],
            source_phrases=len(bank["origins"]),
            generation_coverage=coverage,
            **extra,
        )
        write_json(stage / "summary.json", summaries[experiment])
        emit(
            f"{experiment}: {rec['state']['resolved_cells']:,}/{rec['state']['planned_cells']:,} cells reused; {len(rec['state']['unresolved']):,} unresolved"
        )
    preservation = verify_preserved(inputs["provenance"])
    write_json(out / "source_preservation.json", preservation)
    summary = dict(
        version=VERSION,
        demo=bool(inputs["config"].get("demo")),
        api_calls=0,
        source_writes=0,
        paid_cost_usd=0,
        primary_experiments_separate=True,
        pooled_bank_secondary=True,
        human=summaries["human"],
        task=summaries["task"],
        scientifically_complete=False,
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(out / "summary.json", summary)
    (out / "README.md").write_text(
        "# Corrected independent experiments\n\n[Experiment 1: human methodology benchmark](human/scientific_report.md)\n\n[Experiment 2: primary task feature space](task/scientific_report.md)\n\nThe pooled study is preserved as secondary exploration. Each directory has its own bank identity, matrix scope, provenance, mappings and judgment reuse audit. Unknowns remain ?. No model calls or source writes occurred. Provisional semantic annotations are metadata; no preference model or causal feature-use claim is inferred.\n"
    )
    manifest = dict(
        identity=identity,
        api_calls=0,
        source_writes=0,
        artifacts={
            str(p.relative_to(out)): file_hash(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )
    write_json(out / "manifest.json", manifest)
    verify_export(out)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "latest.json",
        dict(
            output_dir=str(out),
            primary_human_report=str(out / "human/scientific_report.md"),
            primary_task_report=str(out / "task/scientific_report.md"),
            api_calls=0,
        ),
    )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pooled-run",
        help="Preserved pooled project whose evidence is reconstructed offline",
    )
    p.add_argument("--output-root", default="artifacts/feature_space_experiments")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument(
        "--verify",
        metavar="EXPORT",
        help="Verify an existing sealed corrected export and unchanged source prefixes",
    )
    args = p.parse_args()
    if args.verify:
        print(json.dumps(verify_export(args.verify), indent=2))
        return
    root = args.pooled_run
    if not root:
        pointer = Path("artifacts/feature_space_full/controller_latest.json")
        root = json.loads(pointer.read_text())["project_dir"]
    print(
        "Offline experimental separation: no provider calls; original jobs, artifacts and ledgers are read-only",
        flush=True,
    )
    inputs = load_inputs(root)
    out = reconstruct(
        inputs,
        args.output_root,
        render=not args.no_figures,
        progress=lambda s: print(s, flush=True),
    )
    print(str(out), flush=True)


if __name__ == "__main__":
    main()

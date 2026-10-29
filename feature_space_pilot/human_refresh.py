"""Offline human RSA refresh using sealed, independently sourced judgments."""

import argparse
from collections import Counter
import csv
from importlib.metadata import version as package_version
import json
from pathlib import Path
import platform
import numpy as np
import scipy
from .io import digest, file_hash, write_json, write_csv
from .experiments.banks import experiment_of
from .experiments.reports import csv_rows
from .experiments.sources import verify_preserved
from .collaborator.run import load_bundle, _holm, _matrix
from .collaborator.geometry import (
    cosine_matrix,
    cosine_pair,
    human_vectors,
    rsa_test,
    discovery_vectors,
)
from .collaborator.qualitative import linguistic_feature_mask

VERSION = "latest_human_applicability_rsa_v1"
FAMILIES = ("P0", "P1", "P2", "Combined")


def alignment_coverage(full_frequency, existing_frequency):
    rows = []
    if set(full_frequency) != set(existing_frequency):
        raise ValueError("Alignment coverage scope mismatch")
    for family, full in full_frequency.items():
        full, scored = np.asarray(full), np.asarray(existing_frequency[family])
        if full.shape != scored.shape or np.any(scored < 0) or np.any(scored > full):
            raise ValueError("Existing alignment exceeds full generation scope")
        total, covered = int(full.sum()), int(scored.sum())
        rows.append(
            dict(
                prompt_family=family,
                full_generation_units=total,
                units_in_existing_alignment_scope=covered,
                additional_unscored_units=total - covered,
                additional_units_are_unmatched=False,
                fraction_in_existing_scope=covered / total if total else None,
            )
        )
    return rows


def validate_output_root(output_root, protected):
    output_root = Path(output_root).resolve()
    if any(
        output_root == Path(p).resolve()
        or output_root.is_relative_to(Path(p).resolve())
        for p in protected
    ):
        raise ValueError("Refresh output must be separate from original artifacts")


def observability_geometry(matrix):
    x = np.asarray(matrix)
    if x.ndim != 2 or not np.isin(x, [-1, 0, 1]).all():
        raise ValueError("Judgments must be binary with explicit -1 unknowns")
    return dict(
        cosine_matrix((x >= 0).astype(int)),
        mask_is_applicability=False,
        mask_definition="1 = definitive observed judgment; 0 = unresolved, not property absence",
    )


def select_representation(
    items, bank, matrix, row_ids, column_ids, selected_ids, family, *, origin_scope
):
    """Prompt pools are determined by independent generation, never judge labels."""
    if bank.get("experiment") != "human" or any(
        experiment_of(i) != "human" for i in items
    ):
        raise ValueError(
            "Only the independent human experiment may enter this comparison"
        )
    item_ids = [i["item_id"] for i in items]
    features = [f["feature_id"] for f in bank["features"]]
    item_set, feature_set = set(item_ids), set(features)
    if (
        len(set(row_ids)) != len(row_ids)
        or set(row_ids) != set(item_ids)
        or len(set(column_ids)) != len(column_ids)
        or set(column_ids) != set(features)
        or len(set(selected_ids)) != len(selected_ids)
        or not set(selected_ids) <= set(item_ids)
    ):
        raise ValueError("Representation identity mismatch")
    x = np.asarray(matrix)
    if x.shape != (len(row_ids), len(column_ids)) or not np.isin(x, [-1, 0, 1]).all():
        raise ValueError("Judgments must be binary with explicit -1 unknowns")
    if family not in FAMILIES or origin_scope not in {"all_human", "vinson_only"}:
        raise ValueError("Undeclared prompt or discovery-origin scope")
    candidates = set()
    for origin in bank["origins"]:
        if (
            origin["item_id"] not in item_set
            or origin["canonical_id"] not in feature_set
        ):
            raise ValueError("Feature origin outside human identity scope")
        if (family == "Combined" or origin["prompt_family"] == family) and (
            origin_scope == "all_human" or origin["item_id"] in selected_ids
        ):
            candidates.add(origin["canonical_id"])
    ids = [fid for fid in features if fid in candidates]
    ri = {iid: n for n, iid in enumerate(row_ids)}
    ci = {fid: n for n, fid in enumerate(column_ids)}
    return dict(
        matrix=x[np.ix_([ri[i] for i in selected_ids], [ci[f] for f in ids])].copy(),
        item_ids=list(selected_ids),
        feature_ids=ids,
        family=family,
        origin_scope=origin_scope,
        weighting="binary automated applicability",
    )


def judged_geometry(matrix, *, method):
    """Use complete columns or explicitly conditional pairwise observations."""
    x = np.asarray(matrix)
    if x.ndim != 2 or not np.isin(x, [-1, 0, 1]).all():
        raise ValueError("Judgments must be binary with explicit -1 unknowns")
    known = x >= 0
    n, m = x.shape
    complete = np.flatnonzero(known.all(axis=0))
    counts = np.zeros((n, n), dtype=int)
    if method == "complete_columns":
        geometry = cosine_matrix(x[:, complete])
        counts[:] = len(complete)
        retained = complete.tolist()
    elif method == "jointly_observed":
        similarity = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i, n):
                mask = known[i] & known[j]
                counts[i, j] = counts[j, i] = int(mask.sum())
                value = cosine_pair(x[i, mask], x[j, mask])
                if value is not None:
                    similarity[i, j] = similarity[j, i] = value
        upper = similarity[np.triu_indices(n, 1)]
        finite = upper[np.isfinite(upper)]
        geometry = dict(
            similarity=similarity,
            zero_norm_rows=np.flatnonzero(~np.isfinite(np.diag(similarity))).tolist(),
            defined_pairs=len(finite),
            undefined_pairs=len(upper) - len(finite),
            zero_offdiagonal_pairs=int((finite == 0).sum()),
            zero_offdiagonal_proportion=(
                float((finite == 0).mean()) if len(finite) else None
            ),
        )
        retained = list(range(m))
    else:
        raise ValueError("Unknown missing-data method")
    upper_counts = counts[np.triu_indices(n, 1)]
    return dict(
        **geometry,
        n_candidate_features=m,
        n_retained_features=len(retained),
        retained_column_indices=retained,
        n_complete_columns=len(complete),
        known_cells=int(known.sum()),
        unknown_cells=int((~known).sum()),
        known_fraction=float(known.mean()) if known.size else None,
        shared_known_counts=counts,
        shared_known_min=int(upper_counts.min()) if len(upper_counts) else 0,
        shared_known_median=float(np.median(upper_counts)) if len(upper_counts) else 0,
        shared_known_max=int(upper_counts.max()) if len(upper_counts) else 0,
        common_coordinate_space=method == "complete_columns",
        unknown_is_zero=False,
    )


def verify_artifacts(out):
    """Verify sealed outputs without requiring historical mutable runtime views."""
    out = Path(out).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    actual = {
        str(p.relative_to(out))
        for p in out.rglob("*")
        if p.is_file() and p != out / "manifest.json"
    }
    if actual != set(manifest["artifacts"]):
        raise ValueError("Sealed artifact inventory mismatch")
    for name, expected in manifest["artifacts"].items():
        if file_hash(out / name) != expected:
            raise ValueError("Sealed artifact hash mismatch: " + name)
    return dict(verified=True, artifacts=len(actual))


def verify(out):
    result = verify_artifacts(out)
    result["source_preservation"] = verify_preserved(
        json.loads((Path(out) / "source_checkpoint.json").read_text())
    )
    return dict(result, api_calls=0, source_writes=0)


def _read_matrix(path):
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames[1:]
        rows = list(reader)
    matrix = np.array(
        [[int(r[c]) if r[c] in {"0", "1"} else -1 for c in columns] for r in rows],
        dtype=np.int8,
    )
    if any(r[c] not in {"0", "1", "?"} for r in rows for c in columns):
        raise ValueError("Undeclared applicability-cell encoding")
    return [r["item_id"] for r in rows], columns, matrix


def _format(value, digits=3):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def _figures(
    out, human_similarity, results, geometries, items, validation, full_output
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [i["text"] for i in items]
    sizes = len(items)

    def draw_grid(panels, title, stem):
        fig, axes = plt.subplots(
            1, len(panels), figsize=(4.6 * len(panels), 6.7), sharey=True
        )
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#d7dce2")
        for ax, (label, values) in zip(axes, panels):
            display = values.copy()
            np.fill_diagonal(display, np.nan)
            image = ax.imshow(np.ma.masked_invalid(display), vmin=0, vmax=1, cmap=cmap)
            ax.set_title(label, fontsize=10)
            ax.set_xticks(range(sizes), labels, rotation=90, fontsize=7)
            ax.set_yticks(range(sizes), labels, fontsize=7)
        fig.suptitle(
            title + " · same 36 Vinson concepts · gray = undefined/diagonal",
            fontsize=14,
            y=0.97,
        )
        fig.subplots_adjust(left=0.09, right=0.94, bottom=0.32, top=0.86, wspace=0.08)
        fig.colorbar(
            image,
            cax=fig.add_axes([0.955, 0.33, 0.009, 0.5]),
            label="Cosine similarity",
        )
        for ext in ["png", "svg"]:
            fig.savefig(out / f"figure_{stem}.{ext}", dpi=180)
        plt.close(fig)

    generation_panels = [("Human norms", human_similarity)]
    ids = [i["item_id"] for i in items]
    for row in full_output:
        family = row["prompt_family"]
        with (
            validation / "human" / f"similarity_all_output_frequency_{family}.csv"
        ).open() as f:
            reader = csv.DictReader(f)
            coordinates = reader.fieldnames[1:]
            records = list(reader)
        if coordinates != ids or [r["item_id"] for r in records] != ids:
            raise ValueError("Generation figure concept order mismatch")
        values = np.array(
            [[float(r[c]) if r[c] != "?" else np.nan for c in ids] for r in records]
        )
        generation_panels.append(
            (
                f"{family}: ρ={_format(row['rho'])}\n{row['observed_active_coordinates']:,} active coordinates",
                values,
            )
        )
    draw_grid(
        generation_panels,
        "All accepted generated features, frequency weighted",
        "generation_full_output",
    )
    for method, title in [
        ("complete_columns", "Fixed complete-column applicability"),
        ("jointly_observed", "Conditional cosine on jointly observed features"),
    ]:
        panels = [("Human norms", human_similarity)]
        for family in FAMILIES:
            row = next(
                r
                for r in results
                if r["origin_scope"] == "all_human"
                and r["family"] == family
                and r["method"] == method
                and not r["content_only"]
            )
            extra = (
                f"{row['n_retained_features']:,} columns"
                if method == "complete_columns"
                else f"median {row['shared_known_median']:,.0f} shared / pair"
            )
            panels.append(
                (
                    f"{family}: ρ={_format(row['rho'])}\n{extra}",
                    geometries["all_human", family, method, False]["similarity"],
                )
            )
        draw_grid(panels, title, method)
    columns = ["Definitive coverage", "Complete columns / candidate columns"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(4)
    base = [
        next(
            r
            for r in results
            if r["origin_scope"] == "all_human"
            and r["family"] == f
            and r["method"] == "complete_columns"
            and not r["content_only"]
        )
        for f in FAMILIES
    ]
    for shift, values, label, color in [
        (-0.18, [r["known_fraction"] for r in base], columns[0], "#168888"),
        (
            0.18,
            [r["n_complete_columns"] / r["n_candidate_features"] for r in base],
            columns[1],
            "#b87542",
        ),
    ]:
        bars = ax.bar(x + shift, values, width=0.34, label=label, color=color)
        ax.bar_label(bars, labels=[f"{100*v:.1f}%" for v in values], padding=3)
    ax.set_xticks(x, FAMILIES)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Proportion")
    ax.set_title("Judgment coverage is not a semantic-similarity score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(out / f"figure_coverage.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def refresh(
    source_export,
    validation_export,
    baseline_export,
    output_root,
    *,
    n_permutations=9999,
    seed=20261028,
    render=True,
):
    source, validation, baseline = map(
        lambda p: Path(p).resolve(), [source_export, validation_export, baseline_export]
    )
    bundle = load_bundle(source)
    verify_artifacts(validation)
    verify_artifacts(baseline)
    source_identity = json.loads((validation / "manifest.json").read_text())[
        "identity"
    ]["provenance"]
    if source_identity["source_export"] != str(source):
        raise ValueError("Refreshed validation uses a different source export")
    protected = {
        source,
        validation,
        baseline,
        Path(bundle["provenance"]["pooled_run"]),
        Path(bundle["provenance"]["generation_run"]),
    }
    protected.update(Path(s["path"]) for s in bundle["provenance"]["original_sources"])
    protected.add(
        Path(
            json.loads((baseline / "source_checkpoint.json").read_text())[
                "source_export"
            ]
        )
    )
    if bundle.get("semantic", {}).get("source_path"):
        protected.add(Path(bundle["semantic"]["source_path"]))
    output_root = Path(output_root).resolve()
    validate_output_root(output_root, protected)
    source_files = {
        str(p / "manifest.json"): file_hash(p / "manifest.json")
        for p in [validation, baseline]
    }
    identity = dict(
        version=VERSION,
        source_manifest=file_hash(source / "manifest.json"),
        validation_manifest=file_hash(validation / "manifest.json"),
        baseline_manifest=file_hash(baseline / "manifest.json"),
        code={
            str(p): file_hash(p)
            for p in [
                Path(__file__),
                Path(__file__).parent / "collaborator/geometry.py",
                Path(__file__).parent / "collaborator/qualitative.py",
            ]
        },
        n_permutations=n_permutations,
        seed=seed,
        render=render,
        numpy=np.__version__,
        scipy=scipy.__version__,
        matplotlib=package_version("matplotlib") if render else None,
        python=platform.python_version(),
    )
    out = output_root / ("human_comparison_" + digest(identity)[:16])
    if (out / "manifest.json").exists():
        verify(out)
        return out
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preserve incomplete output and choose another output root")
    out.mkdir(parents=True)
    provenance = bundle["provenance"]
    provenance["immutable_files"].update(source_files)
    for p in [validation, baseline]:
        for name, h in json.loads((p / "manifest.json").read_text())[
            "artifacts"
        ].items():
            provenance["immutable_files"][str(p / name)] = h
    write_json(out / "source_checkpoint.json", provenance)
    rows, columns, matrix = _read_matrix(source / "human/applicability_matrix.csv")
    items = bundle["human_items"]
    ids = [i["item_id"] for i in items]
    human = human_vectors(items, bundle["raw_vinson"])
    human_similarity = cosine_matrix(human["matrix"])["similarity"]
    _matrix(out / "human_similarity.csv", ids, ids, human_similarity)
    _matrix(
        out / "human_frequency_vectors.csv",
        ids,
        [f"vf_{i+1:04d}" for i in range(1029)],
        human["matrix"],
    )
    csv_rows(out / "concept_order.csv", human["crosswalk"])
    bank = bundle["human_bank"]
    features = {f["feature_id"]: f for f in bank["features"]}
    results = []
    geometries = {}
    membership = []
    (out / "permutation_nulls").mkdir()
    masks = []
    observation_diagnostics = []
    for scope in ["all_human", "vinson_only"]:
        for family in FAMILIES:
            selected = select_representation(
                bundle["full_human_items"],
                bank,
                matrix,
                rows,
                columns,
                ids,
                family,
                origin_scope=scope,
            )
            fids = selected["feature_ids"]
            x = selected["matrix"]
            observed = observability_geometry(x)
            diagnostic = rsa_test(
                human_similarity,
                observed["similarity"],
                ids,
                ids,
                n_permutations=n_permutations,
                seed=seed,
            )
            tag = f"{scope}_{family}_observation_mask"
            np.save(
                out / "permutation_nulls" / f"{tag}.npy",
                np.array(diagnostic.pop("null_rhos")),
            )
            _matrix(out / f"similarity_{tag}.csv", ids, ids, observed["similarity"])
            observation_diagnostics.append(
                dict(
                    origin_scope=scope,
                    family=family,
                    **diagnostic,
                    mask_is_applicability=False,
                    mask_definition=observed["mask_definition"],
                )
            )
            print(
                f"Judged human geometry: {scope}, {family}, {len(fids):,} candidate features",
                flush=True,
            )
            write_csv(
                out / f"applicability_{scope}_{family}.csv",
                [
                    dict(
                        item_id=iid,
                        **{
                            fid: int(v) if v >= 0 else "?" for fid, v in zip(fids, x[i])
                        },
                    )
                    for i, iid in enumerate(ids)
                ],
                ["item_id"] + fids,
            )
            linguistic = linguistic_feature_mask([features[fid] for fid in fids])
            membership.extend(
                dict(
                    origin_scope=scope,
                    family=family,
                    feature_id=fid,
                    proposition=features[fid]["text"],
                    linguistic_screen_flag=bool(flag),
                )
                for fid, flag in zip(fids, linguistic)
            )
            for content_only in [False, True]:
                keep = ~linguistic if content_only else np.ones(len(fids), dtype=bool)
                filtered = x[:, keep]
                kept_fids = np.array(fids)[keep]
                for method in ["complete_columns", "jointly_observed"]:
                    geometry = judged_geometry(filtered, method=method)
                    key = scope, family, method, content_only
                    geometries[key] = geometry
                    stat = rsa_test(
                        human_similarity,
                        geometry["similarity"],
                        ids,
                        ids,
                        n_permutations=n_permutations,
                        seed=seed,
                    )
                    within = rsa_test(
                        human_similarity,
                        geometry["similarity"],
                        ids,
                        ids,
                        n_permutations=n_permutations,
                        seed=seed,
                        strata=[i["selection_stratum"] for i in items],
                    )
                    tag = "_".join(
                        [scope, family, method, "content" if content_only else "all"]
                    )
                    np.save(
                        out / "permutation_nulls" / f"{tag}.npy",
                        np.array(stat.pop("null_rhos")),
                    )
                    np.save(
                        out / "permutation_nulls" / f"{tag}_within_type.npy",
                        np.array(within.pop("null_rhos")),
                    )
                    row = dict(
                        origin_scope=scope,
                        family=family,
                        method=method,
                        content_only=content_only,
                        **{
                            k: v
                            for k, v in geometry.items()
                            if k
                            not in {
                                "similarity",
                                "retained_column_indices",
                                "shared_known_counts",
                            }
                        },
                        **stat,
                        within_type_rho=within["rho"],
                        within_type_p=within["permutation_p_two_sided"],
                    )
                    results.append(row)
                    _matrix(
                        out / f"similarity_{tag}.csv", ids, ids, geometry["similarity"]
                    )
                    _matrix(
                        out / f"shared_known_{tag}.csv",
                        ids,
                        ids,
                        geometry["shared_known_counts"],
                    )
                    if method == "complete_columns":
                        masks.extend(
                            dict(
                                origin_scope=scope,
                                family=family,
                                content_only=content_only,
                                feature_id=str(kept_fids[n]),
                                proposition=features[kept_fids[n]]["text"],
                            )
                            for n in geometry["retained_column_indices"]
                        )
    for scope in ["all_human", "vinson_only"]:
        for method in ["complete_columns", "jointly_observed"]:
            for content in [False, True]:
                _holm(
                    [
                        r
                        for r in results
                        if r["origin_scope"] == scope
                        and r["method"] == method
                        and r["content_only"] == content
                    ]
                )
    csv_rows(out / "rsa_applicability.csv", results)
    csv_rows(out / "rsa_observation_mask.csv", observation_diagnostics)
    csv_rows(out / "feature_pool_membership.csv", membership)
    csv_rows(out / "complete_column_audit.csv", masks)
    old_summary = json.loads((baseline / "summary.json").read_text())
    new_summary = json.loads((validation / "summary.json").read_text())
    coverage = json.loads((source / "human/summary.json").read_text())
    old_source = Path(
        json.loads((baseline / "source_checkpoint.json").read_text())["source_export"]
    )
    old_coverage = json.loads((old_source / "human/summary.json").read_text())
    old_matrix_ids, old_columns, old_matrix = _read_matrix(
        old_source / "human/applicability_matrix.csv"
    )
    if old_columns != columns or old_matrix_ids != rows:
        raise ValueError("Before/after human matrix coordinates differ")
    transitions = Counter()
    for a, b in zip(old_matrix.ravel(), matrix.ravel()):
        transitions[
            (
                "unknown_to_binary"
                if a < 0 and b >= 0
                else (
                    "binary_to_unknown"
                    if a >= 0 and b < 0
                    else (
                        "changed_binary"
                        if a >= 0 and b >= 0 and a != b
                        else "unchanged"
                    )
                )
            )
        ] += 1
    # The historical checkpoint's source views are mutable runtime files; verify
    # only its sealed output bytes, then pin those artifacts into this refresh.
    verify_artifacts(old_source)
    for name, h in json.loads((old_source / "manifest.json").read_text())[
        "artifacts"
    ].items():
        provenance["immutable_files"][str(old_source / name)] = h
    provenance["immutable_files"][str(old_source / "manifest.json")] = file_hash(
        old_source / "manifest.json"
    )
    write_json(out / "source_checkpoint.json", provenance)
    full_output = [
        r
        for r in new_summary["human"]["rsa_sensitivities"]
        if r["variant"] == "all_output_frequency"
    ]
    _holm(full_output)
    full_discovery = discovery_vectors(
        items,
        bank,
        bundle["human_generations"],
        bundle["human_jobs"],
        comparison_cap=None,
    )
    balanced_discovery = discovery_vectors(
        items,
        bank,
        bundle["human_generations"],
        bundle["human_jobs"],
        comparison_cap=10,
    )
    semantic_coverage = alignment_coverage(
        full_discovery["frequency"], balanced_discovery["frequency"]
    )
    csv_rows(out / "semantic_alignment_generation_coverage.csv", semantic_coverage)
    selected_full = select_representation(
        bundle["full_human_items"],
        bank,
        matrix,
        rows,
        columns,
        ids,
        "Combined",
        origin_scope="all_human",
    )["matrix"]
    summary = dict(
        version=VERSION,
        api_calls=0,
        additional_paid_usd=0,
        source_writes=0,
        n_permutations=n_permutations,
        seed=seed,
        source_export=str(source),
        validation_export=str(validation),
        baseline_export=str(baseline),
        human_feature_bank=len(features),
        human_targets=len(rows),
        vinson_targets=len(ids),
        human_norm_coordinates=1029,
        accepted_generation_calls=coverage["generation_coverage"],
        prior_definitive_human_cells=old_coverage["reused_binary_cells"],
        definitive_human_cells=coverage["reused_binary_cells"],
        planned_human_cells=coverage["planned_cells"],
        vinson_definitive_cells=int((selected_full >= 0).sum()),
        vinson_planned_cells=int(selected_full.size),
        before_after_cell_transitions=dict(transitions),
        discovery_primary=new_summary["human"]["primary_rsa"],
        discovery_full_output=full_output,
        alignment=new_summary["human"]["aligned_cosine"],
        semantic_alignment_scope=semantic_coverage,
        observation_mask_rsa=observation_diagnostics,
        generation_rsa_changed=old_summary["human"]["primary_rsa"]
        != new_summary["human"]["primary_rsa"],
        alignment_changed=old_summary["human"]["aligned_cosine"]
        != new_summary["human"]["aligned_cosine"],
        harpaintner_available=False,
        participant_similarity_ceiling=None,
    )
    write_json(out / "summary.json", summary)
    csv_rows(out / "rsa_generation_full_output.csv", full_output)
    if render:
        _figures(
            out, human_similarity, results, geometries, items, validation, full_output
        )
    _report(out, summary, results)
    write_json(out / "source_preservation.json", verify_preserved(provenance))
    write_json(
        out / "manifest.json",
        dict(
            identity=identity,
            artifacts={
                str(p.relative_to(out)): file_hash(p)
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
        ),
    )
    verify(out)
    write_json(
        output_root / "latest.json",
        dict(
            output_dir=str(out), report=str(out / "scientific_report.md"), api_calls=0
        ),
    )
    return out


def _report(out, s, results):
    text = f"""# Human–LLM comparison after the applicability queue finished

## What changed

All {s['accepted_generation_calls']['completed']} human-generation calls were already available. Human-only applicability coverage expanded from {s['prior_definitive_human_cells']:,} to {s['definitive_human_cells']:,} definitive cells out of {s['planned_human_cells']:,}. The independent human vocabulary has {s['human_feature_bank']:,} coordinates; no task-only feature enters. For the 36 typed Vinson concepts, {s['vinson_definitive_cells']:,}/{s['vinson_planned_cells']:,} cells are definitive. Unknowns remain ? throughout, not zero. The 24 Harpaintner concepts have no feature-level human norms and are not given invented comparisons.

The earlier primary discovery RSA used the first ten outputs per call to balance generation yield. This refresh foregrounds the full accepted outputs as requested, retains that balanced analysis, reruns supervised semantic alignment, and adds an applicability-based geometry. No new inference was needed.

## Generation-based RSA using every accepted feature

Human vectors retain original frequencies and all 1,029 published coordinates. LLM weights count the repetitions generating a canonical feature, once per repetition. All three repetitions remain. Each family has the same number of generation calls; accepted output lengths differ. These independent generation spaces and the first-ten results do not change merely because applicability judgments finished.

| Prompt | Full-output rho | Permutation p | Holm p | Nonzero feature coordinates | Zero concept-pair similarities |
|---|---:|---:|---:|---:|---:|
"""
    for r in s["discovery_full_output"]:
        text += f"| {r['prompt_family']} | {_format(r['rho'])} | {_format(r['permutation_p_two_sided'],4)} | {_format(r['holm_p'],4)} | {r['observed_active_coordinates']:,} | {100*r['zero_offdiagonal_proportion']:.1f}% |\n"
    text += """
These are limited positive correspondences, constrained by extreme exact-feature sparsity. A low RSA here is not evidence that the model lacks semantic structure, and a small p-value does not establish broad recovery. The first-ten primary table, content-only, binary, surface-form and all-output results are retained in the refreshed validation export.

## New applicability-based comparison

For each prompt, the coordinate pool contains features independently generated by that prompt anywhere in the human experiment; Combined is the union, not a sum of duplicate columns. Judgments are binary applicability weights. This tests a generator-plus-judge procedure, rather than spontaneous feature production alone. All 36 concepts are retained in the same published typed-word order. No human property labels define the LLM coordinates.

Two explicit missing-data analyses are reported:

1. **Complete columns:** a single shared coordinate set whose values are definitive for all 36 concepts. This is a valid fixed-space cosine comparison, but feature selection by complete judge coverage can bias the selected vocabulary. It uses a small subset of the full candidate inventory.
2. **Jointly observed features:** each concept pair uses only columns definitive for both concepts. This draws on much more of the judged dataset, but different pairs use different coordinates. It is a conditional, exploratory similarity—not an estimate from a fully observed common feature space. Its RSA can reflect structured ambiguity/observation patterns. It cannot establish full-dataset recovery or a superior elicitation prompt.

Empty vectors and zero norms yield undefined cosines and unavailable full-cohort RSA, never zero. We do not drop difficult concepts to improve a score. Prompt-specific and Combined results share discoveries and are not independent replications.

| Pool | Candidate features | Definitive coverage | Complete columns | Fixed-space rho / p | Joint-observation rho / p | Median coordinates per pair |
|---|---:|---:|---:|---:|---:|---:|
"""
    for family in FAMILIES:
        rows = [
            r
            for r in results
            if r["origin_scope"] == "all_human"
            and r["family"] == family
            and not r["content_only"]
        ]
        fixed = next(r for r in rows if r["method"] == "complete_columns")
        joint = next(r for r in rows if r["method"] == "jointly_observed")
        text += f"| {family} | {fixed['n_candidate_features']:,} | {100*fixed['known_fraction']:.1f}% | {fixed['n_complete_columns']:,} | {_format(fixed['rho'])} / {_format(fixed['permutation_p_two_sided'],4)} | {_format(joint['rho'])} / {_format(joint['permutation_p_two_sided'],4)} | {joint['shared_known_median']:,.0f} |\n"
    combined_fixed = next(
        r
        for r in results
        if r["origin_scope"] == "all_human"
        and r["family"] == "Combined"
        and r["method"] == "complete_columns"
        and not r["content_only"]
    )
    combined_joint = next(
        r
        for r in results
        if r["origin_scope"] == "all_human"
        and r["family"] == "Combined"
        and r["method"] == "jointly_observed"
        and not r["content_only"]
    )
    text += f"\nThe larger conditional representation has positive correspondence (Combined rho {_format(combined_joint['rho'])}), but the fixed complete-column comparison does not show positive recovery (rho {_format(combined_fixed['rho'])}, p {_format(combined_fixed['permutation_p_two_sided'],4)}). Only {combined_fixed['n_complete_columns']:,}/{combined_fixed['n_candidate_features']:,} columns survive complete-case selection. P2 has zero-norm concepts under complete-column selection, so its full-cohort fixed-space RSA is unavailable. These divergent results make missing-data handling a substantive limitation. They do not support a missingness-robust validation or a clean ranking of prompt quality.\n"
    observation = next(
        r
        for r in s["observation_mask_rsa"]
        if r["origin_scope"] == "all_human" and r["family"] == "Combined"
    )
    text += f"\nA separate observation-mask diagnostic compares which coordinates received definitive judgments, ignoring the 0/1 applicability values. Combined mask RSA is {_format(observation['rho'])} (p {_format(observation['permutation_p_two_sided'],4)}). The mask uses 1=observed and 0=unresolved; this is not a zero-filled applicability representation. A weak mask correlation does not establish random missingness or remove the conditional-cosine selection limitation. All prompt/scope mask tests are exported.\n"
    text += f"""
All tests correlate the 630 upper-triangle pairs using Spearman average ranks and {s['n_permutations']:,} concept-label permutations with the same shuffle on both axes, seed {s['seed']}. Tests never treat pairs as independent. Holm adjustment covers each four-pool method/scope/content group; the CSV contains adjusted values. Exploratory within-original-word-type permutations are also exported. The full sensitivity grid includes removing lexical grammatical/metalinguistic columns and restricting discovery origins to the 36 Vinson concepts. No favorable subset or p-value is promoted as the sole result.

## Semantic feature alignment remains a separate supervised analysis

"""
    text += (
        "| Prompt | Mean aligned cosine | Defined paired concepts |\n|---|---:|---:|\n"
    )
    for family, row in s["alignment"].items():
        # Exact family-summary keys are also available in the linked validation summary.
        value = row.get("mean_cosine", row.get("paired_mean_cosine"))
        text += (
            f"| {family} | {_format(value)} | {row['n_complete_paired_concepts']} |\n"
        )
    text += "\nThe original semantic matcher evaluated the first-ten comparison scope. Coverage of all accepted canonical proposition/repetition units is:\n\n| Prompt | Full generation units | Within existing matching scope | Additional unscored units |\n|---|---:|---:|---:|\n"
    for row in s["semantic_alignment_scope"]:
        text += f"| {row['prompt_family']} | {row['full_generation_units']:,} | {row['units_in_existing_alignment_scope']:,} | {row['additional_unscored_units']:,} |\n"
    text += "\nAdditional unscored units are not classified as unmatched or invalid. The aligned cosines above remain results for the original matching scope, not a newly completed full-output alignment. Full-output P0 alignment would require additional validated semantic matching, which was not launched.\n"
    text += """
Alignment uses existing automated mappings to the human feature vocabulary, with a unit-mass limit per proposition/repetition, conservative sense filtering, unmatched-feature exports and single-label sensitivity. Applicability completion adds no new semantic-matching judgments. This is supervised correspondence, not independent RSA. Unmatched generated properties are not necessarily invalid. Neither source paper supplies a comparable participant-level cosine/RSA ceiling; researcher coding agreement is not that ceiling. Harpaintner category comparison remains unavailable pending reviewed translations and feature-category coding.

## Outputs and interpretation

- `summary.json`: coverage change, generation RSA and alignment before/after checks.
- `rsa_applicability.csv`: every method, scope, prompt, content sensitivity, sparsity, sample size and permutation p-value.
- `applicability_all_human_Combined.csv`: all 36 × 5,233 human-only judged coordinates, retaining ?.
- `similarity_*.csv`, `shared_known_*.csv`, `permutation_nulls/`: every similarity matrix, pair-specific denominator and null distribution.
- `feature_pool_membership.csv`, `complete_column_audit.csv`: exact generation-provenance pool membership and selected complete columns.
- `figure_complete_columns.*`, `figure_jointly_observed.*`, `figure_coverage.*`: scalable SVG and PNG figures. Gray denotes undefined values; coverage is not a validity score.
- `figure_generation_full_output.*`: independent human/LLM similarity matrices using every accepted generated feature.

The empirical question is which aspect of correspondence survives full output, prompt balancing, word-type control, grammar screening and missing-data assumptions. These are descriptive representations of model outputs/judgments, not evidence that these features cause model preferences. Primary task inventories remain target-specific and unchanged.

"""
    text += f"Refreshed original validation: [{s['validation_export']}]({s['validation_export']}/human/scientific_report.md).\n\n"
    text += "Reproduce offline with the exact source/validation/baseline paths recorded in `summary.json` and `--permutations 9999 --seed 20261028`. All source files and journals are checked before and after; original results remain preserved. Additional paid cost: $0.\n"
    (out / "scientific_report.md").write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-export")
    p.add_argument("--validation-export")
    p.add_argument(
        "--baseline-export",
        default="artifacts/feature_space_collaborator/followup_23476f15019ef1c1",
    )
    p.add_argument(
        "--output-root", default="artifacts/feature_space_human_refresh/results"
    )
    p.add_argument("--permutations", type=int, default=9999)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--verify")
    a = p.parse_args()
    if a.verify:
        print(json.dumps(verify(a.verify), indent=2))
        return
    root = Path("artifacts/feature_space_human_refresh")
    source = (
        a.source_export
        or json.loads((root / "checkpoints/latest.json").read_text())["output_dir"]
    )
    validation = (
        a.validation_export
        or json.loads((root / "validation/latest.json").read_text())["output_dir"]
    )
    print(
        "Offline refresh: all human-only features, explicit unknowns, no provider calls",
        flush=True,
    )
    print(
        refresh(
            source,
            validation,
            a.baseline_export,
            a.output_root,
            n_permutations=a.permutations,
            seed=a.seed,
            render=not a.no_figures,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

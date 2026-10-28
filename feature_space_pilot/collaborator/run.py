"""Offline human validation and target-local task analysis; no paid execution path."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
import platform
from importlib.metadata import version as package_version
from pathlib import Path
import numpy as np
import scipy
from ..io import digest, file_hash, write_json, write_csv
from ..source_adapters import load_vinson_zip
from ..experiments.run import verify_export
from ..experiments.sources import _matching, verify_preserved
from ..experiments.reports import csv_rows
from ..task_focus.run import ORDER
from .geometry import (
    FAMILIES,
    human_vectors,
    discovery_vectors,
    cosine_matrix,
    rsa_test,
)
from .alignment import align_features
from .tasks import local_inventories, discovery_compositions
from .qualitative import qualitative_comparison, linguistic_feature_mask
from .literature import reliability_summary
from .presentation import prompt_html, write_reports, render_figures

VERSION = "collaborator_validation_target_local_v1"
ROOT = Path(__file__).resolve().parents[2]


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_bundle(source):
    source = Path(source).resolve()
    verify_export(source)
    old = json.loads((source / "manifest.json").read_text())
    provenance = deepcopy(json.loads((source / "source_checkpoint.json").read_text()))
    provenance["immutable_files"][str(source / "manifest.json")] = file_hash(
        source / "manifest.json"
    )
    for name, h in old["artifacts"].items():
        provenance["immutable_files"][str(source / name)] = h
    hsource = next(
        r for r in provenance["original_sources"] if r["experiment"] == "human"
    )
    registry_path = Path(hsource["path"]) / "frozen_source_registry.json"
    m = json.loads((Path(hsource["path"]) / "manifest.json").read_text())
    if file_hash(registry_path) != m["artifacts"]["frozen_source_registry.json"]:
        raise ValueError("Frozen source registry mismatch")
    provenance["immutable_files"][str(registry_path)] = file_hash(registry_path)
    registry = json.loads(registry_path.read_text())
    spec = next(s for s in registry["sources"] if s["dataset_id"] == "vinson_2008")
    raw_path = ROOT / spec["raw_path"]
    if file_hash(raw_path) != spec["raw_sha256"]:
        raise ValueError("Original Vinson raw archive hash mismatch")
    provenance["immutable_files"][str(raw_path)] = file_hash(raw_path)
    raw = load_vinson_zip(raw_path)
    full_human = json.loads((source / "human/stimuli.json").read_text())
    selected = [i for i in full_human if i.get("source_dataset") == "vinson_2008"]
    tasks = sorted(
        json.loads((source / "task/stimuli.json").read_text()),
        key=lambda i: (
            ORDER.index(i["item_id"]) if i["item_id"] in ORDER else len(ORDER)
        ),
    )
    if len(selected) != 36 or len(tasks) != 15 or len(raw["features"]) != 1029:
        raise ValueError(
            "Declared 36-concept / 15-target / 1,029-feature scope mismatch"
        )
    validation = json.loads((source / "human/human_validation.json").read_text())
    semantic = _matching(validation["matching_source_path"], provenance)
    provenance.update(
        source_export=str(source),
        source_export_manifest_sha256=file_hash(source / "manifest.json"),
        raw_vinson_sha256=spec["raw_sha256"],
    )
    bundle = dict(
        human_items=selected,
        full_human_items=full_human,
        human_bank=json.loads((source / "human/human_feature_bank.json").read_text()),
        human_generations=_jsonl(source / "human/generation_provenance.jsonl"),
        human_jobs=_jsonl(source / "human/generation_jobs.jsonl"),
        task_items=tasks,
        task_bank=json.loads((source / "task/task_feature_bank.json").read_text()),
        task_generations=_jsonl(source / "task/generation_provenance.jsonl"),
        task_jobs=_jsonl(source / "task/generation_jobs.jsonl"),
        historical_task_cells=list(
            csv.DictReader((source / "task/applicability_cells.csv").open())
        ),
        semantic=semantic,
        raw_vinson=raw,
        abstract_reference=list(
            csv.DictReader((source / "human/aggregate_category_reference.csv").open())
        ),
        demo=False,
        source_export=str(source),
        provenance=provenance,
    )
    return bundle


def _matrix(path, ids, columns, matrix):
    def value(v):
        return (
            "?" if not np.isfinite(v) else int(v) if float(v).is_integer() else float(v)
        )

    write_csv(
        path,
        [
            dict(
                item_id=iid,
                **{str(c): value(matrix[n, k]) for k, c in enumerate(columns)},
            )
            for n, iid in enumerate(ids)
        ],
        ["item_id"] + list(map(str, columns)),
    )


def _holm(rows):
    selected = sorted(
        (r for r in rows if r["permutation_p_two_sided"] is not None),
        key=lambda r: r["permutation_p_two_sided"],
    )
    previous = 0
    for n, r in enumerate(selected):
        previous = max(
            previous, min(1, (len(selected) - n) * r["permutation_p_two_sided"])
        )
        r["holm_p"] = previous
    for r in rows:
        r.setdefault("holm_p", None)


def _human(out, bundle, *, n_permutations, seed):
    items = bundle["human_items"]
    bank = bundle["human_bank"]
    h = human_vectors(items, bundle["raw_vinson"])
    d = discovery_vectors(
        items, bank, bundle["human_generations"], bundle["human_jobs"]
    )
    ids = h["item_ids"]
    hs = cosine_matrix(h["matrix"])
    feature_ids = [f"vf_{n+1:04d}" for n in range(len(h["feature_labels"]))]
    _matrix(out / "human_frequency_vectors.csv", ids, feature_ids, h["matrix"])
    csv_rows(
        out / "human_feature_coordinates.csv",
        [
            dict(feature_id=fid, original_index=n + 1, label=label)
            for n, (fid, label) in enumerate(zip(feature_ids, h["feature_labels"]))
        ],
    )
    csv_rows(
        out / "concept_order_and_raw_crosswalk.csv",
        [
            dict(order=n, **r, stimulus=items[n]["text"])
            for n, r in enumerate(h["crosswalk"])
        ],
    )
    csv_rows(out / "generation_coverage.csv", d["generation_coverage"])
    csv_rows(out / "generation_occurrences.csv", d["occurrences"])
    csv_rows(out / "llm_feature_coordinates.csv", bank["features"])
    _matrix(out / "human_similarity.csv", ids, ids, hs["similarity"])
    linguistic = linguistic_feature_mask(bank["features"])
    csv_rows(
        out / "metalinguistic_column_audit.csv",
        [
            dict(
                feature_id=f["feature_id"],
                proposition=f["text"],
                linguistic_screen_flag=bool(flag),
            )
            for f, flag in zip(bank["features"], linguistic)
        ],
    )
    surface_texts = sorted(
        {r["proposition"].casefold().strip().rstrip(".") for r in d["occurrences"]}
    )
    surface_index = {t: n for n, t in enumerate(surface_texts)}
    ii = {iid: n for n, iid in enumerate(ids)}
    surface = {
        f: np.zeros((len(ids), len(surface_texts)), dtype=np.int16) for f in FAMILIES
    }
    for iid, f, rep, text in {
        (
            r["item_id"],
            r["prompt_family"],
            r["replicate"],
            r["proposition"].casefold().strip().rstrip("."),
        )
        for r in d["occurrences"]
    }:
        surface[f][ii[iid], surface_index[text]] += 1
    csv_rows(
        out / "surface_feature_coordinates.csv",
        [
            dict(feature_id="surface_" + str(n), proposition=text)
            for n, text in enumerate(surface_texts)
        ],
    )
    all_output = discovery_vectors(
        items,
        bank,
        bundle["human_generations"],
        bundle["human_jobs"],
        comparison_cap=None,
    )
    variants = {
        "frequency": d["frequency"],
        "binary": d["binary"],
        "content_frequency": {f: m[:, ~linguistic] for f, m in d["frequency"].items()},
        "content_binary": {f: m[:, ~linguistic] for f, m in d["binary"].items()},
        "frequency_within_type_null": d["frequency"],
        "content_frequency_within_type_null": {
            f: m[:, ~linguistic] for f, m in d["frequency"].items()
        },
        "surface_frequency": surface,
        "surface_binary": {f: (m > 0).astype(int) for f, m in surface.items()},
        "all_output_frequency": all_output["frequency"],
    }
    primary = []
    sensitivities = []
    similarities = {"Human": hs["similarity"]}
    cache = {}
    positive_pairs = []
    (out / "permutation_nulls").mkdir()
    for variant, matrices in variants.items():
        for family, matrix in matrices.items():
            cs = cosine_matrix(matrix)
            groups = (
                [i["selection_stratum"] for i in items]
                if variant.endswith("within_type_null")
                else None
            )
            key = digest([np.nan_to_num(cs["similarity"], nan=-2).tolist(), groups])
            if key not in cache:
                cache[key] = rsa_test(
                    hs["similarity"],
                    cs["similarity"],
                    ids,
                    d["item_ids"],
                    n_permutations=n_permutations,
                    seed=seed,
                    strata=groups,
                )
            test = cache[key]
            if not all(
                r["complete"]
                for r in d["generation_coverage"]
                if r["prompt_family"] == family
            ):
                test = dict(
                    test,
                    available=False,
                    rho=None,
                    permutation_p_two_sided=None,
                    reason="Incomplete generation cohort",
                )
            row = {k: v for k, v in test.items() if k != "null_rhos"}
            row.update(
                prompt_family=family,
                variant=variant,
                n_feature_coordinates=matrix.shape[1],
                observed_active_coordinates=int((matrix != 0).any(axis=0).sum()),
                zero_offdiagonal_proportion=cs["zero_offdiagonal_proportion"],
                zero_norm_concepts=[ids[n] for n in cs["zero_norm_rows"]],
            )
            np.save(
                out / "permutation_nulls" / (variant + "_" + family + ".npy"),
                np.asarray(test["null_rhos"]),
            )
            _matrix(
                out / ("similarity_" + variant + "_" + family + ".csv"),
                ids,
                ids,
                cs["similarity"],
            )
            if variant == "frequency":
                primary.append(row)
                similarities[family] = cs["similarity"]
                _matrix(
                    out / ("llm_frequency_vectors_" + family + ".csv"),
                    ids,
                    d["feature_ids"],
                    matrix,
                )
                _matrix(
                    out / ("llm_binary_vectors_" + family + ".csv"),
                    ids,
                    d["feature_ids"],
                    d["binary"][family],
                )
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        if (
                            not np.isfinite(cs["similarity"][a, b])
                            or cs["similarity"][a, b] <= 0
                        ):
                            continue
                        ks = np.flatnonzero((matrix[a] > 0) & (matrix[b] > 0))
                        for k in ks:
                            f = bank["features"][k]
                            positive_pairs.append(
                                dict(
                                    prompt_family=family,
                                    left_id=ids[a],
                                    right_id=ids[b],
                                    left_concept=items[a]["text"],
                                    right_concept=items[b]["text"],
                                    llm_cosine=float(cs["similarity"][a, b]),
                                    human_cosine=float(hs["similarity"][a, b]),
                                    feature_id=f["feature_id"],
                                    canonical_proposition=f["text"],
                                    linguistic_screen_flag=bool(linguistic[k]),
                                    left_original_propositions=sorted(
                                        {
                                            o["proposition"]
                                            for o in d["occurrences"]
                                            if o["item_id"] == ids[a]
                                            and o["prompt_family"] == family
                                            and o["feature_id"] == f["feature_id"]
                                        }
                                    ),
                                    right_original_propositions=sorted(
                                        {
                                            o["proposition"]
                                            for o in d["occurrences"]
                                            if o["item_id"] == ids[b]
                                            and o["prompt_family"] == family
                                            and o["feature_id"] == f["feature_id"]
                                        }
                                    ),
                                )
                            )
            else:
                sensitivities.append(row)
    _holm(primary)
    csv_rows(out / "rsa_primary.csv", primary)
    csv_rows(out / "rsa_sensitivities.csv", sensitivities)
    csv_rows(out / "shared_pair_feature_evidence.csv", positive_pairs)
    csv_rows(
        out / "upper_triangle_similarities.csv",
        [
            dict(
                left_id=ids[a],
                right_id=ids[b],
                **{
                    key: float(m[a, b]) if np.isfinite(m[a, b]) else None
                    for key, m in similarities.items()
                },
            )
            for a in range(len(ids))
            for b in range(a + 1, len(ids))
        ],
    )
    aligned = align_features(items, h, d, bundle["semantic"])
    csv_rows(out / "semantic_inventory_audit.csv", aligned["inventory_audit"])
    csv_rows(out / "semantic_mapping_audit.csv", aligned["mapping_audit"])
    csv_rows(
        out / "unmatched_generated_features.csv",
        [r for r in aligned["mapping_audit"] if r["unmatched"] or r["unscored"]],
    )
    csv_rows(out / "feature_aligned_cosine.csv", aligned["concept_scores"])
    concept_scores = []
    for iid in ids:
        values = {
            f: next(
                r
                for r in aligned["concept_scores"]
                if r["item_id"] == iid and r["prompt_family"] == f
            )["cosine"]
            for f in FAMILIES
        }
        defined = [v for v in values.values() if v is not None]
        concept_scores.append(
            dict(
                item_id=iid,
                **values,
                mean_all_three=float(np.mean(defined)) if len(defined) == 3 else None,
                mean_defined_prompts=float(np.mean(defined)) if defined else None,
                n_defined_prompts=len(defined),
            )
        )
    csv_rows(out / "feature_aligned_cosine_by_concept.csv", concept_scores)
    for f in FAMILIES:
        _matrix(
            out / ("aligned_frequency_vectors_" + f + ".csv"),
            ids,
            feature_ids,
            aligned["frequency"][f],
        )
        _matrix(
            out / ("aligned_single_label_only_vectors_" + f + ".csv"),
            ids,
            feature_ids,
            aligned["single_label_only"][f],
        )
    write_json(
        out / "alignment_summary.json",
        dict(
            families=aligned["family_summary"],
            paired_concepts=aligned["paired_concept_ids"],
            supervised=True,
            human_verified=False,
            policy=aligned["policy"],
        ),
    )
    abstract = [
        i
        for i in bundle.get("full_human_items", [])
        if i.get("source_dataset") == "harpaintner_2018"
    ]
    abstract_status = dict(
        available=False,
        categories=["SM", "IS/E", "SC", "VA"],
        n_targets=len(abstract),
        reason="Reviewed English translations and generated-feature category coding are incomplete. No individual human feature strings exist in the supplied category summaries.",
        preserved_stimuli=abstract,
        participant_feature_vectors_invented=False,
    )
    write_json(out / "harpaintner_status.json", abstract_status)
    csv_rows(
        out / "harpaintner_aggregate_reference.csv",
        bundle.get("abstract_reference", []),
    )
    write_json(
        out / "human_stimulus_provenance.json", bundle.get("full_human_items", items)
    )
    return dict(
        vectors=h,
        discovery=d,
        human_similarity=hs,
        similarities=similarities,
        rsa_primary=primary,
        rsa_sensitivities=sensitivities,
        alignment=aligned,
        shared_pair_evidence=positive_pairs,
        abstract_status=abstract_status,
    )


def verify(out):
    out = Path(out).resolve()
    m = json.loads((out / "manifest.json").read_text())
    names = {
        str(p.relative_to(out))
        for p in out.rglob("*")
        if p.is_file() and p != out / "manifest.json"
    }
    if names != set(m["artifacts"]):
        raise ValueError("Follow-up artifact inventory changed")
    for name, h in m["artifacts"].items():
        if file_hash(out / name) != h:
            raise ValueError("Follow-up artifact hash mismatch: " + name)
    preserved = verify_preserved(
        json.loads((out / "source_checkpoint.json").read_text())
    )
    return dict(
        verified=True,
        artifacts=len(names),
        source_preservation=preserved,
        model_api_calls=0,
    )


def analyze(
    bundle,
    output_root,
    *,
    n_permutations=9999,
    seed=20261028,
    render=True,
    progress=None,
):
    output_root = Path(output_root).resolve()
    source = Path(bundle["source_export"]).resolve()
    protected = {source}
    protected.update(
        Path(bundle["provenance"][key]).resolve()
        for key in ("pooled_run", "generation_run")
        if bundle["provenance"].get(key)
    )
    protected.update(
        Path(r["path"]).resolve()
        for r in bundle["provenance"].get("original_sources", [])
    )
    if bundle.get("semantic", {}).get("source_path"):
        protected.add(Path(bundle["semantic"]["source_path"]).resolve())
    if any(output_root == p or output_root.is_relative_to(p) for p in protected):
        raise ValueError("Follow-up outputs must be separate from original results")

    def emit(message):
        if progress:
            progress(message)

    provenance = deepcopy(bundle["provenance"])
    review_path = ROOT / "data/feature_space_pilot/reliability_review.json"
    provenance["immutable_files"][str(review_path)] = file_hash(review_path)
    evidence = json.loads(review_path.read_text())
    literature = reliability_summary(evidence)
    dependencies = [
        "source_adapters.py",
        "prompts.py",
        "full/bank.py",
        "full/equivalence.py",
        "composition.py",
        "io.py",
        "experiments/banks.py",
        "experiments/sources.py",
    ]
    code = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    code.update(
        {
            "support/" + name: file_hash(ROOT / "feature_space_pilot" / name)
            for name in dependencies
        }
    )
    identity = dict(
        version=VERSION,
        provenance=provenance,
        code=code,
        n_permutations=n_permutations,
        seed=seed,
        render=render,
        numpy=np.__version__,
        scipy=scipy.__version__,
        matplotlib=package_version("matplotlib") if render else None,
        python=platform.python_version(),
        demo=bundle.get("demo", False),
    )
    out = output_root / ("followup_" + digest(identity)[:16])
    if (out / "manifest.json").exists():
        verify(out)
        return out
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Preserve the incomplete output and choose a different output root"
        )
    (out / "human").mkdir(parents=True)
    (out / "task").mkdir()
    write_json(out / "source_checkpoint.json", provenance)
    emit(
        "Human benchmark: published weighted vectors, independent cosine RSA, permutation tests, supervised alignment"
    )
    human = _human(out / "human", bundle, n_permutations=n_permutations, seed=seed)
    write_json(out / "human/published_reliability.json", literature)
    emit(
        "Task experiment: reconstructing each target's own feature bank and reusing only native judgments"
    )
    local = local_inventories(
        bundle["task_items"],
        bundle["task_bank"],
        bundle["task_generations"],
        bundle["historical_task_cells"],
    )
    counts = []
    for iid, target in local["targets"].items():
        folder = out / "task/target_inventories" / iid
        folder.mkdir(parents=True)
        write_json(folder / "inventory.json", target["bank"])
        write_json(folder / "target.json", target["item"])
        csv_rows(folder / "generation_origins.csv", target["bank"]["origins"])
        csv_rows(
            folder / "judgments.csv",
            [
                dict(j, applies="?" if j["applies"] is None else j["applies"])
                for j in target["judgments"]
            ],
        )
        write_json(
            folder / "scope.json",
            dict(
                item_id=iid,
                bank_id=target["bank"]["bank_id"],
                bank_hash=target["bank"]["inventory_hash"],
                feature_ids=[f["feature_id"] for f in target["bank"]["features"]],
                primary=True,
                scope_policy=target["primary_scope"],
            ),
        )
        counts.append(
            dict(
                item_id=iid,
                level=target["item"].get("level"),
                task_id=target["item"].get("task_id"),
                features=len(target["judgments"]),
                yes=sum(j["applies"] == 1 for j in target["judgments"]),
                no=sum(j["applies"] == 0 for j in target["judgments"]),
                unresolved=sum(j["applies"] is None for j in target["judgments"]),
            )
        )
    csv_rows(out / "task/target_inventory_summary.csv", counts)
    csv_rows(
        out / "task/primary_local_judgments.csv",
        [
            dict(j, applies="?" if j["applies"] is None else j["applies"])
            for j in local["judgments"]
        ],
    )
    qual = qualitative_comparison(
        bundle["task_items"], local, bundle["task_generations"], bundle["task_jobs"]
    )
    for name, key in (
        ("generation_features.csv", "raw_features"),
        ("generation_call_metrics.csv", "call_metrics"),
        ("target_prompt_metrics.csv", "target_prompt_metrics"),
        ("prompt_summary.csv", "prompt_summary"),
        ("prompt_feature_patterns.csv", "prompt_feature_patterns"),
        ("near_duplicate_candidates.csv", "near_duplicate_pairs"),
        ("manual_review.csv", "manual_review"),
    ):
        csv_rows(out / "task" / name, qual[key])
    level_rows = []
    for level in sorted(
        {r["level"] or "unspecified" for r in qual["target_prompt_metrics"]}
    ):
        for f in FAMILIES:
            rs = [
                r
                for r in qual["target_prompt_metrics"]
                if (r["level"] or "unspecified") == level and r["prompt_family"] == f
            ]
            numeric = [
                k
                for k, v in qual["prompt_summary"][0].items()
                if k
                not in {
                    "prompt_family",
                    "n_complete_paired_targets",
                    "mean_canonical_features",
                }
            ]
            level_rows.append(
                dict(
                    level=level,
                    prompt_family=f,
                    n_targets=len(rs),
                    complete_target_count=sum(r["coverage_complete"] for r in rs),
                    **{
                        k: (
                            float(np.mean([r[k] for r in rs if r[k] is not None]))
                            if any(r[k] is not None for r in rs)
                            else None
                        )
                        for k in numeric
                    },
                )
            )
    csv_rows(out / "task/level_prompt_summary.csv", level_rows)
    compositions = discovery_compositions(
        bundle["task_items"], local, bundle["task_generations"]
    )
    csv_rows(out / "task/discovery_compositions.csv", compositions["comparisons"])
    csv_rows(
        out / "task/discovery_composition_features.csv", compositions["feature_rows"]
    )
    plan = compositions["targeted_validation_plan"]
    historical = {
        (r["item_id"], r["feature_id"]): r for r in bundle["historical_task_cells"]
    }
    for cell in plan["logical_cells"]:
        old = historical.get((cell["item_id"], cell["comparison_id"]))
        cell["historical_compatible_response_available"] = bool(
            old and old["status"] == "reused_binary"
        )
    plan["historical_resolved_cells_available_for_separate_review"] = sum(
        c["historical_compatible_response_available"] for c in plan["logical_cells"]
    )
    write_json(out / "task/targeted_validation_design.json", plan)
    prompt_html(
        out / "task",
        bundle["task_items"],
        qual,
        bundle["task_jobs"],
        demo=bundle.get("demo", False),
    )
    write_reports(out, bundle, human, local, qual, compositions, literature)
    if render:
        emit(
            "Rendering human similarity matrices, aligned cosines and target-local task figures"
        )
        render_figures(out, bundle, human, local, qual, compositions)
    preserved = verify_preserved(provenance)
    write_json(out / "source_preservation.json", preserved)
    summary = dict(
        version=VERSION,
        demo=bundle.get("demo", False),
        model_api_calls=0,
        additional_paid_usd=0,
        original_writes=0,
        human=dict(
            n_concepts=len(human["vectors"]["item_ids"]),
            human_feature_coordinates=len(human["vectors"]["feature_labels"]),
            primary_rsa=human["rsa_primary"],
            rsa_sensitivities=human["rsa_sensitivities"],
            aligned_cosine=human["alignment"]["family_summary"],
            matching_inventory_mismatches=sum(
                bool(r["unexpected_target_labels"] or r["missing_target_labels"])
                for r in human["alignment"]["inventory_audit"]
            ),
            participant_similarity_ceiling=None,
            harpaintner_available=False,
        ),
        task=dict(
            primary_cells=local["primary_cell_count"],
            historical_cells=local["historical_cross_product_cells"],
            reused_binary_cells=local["reused_binary_cells"],
            statuses=local["statuses"],
            fixed_prompt_comparison_targets=qual["fixed_complete_target_ids"],
            prompt_summary=qual["prompt_summary"],
            manual_review_cases=len(qual["manual_review"]),
            confirmed_interactions=0,
            new_requests=0,
        ),
    )
    write_json(out / "summary.json", summary)
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
            output_dir=str(out),
            human_report=str(out / "human/scientific_report.md"),
            task_report=str(out / "task/scientific_report.md"),
            full_prompt_comparison=str(out / "task/prompt_comparison.html"),
            model_api_calls=0,
        ),
    )
    emit(f"Complete: {out}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-export")
    p.add_argument("--output-root", default="artifacts/feature_space_collaborator")
    p.add_argument("--permutations", type=int, default=9999)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--verify", metavar="EXPORT")
    args = p.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
        return
    source = (
        args.source_export
        or json.loads(
            (ROOT / "artifacts/feature_space_experiments/latest.json").read_text()
        )["output_dir"]
    )
    print(
        "Offline analysis: zero model calls; independent human geometry and target-local task inventories",
        flush=True,
    )
    bundle = load_bundle(source)
    analyze(
        bundle,
        args.output_root,
        n_permutations=args.permutations,
        seed=args.seed,
        render=not args.no_figures,
        progress=lambda s: print(s, flush=True),
    )


if __name__ == "__main__":
    main()

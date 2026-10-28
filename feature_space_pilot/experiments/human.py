"""Methodological validation against the available human references, kept separate."""

from collections import Counter, defaultdict
import numpy as np
from ..prompts import normalize_phrase
from ..matching.analysis import aggregate, paired_summary
from ..full.analysis import discovery_report, saturation_summary


def stratum(item):
    if item["target_type"] == "concept":
        return "abstract concepts"
    if item["target_type"] == "object":
        return "concrete objects"
    return "events/actions"


def analyze_human(items, bank, generations, jobs, state, semantic, *, replicates=3):
    if any(i["target_type"] not in {"concept", "event", "object"} for i in items):
        raise ValueError("Non-human stimulus in methodology benchmark")
    im = {i["item_id"]: i for i in items}
    curves = discovery_report(items, bank, generations, jobs, state)
    for r in curves:
        r["stratum"] = stratum(im[r["item_id"]])
    source_categories = Counter(stratum(i) for i in items)
    result = dict(
        generation_models=sorted(
            {
                o["generation_model"]
                for o in bank["origins"]
                if o.get("generation_model")
            }
        ),
        reference_scope=dict(
            n_items=len(items),
            target_groups=dict(source_categories),
            individual_feature_norm_targets=0,
            abstract_category_only_targets=source_categories["abstract concepts"],
            abstract_individual_recall_available=False,
            abstract_category_comparison_available=False,
            abstract_limitation="Harpaintner supplies aggregate category proportions, not per-item human feature lists. English translations and generated category coding remain unreviewed.",
        ),
        family_summary={},
        stratified_recovery=[],
        semantic_recovery=[],
        semantic_pairs=[],
        discovery_curves=curves,
        saturation_summary=saturation_summary(curves, replicates=replicates),
        semantic_available=False,
        precision_against_truth=None,
        human_reviewed=False,
        validity_limit="Human production norms are incomplete reference lists, not truth labels. Unmatched propositions are not automatically false; semantic correspondence is not feature validity or preference use.",
    )
    if not semantic:
        return result
    inv = semantic["inventory"]
    prop = {(p["item_id"], p["proposition_id"]): p["text"] for p in inv["propositions"]}
    origins = {
        (o["generation_job_id"], o["raw_feature_index"]): o for o in bank["origins"]
    }
    for o in inv["origins"]:
        original = origins.get((o["generation_job_id"], o["raw_feature_index"]))
        if (
            original is None
            or original["item_id"] != o["item_id"]
            or original["prompt_family"] != o["prompt_family"]
            or original["replicate"] != o["replicate"]
            or original["normalized"]
            != normalize_phrase(prop[o["item_id"], o["proposition_id"]])
        ):
            raise ValueError(
                "Semantic matcher lacks compatible human generation provenance"
            )
    scored = aggregate(inv, semantic["records"])
    final = [r for r in scored["recovery"] if r["replicates_included"] == replicates]
    families, paired = paired_summary(final)
    groups = defaultdict(list)
    for r in final:
        groups[stratum(im[r["item_id"]]), r["prompt_family"]].append(r)
    strata = []
    for (group, family), rs in sorted(groups.items()):
        ids = {r["item_id"] for r in rs if r["recall"] is not None}
        common = set.intersection(
            *(
                {
                    r["item_id"]
                    for r in final
                    if stratum(im[r["item_id"]]) == group
                    and r["prompt_family"] == f
                    and r["recall"] is not None
                }
                for f in ("P0", "P1", "P2")
            )
        )
        vals = [r["recall"] for r in rs if r["item_id"] in common]
        strata.append(
            dict(
                group=group,
                prompt_family=family,
                n_complete_items=len(vals),
                item_ids=sorted(common),
                mean_recall=float(np.mean(vals)) if vals else None,
            )
        )
    decisions = {
        (r["item_id"], row["proposition_id"]): row
        for r in semantic["records"]
        if r.get("parse_status") == "ok" and not r.get("is_control")
        for row in r["results"]
    }
    alignment = []
    for family in ("P0", "P1", "P2"):
        ids = {
            (o["item_id"], o["proposition_id"])
            for o in inv["origins"]
            if o["prompt_family"] == family
        }
        rows = [decisions[key] for key in ids if key in decisions]
        alignment.append(
            dict(
                prompt_family=family,
                propositions=len(ids),
                scored=len(rows),
                with_clear_human_alignment=sum(bool(r["matches"]) for r in rows),
                only_related_or_uncertain=sum(
                    not r["matches"] and bool(r["related"] or r["uncertain"])
                    for r in rows
                ),
                without_list_alignment=sum(
                    not (r["matches"] or r["related"] or r["uncertain"]) for r in rows
                ),
                interpretation="Alignment diagnostic, not precision against truth; one proposition may align to several supplied labels.",
            )
        )
    result.update(
        semantic_available=True,
        matching_source_path=semantic.get("source_path"),
        matching_analysis_id=semantic.get("manifest", {}).get("analysis_id"),
        matching_grader=semantic.get("manifest", {}).get("grader"),
        family_summary=families,
        paired_item_ids=paired,
        stratified_recovery=strata,
        semantic_recovery=scored["recovery"],
        semantic_pairs=scored["pairs"],
        alignment_diagnostics=alignment,
        comparison_cap=inv["comparison_cap"],
        planned_propositions=scored["n_propositions"],
        scored_propositions=scored["n_scored_propositions"],
        reference_item_ids=sorted(i["item_id"] for i in inv["items"]),
    )
    result["reference_scope"]["individual_feature_norm_targets"] = len(inv["items"])
    return result

"""Secondary, supervised alignment into the original human feature coordinates."""

from collections import defaultdict
import numpy as np
from ..prompts import normalize_phrase
from .geometry import FAMILIES, cosine_pair


def align_features(items, human, discovery, semantic):
    ids = [i["item_id"] for i in items]
    if ids != human["item_ids"] or ids != [
        i["item_id"] for i in semantic["inventory"]["items"]
    ]:
        raise ValueError(
            "Alignment concept order differs from human and matching provenance"
        )
    ii = {iid: n for n, iid in enumerate(ids)}
    labels = defaultdict(list)
    for n, label in enumerate(human["feature_labels"]):
        labels[normalize_phrase(label)].append(n)
    coordinates = {
        (iid, label): [k for k in ks if human["matrix"][n, k] > 0]
        for iid, n in ii.items()
        for label, ks in labels.items()
    }
    n_features = len(human["feature_labels"])
    inv = semantic["inventory"]
    props = {
        (p["item_id"], p["proposition_id"]): p["text"] for p in inv["propositions"]
    }
    norms = {
        (r["item_id"], r["human_id"]): normalize_phrase(r["label"])
        for r in inv["norm_features"]
    }
    source_origins = {
        (o["generation_job_id"], o["raw_feature_index"]): o for o in inv["origins"]
    }
    decisions = {}
    for r in semantic["records"]:
        if r.get("parse_status") != "ok" or r.get("is_control"):
            continue
        for decision in r["results"]:
            key = r["item_id"], decision["proposition_id"]
            if key in decisions:
                raise ValueError("Duplicate successful matching decision")
            decisions[key] = (decision, r.get("record_hash"))
    inventory_audit = []
    for iid, n in ii.items():
        expected = {
            normalize_phrase(label)
            for k, label in enumerate(human["feature_labels"])
            if human["matrix"][n, k] > 0
        }
        offered = {label for (target, _), label in norms.items() if target == iid}
        inventory_audit.append(
            dict(
                item_id=iid,
                expected_labels=len(expected),
                offered_labels=len(offered),
                missing_target_labels=sorted(expected - offered),
                unexpected_target_labels=sorted(offered - expected),
                ambiguous_codebook_labels=sorted(
                    label for label in labels if len(coordinates[iid, label]) > 1
                ),
                policy="Keep clear labels only when present in the exact published word-plus-type frequency vector; other-sense labels are audited, not treated as human target productions",
            )
        )
    groups = defaultdict(list)
    for o in discovery["occurrences"]:
        old = source_origins.get((o["generation_job_id"], o["raw_feature_index"]))
        if old is None:
            groups[
                o["item_id"], o["prompt_family"], o["replicate"], o["feature_id"]
            ].append((o, None, None, None))
            continue
        key = o["item_id"], old["proposition_id"]
        if (
            old["item_id"] != o["item_id"]
            or old["prompt_family"] != o["prompt_family"]
            or old["replicate"] != o["replicate"]
            or normalize_phrase(props[key]) != normalize_phrase(o["proposition"])
        ):
            raise ValueError(
                "Matching generation provenance differs from original human proposition"
            )
        decision, receipt = decisions.get(key, (None, None))
        groups[
            o["item_id"], o["prompt_family"], o["replicate"], o["feature_id"]
        ].append((o, decision, receipt, old["proposition_id"]))
    reps = discovery["replicates"]
    by_rep = {f: np.zeros((len(ids), reps, n_features), dtype=float) for f in FAMILIES}
    singleton = {
        f: np.zeros((len(ids), reps, n_features), dtype=float) for f in FAMILIES
    }
    audit = []
    for (iid, family, rep, fid), rows in sorted(groups.items()):
        n = ii[iid]
        clear_sets = []
        dropped = set()
        related = set()
        uncertain = set()
        all_clear = set()
        unscored = any(d is None for _, d, _, _ in rows)
        for origin, decision, receipt, pid in rows:
            if decision is None:
                continue
            valid = set()
            for hid in decision["matches"]:
                label = norms.get((iid, hid))
                if label is None:
                    raise ValueError("Unknown supplied human match ID")
                all_clear.add(label)
                if len(coordinates.get((iid, label), [])) != 1:
                    dropped.add(label)
                else:
                    valid.add(label)
            clear_sets.append(valid)
            related.update(norms[iid, hid] for hid in decision["related"])
            uncertain.update(norms[iid, hid] for hid in decision["uncertain"])
        # Conservative within-call paraphrase handling: require agreement, not a union.
        clear = set.intersection(*clear_sets) if clear_sets and not unscored else set()
        weights = {label: 1 / len(clear) for label in sorted(clear)} if clear else {}
        for label, weight in weights.items():
            by_rep[family][n, rep, coordinates[iid, label][0]] += weight
        if len(clear) == 1:
            singleton[family][n, rep, coordinates[iid, next(iter(clear))][0]] += 1
        audit.append(
            dict(
                item_id=iid,
                prompt_family=family,
                replicate=rep,
                feature_id=fid,
                propositions=sorted({o["proposition"] for o, _, _, _ in rows}),
                generation_origins=[
                    dict(
                        generation_job_id=o["generation_job_id"],
                        raw_feature_index=o["raw_feature_index"],
                    )
                    for o, _, _, _ in rows
                ],
                proposition_ids=sorted({pid for _, _, _, pid in rows if pid}),
                source_receipt_hashes=sorted(
                    {receipt for _, _, receipt, _ in rows if receipt}
                ),
                raw_clear_labels=sorted(all_clear),
                dropped_other_sense_or_noncodebook_labels=sorted(
                    label
                    for label in dropped
                    if len(coordinates.get((iid, label), [])) == 0
                ),
                dropped_ambiguous_codebook_labels=sorted(
                    label
                    for label in dropped
                    if len(coordinates.get((iid, label), [])) > 1
                ),
                eligible_clear_labels=sorted(clear),
                related_labels=sorted(related),
                uncertain_labels=sorted(uncertain),
                label_weights=weights,
                unscored=unscored,
                unmatched=not clear and not unscored,
                multi_label_review_required=len(all_clear) > 1,
                canonical_match_disagreement=len({tuple(sorted(s)) for s in clear_sets})
                > 1,
                human_verified=False,
            )
        )
    frequency = {f: np.minimum(m, 1).sum(axis=1) / reps for f, m in by_rep.items()}
    singleton = {f: np.minimum(m, 1).sum(axis=1) / reps for f, m in singleton.items()}
    scores = []
    for iid, n in ii.items():
        for f in FAMILIES:
            rs = [r for r in audit if r["item_id"] == iid and r["prompt_family"] == f]
            unscored = sum(r["unscored"] for r in rs)
            scores.append(
                dict(
                    item_id=iid,
                    prompt_family=f,
                    canonical_generation_units=len(rs),
                    matched_units=sum(bool(r["label_weights"]) for r in rs),
                    unmatched_units=sum(r["unmatched"] for r in rs),
                    unscored_units=unscored,
                    multi_label_units=sum(r["multi_label_review_required"] for r in rs),
                    cosine=(
                        cosine_pair(human["matrix"][n], frequency[f][n])
                        if rs and not unscored
                        else None
                    ),
                    cosine_binary=(
                        cosine_pair(
                            human["matrix"][n], (frequency[f][n] > 0).astype(float)
                        )
                        if rs and not unscored
                        else None
                    ),
                    cosine_single_label_only=(
                        cosine_pair(human["matrix"][n], singleton[f][n])
                        if rs and not unscored
                        else None
                    ),
                    human_verified=False,
                )
            )
    family_summary = {}
    common = {
        iid
        for iid in ids
        if all(
            next(r for r in scores if r["item_id"] == iid and r["prompt_family"] == f)[
                "cosine"
            ]
            is not None
            for f in FAMILIES
        )
    }
    for f in FAMILIES:
        rs = [r for r in scores if r["prompt_family"] == f]
        vals = [r["cosine"] for r in rs if r["item_id"] in common]
        family_summary[f] = dict(
            n_complete_paired_concepts=len(vals),
            mean_cosine=float(np.mean(vals)) if vals else None,
            generation_units=sum(r["canonical_generation_units"] for r in rs),
            matched_units=sum(r["matched_units"] for r in rs),
            unmatched_units=sum(r["unmatched_units"] for r in rs),
            unscored_units=sum(r["unscored_units"] for r in rs),
        )
    return dict(
        item_ids=ids,
        frequency=frequency,
        single_label_only=singleton,
        mapping_audit=audit,
        inventory_audit=inventory_audit,
        concept_scores=scores,
        family_summary=family_summary,
        paired_concept_ids=sorted(common),
        supervised_alignment=True,
        human_verified=False,
        policy="One canonical proposition per repetition has total mass at most one, equally divided across eligible clear matches; paraphrase matches are intersected within a unit. Related/uncertain labels do not count. Label mass is capped at one per repetition and averaged over three repetitions. Other-sense labels and labels with multiple active original coordinates are excluded, not guessed; all 1,029 published coordinates remain intact. Single-label-only and binary sensitivities are separate.",
    )

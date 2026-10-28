"""Reconstruct local vocabularies from generation provenance, never pooled text filters."""

from collections import defaultdict
from copy import deepcopy
from ..io import digest
from ..prompts import normalize_phrase
from ..full.bank import build_bank, proof_key, protected_compatible

VERSION = "separate_experiments_v1"
HUMAN_TYPES = {"concept", "event", "object"}
TASK_TYPES = {"domain", "task_type", "task_instance", "situation", "situated_task"}


def experiment_of(item):
    kind = item["target_type"]
    if kind in HUMAN_TYPES:
        return "human"
    if kind in TASK_TYPES:
        return "task"
    raise ValueError("Undeclared experiment target type: " + kind)


def meaning_proof(left, right, reviews):
    """An ID change is free; a meaning change requires direct, protected evidence."""
    a, b = normalize_phrase(left), normalize_phrase(right)
    if a == b:
        return dict(method="exact_proposition", evidence=[])
    pair = frozenset((proof_key(a), proof_key(b)))
    relevant = [
        r
        for r in reviews
        if frozenset((proof_key(r["left"]), proof_key(r["right"]))) == pair
    ]
    if any(
        r["decision"] in {"distinct", "related", "uncertain", "pending"}
        for r in relevant
    ):
        return None
    if proof_key(a) == proof_key(b):
        return dict(method="declared_conservative_rewrite", evidence=[])
    if not protected_compatible(a, b):
        return None
    for r in relevant:
        if r["decision"] != "merge":
            continue
        if r.get("review_kind") == "automated_bidirectional_entailment":
            from ..full.equivalence import validate_automated_review

            try:
                validate_automated_review(r)
            except ValueError:
                continue
        elif not r.get("reviewer") or not r.get("reason"):
            continue
        return dict(
            method="direct_reviewed_equivalence",
            evidence=[r.get("source_receipt_hash", digest(r))],
        )
    return None


def separate_banks(items, generations, pooled):
    item_experiment = {i["item_id"]: experiment_of(i) for i in items}
    if len(item_experiment) != len(items):
        raise ValueError("Duplicate experiment item identity")
    if any(g["item_id"] not in item_experiment for g in generations):
        raise ValueError("Generation outside declared experiment stimuli")
    pooled_origins = {}
    for o in pooled["origins"]:
        key = (
            o["source_run_id"],
            o["generation_job_id"],
            o.get("attempt", 0),
            o["raw_feature_index"],
        )
        if key in pooled_origins:
            raise ValueError("Duplicate pooled provenance")
        pooled_origins[key] = o
    old_features = {f["feature_id"]: f for f in pooled["features"]}
    output = {}
    for experiment in ("human", "task"):
        local = [
            deepcopy(g)
            for g in generations
            if item_experiment[g["item_id"]] == experiment
        ]
        base = build_bank(local)
        available = {o["normalized"] for o in base["origins"]} | {
            o["equivalence_key"] for o in base["origins"]
        }
        accepted, audit = [], []
        for raw in pooled.get("review_decisions", []):
            r = {
                k: deepcopy(v) for k, v in raw.items() if not k.startswith("submitted_")
            }
            status, reason = (
                "transferred",
                "Both endpoints are locally discovered propositions or declared rewrite aliases",
            )
            if not all(normalize_phrase(r[k]) in available for k in ("left", "right")):
                status, reason = (
                    "not_local_pair",
                    "At least one endpoint is not independently discovered in this experiment; no bridge imported",
                )
            else:
                try:
                    if r["decision"] not in {
                        "merge",
                        "distinct",
                        "related",
                        "uncertain",
                        "pending",
                    }:
                        raise ValueError("Invalid equivalence decision")
                    if r.get("review_kind") == "automated_bidirectional_entailment":
                        from ..full.equivalence import validate_automated_review

                        validate_automated_review(r)
                    elif r["decision"] != "pending" and (
                        not r.get("reviewer") or not r.get("reason")
                    ):
                        raise ValueError("Named reviewer and reason required")
                    if r["decision"] == "merge" and not protected_compatible(
                        r["left"], r["right"]
                    ):
                        raise ValueError("Review changes protected meaning")
                except ValueError as e:
                    status, reason = "unsafe_review", str(e)
            audit.append(
                dict(
                    left=r["left"],
                    right=r["right"],
                    decision=r["decision"],
                    status=status,
                    reason=reason,
                    evidence_hash=digest(raw),
                )
            )
            if status == "transferred":
                accepted.append(r)
        bank = build_bank(local, reviews=accepted)
        rename = {
            f["feature_id"]: ("hf_" if experiment == "human" else "tf_")
            + digest([VERSION, experiment, f["text"], f["equivalence_keys"]])[:20]
            for f in bank["features"]
        }
        old_ids = defaultdict(set)
        for o in bank["origins"]:
            key = (
                o["source_run_id"],
                o["generation_job_id"],
                o["attempt"],
                o["raw_feature_index"],
            )
            old = pooled_origins.get(key)
            if (
                old is None
                or old["phrase"] != o["phrase"]
                or old["item_id"] != o["item_id"]
            ):
                raise ValueError(
                    "Local generation provenance absent or changed in pooled checkpoint"
                )
            o["pooled_feature_id"] = old["canonical_id"]
            o["local_content_id"] = o["canonical_id"]
            o["canonical_id"] = rename[o["canonical_id"]]
            old_ids[o["canonical_id"]].add(old["canonical_id"])
        mappings = []
        for f in bank["features"]:
            f["local_content_id"] = f["feature_id"]
            f["feature_id"] = rename[f["feature_id"]]
            f["experiment"] = experiment
            f["pooled_feature_ids"] = sorted(old_ids[f["feature_id"]])
            for old_id in f["pooled_feature_ids"]:
                old = old_features[old_id]
                proof = meaning_proof(
                    old["text"], f["text"], pooled.get("review_decisions", [])
                )
                mappings.append(
                    dict(
                        old_feature_id=old_id,
                        old_proposition=old["text"],
                        feature_id=f["feature_id"],
                        proposition=f["text"],
                        reusable_meaning=proof is not None,
                        proof=proof,
                        review_required=proof is None,
                    )
                )
        identity = dict(
            version=VERSION,
            experiment=experiment,
            local_inventory_hash=bank["inventory_hash"],
            consolidation_evidence_hash=digest(bank["review_decisions"]),
            blocked_chains_hash=digest(bank["blocked_chains"]),
            provenance=sorted(
                (
                    o["source_run_id"],
                    o["source_record_hash"],
                    o["raw_feature_index"],
                    o["canonical_id"],
                )
                for o in bank["origins"]
            ),
        )
        bank.update(
            experiment=experiment,
            bank_id=experiment + "_feature_bank_" + digest(identity)[:16],
            inventory_hash=digest(identity),
            identity=identity,
            consolidation_transfer_audit=audit,
            old_to_new=mappings,
            primary=True,
            pooled_bank_is_secondary=True,
        )
        for f in bank["features"]:
            f["inventory_hash"] = bank["inventory_hash"]
        output[experiment] = bank
    return output

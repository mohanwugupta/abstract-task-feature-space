"""Primary applicability is target-local; cross-target discovery is descriptive."""

from collections import Counter, defaultdict
import json
from ..experiments.banks import separate_banks
from ..composition import matched_sampling_allocation
from ..prompts import normalize_phrase


def _list(value):
    return json.loads(value) if isinstance(value, str) and value else value or []


def local_inventories(items, historical_bank, generations, historical_cells):
    if historical_bank.get("experiment") != "task":
        raise ValueError("Target inventories require task-only discovery provenance")
    cells = {}
    for r in historical_cells:
        key = r["item_id"], r["feature_id"]
        if key in cells:
            raise ValueError("Duplicate historical task cell")
        cells[key] = r
    targets = {}
    all_judgments = []
    for item in items:
        iid = item["item_id"]
        gs = [g for g in generations if g["item_id"] == iid]
        bank = separate_banks([item], gs, historical_bank)["task"]
        mapping = defaultdict(list)
        for m in bank["old_to_new"]:
            mapping[m["feature_id"]].append(m)
        judgments = []
        for feature in bank["features"]:
            fid = feature["feature_id"]
            compatible = []
            blocked = []
            for m in mapping[fid]:
                old = cells.get((iid, m["old_feature_id"]))
                if not m["reusable_meaning"]:
                    blocked.append(m)
                elif old is not None:
                    compatible.append((m, old))
            values = {
                int(r["applies"])
                for _, r in compatible
                if r["status"] == "reused_binary" and str(r["applies"]) in {"0", "1"}
            }
            value = next(iter(values)) if len(values) == 1 else None
            status = (
                "reused_binary"
                if value is not None
                else (
                    "conflicting_resolved_evidence"
                    if len(values) > 1
                    else (
                        "ambiguous"
                        if any(r["status"] == "ambiguous" for _, r in compatible)
                        else (
                            next(
                                (
                                    state
                                    for state in (
                                        "conflicting_resolved_evidence",
                                        "usage_unknown",
                                        "invalid",
                                        "incompatible_evidence",
                                        "missing",
                                    )
                                    if any(r["status"] == state for _, r in compatible)
                                ),
                                "invalid",
                            )
                            if compatible
                            else "meaning_requires_review" if blocked else "missing"
                        )
                    )
                )
            )
            row = dict(
                item_id=iid,
                feature_id=fid,
                proposition=feature["text"],
                applies=value,
                status=status,
                independently_discovered_for_target=True,
                target_bank_id=bank["bank_id"],
                source_experiment_bank_id=historical_bank["bank_id"],
                source_bank_hash=historical_bank["inventory_hash"],
                source_old_feature_ids=sorted(
                    {m["old_feature_id"] for m, _ in compatible}
                ),
                source_record_hashes=sorted(
                    {
                        h
                        for _, r in compatible
                        for h in _list(r.get("source_record_hashes"))
                    }
                ),
                source_provider_feature_ids=sorted(
                    {h for _, r in compatible for h in _list(r.get("old_feature_ids"))}
                ),
                meaning_proofs=[m["proof"] for m, _ in compatible],
                new_request=False,
            )
            judgments.append(row)
            all_judgments.append(row)
        # Comparison IDs identify equivalent discoveries, never licenses for new cells.
        comparison = {}
        for f in bank["features"]:
            ms = [m for m in mapping[f["feature_id"]] if m["reusable_meaning"]]
            comparison[f["feature_id"]] = (
                ms[0]["old_feature_id"]
                if len(ms) == 1
                else "local/" + iid + "/" + f["feature_id"]
            )
        targets[iid] = dict(
            item=item,
            bank=bank,
            judgments=judgments,
            comparison_ids=comparison,
            primary_scope="Only this target's original accepted generations; P0/P1/P2 and repetitions pooled locally",
        )
    return dict(
        targets=targets,
        judgments=all_judgments,
        primary_cell_count=len(all_judgments),
        comparison_registry_texts={
            f["feature_id"]: f["text"] for f in historical_bank["features"]
        },
        historical_cross_product_cells=len(items) * len(historical_bank["features"]),
        reused_binary_cells=sum(r["applies"] is not None for r in all_judgments),
        statuses=dict(Counter(r["status"] for r in all_judgments)),
        new_requests=0,
        source_reuse_policy="Sealed experiment-specific evidence chain, same exact target, explicit safe proposition mapping and unchanged original rubric/model/prompt provenance",
    )


def discovery_compositions(items, local, generations, *, replicates=3):
    accepted = [g for g in generations if g.get("parse_status") == "ok"]
    origin_maps = {}
    comparison_text = dict(local["comparison_registry_texts"])
    native = {}
    for iid, target in local["targets"].items():
        js = {j["feature_id"]: j for j in target["judgments"]}
        proofs = {
            (m["feature_id"], m["old_feature_id"]): m["reusable_meaning"]
            for m in target["bank"]["old_to_new"]
        }
        origin_maps[iid] = {}
        for o in target["bank"]["origins"]:
            cid = (
                o["pooled_feature_id"]
                if proofs.get((o["canonical_id"], o["pooled_feature_id"]))
                else "local/" + iid + "/" + o["canonical_id"]
            )
            origin_maps[iid][o["generation_job_id"], o["raw_feature_index"]] = cid
            j = js[o["canonical_id"]]
            comparison_text.setdefault(cid, j["proposition"])
            native[iid, cid] = j
    rows = []
    feature_rows = []
    plans = {}
    for index, item in enumerate(
        i for i in items if i["target_type"] == "situated_task"
    ):
        roles = dict(
            T=item["parent_id"], S=item["situation_ids"][0], TS=item["item_id"]
        )
        allocation = matched_sampling_allocation(replicates, composition_index=index)
        for family in ("P0", "P1", "P2"):
            for mode in ("equal_calls", "full_samples"):
                quotas = (
                    dict(
                        T=allocation[family]["task"],
                        S=allocation[family]["situation"],
                        TS=replicates,
                    )
                    if mode == "equal_calls"
                    else {k: replicates for k in roles}
                )
                samples = {
                    role: [
                        g
                        for g in accepted
                        if g["item_id"] == iid
                        and g["prompt_family"] == family
                        and g["replicate"] < quotas[role]
                    ]
                    for role, iid in roles.items()
                }
                sets = {}
                for role, gs in samples.items():
                    iid = roles[role]
                    target = local["targets"][iid]
                    sets[role] = {
                        origin_maps[iid][g["job_id"], k]
                        for g in gs
                        for k in range(len(g["features"]))
                    }
                component = sets["T"] | sets["S"]
                direct = sets["TS"]
                union = component | direct
                missing = [role for role in roles if len(samples[role]) != quotas[role]]
                rows.append(
                    dict(
                        item_id=item["item_id"],
                        task_id=item["task_id"],
                        situation_id=roles["S"],
                        prompt_family=family,
                        sampling_mode=mode,
                        component_count=len(component),
                        direct_count=len(direct),
                        shared_count=len(component & direct),
                        direct_only_count=len(direct - component),
                        component_only_count=len(component - direct),
                        component_calls=len(samples["T"]) + len(samples["S"]),
                        direct_calls=len(samples["TS"]),
                        planned_allocation=quotas,
                        missing_roles=missing,
                        generation_coverage_complete=not missing,
                        interaction_status="not_established_by_discovery",
                        discovery_jaccard=(
                            len(component & direct) / len(union) if union else None
                        ),
                    )
                )
                for cid in sorted(union):
                    membership = (
                        "shared"
                        if cid in component & direct
                        else "direct_only" if cid in direct else "component_only"
                    )
                    data = dict(
                        item_id=item["item_id"],
                        prompt_family=family,
                        sampling_mode=mode,
                        comparison_id=cid,
                        proposition=comparison_text[cid],
                        membership=membership,
                        interaction_confirmed=False,
                    )
                    for role, iid in roles.items():
                        j = native.get((iid, cid))
                        data[role + "_generated"] = cid in sets[role]
                        data[role + "_applicability"] = j["applies"] if j else None
                        data[role + "_judgment_status"] = (
                            j["status"] if j else "not_generated_not_in_primary_scope"
                        )
                    feature_rows.append(data)
                    if membership == "direct_only" and mode == "full_samples":
                        for role in ("T", "S", "TS"):
                            iid = roles[role]
                            j = native.get((iid, cid))
                            if j is None or j["applies"] is None:
                                plans.setdefault(
                                    (iid, cid),
                                    dict(
                                        item_id=iid,
                                        comparison_id=cid,
                                        proposition=comparison_text[cid],
                                        reason=(
                                            "Not locally generated"
                                            if j is None
                                            else "Unresolved own-target judgment"
                                        ),
                                        combined_item_id=item["item_id"],
                                        role=role,
                                        execute=False,
                                    ),
                                )
    return dict(
        comparisons=rows,
        feature_rows=feature_rows,
        new_requests=0,
        discovery_identity_policy="Original task-only canonical IDs attached to actual accepted phrase origins compare discovered sets. Local reconsolidation does not erase a shared origin. This registry never creates primary applicability cells; local inventories can have different counts.",
        targeted_validation_plan=dict(
            logical_cells=list(plans.values()),
            logical_cell_count=len(plans),
            purpose="A separate, selective design for combined-only discoveries. Reuse compatible historical exploratory judgments before considering new calls; audit grounding. No interaction inference from discovery absence.",
            requires_new_cost_preflight_and_authorization_for_paid_calls=True,
            execution_enabled=False,
        ),
    )

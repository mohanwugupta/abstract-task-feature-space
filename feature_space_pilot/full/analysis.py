"""Global structure and composition with generation distinct from applicability."""

from collections import defaultdict
import numpy as np
from ..composition import matched_sampling_allocation
from ..metrics import rdm, geometry_comparison
from ..prompts import normalize_phrase


def task_contrast(items, features, state):
    """Compare concrete task instances only on independently resolved cells."""
    from itertools import combinations

    instances = [
        (n, i) for n, i in enumerate(items) if i.get("target_type") == "task_instance"
    ]
    rows = []
    for (a, left), (b, right) in combinations(instances, 2):
        for k, f in enumerate(features):
            x, y = (int(state["matrix"][n, k]) for n in (a, b))
            comparison = (
                "unresolved"
                if min(x, y) < 0
                else (
                    "shared"
                    if x == y == 1
                    else (
                        left["task_id"] + "_only"
                        if x == 1
                        else right["task_id"] + "_only" if y == 1 else "neither"
                    )
                )
            )
            rows.append(
                dict(
                    left_item=left["item_id"],
                    right_item=right["item_id"],
                    feature_id=f["feature_id"],
                    proposition=f["text"],
                    left_applies=x if x >= 0 else None,
                    right_applies=y if y >= 0 else None,
                    comparison=comparison,
                )
            )
    return rows


def observed_usage(source, primary_records, settings):
    """Descriptive receipt usage; no transfer of token rates across models/tasks."""
    groups = defaultdict(list)
    seen = set()
    for s in source["snapshots"]:
        for stage, name, model_key in [
            ("generation", "generations", "generator"),
            ("judgment", "judgments", "judge"),
        ]:
            for r in s.get(name, []):
                model = s["config"].get(model_key, {}).get("model_id", "unknown")
                key = (stage, r["job_id"], r.get("attempt", 0))
                if key not in seen:
                    groups[(stage, model)].append(r)
                    seen.add(key)
    for r in primary_records:
        key = ("judgment", r["job_id"], r.get("attempt", 0))
        if key not in seen:
            groups[("judgment", settings["model_id"])].append(r)
            seen.add(key)
    rows = []
    for (stage, model), records in sorted(groups.items()):
        known = [
            r["usage"]
            for r in records
            if all(
                type(r.get("usage", {}).get(k)) is int
                for k in ["input_tokens", "output_tokens"]
            )
        ]
        rows.append(
            dict(
                stage=stage,
                model_id=model,
                known_usage_requests=len(known),
                missing_usage_requests=len(records) - len(known),
                input_tokens_known=sum(u["input_tokens"] for u in known),
                output_tokens_known=sum(u["output_tokens"] for u in known),
                mean_input_tokens_known=(
                    np.mean([u["input_tokens"] for u in known]).item()
                    if known
                    else None
                ),
                mean_output_tokens_known=(
                    np.mean([u["output_tokens"] for u in known]).item()
                    if known
                    else None
                ),
            )
        )
    return dict(
        groups=rows,
        configured_global_judge=settings["model_id"],
        global_judge_atomic_usage_available=any(
            r["stage"] == "judgment"
            and r["model_id"] == settings["model_id"]
            and r["known_usage_requests"]
            for r in rows
        ),
        interpretation="Observed receipt usage only. Generation, semantic matching and different judge models do not calibrate global atomic output usage. Frozen output reservations include reasoning and are not reduced from unrelated tasks.",
    )


def saturation_summary(curves, *, replicates=3):
    """Fixed complete cohort across families and every discovery-curve point."""
    selected = [r for r in curves if r["sampling_mode"] == "protocol_comparison_cap"]
    output = []
    required = {(f, n) for f in ["P0", "P1", "P2"] for n in range(1, replicates + 1)}
    for stratum in sorted({r["stratum"] for r in selected}):
        rows = [r for r in selected if r["stratum"] == stratum]
        complete = defaultdict(set)
        for r in rows:
            if r["generation_complete"]:
                complete[r["item_id"]].add(
                    (r["prompt_family"], r["replicates_included"])
                )
        ids = sorted(i for i, points in complete.items() if required <= points)
        for family, n in sorted(required):
            values = [
                r["canonical_discoveries"]
                for r in rows
                if r["item_id"] in ids
                and r["prompt_family"] == family
                and r["replicates_included"] == n
            ]
            output.append(
                dict(
                    stratum=stratum,
                    prompt_family=family,
                    replicates_included=n,
                    item_ids=ids,
                    n_items=len(ids),
                    mean_canonical_discoveries=(
                        float(np.mean(values)) if values else None
                    ),
                )
            )
    return output


def classify_composition(task, situation, direct, values, *, reviews=None):
    t, s, d = map(set, (task, situation, direct))
    component = t | s
    union = component | d

    def val(level, f):
        x = values.get(level, {}).get(f)
        return x if type(x) is int and x in (0, 1) else None

    gain = {f for f in d - component if val("TS", f) == 1}
    gaps = {f for f in gain if val("T", f) == 1 or val("S", f) == 1}
    candidates = {f for f in gain if val("T", f) == 0 and val("S", f) == 0}
    unresolved = {
        f
        for f in union
        if val("TS", f) is None
        or f in gain
        and (val("T", f) is None or val("S", f) is None)
    }
    confirmed = []
    for f, review in (reviews or {}).items():
        if (
            not review.get("reviewer")
            or not review.get("note")
            or type(review.get("confirmed")) is not bool
        ):
            raise ValueError(
                "Named relational reviewer, note and explicit boolean required"
            )
        if review["confirmed"] and f in candidates:
            confirmed.append(f)
    return dict(
        recovered_both=sorted(component & d),
        component_bank=sorted(component),
        direct_bank=sorted(d),
        direct_only_discovered=sorted(d - component),
        component_only_discovered=sorted(component - d),
        component_generated_but_inapplicable=sorted(
            f for f in component if val("TS", f) == 0
        ),
        direct_only_applicable=sorted(gain),
        direct_only_inapplicable=sorted(f for f in d - component if val("TS", f) == 0),
        constituent_discovery_gaps=sorted(gaps),
        interaction_candidates=sorted(candidates),
        confirmed_interactions=sorted(confirmed),
        unresolved_features=sorted(unresolved),
        component_applicable=sorted(f for f in component if val("TS", f) == 1),
        union_applicable=sorted(f for f in union if val("TS", f) == 1),
        jaccard_discovery=len(component & d) / len(union) if union else None,
        available=not unresolved,
        interpretation="Discovery novelty is not emergence. Interaction candidates require independent constituent/combination cells and named relational interpretation.",
    )


def structure_summary(items, features, state, *, legacy_subsets=None):
    a = state["matrix"]
    rows = []
    columns = []
    for n, item in enumerate(items):
        known = a[n] >= 0
        complete = bool(known.all())
        positive = int((a[n] == 1).sum())
        rows.append(
            dict(
                item_id=item["item_id"],
                target=item.get("text", item["item_id"]),
                target_type=item.get("target_type"),
                level=item.get("level"),
                task_id=item.get("task_id"),
                applicable_features=positive if complete else None,
                observed_applicable_lower_bound=positive,
                judged_cells=int(known.sum()),
                unresolved_cells=int((~known).sum()),
                complete=complete,
            )
        )
    for n, f in enumerate(features):
        complete = bool((a[:, n] >= 0).all())
        positive = int((a[:, n] == 1).sum())
        columns.append(
            dict(
                feature_id=f["feature_id"],
                text=f.get("text", f["feature_id"]),
                generation_origins=f.get("n_origins"),
                applicable_items=positive if complete else None,
                observed_applicable_lower_bound=positive,
                judged_items=int((a[:, n] >= 0).sum()),
                unresolved_items=int((a[:, n] < 0).sum()),
            )
        )
    geometry = dict(
        available=False,
        reason="Global applicability rows are incomplete; unknown values are never zero-filled",
    )
    if state["complete"]:
        geometry = dict(
            available=True,
            metric="jaccard",
            item_ids=state["item_ids"],
            rdm=rdm(a, metric="jaccard").tolist(),
            zero_vector_policy="both_empty=0;one_empty=1",
        )
    comparisons = {}
    idx = {f["feature_id"]: n for n, f in enumerate(features)}
    for name, ids in (legacy_subsets or {}).items():
        cols = [idx[f] for f in sorted(set(ids)) if f in idx]
        result = dict(
            comparison_policy="same_global_judge_and_item_scope",
            canonical_legacy_columns=len(cols),
            full_columns=len(features),
            original_model_scores_are_not_pooled=True,
        )
        if not state["complete"] or not cols:
            result.update(
                available=False,
                reason="Complete same-judge rows and mapped legacy columns required",
            )
        else:
            result.update(
                geometry_comparison(
                    a[:, cols], a, metric="jaccard", bootstrap=200, seed=42
                )
            )
            result["applicable_full"] = (a == 1).sum(axis=1).tolist()
            result["applicable_legacy_subset"] = (a[:, cols] == 1).sum(axis=1).tolist()
        comparisons[name] = result
    return dict(
        planned_cells=state["planned_cells"],
        resolved_cells=state["resolved_cells"],
        complete=state["complete"],
        items=rows,
        features=columns,
        features_with_observed_applicability=int(np.any(a == 1, axis=0).sum()),
        features_with_complete_zero_applicability=int(np.all(a == 0, axis=0).sum()),
        geometry=geometry,
        legacy_comparisons=comparisons,
    )


def composition_report(
    items, bank, generations, state, *, replicates=3, relational_reviews=None
):
    ii = {i["item_id"]: n for n, i in enumerate(items)}
    ff = {f["feature_id"]: n for n, f in enumerate(bank["features"])}
    mapping = {o["normalized"]: o["canonical_id"] for o in bank["origins"]}
    texts = {f["feature_id"]: f["text"] for f in bank["features"]}
    records = [r for r in generations if r.get("parse_status") == "ok"]
    result = []

    def feats(rs):
        return {mapping[normalize_phrase(f)] for r in rs for f in r["features"]}

    def tokens(rs):
        values = [
            r.get("usage", {}).get(k)
            for r in rs
            for k in ["input_tokens", "output_tokens"]
        ]
        return sum(values) if values and all(type(v) is int for v in values) else None

    for index, item in enumerate(
        i for i in items if i.get("target_type") == "situated_task"
    ):
        ids = dict(T=item["parent_id"], S=item["situation_ids"][0], TS=item["item_id"])
        allocation = matched_sampling_allocation(replicates, composition_index=index)
        for family in ["P0", "P1", "P2"]:
            for mode in ["equal_calls", "full_samples"]:
                limits = (
                    dict(
                        T=allocation[family]["task"],
                        S=allocation[family]["situation"],
                        TS=replicates,
                    )
                    if mode == "equal_calls"
                    else {k: replicates for k in ids}
                )
                rs = {
                    k: [
                        r
                        for r in records
                        if r["item_id"] == iid
                        and r["prompt_family"] == family
                        and r["replicate"] < limits[k]
                    ]
                    for k, iid in ids.items()
                }
                banks = {k: feats(v) for k, v in rs.items()}
                union = set().union(*banks.values())
                values = {
                    k: {
                        f: (
                            int(state["matrix"][ii[iid], ff[f]])
                            if state["matrix"][ii[iid], ff[f]] >= 0
                            else None
                        )
                        for f in union
                    }
                    for k, iid in ids.items()
                }
                analysis = classify_composition(
                    banks["T"],
                    banks["S"],
                    banks["TS"],
                    values,
                    reviews=(relational_reviews or {}).get(item["item_id"], {}),
                )
                missing = [k for k in ids if len(rs[k]) != limits[k]]
                analysis.update(
                    item_id=item["item_id"],
                    task_id=item["task_id"],
                    task_item_id=ids["T"],
                    situation_id=ids["S"],
                    prompt_family=family,
                    sampling_mode=mode,
                    sampling_allocation=limits,
                    component_calls=len(rs["T"]) + len(rs["S"]),
                    direct_calls=len(rs["TS"]),
                    component_tokens=tokens(rs["T"] + rs["S"]),
                    direct_tokens=tokens(rs["TS"]),
                    generation_coverage_complete=not missing,
                    missing_generation_roles=missing,
                    available=not missing and analysis["available"],
                    primary_equal_call_comparison=mode == "equal_calls",
                    examples={
                        k: [dict(feature_id=f, text=texts[f]) for f in analysis[k]]
                        for k in [
                            "direct_only_applicable",
                            "constituent_discovery_gaps",
                            "interaction_candidates",
                            "component_generated_but_inapplicable",
                        ]
                    },
                )
                result.append(analysis)
    return result


def discovery_report(items, bank, generations, jobs, state):
    mapping = {o["normalized"]: o["canonical_id"] for o in bank["origins"]}
    ii = {i["item_id"]: n for n, i in enumerate(items)}
    ff = {f["feature_id"]: n for n, f in enumerate(bank["features"])}
    itemmap = {i["item_id"]: i for i in items}
    planned = defaultdict(list)
    observed = defaultdict(list)
    for j in jobs:
        planned[j["item_id"], j["prompt_family"]].append(j)
    for r in generations:
        if r.get("parse_status") == "ok":
            observed[r["item_id"], r["prompt_family"]].append(r)
    rows = []
    for (iid, family), js in sorted(planned.items()):
        maxrep = max(j["replicate"] for j in js) + 1
        for mode in ["all_generated", "protocol_comparison_cap"]:
            cap = 4 if itemmap[iid].get("source_dataset") == "harpaintner_2018" else 10
            for n in range(1, maxrep + 1):
                rs = [r for r in observed[iid, family] if r["replicate"] < n]
                expected = sum(j["replicate"] < n for j in js)
                complete = len(rs) == expected
                phrases = [
                    f
                    for r in rs
                    for f in (
                        r["features"]
                        if mode == "all_generated"
                        else r["features"][:cap]
                    )
                ]
                ids = {mapping[normalize_phrase(f)] for f in phrases}
                values = [int(state["matrix"][ii[iid], ff[f]]) for f in ids]
                rows.append(
                    dict(
                        item_id=iid,
                        prompt_family=family,
                        replicates_included=n,
                        sampling_mode=mode,
                        comparison_cap_per_call=(
                            None if mode == "all_generated" else cap
                        ),
                        generation_calls=len(rs),
                        planned_calls=expected,
                        generation_complete=complete,
                        raw_produced=len(phrases),
                        observed_canonical_discoveries=len(ids),
                        canonical_discoveries=len(ids) if complete else None,
                        observed_applicable_discoveries=sum(v == 1 for v in values),
                        unjudged_discoveries=sum(v < 0 for v in values),
                        applicable_discoveries=(
                            sum(v == 1 for v in values)
                            if complete and all(v >= 0 for v in values)
                            else None
                        ),
                        generation_model=rs[0].get("generation_model") if rs else None,
                        stratum=(
                            "human"
                            if itemmap[iid].get("target_type")
                            in {"object", "concept", "event"}
                            else "task"
                        ),
                    )
                )
    return rows

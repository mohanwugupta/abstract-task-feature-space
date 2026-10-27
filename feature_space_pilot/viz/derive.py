"""Pure, shared figure semantics. Discovery is never a substitute for applicability."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from .load import CellState, SYMBOLS
from ..io import digest, file_hash
from ..metrics import rdm
from ..prompts import normalize_phrase
from ..references import load_reference, validate_item_alignment

FAMILIES = ("P0", "P1", "P2")
LEVELS = ("domain", "task_type", "task_instance", "situated_task", "situation")


def item_label(item):
    family = {"poem": "Poem", "finance": "Reconciliation"}.get(
        item.get("task_id"), "Situation"
    )
    level = {
        "domain": "Domain",
        "task_type": "Task type",
        "task_instance": "Instance",
        "situated_task": "Situated task",
        "situation": "Independent situation",
    }.get(item["target_type"], item["target_type"].title())
    situation = {
        "S_deadline": "Deadline",
        "S_review": "Review",
        "S_both": "Deadline + review",
    }
    if item["target_type"] == "situation":
        return situation.get(item["item_id"], item["text"])
    suffix = (
        " · " + situation.get(item["situation_ids"][0], item["situation_ids"][0])
        if item.get("situation_ids")
        else ""
    )
    return f"{family} · {level}{suffix}" if item.get("task_id") else item["text"]


def ordered_task_ids(run):
    task_order = {"poem": 0, "finance": 1, None: 2}
    situation_order = {"S_deadline": 0, "S_review": 1, "S_both": 2}
    return sorted(
        run.task_item_ids,
        key=lambda i: (
            task_order.get(run.items[i].get("task_id"), 3),
            LEVELS.index(run.items[i]["target_type"]),
            situation_order.get((run.items[i].get("situation_ids") or [i])[0], 9),
            i,
        ),
    )


def select_features(
    run,
    *,
    prompts=(),
    origins=(),
    levels=(),
    min_count=0,
    search="",
    include_unreviewed=True,
):
    jobs = defaultdict(set)
    for r in run.candidates:
        if not r["selected"]:
            continue
        if prompts and r["prompt_family"] not in prompts:
            continue
        if origins and r["origin_item"] not in origins:
            continue
        if levels and run.items[r["origin_item"]]["level"] not in levels:
            continue
        jobs[r["feature_id"]].add(r["job_id"])
    filtered = bool(prompts or origins or levels or min_count)
    excluded = set()
    if not include_unreviewed:
        for row in composition_rows(run):
            excluded |= {
                f for f, s in row["interaction_status"].items() if s == "unreviewed"
            }
    return sorted(
        (
            f
            for f in run.features
            if (not filtered or len(jobs[f]) >= max(1, min_count))
            and f not in excluded
            and (
                not search
                or search.casefold()
                in (
                    f
                    + " "
                    + run.features[f]["text"]
                    + " "
                    + " ".join(
                        r["phrase"] for r in run.candidates if r["feature_id"] == f
                    )
                ).casefold()
            )
        ),
        key=lambda f: (run.features[f]["text"], f),
    )


def atlas(run, left, right, feature_ids=None):
    ids = list(feature_ids if feature_ids is not None else select_features(run))
    result = {
        k: []
        for k in [
            "shared",
            "left_only",
            "right_only",
            "neither",
            "unresolved",
            "out_of_scope",
        ]
    }
    for f in ids:
        a, b = run.cell(left, f), run.cell(right, f)
        row = dict(
            feature_id=f,
            text=run.features[f]["text"],
            left_item=left,
            right_item=right,
            left_state=SYMBOLS[a],
            right_state=SYMBOLS[b],
        )
        group = (
            "unresolved"
            if CellState.UNRESOLVED in (a, b)
            else (
                "out_of_scope"
                if CellState.OUT_OF_SCOPE in (a, b)
                else (
                    "shared"
                    if a == b == 1
                    else (
                        "left_only" if a == 1 else "right_only" if b == 1 else "neither"
                    )
                )
            )
        )
        result[group].append(row)
    result.update(
        n_selected=len(ids),
        n_bank=len(run.features),
        n_resolved=sum(
            len(result[k]) * 2 for k in ["shared", "left_only", "right_only", "neither"]
        ),
        scope_note="Origin filters select vocabulary; 0/1 states always come from independent judgments.",
    )
    return result


def matrix_data(
    run, item_ids=None, feature_ids=None, *, cluster=False, present_only=False
):
    items = list(item_ids if item_ids is not None else ordered_task_ids(run))
    features = list(feature_ids if feature_ids is not None else select_features(run))
    if present_only:
        features = [f for f in features if any(run.cell(i, f) == 1 for i in items)]
    values = [[int(run.cell(i, f)) for f in features] for i in items]
    complete = all(v in (0, 1) for row in values for v in row)
    if cluster and complete and len(features) > 1 and items:
        distances = rdm(np.asarray(values).T, metric="jaccard")
        order = leaves_list(
            linkage(
                squareform(distances, checks=True),
                method="average",
                optimal_ordering=True,
            )
        ).tolist()
        features = [features[j] for j in order]
        values = [[row[j] for j in order] for row in values]
    return dict(
        item_ids=items,
        feature_ids=features,
        values=values,
        states=[[SYMBOLS[CellState(v)] for v in row] for row in values],
        n_items=len(items),
        n_features=len(features),
        n_bank=len(run.features),
        n_resolved=sum(v in (0, 1) for row in values for v in row),
        clustering_available=complete,
        ordering=(
            "binary Jaccard / average linkage"
            if cluster and complete
            else "fixed conceptual order"
        ),
        item_labels=[item_label(run.items[i]) for i in items],
        feature_labels=[run.features[f]["text"] for f in features],
    )


def _review_status(run, iid, fid):
    decision = run.manual_audits.get(iid, {}).get(fid)
    if decision is None:
        rows = [
            r
            for r in run.interaction_reviews
            if r.get("item_id") == iid
            and r.get("feature_id") == fid
            and r.get("relational_confirmed", "").strip()
        ]
        if rows:
            r = rows[-1]
            value = r["relational_confirmed"].casefold()
            if value not in {"true", "false", "1", "0"}:
                raise ValueError("Invalid manual review flag")
            decision = dict(
                confirmed=value in {"true", "1"},
                reviewer=r.get("reviewer"),
                note=r.get("note"),
            )
    if decision is None:
        return "unreviewed"
    if (
        not isinstance(decision, dict)
        or not decision.get("reviewer")
        or not decision.get("note")
        or type(decision.get("confirmed")) is not bool
    ):
        raise ValueError(
            "Interaction confirmation requires named manual reviewer and note"
        )
    return "confirmed" if decision["confirmed"] else "rejected"


def composition_rows(run):
    result = []
    candidates_by_job = defaultdict(set)
    for r in run.candidates:
        if r["selected"]:
            candidates_by_job[r["job_id"]].add(r["feature_id"])
    for row in run.composition_audit:
        iid = row["item_id"]
        if iid not in run.items or run.items[iid]["target_type"] != "situated_task":
            raise ValueError("Unknown composition item")
        target = run.items[iid]
        t = target["parent_id"]
        if len(target["situation_ids"]) != 1:
            raise ValueError(
                "Composition requires one standalone situation; S_both is one combined item"
            )
        s = target["situation_ids"][0]
        if s != row.get("situation_id"):
            raise ValueError("Composition situation diverges from item registry")
        allocation = row.get("sampling_allocation")
        if not allocation:
            n = run.config.get("replicates", 10)
            if n % 2:
                raise ValueError("Odd composition budget lacks frozen allocation")
            allocation = {
                f: dict(task=n // 2, situation=n // 2, direct=n) for f in FAMILIES
            }
        groups = {}
        counts = {}
        tokens = {}
        for role, item in [("task", t), ("situation", s), ("direct", iid)]:
            records = [
                r
                for r in run.generations
                if r["item_id"] == item
                and r["replicate"] < allocation[r["prompt_family"]][role]
            ]
            groups[role] = (
                set().union(*(candidates_by_job[r["job_id"]] for r in records))
                if records
                else set()
            )
            counts[role] = len(records)
            tokens[role] = sum(
                (r.get("usage", {}).get("input_tokens") or 0)
                + (r.get("usage", {}).get("output_tokens") or 0)
                for r in records
            )
        if (
            counts["task"] + counts["situation"] != counts["direct"]
            or row.get("component_calls") != counts["task"] + counts["situation"]
            or row.get("direct_calls") != counts["direct"]
        ):
            raise ValueError(
                "Composition generation calls/budget unmatched or audit diverges"
            )
        for family in FAMILIES:
            if (
                allocation[family]["task"] + allocation[family]["situation"]
                != allocation[family]["direct"]
            ):
                raise ValueError("Unmatched per-family call budget")
        component = groups["task"] | groups["situation"]
        direct = groups["direct"]
        union = component | direct
        for item in [t, s, iid]:
            if any(run.cell(item, f) not in (0, 1) for f in union):
                raise ValueError(
                    "Composition blocked: missing or out-of-scope applicability"
                )
        baseline = {f for f in component if run.cell(iid, f) == 1}
        direct_applicable = {f for f in direct if run.cell(iid, f) == 1}
        delta = direct_applicable - baseline
        interactions = {f for f in delta if run.cell(t, f) == run.cell(s, f) == 0}
        statuses = {f: _review_status(run, iid, f) for f in sorted(interactions)}
        expected = dict(
            component_bank=component,
            direct_bank=direct,
            component_applicable=baseline,
            union_applicable=baseline | direct_applicable,
            newly_discovered=direct - component,
            incremental_applicable=delta,
            interaction_candidates=interactions,
            confirmed_interactions={
                f for f, state in statuses.items() if state == "confirmed"
            },
        )
        for key, values in expected.items():
            if set(row.get(key, [])) != values:
                raise ValueError(f"Composition audit diverges for {iid}: {key}")
        if delta != (baseline | direct_applicable) - baseline:
            raise ValueError("Incremental identity failed")
        result.append(
            dict(
                row,
                direct_applicable=sorted(direct_applicable),
                interaction_status=statuses,
                examples=[
                    dict(
                        feature_id=f,
                        text=run.features[f]["text"],
                        status=statuses.get(f, "incremental applicable"),
                        task_evidence=run.judgments[t, f]["evidence"],
                        situation_evidence=run.judgments[s, f]["evidence"],
                        situated_evidence=run.judgments[iid, f]["evidence"],
                    )
                    for f in sorted(delta)
                ],
                situation_interpretation=(
                    "One combined deadline-and-review situation, not a factorial interaction estimate"
                    if s == "S_both"
                    else "Independent situation modifier"
                ),
                n_component_discovered=len(component),
                n_direct_discovered=len(direct),
                component_tokens=tokens["task"] + tokens["situation"],
                direct_tokens=tokens["direct"],
                gain_per_1000_direct_tokens=(
                    1000 * len(delta) / tokens["direct"] if tokens["direct"] else None
                ),
            )
        )
    return result


def discovery_rows(run, *, representation="canonical", item_ids=None, facet=None):
    if representation not in {"raw", "canonical", "applicable"}:
        raise ValueError("Unknown discovery representation")
    selected = set(item_ids or run.items)
    candidates = defaultdict(list)
    for r in run.candidates:
        candidates[r["job_id"]].append(r)
    groups = defaultdict(list)
    item_order = {i: n for n, i in enumerate(run.items)}
    for r in run.generations:
        if r["item_id"] not in selected:
            continue
        group = run.items[r["item_id"]]["level"] if facet == "level" else "pooled"
        groups[group, r["prompt_family"]].append(r)
    output = []
    for (group, family), records in sorted(groups.items()):
        seen = set()
        excluded = set()
        produced = 0
        tokens = 0
        unknown = 0
        for call, r in enumerate(
            sorted(
                records,
                key=lambda r: (r["replicate"], item_order[r["item_id"]], r["job_id"]),
            ),
            1,
        ):
            proposals = candidates[r["job_id"]]
            produced += len(r["features"])
            excluded |= {
                normalize_phrase(p["phrase"]) for p in proposals if not p["selected"]
            }
            if representation == "raw":
                seen |= {normalize_phrase(f) for f in r["features"]}
            else:
                for p in proposals:
                    if not p["selected"]:
                        continue
                    if representation == "canonical":
                        seen.add(p["feature_id"])
                    else:
                        state = run.cell(r["item_id"], p["feature_id"])
                        if state == 1:
                            seen.add(p["feature_id"])
                        elif state not in (0, 1):
                            unknown += 1
            usage = r.get("usage", {})
            tokens += (usage.get("input_tokens") or 0) + (
                usage.get("output_tokens") or 0
            )
            caps = sorted(
                {
                    run.jobs[x["job_id"]]["cap"]
                    for x in records
                    if x["job_id"] in run.jobs
                }
            )
            output.append(
                dict(
                    group=group,
                    prompt_family=family,
                    call_count=call,
                    replicate=r["replicate"],
                    item_id=r["item_id"],
                    job_id=r["job_id"],
                    unique=len(seen),
                    produced=produced,
                    excluded_proposals=len(excluded),
                    unresolved_proposals=unknown,
                    n_items=len({x["item_id"] for x in records}),
                    total_calls=len(records),
                    output_caps=caps,
                    tokens=tokens,
                    representation=representation,
                    availability=(
                        "partial unresolved judgments" if unknown else "available"
                    ),
                    denominator="Deterministic pooled union ordered by replicate, frozen item order, job ID; not summed per-item counts",
                )
            )
    return output


def contribution_rows(run, *, by="prompt_family", item_ids=None):
    if by not in {"prompt_family", "level"}:
        raise ValueError("Unknown contribution grouping")
    allowed = set(item_ids or run.items)
    members = defaultdict(set)
    selected = set()
    texts = {}
    origin_groups = defaultdict(set)
    for r in run.candidates:
        if r["origin_item"] not in allowed:
            continue
        group = (
            r["prompt_family"]
            if by == "prompt_family"
            else run.items[r["origin_item"]]["level"]
        )
        key = (
            ("selected", r["feature_id"])
            if r["selected"]
            else ("excluded", normalize_phrase(r["phrase"]))
        )
        members[group].add(key)
        origin_groups[key].add(group)
        texts[key] = (
            run.features[r["feature_id"]]["text"] if r["selected"] else r["phrase"]
        )
    result = []
    for group, keys in sorted(members.items()):
        confirmed = {k for k in keys if k[0] == "selected"}
        result.append(
            dict(
                group=group,
                selected_unique=len(confirmed),
                excluded_unique=len(keys - confirmed),
                unique_to_group=sum(len(origin_groups[k]) == 1 for k in keys),
                shared_elsewhere=sum(len(origin_groups[k]) > 1 for k in keys),
                applicable_cross_item_cells=sum(
                    run.cell(i, k[1]) == 1 for k in confirmed for i in allowed
                ),
                examples=[
                    dict(
                        feature_id=k[1] if k[0] == "selected" else None,
                        text=texts[k],
                        selected=k[0] == "selected",
                    )
                    for k in sorted(keys)[:8]
                ],
            )
        )
    return result


def _external_path(run, value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else Path(run.config.get("root", run.path)) / p


def human_sources(run):
    if run.demo:
        return [
            dict(
                source_dataset=s.get("dataset_id", "synthetic_demo"),
                available=False,
                reason="Synthetic fixture; awaiting verified empirical human sources",
                categories_available=False,
                participant_available=False,
            )
            for s in run.sources
        ] or [
            dict(
                source_dataset="human_reference",
                available=False,
                reason="Synthetic demonstration; human scientific panels are unavailable",
                categories_available=False,
                participant_available=False,
            )
        ]
    output = []
    for spec in run.sources:
        dataset = spec.get("dataset_id", "unknown")
        items = [i for i, x in run.items.items() if x["source_dataset"] == dataset]
        record = dict(
            source_dataset=dataset,
            available=False,
            items=items,
            categories_available=False,
            participant_available=False,
        )
        if spec.get("kind") != "human" or not spec.get("verified"):
            output.append(
                dict(
                    record,
                    reason=spec.get(
                        "blocker", "Reference provenance or codebook is not verified"
                    ),
                )
            )
            continue
        if not items:
            output.append(dict(record, reason="No aligned human targets in this run"))
            continue
        try:
            ref = load_reference(
                dict(spec, path=str(_external_path(run, spec["path"]))), live=True
            )
            for iid in items:
                validate_item_alignment(run.items[iid], ref)
            recovery = [
                r
                for r in run.human_summary.get("recovery", [])
                if r["item_id"] in items and r.get("available", True)
            ]
            profile = [
                r
                for r in run.human_summary.get("category_profiles", [])
                if r["item_id"] in items
            ]
            category_file = run.evaluation.get(
                "category_coding_file"
            ) or run.config.get("category_coding_file")
            category_hash = run.evaluation.get(
                "category_coding_sha256"
            ) or run.config.get("category_coding_sha256")
            coded = False
            if category_file:
                p = _external_path(run, category_file)
                if not category_hash or file_hash(p) != category_hash:
                    raise ValueError("Category coding hash mismatch")
                with p.open(newline="") as f:
                    codes = list(csv.DictReader(f))
                if any(not r.get("reviewer") for r in codes):
                    raise ValueError("Category coding requires a named reviewer")
                allowed = {r.get("category") for r in ref["rows"] if r.get("category")}
                if any(
                    r["category"] not in allowed for r in codes if r["item_id"] in items
                ):
                    raise ValueError("Incompatible source category coding")
                coded = bool(codes)
            output.append(
                dict(
                    record,
                    available=True,
                    reason=None,
                    reference=ref,
                    recovery=recovery,
                    profiles=profile,
                    categories_available=coded,
                    participant_available=ref["data_level"] == "participant_features"
                    and all(r.get("participant_id") for r in ref["rows"]),
                )
            )
        except (ValueError, KeyError, OSError) as error:
            output.append(dict(record, reason=str(error)))
    return output or [
        dict(
            source_dataset="human_reference",
            available=False,
            reason="No verified reference registry in this run",
            categories_available=False,
            participant_available=False,
        )
    ]


def geometry_data(run, item_ids, feature_ids=None):
    ids = list(item_ids)
    features = list(
        feature_ids if feature_ids is not None else run.scope["feature_ids"]
    )
    values = [[run.cell(i, f) for f in features] for i in ids]
    if any(v not in (0, 1) for row in values for v in row):
        return dict(
            available=False,
            reason="Complete aligned binary scope required; no zero filling",
        )
    return dict(
        available=True,
        item_ids=ids,
        feature_ids=features,
        metric="jaccard",
        range=[0, 1],
        rdm=rdm(values, metric="jaccard").tolist(),
        representation="Generated judged binary features",
    )


def embedding_data(run, item_ids=None):
    value = run.evaluation.get("embedding_file") or run.config.get("embedding_file")
    expected = run.evaluation.get("embedding_sha256") or run.config.get(
        "embedding_sha256"
    )
    if not value:
        return dict(
            available=False,
            reason="Pinned raw embedding vectors are unavailable; summary correlations cannot reconstruct an RDM",
        )
    path = _external_path(run, value)
    if not expected or file_hash(path) != expected:
        raise ValueError("Embedding input hash mismatch")
    record = json.loads(path.read_text())
    if not all(record.get(k) for k in ["model_id", "revision", "source_url"]):
        raise ValueError("Pinned documented embedding model required")
    vectors = {r["item_id"]: r for r in record["items"]}
    if len(vectors) != len(record["items"]) or set(vectors) != set(run.items):
        raise ValueError("Embedding item alignment mismatch")
    for i, item in run.items.items():
        if vectors[i]["text_hash"] != digest(item["text"]):
            raise ValueError("Embedding text alignment mismatch")
    ids = list(item_ids if item_ids is not None else run.items)
    matrix = np.asarray([vectors[i]["vector"] for i in ids], dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Invalid embedding vectors")
    return dict(
        available=True,
        item_ids=ids,
        metric="cosine",
        range=[0, 2],
        rdm=rdm(matrix, metric="cosine").tolist(),
        model_id=record["model_id"],
        revision=record["revision"],
    )


def human_geometry(run, source):
    if not source["available"]:
        return dict(available=False, reason=source["reason"])
    ref = source["reference"]
    if ref["data_level"] == "category_counts":
        return dict(
            available=False, reason="Category summaries do not supply item-feature RDMs"
        )
    ids = source["items"]
    features = sorted({r["feature"] for r in ref["rows"]})
    observed = {
        (r["item"], r["feature"])
        for r in ref["rows"]
        if float(r.get("frequency", 1)) > 0
    }
    matrix = [
        [
            int((run.items[i].get("source_text", run.items[i]["text"]), f) in observed)
            for f in features
        ]
        for i in ids
    ]
    return dict(
        available=True,
        item_ids=ids,
        metric="jaccard",
        range=[0, 1],
        rdm=rdm(matrix, metric="jaccard").tolist(),
        source_dataset=source["source_dataset"],
    )


def compare_rdms(left, right, *, item_ids, other_item_ids=None, seed=42, bootstrap=200):
    if other_item_ids is not None and list(item_ids) != list(other_item_ids):
        raise ValueError("RDM item order mismatch")
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    n = len(item_ids)
    if (
        a.shape != (n, n)
        or b.shape != (n, n)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Invalid aligned RDM shape/values")
    if n < 4:
        return dict(available=False, reason="At least four aligned items required")

    def rho(x, y):
        if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return None
        return float(spearmanr(x, y).statistic)

    ix = np.triu_indices(n, 1)
    value = rho(a[ix], b[ix])
    if value is None:
        return dict(available=False, reason="Constant upper-triangle distances")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap):
        indices = rng.integers(0, n, n)
        l, r = ix
        keep = indices[l] != indices[r]
        result = rho(
            a[indices[l[keep]], indices[r[keep]]], b[indices[l[keep]], indices[r[keep]]]
        )
        if result is not None and np.isfinite(result):
            samples.append(result)
    return dict(
        available=True,
        spearman=value,
        ci=np.quantile(samples, [0.025, 0.975]).tolist() if samples else None,
        bootstrap_unit="items_excluding_same_original_item",
        valid_resamples=len(samples),
        item_ids=list(item_ids),
        seed=seed,
        caveat="Dependent pairs; descriptive item-bootstrap uncertainty, not evidence of internal encoding",
    )

"""Within-target similarity and development-only calibration against a model reference."""

from collections import defaultdict
import re
import numpy as np
from ..io import digest
from ..prompts import normalize_phrase

VERSION = "matching_baselines_v1"
REFERENCE = "automated_luna_decisions_not_human_gold"


def lexical_scores(proposition, label):
    a, b = normalize_phrase(proposition), normalize_phrase(label)
    x, y = set(re.findall(r"[^\W_]+", a)), set(re.findall(r"[^\W_]+", b))
    return dict(
        exact=float(bool(a) and a == b),
        token_jaccard=len(x & y) / len(x | y) if x | y else 0.0,
        label_token_coverage=len(x & y) / len(y) if y else 0.0,
    )


def embedding_texts(inventory):
    items = {i["item_id"]: i for i in inventory["items"]}
    result = {}
    for rows, key in [
        (inventory["norm_features"], "label"),
        (inventory["propositions"], "text"),
    ]:
        for row in rows:
            text = f"Target: {items[row['item_id']]['text']}\nFeature: {row[key]}"
            tid = digest(text)
            result[tid] = dict(text_id=tid, text=text)
    return sorted(result.values(), key=lambda x: x["text_id"])


def cosine_scores(inventory, vectors):
    items = {i["item_id"]: i for i in inventory["items"]}

    def vector(i, text):
        key = digest(f"Target: {items[i]['text']}\nFeature: {text}")
        if key not in vectors:
            raise ValueError("Missing baseline embedding: " + key)
        v = np.asarray(vectors[key], dtype=float)
        if v.ndim != 1 or not np.isfinite(v).all() or not np.linalg.norm(v) > 0:
            raise ValueError("Invalid or zero baseline embedding")
        return v / np.linalg.norm(v)

    result = {}
    for item in inventory["items"]:
        iid = item["item_id"]
        props = [p for p in inventory["propositions"] if p["item_id"] == iid]
        norms = [h for h in inventory["norm_features"] if h["item_id"] == iid]
        a = np.stack([vector(iid, p["text"]) for p in props])
        b = np.stack([vector(iid, h["label"]) for h in norms])
        if a.shape[1] != b.shape[1]:
            raise ValueError("Embedding dimensions differ")
        scores = np.clip(a @ b.T, -1, 1)
        for n, p in enumerate(props):
            for m, h in enumerate(norms):
                result[iid, p["proposition_id"], h["human_id"]] = float(scores[n, m])
    return result


def build_pairs(inventory, decisions, *, embedding=None):
    items = {i["item_id"]: i for i in inventory["items"]}
    norms = defaultdict(list)
    for h in inventory["norm_features"]:
        norms[h["item_id"]].append(h)
    rows = []
    for p in inventory["propositions"]:
        iid, pid = p["item_id"], p["proposition_id"]
        decision = decisions.get((iid, pid))
        for h in norms[iid]:
            hid = h["human_id"]
            relation = "unscored"
            if decision is not None:
                relation = next(
                    (
                        name
                        for name in ["matches", "related", "uncertain"]
                        if hid in decision[name]
                    ),
                    "no_match",
                )
            row = dict(
                item_id=iid,
                target=items[iid]["text"],
                split=items[iid]["split"],
                proposition_id=pid,
                proposition=p["text"],
                human_id=hid,
                human_label=h["label"],
                judge_relation=relation,
                judge_label=(
                    1
                    if relation == "matches"
                    else 0 if relation in {"related", "no_match"} else None
                ),
                **lexical_scores(p["text"], h["label"]),
            )
            if embedding is not None:
                row["embedding_cosine"] = embedding[iid, pid, hid]
            rows.append(row)
    return rows


def metrics(rows, method, threshold):
    tp = fp = tn = fn = 0
    for r in rows:
        if r["judge_label"] is None:
            continue
        predicted = r[method] >= threshold
        positive = r["judge_label"] == 1
        tp += int(predicted and positive)
        fp += int(predicted and not positive)
        tn += int(not predicted and not positive)
        fn += int(not predicted and positive)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return dict(
        threshold=threshold,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision_vs_judge=precision,
        recall_vs_judge=recall,
        f1_vs_judge=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        agreement=(tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn else None,
        balanced_agreement=(
            (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None
        ),
    )


def methods(rows):
    return ["exact", "token_jaccard", "label_token_coverage"] + (
        ["embedding_cosine"] if rows and "embedding_cosine" in rows[0] else []
    )


def grid(method):
    return (
        [1.0]
        if method == "exact"
        else [float(n) / 100 for n in range(0, 101, 5)] + [1.000001]
    )


def calibrate(rows, inventory):
    dev = {i["item_id"] for i in inventory["items"] if i["split"] == "development"}
    part = [r for r in rows if r["item_id"] in dev]
    if (
        not dev
        or {r["item_id"] for r in part} != dev
        or any(r["judge_relation"] == "unscored" for r in part)
    ):
        return dict(
            available=False,
            reason="Waiting for complete judging of every development target",
            methods={},
        )
    if {r["judge_label"] for r in part if r["judge_label"] is not None} != {0, 1}:
        return dict(
            available=False,
            reason="Both clear-match and nonmatch development pairs required",
            methods={},
        )
    result = {}
    for name in methods(rows):
        scores = [metrics(part, name, t) for t in grid(name)]
        # Resolve equal F1 toward fewer false positives, then a higher threshold.
        result[name] = max(
            scores, key=lambda x: (x["f1_vs_judge"] or 0, -x["fp"], x["threshold"])
        )
    return dict(
        available=True,
        reference_kind=REFERENCE,
        selection="maximum development F1; ties fewer FP then higher threshold",
        development_item_ids=sorted(dev),
        methods=result,
    )


def evaluate(rows, calibration, *, split):
    part = [r for r in rows if r["split"] == split]
    # Use only targets for which every generated proposition was actually graded.
    incomplete = {r["item_id"] for r in part if r["judge_relation"] == "unscored"}
    part = [r for r in part if r["item_id"] not in incomplete]
    valid = [r for r in part if r["judge_label"] is not None]
    return dict(
        available=bool(calibration["available"] and valid),
        reference_kind=REFERENCE,
        item_ids=sorted({r["item_id"] for r in part}),
        incomplete_item_ids=sorted(incomplete),
        n_pairs=len(valid),
        uncertain_pairs_excluded=sum(r["judge_relation"] == "uncertain" for r in part),
        judge_positive_pairs=sum(r["judge_label"] == 1 for r in valid),
        methods=(
            {
                name: metrics(valid, name, s["threshold"])
                for name, s in calibration.get("methods", {}).items()
            }
            if calibration["available"]
            else {}
        ),
    )

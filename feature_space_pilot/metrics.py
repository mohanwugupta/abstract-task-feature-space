"""Evaluation-only metrics; human feature production is never treated as truth."""

from __future__ import annotations
import math
import random
import numpy as np
from scipy.stats import spearmanr
from .prompts import normalize_phrase


def unavailable(reason):
    return dict(available=False, reason=reason)


def rdm(matrix, *, metric="jaccard"):
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Invalid feature matrix")
    if metric not in {"jaccard", "cosine"}:
        raise ValueError("Unknown distance metric")
    n = len(x)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            if metric == "jaccard":
                a = x[i] != 0
                b = x[j] != 0
                u = np.logical_or(a, b).sum()
                v = 1 - np.logical_and(a, b).sum() / u if u else 0.0
            else:
                a = np.linalg.norm(x[i])
                b = np.linalg.norm(x[j])
                v = (
                    1 - np.dot(x[i], x[j]) / (a * b)
                    if a and b
                    else (0.0 if not a and not b else 1.0)
                )
            d[i, j] = d[j, i] = float(np.clip(v, 0, 2))
    return d


def _rho(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    v = float(spearmanr(a, b).statistic)
    return v if math.isfinite(v) else None


def geometry_comparison(human, generated, *, metric="jaccard", bootstrap=500, seed=42):
    if human is None or generated is None:
        return unavailable("Verified item-level data unavailable")
    if len(human) != len(generated):
        raise ValueError("Geometry item order/length mismatch")
    n = len(human)
    if n < 4:
        return unavailable("At least four aligned items required")
    a, b = rdm(human, metric=metric), rdm(generated, metric=metric)
    pairs = np.triu_indices(n, 1)
    observed = _rho(a[pairs], b[pairs])
    if observed is None:
        return unavailable("Constant or insufficient item distances")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap):
        indices = rng.integers(0, n, n)
        # Resample items, not dependent pair entries. Exclude copies of the same original item.
        left, right = np.triu_indices(n, 1)
        keep = indices[left] != indices[right]
        values = _rho(
            a[indices[left[keep]], indices[right[keep]]],
            b[indices[left[keep]], indices[right[keep]]],
        )
        if values is not None:
            samples.append(values)
    return dict(
        available=True,
        spearman=observed,
        ci=np.quantile(samples, [0.025, 0.975]).tolist() if samples else None,
        bootstrap_unit="items_excluding_same_original_item",
        valid_resamples=len(samples),
        metric=metric,
        zero_vector_policy="both_empty=0;one_empty=1",
        caveat="Item pairs are dependent; small item-resample intervals are descriptive.",
    )


def human_recovery(generated, human, *, aligned_language, equivalences=None):
    if not aligned_language:
        raise ValueError("Aligned language and reviewed mapping required")
    equivalences = equivalences or {}

    def key(p):
        p = normalize_phrase(p)
        return equivalences.get(p, p)

    g = {key(x) for x in generated}
    h = {key(x) for x in human}
    matched = g & h
    return dict(
        precision_relative_to_lists=len(matched) / len(g) if g else None,
        recall_relative_to_lists=len(matched) / len(h) if h else None,
        matched=sorted(matched),
        novel_not_false=sorted(g - h),
        matching_rule="frozen_reviewed_equivalences_or_exact_NFC_v1",
        interpretation="Recovery of listed human productions, not truth or internal model use",
    )


def category_profile(counts, *, allowed_categories):
    if set(counts) - set(allowed_categories):
        raise ValueError("Unsupported source category")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in counts.values()):
        raise ValueError("Invalid category count")
    total = sum(float(v) for v in counts.values())
    return {k: float(v) / total if total else 0.0 for k, v in sorted(counts.items())}


def split_half(participant_records, *, seed=42, repeats=100):
    if not participant_records:
        return unavailable("Participant-level records unavailable")
    people = sorted({r["participant_id"] for r in participant_records})
    if len(people) < 4:
        return unavailable("At least four participants required")
    rng = random.Random(seed)
    scores = []
    for _ in range(repeats):
        shuffled = people[:]
        rng.shuffle(shuffled)
        a = set(shuffled[: len(people) // 2])
        left = {
            (r["item"], normalize_phrase(r["feature"]))
            for r in participant_records
            if r["participant_id"] in a
        }
        right = {
            (r["item"], normalize_phrase(r["feature"]))
            for r in participant_records
            if r["participant_id"] not in a
        }
        scores.append(len(left & right) / len(left | right) if left | right else 1.0)
    return dict(
        available=True,
        mean_jaccard=float(np.mean(scores)),
        participants=len(people),
        repeats=repeats,
    )


def discovery_curves(records, judgments=None):
    judgments = judgments or {}
    groups = {}
    for r in records:
        groups.setdefault((r["item_id"], r["prompt_family"]), []).append(r)
    rows = []
    for (item, family), group in sorted(groups.items()):
        seen = set()
        total = 0
        for r in sorted(group, key=lambda r: r["replicate"]):
            features = {normalize_phrase(f) for f in r["features"]}
            seen |= features
            total += len(r["features"])
            rows.append(
                dict(
                    item_id=item,
                    prompt_family=family,
                    replicate=r["replicate"],
                    unique_discovered=len(seen),
                    produced=total,
                    lexical_redundancy=1 - len(seen) / total if total else 0,
                    valid_unique=sum(judgments.get((item, p)) == 1 for p in seen),
                )
            )
    return rows

"""Weighted cosine geometry and concept-label permutation RSA."""

from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
from ..prompts import normalize_phrase

FAMILIES = ("P0", "P1", "P2")


def human_vectors(items, raw):
    if len({i["item_id"] for i in items}) != len(items):
        raise ValueError("Duplicate concept identity")
    matrix = []
    crosswalk = []
    for item in items:
        kind = item.get("selection_stratum")
        if kind not in {"actionN", "actionV", "object"}:
            raise ValueError("Published word type is required")
        candidates = [
            i
            for i in raw["items"]
            if i["item"] == item["source_text"] and i["stratum"] == kind
        ]
        if len(candidates) != 1:
            raise ValueError("Human word-plus-type identity is not unique")
        source = candidates[0]
        vector = raw["weights"][source["index"] - 1]
        if len(vector) != len(raw["features"]) or any(v < 0 for v in vector):
            raise ValueError("Invalid published frequency vector")
        matrix.append(vector)
        crosswalk.append(
            dict(
                item_id=item["item_id"],
                raw_index=source["index"],
                word=source["item"],
                word_type=kind,
            )
        )
    return dict(
        matrix=np.asarray(matrix, dtype=float),
        item_ids=[i["item_id"] for i in items],
        feature_labels=list(raw["features"]),
        crosswalk=crosswalk,
    )


def cosine_matrix(matrix):
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Cosine requires a finite two-dimensional representation")
    norms = np.linalg.norm(x, axis=1)
    denominator = np.outer(norms, norms)
    similarity = np.full((len(x), len(x)), np.nan)
    np.divide(x @ x.T, denominator, out=similarity, where=denominator > 0)
    similarity = np.clip(similarity, -1, 1)
    upper = similarity[np.triu_indices(len(x), 1)]
    valid = upper[np.isfinite(upper)]
    return dict(
        similarity=similarity,
        zero_norm_rows=np.flatnonzero(norms == 0).tolist(),
        defined_pairs=len(valid),
        undefined_pairs=len(upper) - len(valid),
        zero_offdiagonal_pairs=int((valid == 0).sum()),
        zero_offdiagonal_proportion=float((valid == 0).mean()) if len(valid) else None,
        zero_norm_policy="Undefined cosine is NaN, exported as ?; it is not zero similarity",
    )


def cosine_pair(left, right):
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (
        x.shape != y.shape
        or x.ndim != 1
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Aligned cosine vector mismatch")
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(np.dot(x, y) / d, -1, 1)) if d > 0 else None


def rsa_test(
    human,
    llm,
    human_ids,
    llm_ids,
    *,
    n_permutations=9999,
    seed=20261028,
    permutations=None,
    strata=None
):
    if list(human_ids) != list(llm_ids) or len(set(human_ids)) != len(human_ids):
        raise ValueError("RSA concept order/identity mismatch")
    n = len(human_ids)
    h = np.asarray(human, dtype=float)
    l = np.asarray(llm, dtype=float)
    if (
        h.shape != (n, n)
        or l.shape != (n, n)
        or not np.allclose(h, h.T, equal_nan=True)
        or not np.allclose(l, l.T, equal_nan=True)
    ):
        raise ValueError("RSA requires symmetric concept matrices")
    upper = np.triu_indices(n, 1)
    hv = h[upper]
    lv = l[upper]
    result = dict(
        available=False,
        rho=None,
        permutation_p_two_sided=None,
        n_concepts=n,
        n_pairs=len(hv),
        seed=seed,
        n_permutations=0,
        null_rhos=[],
        permutation_unit="concept labels, same permutation on both axes",
        null=(
            "within-original-word-type labels"
            if strata is not None
            else "all concept labels"
        ),
        alternative="two-sided absolute Spearman rho",
        tie_policy="average ranks",
    )
    if len(hv) < 3 or not np.isfinite(hv).all() or not np.isfinite(lv).all():
        return dict(
            result,
            reason="Undefined full-cohort similarities; no pairwise deletion or zero substitution",
        )
    hr = rankdata(hv) - np.mean(rankdata(hv))
    lr = rankdata(lv) - np.mean(rankdata(lv))
    denominator = np.linalg.norm(hr) * np.linalg.norm(lr)
    if denominator == 0:
        return dict(
            result,
            reason="Constant upper-triangle ranks; sparse geometry cannot identify an RSA coefficient",
        )
    rho = float(np.dot(hr, lr) / denominator)
    if permutations is None:
        if type(n_permutations) is not int or n_permutations < 1:
            raise ValueError("Positive permutation count required")
        rng = np.random.default_rng(seed)
        groups = (
            [np.arange(n)]
            if strata is None
            else [np.flatnonzero(np.array(strata) == g) for g in sorted(set(strata))]
        )
        if strata is not None and len(strata) != n:
            raise ValueError("Permutation strata mismatch")
        permutations = np.tile(np.arange(n), (n_permutations, 1))
        for p in permutations:
            for group in groups:
                p[group] = rng.permutation(group)
    permutations = np.asarray(permutations)
    if (
        permutations.ndim != 2
        or permutations.shape[1] != n
        or not len(permutations)
        or not np.issubdtype(permutations.dtype, np.integer)
        or not np.all(np.sort(permutations, axis=1) == np.arange(n))
    ):
        raise ValueError("Invalid concept-label permutation design")
    if strata is not None:
        groups = np.asarray(strata)
        if any(not np.array_equal(groups[p], groups) for p in permutations):
            raise ValueError("Permutation leaves its declared stratum")
    ranks = np.zeros((n, n))
    ranks[upper] = lr
    ranks[(upper[1], upper[0])] = lr
    null = []
    for start in range(0, len(permutations), 512):
        ps = permutations[start : start + 512]
        values = ranks[ps[:, upper[0]], ps[:, upper[1]]]
        null.extend(((values @ hr) / denominator).tolist())
    null = np.asarray(null)
    p = (1 + int((np.abs(null) >= abs(rho) - 1e-12).sum())) / (len(null) + 1)
    return dict(
        result,
        available=True,
        rho=rho,
        permutation_p_two_sided=p,
        n_permutations=len(null),
        null_rhos=null.tolist(),
        p_estimator="(1 + equally/extremely absolute-rho permutations) / (1 + permutations)",
    )


def discovery_vectors(
    items, bank, generations, jobs, *, replicates=3, comparison_cap=10
):
    if bank.get("experiment") != "human":
        raise ValueError("Human RSA requires the independent human feature bank")
    ii = {i["item_id"]: n for n, i in enumerate(items)}
    ff = {f["feature_id"]: n for n, f in enumerate(bank["features"])}
    if len(ii) != len(items):
        raise ValueError("Duplicate concept identity")
    origins = {
        (o["generation_job_id"], o["raw_feature_index"]): o for o in bank["origins"]
    }
    accepted = {}
    occurrences = []
    units = set()
    for g in generations:
        if g["item_id"] not in ii or g.get("parse_status") != "ok":
            continue
        if g["job_id"] in accepted:
            raise ValueError("Duplicate accepted generation job")
        accepted[g["job_id"]] = g
        for index, phrase in enumerate(
            g["features"][:comparison_cap]
            if comparison_cap is not None
            else g["features"]
        ):
            o = origins.get((g["job_id"], index))
            if o is None or o["normalized"] != normalize_phrase(phrase):
                raise ValueError("Human generation/feature-bank provenance mismatch")
            units.add(
                (g["item_id"], g["prompt_family"], g["replicate"], o["canonical_id"])
            )
            occurrences.append(
                dict(
                    item_id=g["item_id"],
                    prompt_family=g["prompt_family"],
                    replicate=g["replicate"],
                    generation_job_id=g["job_id"],
                    raw_feature_index=index,
                    feature_id=o["canonical_id"],
                    proposition=phrase,
                    source_record_hash=o["source_record_hash"],
                )
            )
    frequency = {f: np.zeros((len(ii), len(ff)), dtype=np.int16) for f in FAMILIES}
    for iid, family, rep, fid in units:
        if not 0 <= rep < replicates:
            raise ValueError("Unexpected repetition outside declared scope")
        frequency[family][ii[iid], ff[fid]] += 1
    coverage = []
    for item in items:
        for family in FAMILIES:
            planned = [
                j
                for j in jobs
                if j["item_id"] == item["item_id"] and j["prompt_family"] == family
            ]
            completed = [j for j in planned if j["job_id"] in accepted]
            complete = {j["replicate"] for j in planned} == set(
                range(replicates)
            ) and len(completed) == len(planned) == replicates
            coverage.append(
                dict(
                    item_id=item["item_id"],
                    prompt_family=family,
                    planned_calls=len(planned),
                    accepted_calls=len(completed),
                    complete=complete,
                    input_tokens=sum(
                        accepted[j["job_id"]].get("usage", {}).get("input_tokens", 0)
                        for j in completed
                    ),
                    output_tokens=sum(
                        accepted[j["job_id"]].get("usage", {}).get("output_tokens", 0)
                        for j in completed
                    ),
                )
            )
    return dict(
        item_ids=list(ii),
        feature_ids=list(ff),
        frequency=frequency,
        binary={f: (m > 0).astype(np.int8) for f, m in frequency.items()},
        occurrences=occurrences,
        generation_coverage=coverage,
        complete=all(r["complete"] for r in coverage),
        weighting="Number of independent repetitions containing a canonical feature, counted once per repetition; binary sensitivity uses any occurrence",
        comparison_cap_per_call=comparison_cap,
        replicates=replicates,
    )

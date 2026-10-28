"""Cosine RSA preserves concept identities and the dependence of pairwise data."""

import itertools
import numpy as np
import pytest
from scipy.stats import spearmanr


def test_raw_human_vectors_preserve_typed_words_order_vocabulary_and_weights():
    from feature_space_pilot.collaborator.geometry import human_vectors

    raw = dict(
        features=["move", "force", "sound"],
        items=[
            dict(index=1, item="PULL", stratum="actionN"),
            dict(index=2, item="PULL", stratum="actionV"),
            dict(index=3, item="BELL", stratum="object"),
        ],
        weights=[[2, 7, 0], [9, 3, 0], [0, 1, 8]],
    )
    items = [
        dict(item_id="v", source_text="PULL", selection_stratum="actionV"),
        dict(item_id="n", source_text="PULL", selection_stratum="actionN"),
    ]
    out = human_vectors(items, raw)
    assert out["matrix"].tolist() == [[9, 3, 0], [2, 7, 0]]
    assert out["item_ids"] == ["v", "n"] and out["feature_labels"] == raw["features"]
    assert [r["raw_index"] for r in out["crosswalk"]] == [2, 1]
    with pytest.raises(ValueError, match="type"):
        human_vectors([dict(item_id="x", source_text="PULL")], raw)


def test_cosine_handles_zero_vectors_and_reports_zero_offdiagonal_proportion():
    from feature_space_pilot.collaborator.geometry import cosine_matrix, cosine_pair

    out = cosine_matrix(np.array([[1, 0], [0, 2], [1, 1]], dtype=float))
    assert out["zero_offdiagonal_proportion"] == pytest.approx(1 / 3)
    assert out["similarity"][0, 2] == pytest.approx(1 / np.sqrt(2))
    empty = cosine_matrix(np.array([[0, 0], [1, 0]], dtype=float))
    assert empty["zero_norm_rows"] == [0] and np.isnan(empty["similarity"][0, 1])
    assert cosine_pair([0, 0], [1, 0]) is None
    assert np.isnan(cosine_matrix(np.empty((2, 0)))["similarity"]).all()


def test_rsa_requires_identical_concept_order_and_permutations_shuffle_both_axes():
    from feature_space_pilot.collaborator.geometry import rsa_test

    h = np.array(
        [[1, 0.9, 0.1, 0.4], [0.9, 1, 0.2, 0.3], [0.1, 0.2, 1, 0.8], [0.4, 0.3, 0.8, 1]]
    )
    l = np.array(
        [[1, 0.6, 0.2, 0.4], [0.6, 1, 0.1, 0.3], [0.2, 0.1, 1, 0.9], [0.4, 0.3, 0.9, 1]]
    )
    ids = ["a", "b", "c", "d"]
    perms = np.array(list(itertools.permutations(range(4))))
    out = rsa_test(h, l, ids, ids, permutations=perms)
    tri = np.triu_indices(4, 1)
    expected = np.array(
        [spearmanr(h[tri], l[np.ix_(p, p)][tri]).statistic for p in perms]
    )
    observed = spearmanr(h[tri], l[tri]).statistic
    assert out["rho"] == pytest.approx(observed)
    assert out["null_rhos"] == pytest.approx(expected)
    assert out["permutation_p_two_sided"] == pytest.approx(
        (1 + sum(abs(expected) >= abs(observed) - 1e-12)) / (len(perms) + 1)
    )
    with pytest.raises(ValueError, match="order"):
        rsa_test(h, l, ids, ids[::-1], n_permutations=5)
    with pytest.raises(ValueError, match="permutation"):
        rsa_test(h, l, ids, ids, permutations=np.array([[0, 0, 2, 3]]))


def test_constant_similarity_and_missing_norms_do_not_create_a_numeric_rsa():
    from feature_space_pilot.collaborator.geometry import rsa_test, cosine_matrix

    x = cosine_matrix(np.eye(4))["similarity"]
    out = rsa_test(x, x, list("abcd"), list("abcd"), n_permutations=9)
    assert (
        not out["available"]
        and out["rho"] is None
        and out["permutation_p_two_sided"] is None
    )


def test_llm_generation_weights_are_repetition_counts_not_raw_paraphrase_counts():
    from feature_space_pilot.collaborator.geometry import discovery_vectors
    from feature_space_pilot.full.bank import build_bank
    from test_experiment_banks import item, generation

    items = [item("h", "event", source_dataset="vinson_2008")]
    gs = [
        generation("h", ["requires accuracy", "needs accuracy"], f, n)
        for f in ("P0", "P1", "P2")
        for n in range(3)
    ]
    bank = dict(build_bank(gs), experiment="human")
    jobs = [
        {k: g[k] for k in ("item_id", "job_id", "prompt_family", "replicate")}
        for g in gs
    ]
    out = discovery_vectors(items, bank, gs, jobs, replicates=3, comparison_cap=10)
    assert all(m.tolist() == [[3]] for m in out["frequency"].values())
    assert all(m.tolist() == [[1]] for m in out["binary"].values())
    assert out["complete"] and len(out["generation_coverage"]) == 3
    with pytest.raises(ValueError, match="human"):
        discovery_vectors(items, dict(bank, experiment="task"), gs, jobs)

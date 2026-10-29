"""Latest applicability geometry must preserve scope, provenance and unknowns."""

import itertools
import numpy as np
import pytest


def fixture():
    items = [dict(item_id=i, target_type="object") for i in "abcd"]
    bank = dict(
        experiment="human",
        features=[dict(feature_id=f"h{j}", text=f"property {j}") for j in range(3)],
        origins=[
            dict(item_id=i, canonical_id=f"h{j}", prompt_family=f)
            for i in "abcd"
            for j in range(3)
            for f in ("P0", "P1", "P2")
        ],
    )
    return items, bank


def test_latest_matrix_reorders_by_explicit_identity_and_rejects_task_features():
    from feature_space_pilot.human_refresh import select_representation

    items, bank = fixture()
    x = np.array([[1, 0, -1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    selected = select_representation(
        items,
        bank,
        x[::-1, ::-1],
        list("dcba"),
        ["h2", "h1", "h0"],
        list("abcd"),
        "P0",
        origin_scope="all_human",
    )
    assert selected["matrix"].tolist() == x.tolist()
    assert selected["feature_ids"] == ["h0", "h1", "h2"]
    with pytest.raises(ValueError, match="human"):
        select_representation(
            items,
            dict(bank, experiment="task"),
            x,
            list("abcd"),
            ["h0", "h1", "h2"],
            list("abcd"),
            "P0",
            origin_scope="all_human",
        )
    with pytest.raises(ValueError, match="identity"):
        select_representation(
            items,
            bank,
            x,
            list("abcd"),
            ["h0", "h1", "foreign"],
            list("abcd"),
            "P0",
            origin_scope="all_human",
        )
    with pytest.raises(ValueError, match="binary"):
        select_representation(
            items,
            bank,
            x + 1,
            list("abcd"),
            ["h0", "h1", "h2"],
            list("abcd"),
            "P0",
            origin_scope="all_human",
        )


def test_prompt_and_vinson_only_pools_use_generation_provenance_not_judge_positives():
    from feature_space_pilot.human_refresh import select_representation

    items, bank = fixture()
    bank["origins"] = [
        dict(item_id="a", canonical_id="h0", prompt_family="P0"),
        dict(item_id="d", canonical_id="h1", prompt_family="P0"),
        dict(item_id="a", canonical_id="h2", prompt_family="P2"),
    ]
    x = np.ones((4, 3), dtype=int)
    common = (items, bank, x, list("abcd"), ["h0", "h1", "h2"], list("abc"), "P0")
    assert select_representation(*common, origin_scope="all_human")["feature_ids"] == [
        "h0",
        "h1",
    ]
    assert select_representation(*common, origin_scope="vinson_only")[
        "feature_ids"
    ] == ["h0"]


def test_complete_columns_are_one_fixed_space_and_zero_norms_remain_undefined():
    from feature_space_pilot.human_refresh import judged_geometry

    x = np.array([[1, -1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1]])
    result = judged_geometry(x, method="complete_columns")
    assert result["retained_column_indices"] == [0, 2]
    assert result["similarity"][0, 2] == pytest.approx(1)
    assert result["n_candidate_features"] == 3 and result["n_retained_features"] == 2
    empty = judged_geometry(np.array([[1, -1], [-1, 1]]), method="complete_columns")
    assert empty["n_retained_features"] == 0
    assert np.isnan(empty["similarity"]).all()
    assert empty["zero_norm_rows"] == [0, 1]


def test_jointly_observed_cosine_never_turns_unknown_into_zero():
    from feature_space_pilot.human_refresh import judged_geometry

    x = np.array([[1, -1, 1], [1, 1, -1], [0, 1, 0]])
    result = judged_geometry(x, method="jointly_observed")
    assert result["similarity"][0, 1] == pytest.approx(1)
    assert result["shared_known_counts"][0, 1] == 1
    assert np.isnan(result["similarity"][0, 2])  # shared observations, zero second norm
    assert not result["common_coordinate_space"]
    assert result["unknown_is_zero"] is False
    assert np.array_equal(x, [[1, -1, 1], [1, 1, -1], [0, 1, 0]])


def test_judgment_rsa_uses_existing_dependency_preserving_permutations():
    from feature_space_pilot.human_refresh import judged_geometry
    from feature_space_pilot.collaborator.geometry import rsa_test
    from scipy.stats import spearmanr

    x = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1]])
    similarity = judged_geometry(x, method="jointly_observed")["similarity"]
    human = np.array(
        [[1, 0.8, 0.2, 0.1], [0.8, 1, 0.3, 0.4], [0.2, 0.3, 1, 0.7], [0.1, 0.4, 0.7, 1]]
    )
    ps = np.array(list(itertools.permutations(range(4))))
    result = rsa_test(human, similarity, list("abcd"), list("abcd"), permutations=ps)
    tri = np.triu_indices(4, 1)
    expected = [
        spearmanr(human[tri], similarity[np.ix_(p, p)][tri]).statistic for p in ps
    ]
    assert result["null_rhos"] == pytest.approx(expected)
    assert result["permutation_unit"] == "concept labels, same permutation on both axes"


def test_sealed_export_validation_rejects_changed_outputs_without_writing(tmp_path):
    from feature_space_pilot.human_refresh import verify_artifacts
    from feature_space_pilot.io import file_hash
    import json

    data = tmp_path / "result.csv"
    data.write_text("actual observations\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(artifacts={"result.csv": file_hash(data)}))
    )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert verify_artifacts(tmp_path)["verified"]
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    data.write_text("modified\n")
    with pytest.raises(ValueError, match="hash"):
        verify_artifacts(tmp_path)


def test_full_output_semantic_alignment_coverage_does_not_call_unscored_unmatched():
    from feature_space_pilot.human_refresh import alignment_coverage

    full = {"P0": np.array([[2, 1], [0, 1]])}
    scored = {"P0": np.array([[1, 1], [0, 0]])}
    row = alignment_coverage(full, scored)[0]
    assert row["full_generation_units"] == 4
    assert row["units_in_existing_alignment_scope"] == 2
    assert row["additional_unscored_units"] == 2
    assert row["additional_units_are_unmatched"] is False
    with pytest.raises(ValueError, match="scope"):
        alignment_coverage(scored, full)


def test_observability_is_a_separate_mask_diagnostic_not_zero_filled_features():
    from feature_space_pilot.human_refresh import observability_geometry

    x = np.array([[1, -1, 0], [0, 1, -1], [1, 0, 1]])
    swapped = np.where(x >= 0, 1 - x, -1)
    result = observability_geometry(x)
    assert result["similarity"] == pytest.approx(
        observability_geometry(swapped)["similarity"]
    )
    assert result["similarity"][0, 1] == pytest.approx(0.5)
    assert result["mask_is_applicability"] is False


def test_refresh_output_guard_protects_historical_exports_and_matching_sources(
    tmp_path,
):
    from feature_space_pilot.human_refresh import validate_output_root

    originals = {tmp_path / name for name in ["source", "old_export", "matching"]}
    for original in originals:
        with pytest.raises(ValueError, match="separate"):
            validate_output_root(original / "new", originals)
    validate_output_root(tmp_path / "analysis", originals)

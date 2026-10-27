import pytest
import numpy as np
from feature_space_pilot.metrics import (
    rdm,
    geometry_comparison,
    human_recovery,
    category_profile,
    split_half,
)
from feature_space_pilot.composition import compare_composition


def test_geometry_zero_vectors_and_item_resampling():
    x = np.array([[0, 0], [0, 0], [1, 0], [0, 1], [1, 1]])
    d = rdm(x, metric="cosine")
    assert np.isfinite(d).all() and d[0, 1] == 0 and d[0, 2] == 1
    result = geometry_comparison(x, x, bootstrap=50, seed=42)
    assert result["spearman"] == pytest.approx(1)
    assert result["bootstrap_unit"] == "items_excluding_same_original_item"
    assert result["ci"][0] == pytest.approx(1)
    assert geometry_comparison(None, x)["available"] is False
    assert geometry_comparison(x[:2], x[:2])["available"] is False


def test_recovery_is_list_relative_and_translation_gated():
    r = human_recovery(
        ["has fur", "barks", "new valid property"],
        ["has fur", "barks"],
        aligned_language=True,
    )
    assert r["precision_relative_to_lists"] == pytest.approx(2 / 3)
    assert r["recall_relative_to_lists"] == 1
    assert r["novel_not_false"] == ["new valid property"]
    with pytest.raises(ValueError, match="language"):
        human_recovery(["a"], ["b"], aligned_language=False)
    assert category_profile(
        {"social": 2, "internal": 2}, allowed_categories={"social", "internal"}
    ) == {"internal": 0.5, "social": 0.5}
    with pytest.raises(ValueError, match="category"):
        category_profile({"invented": 1}, allowed_categories={"social"})
    assert split_half(None)["available"] is False


def test_composition_separates_discovery_applicability_and_audited_interactions():
    Gt = {"base"}
    Gs = {"deadline"}
    Gd = {"deadline", "conflict", "invalid"}
    judgments = {
        "T": {"base": 1, "deadline": 0, "conflict": 0, "invalid": 0},
        "S": {"base": 0, "deadline": 1, "conflict": 0, "invalid": 0},
        "TS": {"base": 1, "deadline": 1, "conflict": 1, "invalid": 0},
    }
    r = compare_composition(
        Gt, Gs, Gd, judgments, component_calls=10, direct_calls=10, relational_audit={}
    )
    assert r["newly_discovered"] == ["conflict", "invalid"]
    assert r["incremental_applicable"] == ["conflict"]
    assert (
        r["interaction_candidates"] == ["conflict"]
        and r["confirmed_interactions"] == []
    )
    r = compare_composition(
        Gt,
        Gs,
        Gd,
        judgments,
        component_calls=10,
        direct_calls=10,
        relational_audit={"conflict": True},
    )
    assert r["confirmed_interactions"] == ["conflict"]
    with pytest.raises(ValueError, match="budget"):
        compare_composition(Gt, Gs, Gd, judgments, component_calls=20, direct_calls=10)
    with pytest.raises(ValueError, match="missing"):
        compare_composition(
            Gt,
            Gs,
            Gd,
            {"T": {}, "S": {}, "TS": {}},
            component_calls=10,
            direct_calls=10,
        )


def test_source_category_coding_can_be_reviewed_without_inventing_labels(tmp_path):
    import csv
    from feature_space_pilot.engine import demo_config
    from feature_space_pilot.fixtures import demo_human_items
    from feature_space_pilot.io import file_hash
    from feature_space_pilot.report import _human_analysis

    c = demo_config(tmp_path)
    item = demo_human_items()[0]
    path = tmp_path / "coding.csv"
    path.write_text(
        "item_id,feature_id,category,reviewer\n"
        + item["item_id"]
        + ",f,concept,reviewer-1\n"
    )
    c.update(category_coding_file=str(path), category_coding_sha256=file_hash(path))
    generations = [
        dict(
            item_id=item["item_id"],
            prompt_family="P1",
            features=["describes a synthetic abstract concept"],
        )
    ]
    bank = [dict(feature_id="f", text="describes a synthetic abstract concept")]
    judgments = [dict(item_id=item["item_id"], feature_id="f", status="ok", applies=1)]
    result = _human_analysis(c, [item], generations, bank, judgments)
    assert result["category_profiles"][0]["generated_profiles"]["P1"] == {
        "concept": 1.0
    }
    path.write_text(path.read_text().replace(",concept,", ",invented,"))
    c["category_coding_sha256"] = file_hash(path)
    with pytest.raises(ValueError, match="category"):
        _human_analysis(c, [item], generations, bank, judgments)

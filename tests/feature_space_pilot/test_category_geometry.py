import math
import pytest


def test_category_geometry_is_euclidean_proportions_and_never_binary_features():
    from feature_space_pilot.metrics import category_geometry

    h = {f"i{n}": dict(SM=v, VA=1 - v) for n, v in enumerate([0, 0.25, 0.5, 1])}
    coverage = {i: dict(coded_applicable_features=4, applicable_features=4) for i in h}
    result = category_geometry(h, h, coverage, bootstrap=10, seed=42)
    assert result["available"] and result["metric"] == "euclidean_category_proportions"
    assert result["spearman"] == pytest.approx(1)
    assert result["human_rdm"][0][2] == pytest.approx(math.sqrt(0.5))
    assert (
        result["item_ids"] == sorted(h)
        and result["representation"] == "reviewed_category_proportions"
    )
    coverage["i0"]["coded_applicable_features"] = 1
    assert not category_geometry(h, h, coverage)["available"]
    assert not category_geometry(h, {}, coverage)["available"]

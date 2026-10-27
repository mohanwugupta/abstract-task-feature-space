import json
import os
import pytest
from feature_space_pilot.references import load_reference, select_stimuli
from feature_space_pilot.io import file_hash


def source(tmp_path, **changes):
    p = tmp_path / "norms.csv"
    p.write_text(
        "item,feature,language,category,frequency\nFreude,fühlt sich gut an,de,internal,2\n"
    )
    p.chmod(0o444)
    r = dict(
        dataset_id="verified",
        kind="human",
        path=str(p),
        sha256=file_hash(p),
        language="de",
        data_level="item_features",
        columns={
            "item": "item",
            "feature": "feature",
            "language": "language",
            "category": "category",
            "frequency": "frequency",
        },
        participants=10,
        instructions="Verified protocol.",
        license="Research-use verified",
        url="https://example.org/source",
        codebook="https://example.org/codebook",
        verified=True,
    )
    return dict(r, **changes)


def test_reference_gate_provenance_language_and_synthetic(tmp_path):
    r = source(tmp_path)
    assert load_reference(r, live=True)["rows"][0]["feature"] == "fühlt sich gut an"
    for changes, message in [
        ({"kind": "synthetic"}, "synthetic"),
        ({"sha256": "bad"}, "hash"),
        ({"columns": {"item": "absent", "feature": "feature"}}, "columns"),
        ({"language": "en"}, "language"),
        ({"verified": False}, "verified"),
    ]:
        with pytest.raises(ValueError, match=message):
            load_reference(dict(r, **changes), live=True)
    with pytest.raises(ValueError, match="translation"):
        load_reference(r, live=True, model_language="en")
    with pytest.raises(ValueError, match="Missing"):
        load_reference(dict(r, path=str(tmp_path / "missing")), live=True)


def test_reference_requires_read_only_and_real_data(tmp_path):
    r = source(tmp_path)
    os.chmod(r["path"], 0o644)
    with pytest.raises(ValueError, match="read.only"):
        load_reference(r, live=True)


def test_selection_is_label_blind_stratified_and_frozen():
    rows = [
        dict(item=f"word{i}", stratum="a" if i < 5 else "b", feature="secret")
        for i in range(10)
    ]
    a = select_stimuli(rows, {"a": 3, "b": 3}, seed=42)
    b = select_stimuli(list(reversed(rows)), {"a": 3, "b": 3}, seed=42)
    assert a == b and len(a) == 6
    assert sum(x["split"] == "held_out" for x in a) == 2
    with pytest.raises(ValueError, match="coverage"):
        select_stimuli(rows, {"a": 9}, seed=42)

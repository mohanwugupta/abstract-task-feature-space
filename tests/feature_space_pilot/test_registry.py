import copy
import json

import pytest

from feature_space_pilot.registry import load_items, validate_items, seal_item


def item(item_id="t", target_type="task_instance", **kw):
    return seal_item(
        dict(
            item_id=item_id,
            text="An isolated target.",
            target_type=target_type,
            level=target_type,
            parent_id=None,
            task_id=None,
            situation_ids=[],
            source_dataset="synthetic",
            language="en",
            selection_stratum="fixture",
            translation_status="original",
            **kw
        )
    )


def test_registry_rejects_duplicate_ids_missing_parents_and_wrong_axis():
    t = item()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_items([t, t])
    for changes, message in [
        ({"parent_id": "missing"}, "parent"),
        ({"situation_ids": ["missing"]}, "situation"),
        ({"target_type": "fourth_level"}, "target_type"),
    ]:
        x = dict(t, **changes)
        with pytest.raises(ValueError, match=message):
            validate_items([seal_item(x)])


def test_registry_rejects_stale_hash_and_cycles():
    t = item()
    with pytest.raises(ValueError, match="hash"):
        validate_items([dict(t, text="changed")])
    with pytest.raises(ValueError, match="parent|cycle"):
        validate_items([seal_item(dict(t, parent_id="t"))])


def test_lfs_pointer_is_actionable(tmp_path):
    path = tmp_path / "stimuli.jsonl"
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:x\nsize 10\n"
    )
    with pytest.raises(ValueError, match="Git LFS.*git lfs pull"):
        load_items(path)


def test_task_registry_has_15_targets_and_synthetic_reconciliation():
    from feature_space_pilot.fixtures import (
        task_items,
        reconciliation_fixture,
        reconcile,
    )

    items = task_items()
    validate_items(items)
    assert len(items) == 15
    assert len({x["item_id"] for x in items}) == 15
    assert sum(x["target_type"] == "situated_task" for x in items) == 6
    assert sum(x["target_type"] == "task_instance" for x in items) == 2
    f = reconciliation_fixture()
    assert f["synthetic"] is True
    assert len(f["invoices"]) == 20
    assert reconcile(f) == f["reference_report"]
    assert {r["status"] for r in reconcile(f)} == {
        "matched",
        "missing",
        "partial",
        "duplicate",
    }
    for x in items:
        if x["task_id"] == "finance" and x["target_type"] in {
            "task_instance",
            "situated_task",
        }:
            assert "INV-020" in x["text"] and "PAY-001" in x["text"]

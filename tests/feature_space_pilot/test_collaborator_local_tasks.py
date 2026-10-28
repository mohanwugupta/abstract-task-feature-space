"""Local generation licenses judging; absence licenses no applicability inference."""

import copy
from feature_space_pilot.io import digest
from feature_space_pilot.full.bank import build_bank
from feature_space_pilot.experiments.banks import separate_banks
from test_experiment_banks import item, generation


def fixture():
    items = [
        item("poem_T", "task_instance", task_id="poem"),
        item("finance_T", "task_instance", task_id="finance"),
    ]
    gs = [
        generation("poem_T", ["requires accuracy", "uses rhyme"]),
        generation("finance_T", ["requires accuracy", "matches payments"]),
    ]
    bank = separate_banks(items, gs, build_bank(gs))["task"]
    cells = [
        dict(
            item_id=i["item_id"],
            feature_id=f["feature_id"],
            applies="1",
            status="reused_binary",
            source_record_hashes=[digest([i["item_id"], f["feature_id"]])],
            old_feature_ids=["provider-old-id"],
        )
        for i in items
        for f in bank["features"]
    ]
    return items, gs, bank, cells


def test_primary_scope_contains_only_independently_discovered_target_features():
    from feature_space_pilot.collaborator.tasks import local_inventories

    items, gs, bank, cells = fixture()
    before = copy.deepcopy((items, gs, bank, cells))
    out = local_inventories(items, bank, gs, cells)
    assert out["primary_cell_count"] == 4 and out["historical_cross_product_cells"] == 6
    target = out["targets"]["poem_T"]
    assert {f["text"] for f in target["bank"]["features"]} == {
        "requires accuracy",
        "uses rhyme",
    }
    assert all(j["applies"] == 1 for j in target["judgments"])
    assert out["targets"]["finance_T"]["bank"]["bank_id"] != target["bank"]["bank_id"]
    assert (items, gs, bank, cells) == before and out["new_requests"] == 0


def test_unknown_local_judgments_remain_unknown_and_existing_results_are_reused():
    from feature_space_pilot.collaborator.tasks import local_inventories

    items, gs, bank, cells = fixture()
    for r in cells:
        if r["item_id"] == "poem_T":
            r.update(applies="?", status="ambiguous")
    out = local_inventories(items, bank, gs, cells)
    assert all(
        j["applies"] is None and j["status"] == "ambiguous"
        for j in out["targets"]["poem_T"]["judgments"]
    )
    assert out["reused_binary_cells"] == 2 and out["new_requests"] == 0


def test_missing_historical_evidence_is_not_reclassified_as_a_failed_response():
    from feature_space_pilot.collaborator.tasks import local_inventories

    items, gs, bank, cells = fixture()
    for r in cells:
        r.update(applies="?", status="missing")
    out = local_inventories(items, bank, gs, cells)
    assert all(j["status"] == "missing" for j in out["judgments"])


def test_composition_is_discovery_only_and_never_infers_zero_zero_one():
    from feature_space_pilot.collaborator.tasks import (
        local_inventories,
        discovery_compositions,
    )

    items = [
        item("poem_T", "task_instance", task_id="poem"),
        item("S_deadline", "situation"),
        item(
            "poem_S_deadline",
            "situated_task",
            task_id="poem",
            parent_id="poem_T",
            situation_ids=["S_deadline"],
        ),
    ]
    gs = [
        generation(
            i["item_id"],
            (
                ["base property"]
                if i["item_id"] != "poem_S_deadline"
                else ["novel combined property"]
            ),
            f,
            n,
        )
        for i in items
        for f in ("P0", "P1", "P2")
        for n in range(3)
    ]
    bank = separate_banks(items, gs, build_bank(gs))["task"]
    local = local_inventories(items, bank, gs, [])
    out = discovery_compositions(items, local, gs, replicates=3)
    assert len(out["comparisons"]) == 6
    assert all(r["direct_only_count"] == 1 for r in out["comparisons"])
    assert all(
        r["interaction_status"] == "not_established_by_discovery"
        for r in out["comparisons"]
    )
    assert all(
        r["T_applicability"] is None and r["S_applicability"] is None
        for r in out["feature_rows"]
        if r["membership"] == "direct_only"
    )
    assert out["new_requests"] == 0


def test_local_reconsolidation_cannot_erase_a_shared_original_discovery():
    from feature_space_pilot.collaborator.tasks import (
        local_inventories,
        discovery_compositions,
    )
    from test_experiment_banks import review

    items = [
        item("poem_T", "task_instance", task_id="poem"),
        item("S_deadline", "situation"),
        item(
            "poem_S_deadline",
            "situated_task",
            task_id="poem",
            parent_id="poem_T",
            situation_ids=["S_deadline"],
        ),
    ]
    gs = [
        generation(
            i["item_id"],
            ["alpha", "beta"] if i["item_id"] == "poem_S_deadline" else ["alpha"],
            f,
            n,
        )
        for i in items
        for f in ("P0", "P1", "P2")
        for n in range(3)
    ]
    bank = separate_banks(items, gs, build_bank(gs))["task"]
    bank["review_decisions"].append(review("alpha", "beta"))
    local = local_inventories(items, bank, gs, [])
    assert len(local["targets"]["poem_S_deadline"]["bank"]["features"]) == 1
    out = discovery_compositions(items, local, gs)
    assert all(r["shared_count"] >= 1 for r in out["comparisons"])

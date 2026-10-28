"""The scientific unit of discovery is an experiment, never the pooled bank."""

import copy
import pytest
from feature_space_pilot.io import digest
from feature_space_pilot.full.bank import build_bank


def item(iid, kind, text=None, **extra):
    text = text or iid
    return dict(
        item_id=iid,
        target_type=kind,
        text=text,
        language="en",
        content_hash=digest(text),
        **extra,
    )


def generation(iid, phrases, family="P0", rep=0):
    human = iid.startswith("h")
    return dict(
        item_id=iid,
        features=phrases,
        parse_status="ok",
        source_run_id="human_source" if human else "task_source",
        generation_model="fixture-generator",
        prompt_family=family,
        replicate=rep,
        attempt=0,
        job_id=digest([iid, family, rep]),
        record_hash=digest([iid, phrases, family, rep]),
    )


def review(a, b):
    return dict(
        left=a,
        right=b,
        decision="merge",
        reviewer="fixture-reviewer",
        reason="Synthetic direct semantic-equivalence evidence",
    )


def test_banks_separate_provenance_even_for_independently_discovered_same_text():
    from feature_space_pilot.experiments.banks import separate_banks

    items = [item("h", "concept"), item("t", "task_instance")]
    gs = [
        generation("h", ["is warm", "requires accuracy"]),
        generation("t", ["requires accuracy", "supports inspection"]),
    ]
    pooled = build_bank(gs)
    before = copy.deepcopy((items, gs, pooled))
    banks = separate_banks(items, gs, pooled)
    h, t = banks["human"], banks["task"]
    assert {f["text"] for f in t["features"]} == {
        "requires accuracy",
        "supports inspection",
    }
    assert {o["item_id"] for o in h["origins"]} == {"h"}
    assert {o["item_id"] for o in t["origins"]} == {"t"}
    assert h["bank_id"] != t["bank_id"] and h["inventory_hash"] != t["inventory_hash"]
    assert not (
        {f["feature_id"] for f in h["features"]}
        & {f["feature_id"] for f in t["features"]}
    )
    assert all(o["source_record_hash"] and o["generation_model"] for o in t["origins"])
    assert (items, gs, pooled) == before


def test_consolidation_uses_local_direct_evidence_never_human_bridge():
    from feature_space_pilot.experiments.banks import separate_banks

    gs = [generation("h", ["beta"]), generation("t", ["alpha", "gamma"])]
    pooled = build_bank(gs, reviews=[review("alpha", "beta"), review("beta", "gamma")])
    out = separate_banks([item("h", "concept"), item("t", "task_instance")], gs, pooled)
    assert len(out["task"]["features"]) == 2
    assert len(out["task"]["consolidation_transfer_audit"]) == 2
    assert all(
        r["status"] == "not_local_pair"
        for r in out["task"]["consolidation_transfer_audit"]
    )
    gs[1]["features"].append("delta")
    pooled = build_bank(gs, reviews=[review("alpha", "delta")])
    assert (
        len(
            separate_banks(
                [item("h", "concept"), item("t", "task_instance")], gs, pooled
            )["task"]["features"]
        )
        == 2
    )


@pytest.mark.parametrize(
    "a,b",
    [
        ("causes harm", "does not cause harm"),
        ("user rewards assistant", "assistant rewards user"),
        ("requires checking if asked", "requires checking"),
        ("can cause harm", "must cause harm"),
        ("requires 5 minutes", "requires 10 minutes"),
    ],
)
def test_unsafe_pooled_decisions_are_flagged_without_changing_meaning(a, b):
    from feature_space_pilot.experiments.banks import separate_banks

    gs = [generation("t", [a, b])]
    pooled = build_bank(gs)
    pooled["review_decisions"] = [review(a, b)]
    bank = separate_banks([item("t", "task_instance")], gs, pooled)["task"]
    assert len(bank["features"]) == 2
    assert bank["consolidation_transfer_audit"][0]["status"] == "unsafe_review"


def test_local_reconstruction_is_uncapped_and_deterministic():
    from feature_space_pilot.experiments.banks import separate_banks

    gs = [generation("t", [f"distinct operation {n}" for n in range(220)])]
    p = build_bank(gs)
    a = separate_banks([item("t", "task_instance")], gs, p)["task"]
    b = separate_banks([item("t", "task_instance")], list(reversed(gs)), p)["task"]
    assert len(a["features"]) == 220 and a["feature_cap"] is None
    assert a["inventory_hash"] == b["inventory_hash"]


def test_declared_60_and_15_item_experiments_have_independent_cross_products():
    from feature_space_pilot.experiments.banks import separate_banks
    from feature_space_pilot.experiments.evidence import reconstruct_matrix
    from test_experiment_reuse import SETTINGS

    items = [item(f"h{n}", "concept") for n in range(60)] + [
        item(f"t{n}", "task_instance") for n in range(15)
    ]
    gs = [
        generation(
            i["item_id"],
            [
                "independently discovered common property",
                i["item_id"] + " specific property",
            ],
        )
        for i in items
    ]
    banks = separate_banks(items, gs, build_bank(gs))
    h = reconstruct_matrix(items[:60], banks["human"], [], SETTINGS)
    t = reconstruct_matrix(items[60:], banks["task"], [], SETTINGS)
    assert h["state"]["matrix"].shape == (60, 61)
    assert t["state"]["matrix"].shape == (15, 16)
    assert (t["state"]["matrix"] == -1).all()


def test_bank_hash_binds_its_local_consolidation_evidence_only():
    from feature_space_pilot.experiments.banks import separate_banks

    items = [item("h", "concept"), item("t", "task_instance")]
    gs = [generation("h", ["alpha", "beta"]), generation("t", ["task property"])]
    r = dict(review("alpha", "beta"), decision="related")
    pooled = build_bank(gs, reviews=[r])
    a = separate_banks(items, gs, pooled)
    pooled["review_decisions"][0]["reason"] = "Different retained evidence"
    b = separate_banks(items, gs, pooled)
    assert a["human"]["inventory_hash"] != b["human"]["inventory_hash"]
    assert a["task"]["inventory_hash"] == b["task"]["inventory_hash"]

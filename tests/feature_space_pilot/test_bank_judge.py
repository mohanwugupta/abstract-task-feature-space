import json
import pytest
from feature_space_pilot.bank import consolidate
from feature_space_pilot.judge import judge_prompt, parse_judgment, build_matrix
from feature_space_pilot.fixtures import task_items


def records():
    return [
        dict(
            job_id="j1",
            item_id="t",
            prompt_family="P1",
            replicate=0,
            features=[
                "User checks records",
                "user checks records.",
                "User does not check records",
                "Agent checks records",
                "User may check records",
                "User checks records if reviewed",
            ],
        )
    ]


def test_bank_conservative_stable_and_provenance():
    bank, candidates, review = consolidate(records(), cap=200)
    assert len(bank) == 5 and len(candidates) == 6
    assert len(bank[0]["inventory_hash"]) == 64
    assert bank == consolidate(records(), cap=200)[0]
    assert all(c["job_id"] == "j1" for c in candidates)
    assert len({c["feature_id"] for c in candidates}) == 5
    assert review
    assert len(consolidate(records(), cap=2)[0]) == 2
    assert sum(c["selected"] for c in consolidate(records(), cap=2)[1]) >= 2
    with pytest.raises(ValueError, match="role|negation|modality|condition"):
        consolidate(
            records(),
            merges=[
                dict(
                    left="User checks records",
                    right="User does not check records",
                    reviewer="human",
                    decision="merge",
                )
            ],
        )


def test_binary_judge_isolated_strict_and_missing_never_zero():
    item = task_items()[0]
    item["human_features"] = "SECRET"
    feature = dict(feature_id="f", text="has a deadline", origins=["SECRET"])
    p = judge_prompt(item, feature)
    assert "SECRET" not in json.dumps(p)
    good = dict(
        item_id=item["item_id"],
        feature_id="f",
        applies=1,
        status="ok",
        evidence="Explicit deadline",
    )
    assert parse_judgment(json.dumps(good), item["item_id"], "f")["applies"] == 1
    for v in [2, 4, 0.5, True, "1", None]:
        with pytest.raises(ValueError):
            parse_judgment(json.dumps(dict(good, applies=v)), item["item_id"], "f")
    ambiguous = dict(good, applies=None, status="ambiguous")
    with pytest.raises(ValueError):
        parse_judgment(json.dumps(ambiguous), item["item_id"], "f")
    assert (
        parse_judgment(
            json.dumps(ambiguous), item["item_id"], "f", rubric_version="binary_v1"
        )["status"]
        == "ambiguous"
    )
    with pytest.raises(ValueError, match="incomplete"):
        build_matrix(["a"], ["f"], [])
    with pytest.raises(ValueError, match="Duplicate"):
        build_matrix(
            ["a"],
            ["f"],
            [dict(item_id="a", feature_id="f", applies=1, status="ok")] * 2,
        )
    assert build_matrix(
        ["a"], ["f"], [dict(item_id="a", feature_id="f", applies=1, status="ok")]
    ) == [[1]]

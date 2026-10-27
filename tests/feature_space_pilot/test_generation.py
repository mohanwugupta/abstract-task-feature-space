import json
import pytest
from feature_space_pilot.fixtures import demo_human_items, task_items
from feature_space_pilot.prompts import generation_prompt, parse_generation
from feature_space_pilot.planner import plan_generation, pending_jobs

SETTINGS = dict(
    model_id="frontier-snapshot",
    revision="pinned",
    supports_seed=True,
    temperature=0.8,
    max_output_tokens=500,
)


def test_prompts_are_isolated_and_stable():
    item = demo_human_items()[0]
    item.update(
        human_features=["NEVER_LEAK"],
        selection_stratum="NEVER_LEAK",
        source_dataset="NEVER_LEAK",
    )
    for family in ["P0", "P1", "P2"]:
        p = generation_prompt(item, family)
        assert "NEVER_LEAK" not in json.dumps(p)
        assert p == generation_prompt(item, family)
        assert json.loads(p["user"]) == {
            "item_id": item["item_id"],
            "text": item["text"],
            "language": "en",
        }
    p = generation_prompt(
        dict(item, source_dataset="harpaintner_2018", language="de"), "P0"
    )
    assert p["cap"] == 4 and p["protocol"] == "harpaintner_2018_adaptation"
    with pytest.raises(ValueError, match="P0"):
        generation_prompt(dict(item, source_dataset="vinson_2008"), "P0", live=True)


def test_independent_job_plan_2250_paired_seed_and_resume():
    items = demo_human_items() + task_items()
    jobs = plan_generation(items, SETTINGS, replicates=10, seed=71)
    assert len(jobs) == 2250 and len({j["job_id"] for j in jobs}) == 2250
    assert jobs == plan_generation(items, SETTINGS, replicates=10, seed=71)
    paired = {}
    for j in jobs:
        paired.setdefault((j["item_id"], j["replicate"]), set()).add(j["seed"])
    assert all(len(v) == 1 for v in paired.values())
    assert (
        pending_jobs(jobs, [{"job_id": jobs[0]["job_id"], "parse_status": "ok"}])
        == jobs[1:]
    )
    assert (
        len(
            pending_jobs(jobs, [{"job_id": jobs[0]["job_id"], "parse_status": "error"}])
        )
        == 2250
    )
    changed = plan_generation(
        items, dict(SETTINGS, temperature=0.2), replicates=10, seed=71
    )
    assert not set(j["job_id"] for j in changed) & set(j["job_id"] for j in jobs)


@pytest.mark.parametrize(
    "raw",
    [
        "bad",
        '{"item_id":"other","features":[]}',
        '{"item_id":"x","features":["a","a"]}',
        '{"item_id":"x","features":["requires rhyme and has a deadline"]}',
        '{"item_id":"x","features":[1]}',
        '{"item_id":"x","features":[],"extra":1}',
    ],
)
def test_parser_rejects_malformed_wrong_item_redundant_and_bundled(raw):
    with pytest.raises(ValueError):
        parse_generation(raw, "x", cap=10)


def test_parser_preserves_relations_empty_and_caps():
    assert (
        parse_generation('{"item_id":"x","features":[]}', "x", cap=4)["features"] == []
    )
    f = "meeting the deadline can conflict with checking every discrepancy"
    assert parse_generation(json.dumps(dict(item_id="x", features=[f])), "x", cap=4)[
        "features"
    ] == [f]
    with pytest.raises(ValueError, match="cap"):
        parse_generation(json.dumps(dict(item_id="x", features=["a", "b"])), "x", cap=1)

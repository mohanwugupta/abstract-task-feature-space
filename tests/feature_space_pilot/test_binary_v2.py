"""A binary decision is required; historical abstentions retain their old meaning."""

import json
from pathlib import Path
import pytest


def response(value=1, status="ok"):
    return json.dumps(
        dict(
            item_id="h",
            feature_id="f",
            applies=value,
            status=status,
            evidence="Ordinary semantic characterization.",
        )
    )


def test_future_prompt_and_schema_require_a_numeric_binary_decision():
    from feature_space_pilot.judge import judge_prompt, RUBRIC
    from feature_space_pilot.clients import judgment_schema

    prompt = judge_prompt(
        dict(item_id="h", text="the ascent", language="en"),
        dict(feature_id="f", text="reduces willingness to eat"),
    )
    assert RUBRIC == prompt["rubric_version"] == "binary_v2"
    assert "ordinary semantic knowledge" in prompt["system"]
    assert "invented context" in prompt["system"]
    assert "Do not abstain" in prompt["system"]
    assert "status=ambiguous" not in prompt["system"]
    assert "ascent" not in prompt["system"] and "cow" not in prompt["system"]
    schema = judgment_schema("h", "f")
    assert schema["properties"]["applies"] == dict(type="integer", enum=[0, 1])
    assert schema["properties"]["status"]["enum"] == ["ok"]


def test_future_parser_rejects_abstentions_and_noninteger_labels():
    from feature_space_pilot.judge import parse_judgment

    for value in [0, 1]:
        assert parse_judgment(response(value), "h", "f")["applies"] == value
    for value in [None, True, 1.0, 4, "1"]:
        with pytest.raises(ValueError):
            parse_judgment(response(value), "h", "f")
    with pytest.raises(ValueError):
        parse_judgment(response(None, "ambiguous"), "h", "f")


def test_legacy_prompt_schema_and_parser_remain_explicitly_available():
    from feature_space_pilot.judge import judge_prompt, parse_judgment
    from feature_space_pilot.clients import judgment_schema

    prompt = judge_prompt(
        dict(item_id="h", text="the ascent", language="en"),
        dict(feature_id="f", text="has direction"),
        rubric_version="binary_v1",
    )
    source = Path("feature_space_pilot/prompts/binary_v1.txt").read_text().strip()
    assert prompt["system"] == source and prompt["rubric_version"] == "binary_v1"
    assert (
        None
        in judgment_schema("h", "f", rubric_version="binary_v1")["properties"][
            "applies"
        ]["enum"]
    )
    assert (
        parse_judgment(
            response(None, "ambiguous"), "h", "f", rubric_version="binary_v1"
        )["applies"]
        is None
    )


def test_frozen_configs_without_a_rubric_are_legacy_and_future_defaults_are_v2():
    from feature_space_pilot.judge import rubric_for
    from feature_space_pilot.engine import load_config, demo_config

    assert rubric_for({}) == "binary_v1"
    assert (
        rubric_for(load_config("configs/feature_space_pilot.json")["judge"])
        == "binary_v2"
    )
    assert rubric_for(demo_config("/tmp/fixture")["judge"]) == "binary_v2"
    with pytest.raises(ValueError):
        rubric_for(dict(rubric_version="unknown"))


def test_request_identity_and_schema_change_with_the_rubric():
    from feature_space_pilot.full.reuse import cell_job
    from feature_space_pilot.io import digest

    settings = dict(
        provider="fixture",
        model_id="fixture",
        max_output_tokens=400,
        supports_structured_outputs=True,
    )
    item = dict(
        item_id="h", text="the ascent", language="en", content_hash=digest("the ascent")
    )
    feature = dict(feature_id="f", text="has direction")
    old = cell_job(item, feature, settings)
    new = cell_job(item, feature, dict(settings, rubric_version="binary_v2"))
    assert old["rubric_version"] == "binary_v1" and new["rubric_version"] == "binary_v2"
    assert (
        old["job_id"] != new["job_id"]
        and old["payload_sha256"] != new["payload_sha256"]
    )
    assert old["payload"]["response_format"]["json_schema"]["schema"]["properties"][
        "applies"
    ]["enum"] == [0, 1, None]
    assert new["payload"]["response_format"]["json_schema"]["schema"]["properties"][
        "applies"
    ]["enum"] == [0, 1]


def test_legacy_binary_is_not_misrepresented_as_same_protocol_v2_evidence():
    from test_experiment_reuse import fixture, SETTINGS
    from feature_space_pilot.experiments.evidence import reconstruct_matrix

    items, _, banks, source = fixture()
    result = reconstruct_matrix(
        [items[0]], banks["task"], [source], dict(SETTINGS, rubric_version="binary_v2")
    )
    assert result["audit_summary"]["reused_binary_cells"] == 0
    assert source["records"][0]["rubric_version"] == "binary_v1"


def test_newly_loaded_configs_default_to_v2_even_when_the_flag_is_omitted(tmp_path):
    from feature_space_pilot.engine import load_config

    raw = json.loads(Path("configs/feature_space_pilot.json").read_text())
    raw["judge"].pop("rubric_version")
    raw["root"] = str(Path.cwd())
    p = tmp_path / "fresh.json"
    p.write_text(json.dumps(raw))
    assert load_config(p)["judge"]["rubric_version"] == "binary_v2"

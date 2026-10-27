import copy
import csv
import json
from pathlib import Path

import pytest

from feature_space_pilot.engine import load_config, selected_items
from feature_space_pilot.io import file_hash
from feature_space_pilot.prompts import generation_prompt, parse_generation
from feature_space_pilot.clients import generation_schema

ROOT = Path(__file__).resolve().parents[2]


def test_workbook_crosswalk_checks_every_selected_row_and_declares_chain_of_custody():
    from feature_space_pilot.source_adapters import verify_harpaintner_workbook

    result = verify_harpaintner_workbook(
        ROOT / "Data Sheet 1.xlsx",
        ROOT / "human_norm_pilot_abstract_24.csv",
        expected_sha256="a0d94ee950f5dcdb5bd753cb9050ddeda4a96d8493bc63772113bc31a39af293",
    )
    assert result["workbook_items"] == 296 and result["matched_items"] == 24
    assert result["discrepancies"] == [] and not result["publisher_download_verified"]
    assert result["matches"][0]["source_excel_row"] >= 5
    with pytest.raises(ValueError, match="hash"):
        verify_harpaintner_workbook(
            ROOT / "Data Sheet 1.xlsx",
            ROOT / "human_norm_pilot_abstract_24.csv",
            expected_sha256="wrong",
        )


def test_workbook_crosswalk_rejects_a_changed_source_row(tmp_path):
    from feature_space_pilot.source_adapters import verify_harpaintner_workbook

    with (ROOT / "human_norm_pilot_abstract_24.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))
    rows[0]["human_sm"] = "0.5"
    p = tmp_path / "summary.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="workbook|crosswalk"):
        verify_harpaintner_workbook(
            ROOT / "Data Sheet 1.xlsx",
            p,
            expected_sha256=file_hash(ROOT / "Data Sheet 1.xlsx"),
        )


def test_english_primary_separates_generation_from_pending_translation_comparisons():
    from feature_space_pilot.scientific import scientific_audit

    config = load_config(ROOT / "configs/feature_space_pilot.json")
    result = scientific_audit(config)
    assert result["generation_ready"] and not result["blockers"]
    assert result["primary_language_condition"] == "published_english_exploratory"
    assert result["pending_translation_reviews"] == 24
    assert not result["abstract_comparison_ready"]
    items = selected_items(config, "human")
    assert len(items) == 60 and all(i["language"] == "en" for i in items)
    abstract = next(i for i in items if i["item_id"] == "hp18_row0154")
    assert abstract["text"] == "creativity" and abstract["source_text"] == "Kreativität"
    assert abstract["english_text"] == "creativity"
    assert abstract["translation_status"] != "reviewed"
    p = generation_prompt(abstract, "P0", live=True, protocols=config["protocols"])
    assert p["cap"] == 4 and p["protocol"] == "harpaintner_2018_english_adaptation_v1"


def test_vinson_protocol_is_uncapped_at_source_and_uses_published_syntax_markers():
    config = load_config(ROOT / "configs/feature_space_pilot.json")
    items = selected_items(config, "human")
    noun = next(i for i in items if i["item_id"] == "vv_actionN_pull")
    verb = next(i for i in items if i["item_id"] == "vv_actionV_pull")
    assert noun["text"] == "the pull" and verb["text"] == "to pull"
    assert noun["source_text"] == verb["source_text"] == "PULL"
    prompt = generation_prompt(verb, "P0", live=True, protocols=config["protocols"])
    assert (
        prompt["cap"] is None
        and prompt["protocol"] == "vinson_2008_appendix_adaptation_v1"
    )
    assert (
        "maxItems"
        not in generation_schema(verb["item_id"], None)["properties"]["features"]
    )
    text = json.dumps(
        dict(item_id=verb["item_id"], features=[f"property {n}" for n in range(12)])
    )
    assert len(parse_generation(text, verb["item_id"], cap=None)["features"]) == 12
    p = config["protocols"]["vinson_2008"]
    assert p["source_feature_cap"] is None and p["analysis_comparison_cap"] == 10
    assert "dictionary" in prompt["system"] and "association" in prompt["system"]


def test_missing_or_changed_raw_source_never_passes_verified_flag(tmp_path):
    from feature_space_pilot.references import load_reference

    registry = json.loads(
        (ROOT / "data/feature_space_pilot/source_registry.json").read_text()
    )
    spec = next(s for s in registry["sources"] if s["dataset_id"] == "vinson_2008")
    spec = dict(
        spec,
        verified=True,
        path=str(ROOT / spec["path"]),
        raw_path=str(tmp_path / "absent.zip"),
    )
    with pytest.raises(ValueError, match="raw|Raw|materialized"):
        load_reference(spec, live=True)


def test_pending_or_rejected_translation_reviews_cannot_enable_comparison(tmp_path):
    from feature_space_pilot.scientific import load_translation_reviews

    source = ROOT / "pilot_abstract_translation_review.csv"
    review = load_translation_reviews(source, expected_sha256=file_hash(source))
    assert len(review) == 24 and all(not r["approved"] for r in review.values())
    with source.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0].update(decision="approved", reviewed_english="creativity", reviewer="")
    p = tmp_path / "reviews.csv"
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="reviewer"):
        load_translation_reviews(p, expected_sha256=file_hash(p))


def test_explicit_german_condition_keeps_separate_original_p0_route():
    from feature_space_pilot.scientific import scientific_audit

    config = load_config(ROOT / "configs/feature_space_pilot.json")
    config["human_language_condition"] = "german_primary"
    item = next(
        i for i in selected_items(config, "human") if i["item_id"] == "hp18_row0154"
    )
    assert (
        item["text"] == "Kreativität"
        and item["language"] == "de"
        and item["english_text"] == "creativity"
    )
    prompt = generation_prompt(item, "P0", live=True, protocols=config["protocols"])
    assert prompt["protocol"] == "harpaintner_2018_adaptation" and prompt["cap"] == 4
    result = scientific_audit(config)
    assert result["generation_ready"] and result["abstract_comparison_ready"]

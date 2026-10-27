"""Uploaded human inventories integrate without promoting unreviewed evidence."""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from feature_space_pilot.engine import load_config, preflight, selected_items
from feature_space_pilot.io import file_hash
from feature_space_pilot.prompts import generation_prompt
from feature_space_pilot.references import load_reference
from feature_space_pilot.registry import load_items

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "human_norm_pilot_60_stimuli.csv"
ABSTRACT = ROOT / "human_norm_pilot_abstract_24.csv"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, records):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def test_default_csv_inventory_preserves_labels_provenance_and_existing_splits():
    config = load_config(ROOT / "configs/feature_space_pilot.json")
    items = selected_items(config, "human")
    assert config["human_stimuli"].endswith(".csv")
    assert len(items) == 60 and len({i["item_id"] for i in items}) == 60
    assert Counter(i["target_type"] for i in items) == dict(
        concept=24, event=24, object=12
    )
    by_id = {i["item_id"]: i for i in items}
    for old in load_items(ROOT / "data/feature_space_pilot/stimuli_human.jsonl"):
        current = by_id[old["item_id"]]
        assert (current["text"], current["split"]) == (old["text"], old["split"])
    example = by_id["hp18_row0154"]
    assert example["text"] == "creativity" and example["source_text"] == "Kreativität"
    assert example["language"] == "en" and example["source_language"] == "de"
    assert example["source_dataset"] == "harpaintner_2018"
    assert example["source_dataset_original"] == "harpaintner_2018_original_supplement"
    assert example["source_excel_row"] == 154
    assert example["source_file_sha256"] == rows(ABSTRACT)[0]["source_file_sha256"]
    assert (
        example["translation_status"]
        == "published_loose_english_translation_not_independently_reviewed"
    )
    assert example["source_split"] == "unassigned"
    assert not any(k.startswith("human_") for k in example)
    for family in ["P0", "P1", "P2"]:
        prompt = generation_prompt(example, family)
        assert json.loads(prompt["user"]) == dict(
            item_id=example["item_id"], text="creativity", language="en"
        )
        assert (
            "0.8133" not in json.dumps(prompt) and "sensorimotor" not in prompt["user"]
        )


def test_unassigned_abstract_holdouts_are_deterministic_without_resampling(tmp_path):
    from feature_space_pilot.human_inputs import load_human_csv

    original = load_human_csv(INVENTORY, abstract_path=ABSTRACT, seed=42)
    reversed_path = tmp_path / "reordered.csv"
    write_rows(reversed_path, list(reversed(rows(INVENTORY))))
    reordered = load_human_csv(reversed_path, abstract_path=ABSTRACT, seed=42)
    assert [(i["item_id"], i["split"]) for i in original] == [
        (i["item_id"], i["split"]) for i in reordered
    ]
    abstract = [i for i in original if i["target_type"] == "concept"]
    assert Counter(i["split"] for i in abstract) == dict(development=20, held_out=4)
    assert Counter(
        i["selection_stratum"] for i in abstract if i["split"] == "held_out"
    ) == {
        s: 1
        for s in [
            "sensorimotor",
            "internal_state_emotion",
            "social_constellation",
            "verbal_association",
        ]
    }


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("duplicate", "Duplicate"),
        ("summary_disagrees", "disagree"),
        ("nan", "proportion"),
        ("negative", "proportion"),
        ("sum", "sum"),
        ("class", "class"),
        ("split", "split"),
        ("source", "source"),
        ("source_label", "Duplicate"),
        ("translated_label", "Duplicate"),
    ],
)
def test_csv_validation_rejects_unsafe_inventory_and_summary_joins(
    tmp_path, mutation, message
):
    from feature_space_pilot.human_inputs import load_human_csv

    inventory, summary = rows(INVENTORY), rows(ABSTRACT)
    if mutation == "duplicate":
        inventory.append(inventory[0])
    elif mutation == "summary_disagrees":
        summary[0]["item_text"] = "different label"
    elif mutation == "nan":
        inventory[-1]["human_sm"] = "nan"
    elif mutation == "negative":
        inventory[-1]["human_sm"] = "-0.1"
    elif mutation == "sum":
        inventory[-1]["human_sm"] = "0.99"
    elif mutation == "class":
        inventory[0]["item_class"] = "unsupported"
    elif mutation == "split":
        inventory[0]["split"] = "anything"
    elif mutation == "source":
        inventory[-1]["source_dataset"] = "unknown_source"
    elif mutation == "source_label":
        inventory[-1]["original_german"] = inventory[-2]["original_german"]
    elif mutation == "translated_label":
        inventory[-1]["item_text"] = inventory[-2]["item_text"]
    p, q = tmp_path / "inventory.csv", tmp_path / "summary.csv"
    write_rows(p, inventory)
    write_rows(q, summary)
    with pytest.raises(ValueError, match=message):
        load_human_csv(p, abstract_path=q)


def test_default_pins_both_csv_inputs_and_rejects_changes(tmp_path):
    config = load_config(ROOT / "configs/feature_space_pilot.json")
    assert config["human_stimuli_sha256"] == file_hash(INVENTORY)
    assert config["human_abstract_summary_sha256"] == file_hash(ABSTRACT)
    p = tmp_path / "changed.csv"
    p.write_bytes(INVENTORY.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        selected_items(dict(config, human_stimuli=str(p)), "human")
    p.write_bytes(ABSTRACT.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        selected_items(dict(config, human_abstract_summary=str(p)), "human")


def test_category_adapter_preserves_rounded_proportions_without_inventing_features(
    tmp_path,
):
    spec = next(
        s
        for s in json.loads(
            (ROOT / "data/feature_space_pilot/source_registry.json").read_text()
        )["sources"]
        if s["dataset_id"] == "harpaintner_2018"
    )
    assert spec["adapter"] == "human_pilot_category_shares_v1"
    p = tmp_path / "summary.csv"
    p.write_bytes(ABSTRACT.read_bytes())
    p.chmod(0o444)
    ref = load_reference(dict(spec, path=str(p)), live=False)
    assert (
        ref["data_level"] == "category_counts"
        and ref["measurement"] == "rounded_category_proportions"
    )
    assert len(ref["rows"]) == 96 and {r["category"] for r in ref["rows"]} == {
        "SM",
        "IS/E",
        "SC",
        "VA",
    }
    observed = {
        r["category"]: float(r["count"])
        for r in ref["rows"]
        if r["item"] == "Kreativität"
    }
    assert observed == dict(SM=0.8133, **{"IS/E": 0.1, "SC": 0.0, "VA": 0.0867})
    assert all("feature" not in r and r["language"] == "de" for r in ref["rows"])
    with pytest.raises(ValueError, match="verified"):
        load_reference(dict(spec, path=str(p)), live=True)


def test_preflight_plans_all_csv_targets_and_discloses_remaining_scientific_gates():
    plan = preflight(load_config(ROOT / "configs/feature_space_pilot.json"), "human")
    assert (
        plan["selected_item_count"] == 60
        and plan["actual_plannable_generation_jobs"] == 540
    )
    assert not any("sample is not frozen" in b for b in plan["blockers"])
    assert any("reviewed translation" in b for b in plan["blockers"])
    assert any("P0" in b for b in plan["blockers"])
    assert any("category" in w.lower() for w in plan["warnings"])
    assert plan["input_hashes"]["human_stimuli"] == file_hash(INVENTORY)
    assert plan["input_hashes"]["human_abstract_summary"] == file_hash(ABSTRACT)
    assert not plan["live_ready"]
    item = next(
        i
        for i in selected_items(
            load_config(ROOT / "configs/feature_space_pilot.json"), "human"
        )
        if i["target_type"] == "concept"
    )
    assert "awaiting" in generation_prompt(item, "P0")["protocol"]
    with pytest.raises(ValueError, match="P0"):
        generation_prompt(item, "P0", live=True)


@pytest.mark.parametrize("reviewed", [False, True])
def test_category_reports_require_reviewed_item_alignment_and_preserve_availability(
    tmp_path, reviewed
):
    from feature_space_pilot.report import _human_analysis

    config = load_config(ROOT / "configs/feature_space_pilot.json")
    item = next(
        i for i in selected_items(config, "human") if i["item_id"] == "hp18_row0154"
    )
    spec = next(
        s
        for s in json.loads(
            (ROOT / "data/feature_space_pilot/source_registry.json").read_text()
        )["sources"]
        if s["dataset_id"] == "harpaintner_2018"
    )
    p = tmp_path / "reference.csv"
    p.write_bytes(ABSTRACT.read_bytes())
    p.chmod(0o444)
    # Only this isolated test fixture is considered verified; production stays gated.
    spec = dict(spec, path=str(p), verified=True, translation_reviewed=reviewed)
    config.pop("source_registry")
    config["sources"] = [spec]
    if reviewed:
        item = dict(item, translation_status="reviewed")
    result = _human_analysis(config, [item], [], [], [])
    assert result["item_availability"][0]["available"] is reviewed
    assert all(not r.get("available", True) for r in result["recovery"])
    assert all(not r["available"] for r in result["geometry"])
    if reviewed:
        assert result["category_profiles"][0]["human_profile"]["SM"] == pytest.approx(
            0.8133
        )
        assert all(
            not v["available"]
            for v in result["category_profiles"][0]["generated_profiles"].values()
        )
    else:
        assert result["category_profiles"] == []
        assert "translation" in result["item_availability"][0]["reason"]


def test_input_audit_cli_exports_normalized_data_offline_without_mutating_csvs(
    tmp_path,
):
    import subprocess
    import sys

    before = {p.name: file_hash(p) for p in [INVENTORY, ABSTRACT]}
    output = tmp_path / "human-audit"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_space_pilot.human_inputs",
            "--config",
            "configs/feature_space_pilot.json",
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert (
        result["n_items"] == 60
        and result["n_abstract"] == 24
        and result["n_category_rows"] == 96
    )
    assert (
        result["n_unreviewed_translations"] == 24
        and not result["scientific_source_verified"]
    )
    assert len(load_items(output / "stimuli.jsonl")) == 60
    with (output / "abstract_category_shares.csv").open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 96
    assert {p.name: file_hash(p) for p in [INVENTORY, ABSTRACT]} == before
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_space_pilot.human_inputs",
            "--config",
            "configs/feature_space_pilot.json",
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert again.returncode == 0 and json.loads(again.stdout) == result


def test_reviewed_mapping_must_match_exact_stimulus_text():
    from feature_space_pilot.references import validate_item_alignment

    item = dict(
        item_id="x",
        text="creativity",
        source_text="Kreativität",
        language="en",
        translation_status="reviewed",
    )
    ref = dict(
        language="de",
        rows=[dict(item="Kreativität")],
        translation_reviewed=True,
        translation_mapping={"Kreativität": "different meaning"},
    )
    with pytest.raises(ValueError, match="exact reviewed translation mapping"):
        validate_item_alignment(item, ref)


def test_reviewed_category_csv_alignment_is_shared_by_explorer(viz_run_path, tmp_path):
    from dataclasses import replace
    from feature_space_pilot.viz.load import load_run
    from feature_space_pilot.viz.derive import human_sources

    config = load_config(ROOT / "configs/feature_space_pilot.json")
    item = next(
        i for i in selected_items(config, "human") if i["item_id"] == "hp18_row0154"
    )
    spec = next(
        s
        for s in json.loads(
            (ROOT / "data/feature_space_pilot/source_registry.json").read_text()
        )["sources"]
        if s["dataset_id"] == "harpaintner_2018"
    )
    p = tmp_path / "reference.csv"
    p.write_bytes(ABSTRACT.read_bytes())
    p.chmod(0o444)
    spec = dict(spec, path=str(p), verified=True, translation_reviewed=True)
    run = load_run(viz_run_path)
    toy = replace(
        run,
        manifest=dict(run.manifest, demo=False),
        items={item["item_id"]: dict(item, translation_status="reviewed")},
        sources=(spec,),
    )
    assert human_sources(toy)[0]["available"]
    toy = replace(toy, items={item["item_id"]: item})
    assert not human_sources(toy)[0]["available"]


def test_uploaded_csv_hashes_survive_git_checkout_without_line_ending_conversion():
    import subprocess

    for path in [INVENTORY, ABSTRACT]:
        filtered = subprocess.run(
            ["git", "hash-object", "--path", path.name, str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        raw = subprocess.run(
            ["git", "hash-object", "--no-filters", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert (
            filtered.stdout == raw.stdout
        ), "Git must preserve uploaded CSV bytes used by pinned SHA-256 values"

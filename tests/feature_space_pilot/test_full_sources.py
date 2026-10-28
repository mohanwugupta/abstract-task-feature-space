"""Read-only source checkpoints for the expanded pilot."""

import json
from pathlib import Path
import pytest


def source_fixture(root, name="source", *, text="the pencil", complete=True):
    from feature_space_pilot.io import (
        write_json,
        append_jsonl,
        canonical,
        digest,
        file_hash,
    )
    from feature_space_pilot.engine import load_config

    p = root / name
    p.mkdir(parents=True)
    c = load_config("configs/feature_space_pilot.json")
    c["demo"] = True
    item = dict(
        item_id=name,
        text=text,
        language="en",
        content_hash=digest(text),
        target_type="object",
        level="concept",
        parent_id=None,
        task_id=None,
        situation_ids=[],
        source_dataset="fixture",
    )
    job = dict(
        job_id="g_" + name, item_id=name, prompt_family="P0", replicate=0, cap=None
    )
    write_json(p / "frozen_config.json", c)
    append_jsonl(p / "stimuli.jsonl", item)
    append_jsonl(p / "generation_jobs.jsonl", job)
    if complete:
        raw = canonical(
            dict(item_id=name, features=["requires accuracy", "needs accuracy"])
        )
        r = dict(
            job_id=job["job_id"],
            item_id=name,
            prompt_family="P0",
            replicate=0,
            attempt=0,
            parse_status="ok",
            features=["requires accuracy", "needs accuracy"],
            raw_response=raw,
            response_hash=digest(raw),
            usage=dict(input_tokens=100, output_tokens=50),
        )
        r["record_hash"] = digest(r)
        append_jsonl(p / "generations.jsonl", r)
    m = dict(
        run_id=name,
        stage="human",
        demo=True,
        status="partial",
        planned_generation_jobs=1,
        completed_generation_jobs=int(complete),
        model_snapshots=dict(generator=c["generator"], judge=c["judge"]),
        artifacts={q.name: file_hash(q) for q in p.iterdir()},
        journal_checkpoints={},
    )
    if complete:
        q = p / "generations.jsonl"
        m["journal_checkpoints"][q.name] = dict(
            bytes=q.stat().st_size, sha256=file_hash(q)
        )
    write_json(p / "manifest.json", m)
    return p


def test_snapshot_uses_sealed_journal_prefix_without_mutating_source(tmp_path):
    from feature_space_pilot.full.sources import load_source

    source = source_fixture(tmp_path)
    path = source / "generations.jsonl"
    with path.open("a") as f:
        f.write('{"incomplete_append":')
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    snapshot = load_source(source, allow_synthetic=True)
    assert len(snapshot["generations"]) == 1 and snapshot["generations_complete"]
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_source_tampering_and_synthetic_live_inputs_are_rejected(tmp_path):
    from feature_space_pilot.full.sources import load_source

    p = source_fixture(tmp_path)
    with pytest.raises(ValueError, match="synthetic"):
        load_source(p)
    (p / "stimuli.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash"):
        load_source(p, allow_synthetic=True)


def test_global_inventory_keeps_all_items_and_flags_unfinished_generations(tmp_path):
    from feature_space_pilot.full.sources import collect_sources

    a = source_fixture(tmp_path, "human")
    b = source_fixture(tmp_path, "task", complete=False)
    result = collect_sources([a, b], allow_synthetic=True, expected_items=2)
    assert len(result["items"]) == 2 and len(result["generations"]) == 1
    assert result["generations_complete"] is False
    assert (
        result["coverage"]["task"]["completed"] == 0
        and result["coverage"]["task"]["planned"] == 1
    )


def test_duplicate_source_alias_is_not_counted_twice(tmp_path):
    from feature_space_pilot.full.sources import collect_sources

    p = source_fixture(tmp_path)
    result = collect_sources([p, p], allow_synthetic=True, expected_items=1)
    assert len(result["generations"]) == 1 and len(result["items"]) == 1


def test_global_order_preserves_frozen_source_order_for_sampling_allocations(tmp_path):
    from feature_space_pilot.full.sources import collect_sources

    z = source_fixture(tmp_path, "z")
    a = source_fixture(tmp_path, "a")
    result = collect_sources([z, a], allow_synthetic=True, expected_items=2)
    assert [i["item_id"] for i in result["items"]] == ["z", "a"]


def test_available_checkpoint_freezes_every_accepted_feature_and_explicit_missing_jobs(
    tmp_path,
):
    from feature_space_pilot.full.sources import (
        collect_sources,
        available_checkpoint,
        apply_checkpoint,
    )
    from feature_space_pilot.io import append_jsonl, digest, file_hash, write_json

    a = source_fixture(tmp_path, "human")
    b = source_fixture(tmp_path, "task", complete=False)
    source = collect_sources([a, b], allow_synthetic=True, expected_items=2)
    checkpoint = available_checkpoint(source)
    assert len(checkpoint["missing_generation_jobs"]) == 1
    assert checkpoint["missing_generation_jobs"][0]["job_id"] == "g_task"
    # A later successful source generation must not change the frozen vocabulary.
    r = dict(source["generations"][0], job_id="g_task", item_id="task")
    r.pop("source_run_id")
    r.pop("generation_model")
    r.pop("record_hash")
    r["raw_response"] = json.dumps(dict(item_id="task", features=r["features"]))
    r["response_hash"] = digest(r["raw_response"])
    r["record_hash"] = digest(r)
    append_jsonl(b / "generations.jsonl", r)
    m = json.loads((b / "manifest.json").read_text())
    m["artifacts"]["generations.jsonl"] = file_hash(b / "generations.jsonl")
    m["journal_checkpoints"]["generations.jsonl"] = dict(
        bytes=(b / "generations.jsonl").stat().st_size,
        sha256=file_hash(b / "generations.jsonl"),
    )
    write_json(b / "manifest.json", m)
    live = collect_sources([a, b], allow_synthetic=True, expected_items=2)
    frozen = apply_checkpoint(live, checkpoint)
    assert frozen["generation_scope_closed"] is True
    assert frozen["generations_complete"] is False
    assert (
        len(frozen["generations"]) == 1 and frozen["coverage"]["task"]["completed"] == 0
    )
    assert frozen["missing_generation_jobs"] == checkpoint["missing_generation_jobs"]
    checkpoint["generations"][0]["features"] = ["invented data"]
    with pytest.raises(ValueError, match="checkpoint"):
        apply_checkpoint(live, checkpoint)

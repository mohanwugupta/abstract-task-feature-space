import csv
import json
import shutil
from pathlib import Path

import pytest

from feature_space_pilot.clients import FakeClient
from feature_space_pilot.engine import demo_config, run_pipeline
from feature_space_pilot.io import file_hash, write_csv, write_json
from feature_space_pilot.report import seal_manifest


@pytest.fixture
def viz_run_path(tmp_path):
    c = demo_config(tmp_path)
    c["replicates"] = 2
    result = run_pipeline(c, "tasks", client=FakeClient())
    return Path(result["output_dir"])


def test_adapter_is_read_only_and_traces_provenance(viz_run_path):
    from feature_space_pilot.viz.load import load_run, CellState

    before = {p.name: file_hash(p) for p in viz_run_path.iterdir() if p.is_file()}
    run = load_run(viz_run_path)
    assert run.demo and len(run.task_item_ids) == 15
    assert run.cell("poem_T", next(iter(run.features))) in (
        CellState.APPLIES,
        CellState.INAPPLICABLE,
    )
    feature = next(
        f for f in run.features if run.cell("poem_T", f) == CellState.APPLIES
    )
    detail = run.inspect(feature)
    assert detail["text"] == run.features[feature]["text"]
    assert detail["origins"] and detail["judgments"]
    assert all(
        "raw_phrase" in r and "replicate" in r and "prompt_family" in r
        for r in detail["origins"]
    )
    assert {
        p.name: file_hash(p) for p in viz_run_path.iterdir() if p.is_file()
    } == before
    with pytest.raises(ValueError, match="Unknown"):
        run.cell("invented", feature)


def test_adapter_rejects_mutation_duplicate_cells_and_unknown_ids(viz_run_path):
    from feature_space_pilot.viz.load import load_run

    bank = viz_run_path / "feature_bank.csv"
    original = bank.read_bytes()
    bank.write_text("mutated")
    with pytest.raises(ValueError, match="hash"):
        load_run(viz_run_path)
    bank.write_bytes(original)
    judgments = viz_run_path / "binary_judgments.csv"
    rows = list(csv.DictReader(judgments.open()))
    fields = list(rows[0])
    write_csv(judgments, rows + [rows[0]], fields)
    seal_manifest(viz_run_path)
    with pytest.raises(ValueError, match="Duplicate"):
        load_run(viz_run_path)
    rows[0]["feature_id"] = "unknown"
    write_csv(judgments, rows, fields)
    seal_manifest(viz_run_path)
    with pytest.raises(ValueError, match="Unknown"):
        load_run(viz_run_path)


def test_adapter_preserves_four_states_and_requires_scope(viz_run_path):
    from feature_space_pilot.viz.load import load_run, CellState

    path = viz_run_path / "binary_judgments.csv"
    rows = list(csv.DictReader(path.open()))
    fields = list(rows[0])
    missing = rows.pop(0)
    excluded = rows.pop(0)
    write_csv(path, rows, fields)
    (viz_run_path / "binary_matrix.csv").unlink()
    m = json.loads((viz_run_path / "manifest.json").read_text())
    m["status"] = "needs_adjudication"
    write_json(viz_run_path / "manifest.json", m)
    scope = json.loads((viz_run_path / "matrix_scope.json").read_text())
    scope["excluded_combinations"] = [[excluded["item_id"], excluded["feature_id"]]]
    write_json(viz_run_path / "matrix_scope.json", scope)
    seal_manifest(viz_run_path)
    run = load_run(viz_run_path)
    assert run.cell(missing["item_id"], missing["feature_id"]) == CellState.UNRESOLVED
    assert (
        run.cell(excluded["item_id"], excluded["feature_id"]) == CellState.OUT_OF_SCOPE
    )
    assert {run.cell(r["item_id"], r["feature_id"]) for r in rows} == {
        CellState.APPLIES,
        CellState.INAPPLICABLE,
    }
    (viz_run_path / "matrix_scope.json").unlink()
    seal_manifest(viz_run_path)
    with pytest.raises(ValueError, match="scope"):
        load_run(viz_run_path)


def test_adapter_exposes_scope_denominator_and_rejects_pathlike_run_ids(viz_run_path):
    from feature_space_pilot.viz.load import load_run

    run = load_run(viz_run_path)
    assert run.n_in_scope == len(run.scope["item_ids"]) * len(run.scope["feature_ids"])
    manifest = json.loads((viz_run_path / "manifest.json").read_text())
    manifest["run_id"] = "../../outside-output"
    write_json(viz_run_path / "manifest.json", manifest)
    seal_manifest(viz_run_path)
    with pytest.raises(ValueError, match="run.*ID|identity"):
        load_run(viz_run_path)

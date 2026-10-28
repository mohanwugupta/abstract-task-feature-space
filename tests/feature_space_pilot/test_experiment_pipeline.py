"""Offline deliverables are independently scoped, sealed, and preserve source bytes."""

import json
import socket
import pytest
from feature_space_pilot.io import file_hash
from test_experiment_sources import source_fixture


def test_offline_reconstruction_exports_two_banks_matrices_reports_without_network(
    tmp_path, monkeypatch
):
    from feature_space_pilot.experiments.sources import load_inputs
    from feature_space_pilot.experiments.run import reconstruct, verify_export

    root = source_fixture(tmp_path / "empirical_sources")
    before = {str(p): file_hash(p) for p in root.parent.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline reconstruction attempted network/API access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    inputs = load_inputs(root)
    out = reconstruct(inputs, tmp_path / "corrected", render=True)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["api_calls"] == 0 and summary["source_writes"] == 0
    assert summary["human"]["n_items"] == 1 and summary["human"]["n_features"] == 1
    assert summary["task"]["n_items"] == 1 and summary["task"]["n_features"] == 1
    assert (
        json.loads((out / "human/human_feature_bank.json").read_text())[
            "inventory_hash"
        ]
        != json.loads((out / "task/task_feature_bank.json").read_text())[
            "inventory_hash"
        ]
    )
    assert "?" in (out / "task/applicability_matrix.csv").read_text()
    assert all(
        (out / "task" / (stem + ".svg")).is_file()
        for stem in (
            "figure_A_task_feature_space",
            "figure_B_abstraction",
            "figure_C_compositionality",
        )
    )
    assert (out / "human/scientific_report.md").is_file() and (
        out / "task/scientific_report.md"
    ).is_file()
    assert (
        (out / "task/scientific_report.md").read_text().startswith("SYNTHETIC FIXTURE")
    )
    assert (out / "task/diagnostic_set.csv").is_file()
    assert {
        str(p): file_hash(p) for p in root.parent.rglob("*") if p.is_file()
    } == before
    assert verify_export(out)["verified"]
    assert reconstruct(inputs, tmp_path / "corrected", render=True) == out
    (out / "task/applicability_matrix.csv").write_text("altered")
    with pytest.raises(ValueError, match="artifact"):
        verify_export(out)


def test_output_cannot_be_placed_inside_original_empirical_run(tmp_path):
    from feature_space_pilot.experiments.sources import load_inputs
    from feature_space_pilot.experiments.run import reconstruct

    root = source_fixture(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        reconstruct(load_inputs(root), root / "reports", render=False)

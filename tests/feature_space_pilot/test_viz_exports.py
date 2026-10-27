import json
from pathlib import Path
from PIL import Image
import pytest
from feature_space_pilot.io import file_hash


def test_exports_are_deterministic_read_only_and_scientifically_gated(
    viz_run_path, tmp_path
):
    from feature_space_pilot.viz.exports import export_run

    before = {p.name: file_hash(p) for p in viz_run_path.iterdir() if p.is_file()}
    first = export_run(viz_run_path, tmp_path / "figures-a")
    second = export_run(viz_run_path, tmp_path / "figures-b")
    assert first == second
    assert first["source_manifest_sha256"] == file_hash(viz_run_path / "manifest.json")
    assert first["demo"] is True
    out = tmp_path / "figures-a"
    assert (out / "index.html").exists() and (out / "figures_readme.md").exists()
    primary = [r for r in first["figures"] if r["status"] == "rendered"]
    assert len(primary) >= 5
    for row in primary:
        assert row["basename"].startswith("synthetic_")
        for ext in ["png", "svg", "pdf"]:
            assert (out / (row["basename"] + "." + ext)).is_file()
        image = Image.open(out / (row["basename"] + ".png"))
        assert image.width >= 2000 and image.height >= 1000
        assert row["caption"] and row["data_sha256"]
    assert (
        next(r for r in first["figures"] if r["figure_id"] == "05_human_baseline")[
            "status"
        ]
        == "unavailable"
    )
    assert "SYNTHETIC DEMONSTRATION" in (out / "index.html").read_text()
    assert {
        p.name: file_hash(p) for p in viz_run_path.iterdir() if p.is_file()
    } == before
    with pytest.raises(ValueError, match="source|read.only|outside"):
        export_run(viz_run_path, viz_run_path / "figures")


def test_partial_run_exports_missing_cells_and_blocks_composition(
    viz_run_path, tmp_path
):
    import csv
    from feature_space_pilot.io import write_csv, write_json
    from feature_space_pilot.report import seal_manifest
    from feature_space_pilot.viz.exports import export_run

    path = viz_run_path / "binary_judgments.csv"
    rows = list(csv.DictReader(path.open()))
    fields = list(rows[0])
    rows = [r for r in rows if r["item_id"] != "poem_T"]
    write_csv(path, rows, fields)
    (viz_run_path / "binary_matrix.csv").unlink()
    manifest = json.loads((viz_run_path / "manifest.json").read_text())
    manifest["status"] = "needs_adjudication"
    write_json(viz_run_path / "manifest.json", manifest)
    seal_manifest(viz_run_path)
    out = tmp_path / "partial-gallery"
    result = export_run(viz_run_path, out)
    composition = next(
        r for r in result["figures"] if r["figure_id"] == "02_compositionality"
    )
    assert composition["status"] == "blocked" and "missing" in composition["caption"]
    data = json.loads(
        (out / "data_exports/synthetic_figure_03_binary_heatmap.json").read_text()
    )
    assert any(r["item_id"] == "poem_T" and r["state"] == "?" for r in data)

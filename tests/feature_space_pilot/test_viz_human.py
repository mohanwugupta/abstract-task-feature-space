import csv
from pathlib import Path
import pytest
from feature_space_pilot.engine import demo_config, run_pipeline
from feature_space_pilot.io import file_hash, write_csv
from feature_space_pilot.report import seal_manifest, verify_manifest


@pytest.fixture
def human_run(tmp_path):
    c = demo_config(tmp_path)
    c.update(demo_human_count=6, replicates=2, candidate_cap=10)
    out = Path(run_pipeline(c, "human")["output_dir"])
    with (out / "binary_judgments.csv").open() as f:
        reader = csv.DictReader(f)
        rows, fields = list(reader), reader.fieldnames
    write_csv(out / "binary_judgments.csv", rows[1:], fields)
    (out / "binary_matrix.csv").unlink()
    seal_manifest(out)
    return out


def test_human_matrix_uses_all_human_targets_and_preserves_unknowns(human_run):
    from feature_space_pilot.viz.load import load_run
    from feature_space_pilot.viz.derive import matrix_data
    from feature_space_pilot.viz.figures import static_matrix
    import matplotlib.pyplot as plt

    run = load_run(human_run)
    data = matrix_data(run)
    assert set(data["item_ids"]) == set(run.items)
    assert any("?" in row for row in data["states"])
    assert not data["clustering_available"]
    fig, rows, _ = static_matrix(run)
    assert len(rows) > 0 and fig is not None
    assert "human" in fig.texts[1].get_text().lower()
    plt.close(fig)


def test_snapshot_is_verified_independent_and_never_reseals_source(human_run, tmp_path):
    from feature_space_pilot.viz.snapshot import snapshot_run

    before = {p.name: file_hash(p) for p in human_run.iterdir() if p.is_file()}
    snapshot = Path(snapshot_run(human_run, tmp_path / "snapshots"))
    assert snapshot != human_run and snapshot.is_dir()
    assert file_hash(snapshot / "manifest.json") == before["manifest.json"]
    verify_manifest(snapshot)
    assert before == {p.name: file_hash(p) for p in human_run.iterdir() if p.is_file()}
    assert snapshot_run(human_run, tmp_path / "snapshots") == str(snapshot)
    with pytest.raises(ValueError, match="outside|source"):
        snapshot_run(human_run, human_run / "snapshots")


def test_human_gallery_exports_real_panels_and_marks_task_analyses_unavailable(
    human_run, tmp_path
):
    from feature_space_pilot.viz.exports import export_run

    manifest = export_run(human_run, tmp_path / "gallery")
    figures = {row["figure_id"]: row for row in manifest["figures"]}
    assert figures["03_binary_heatmap"]["status"] == "rendered"
    assert figures["07_judgment_coverage"]["status"] == "rendered"
    assert figures["08_feature_applicability"]["status"] == "rendered"
    assert figures["01_task_atlas"]["status"] == "unavailable"
    assert figures["02_compositionality"]["status"] == "unavailable"
    assert "human" in (tmp_path / "gallery/index.html").read_text().lower()


def test_human_explorer_matrix_offers_human_filters(human_run, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FEATURE_SPACE_RUN_DIR", str(human_run))
    app = AppTest.from_file("feature_space_pilot/viz/app.py", default_timeout=30).run()
    app.selectbox(key="page").select("Matrix").run()
    assert not app.exception
    assert "All human targets" in app.selectbox(key="matrix_family").options
    assert any("6 items" in caption.value for caption in app.caption)
    app.selectbox(key="page").select("Coverage").run()
    assert not app.exception and len(app.metric) == 4


def test_coverage_counts_keep_unresolved_denominators(human_run):
    from feature_space_pilot.viz.load import load_run
    from feature_space_pilot.viz.derive import coverage_rows

    run = load_run(human_run)
    items = coverage_rows(run)
    features = coverage_rows(run, by="feature")
    for rows in [items, features]:
        assert sum(row["n_resolved"] for row in rows) == len(run.judgments)
        assert sum(row["missing"] for row in rows) == 1
        assert sum(row["n_in_scope"] for row in rows) == run.n_in_scope


def test_dense_matrix_feature_labels_do_not_overlap(human_run, monkeypatch):
    import matplotlib.pyplot as plt
    from feature_space_pilot.viz.load import load_run
    from feature_space_pilot.viz import figures

    run = load_run(human_run)
    ids = list(run.items)
    labels = [
        "can be negated to deny perception or omission of a particular semantic property and describes a specific relation "
        + str(n)
        for n in range(24)
    ]
    data = dict(
        item_ids=ids,
        feature_ids=list(run.features)[:1] * 24,
        values=[[1] * 24 for i in ids],
        states=[["1"] * 24 for i in ids],
        n_items=len(ids),
        n_features=24,
        n_bank=24,
        n_resolved=len(ids) * 24,
        ordering="fixed",
        item_labels=[run.items[i]["text"] for i in ids],
        feature_labels=labels,
    )
    monkeypatch.setattr(figures, "matrix_data", lambda *args, **kwargs: data)
    fig, _, _ = figures.static_matrix(run)
    fig.canvas.draw()
    bounds = [
        t.get_window_extent(fig.canvas.get_renderer())
        for t in fig.axes[0].get_xticklabels()
    ]
    assert all(right.x0 >= left.x1 for left, right in zip(bounds, bounds[1:]))
    plt.close(fig)

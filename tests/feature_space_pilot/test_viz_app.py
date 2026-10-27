from pathlib import Path


def test_offline_app_smoke_and_all_research_pages(viz_run_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FEATURE_SPACE_RUN_DIR", str(viz_run_path))
    script = Path(__file__).resolve().parents[2] / "feature_space_pilot/viz/app.py"
    app = AppTest.from_file(str(script), default_timeout=30).run()
    assert not app.exception
    assert any("SYNTHETIC DEMONSTRATION" in x.value for x in app.warning)
    for page in [
        "Composition",
        "Matrix",
        "Feature Discovery",
        "Human Baseline",
        "Geometry",
        "Feature Inspector",
        "Overview",
    ]:
        control = next(w for w in app.selectbox if w.key == "page")
        app = control.select(page).run()
        assert not app.exception, page


def test_atlas_feature_button_selects_exact_evidence_and_filters_are_disclosed(
    viz_run_path, monkeypatch
):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FEATURE_SPACE_RUN_DIR", str(viz_run_path))
    script = Path(__file__).resolve().parents[2] / "feature_space_pilot/viz/app.py"
    app = AppTest.from_file(str(script), default_timeout=30).run()
    button = next(b for b in app.button if b.key.startswith("left_"))
    expected = button.key[len("left_") :]
    app = button.click().run()
    assert not app.exception
    assert (
        next(s for s in app.selectbox if s.key == "inspector_atlas").value == expected
    )
    search = next(w for w in app.text_input if w.key == "feature_search")
    app = search.set_value("meeting the deadline").run()
    assert not app.exception
    assert any("Vocabulary selection: 2 /" in c.value for c in app.caption)


def test_one_click_publication_export_is_downloadable_and_read_only(
    viz_run_path, monkeypatch
):
    from streamlit.testing.v1 import AppTest
    from feature_space_pilot.io import file_hash
    from feature_space_pilot.report import verify_manifest

    monkeypatch.setenv("FEATURE_SPACE_RUN_DIR", str(viz_run_path))
    before = file_hash(viz_run_path / "manifest.json")
    app = AppTest.from_file("feature_space_pilot/viz/app.py", default_timeout=60).run()
    button = next(b for b in app.button if b.label == "Export publication figures")
    button.click().run()
    assert not app.exception
    assert any("Publication gallery exported" in s.value for s in app.success)
    gallery = (
        viz_run_path.parent.parent
        / "feature_space_visualizations"
        / viz_run_path.name
        / "v1"
        / "index.html"
    )
    assert gallery.is_file() and "Feature evidence" in gallery.read_text()
    assert any(
        b.label == "Download publication bundle (ZIP)"
        for b in app.get("download_button")
    )
    assert file_hash(viz_run_path / "manifest.json") == before
    verify_manifest(viz_run_path)

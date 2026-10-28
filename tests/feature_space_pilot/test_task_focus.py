import numpy as np


def test_task_pattern_groups_never_turn_unknown_into_zero_or_merge_meanings():
    from feature_space_pilot.task_focus.analysis import pattern_groups

    features = [
        dict(feature_id=f, text=t)
        for f, t in [
            ("x", "requires calculation"),
            ("y", "requires checking"),
            ("z", "unclear"),
            ("none", "not applicable here"),
        ]
    ]
    a = np.array([[1, 1, 1, 0], [0, 0, -1, 0], [1, 1, 0, 0]], dtype=np.int8)
    result = pattern_groups(a, features)
    assert result["complete_columns"] == 3 and result["positive_complete_columns"] == 2
    assert result["unknown_columns"] == 1 and result["all_zero_columns"] == 1
    group = result["groups"][0]
    assert group["feature_ids"] == ["x", "y"] and group["pattern"] == [1, 0, 1]
    assert group["texts"] == ["requires calculation", "requires checking"]
    assert result["semantic_merge"] is False


def test_task_audit_preserves_complete_vocabulary_and_uncertainty(tmp_path):
    from feature_space_pilot.task_focus.analysis import task_audit

    items = [
        dict(item_id="poem_T", task_id="poem", level="task_instance"),
        dict(item_id="finance_T", task_id="finance", level="task_instance"),
    ]
    features = [dict(feature_id=f, text=f) for f in ["a", "b", "c"]]
    a = np.array([[1, 0, -1], [0, 1, 1]], dtype=np.int8)
    report = task_audit(items, features, a)
    assert report["planned_cells"] == 6 and report["resolved_cells"] == 5
    assert report["unresolved_cells"] == 1 and report["items"][0]["unresolved"] == 1
    assert report["items"][0]["applicable_total"] is None
    assert report["items"][1]["applicable_total"] == 2


def test_task_snapshot_verifies_frozen_inputs_and_preserves_unknowns(tmp_path):
    import csv
    import json
    import pytest
    from feature_space_pilot.io import write_json, file_hash
    from feature_space_pilot.task_focus.run import load_snapshot

    root = tmp_path / "source"
    root.mkdir()
    report = root / "report"
    report.mkdir()
    items = [
        dict(item_id=i, task_id=t, target_type="task_instance", level="task_instance")
        for i, t in [("poem_T", "poem"), ("finance_T", "finance")]
    ]
    features = [dict(feature_id=f, text=f) for f in ["a", "b", "c"]]
    write_json(root / "stimuli.json", items)
    write_json(
        root / "frozen_global_bank.json",
        dict(features=features, origins=[], inventory_hash="fixture"),
    )
    with (report / "task_applicability.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "feature_id", "applies"])
        writer.writeheader()
        for item, values in zip(items, [[1, "?", 0], [0, 1, 1]]):
            for f, v in zip(features, values):
                writer.writerow(
                    dict(item_id=item["item_id"], feature_id=f["feature_id"], applies=v)
                )
    write_json(report / "compositionality.json", [])
    (report / "generation_coverage.csv").write_text(
        "item_id,completed,planned\npoem_T,9,9\nfinance_T,9,9\n"
    )
    write_json(
        report / "report_manifest.json",
        dict(artifacts={p.name: file_hash(p) for p in report.iterdir()}),
    )
    write_json(
        root / "full_manifest.json",
        dict(
            analysis_id="fixture",
            demo=True,
            report_dir=str(report),
            frozen_hashes={
                n: file_hash(root / n)
                for n in ["stimuli.json", "frozen_global_bank.json"]
            },
        ),
    )
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    snap = load_snapshot(root)
    assert snap["matrix"].tolist() == [[1, -1, 0], [0, 1, 1]]
    assert {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    (report / "task_applicability.csv").write_text("altered")
    with pytest.raises(ValueError, match="hash"):
        load_snapshot(root)


def test_inheritance_disagreement_is_review_flag_never_an_automatic_correction():
    from feature_space_pilot.task_focus.analysis import inheritance_flags
    items=[dict(item_id='T',text='Write an eight-line poem.',target_type='task_instance'),dict(item_id='TS',text='Write an eight-line poem. Finish quickly.',target_type='situated_task',parent_id='T')]
    features=[dict(feature_id='rhyme',text='requires rhyme'),dict(feature_id='unknown',text='uncertain')]
    a=np.array([[1,1],[0,-1]],dtype=np.int8);before=a.copy()
    flags=inheritance_flags(items,features,a,{'rhyme','unknown'})
    assert len(flags)==1 and flags[0]['feature_id']=='rhyme'
    assert flags[0]['confirmed_error'] is False and flags[0]['review_required'] is True
    assert np.array_equal(a,before)

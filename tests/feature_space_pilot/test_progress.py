import json
from pathlib import Path
from feature_space_pilot.io import canonical


def test_read_only_status_handles_active_journal_tail_and_ambiguous_cells(tmp_path):
    from feature_space_pilot.status import inspect_run, format_status

    run = tmp_path / "live_human_fixture"
    run.mkdir()
    (run / "frozen_config.json").write_text(
        json.dumps(
            dict(
                root=str(tmp_path),
                candidate_cap=2,
                api_cap_usd=100,
                stage_caps=dict(generation=10, judgment=90),
            )
        )
    )
    (run / "generation_jobs.jsonl").write_text(
        canonical(dict(job_id="g", item_id="target")) + "\n"
    )
    (run / "generations.jsonl").write_text(
        canonical(
            dict(job_id="g", parse_status="ok", item_id="target", latency_seconds=2)
        )
        + "\n"
    )
    (run / "judgment_jobs.jsonl").write_text(
        "".join(canonical(dict(job_id=j, item_id="target")) + "\n" for j in ["a", "b"])
    )
    (run / "judgments.jsonl").write_text(
        canonical(
            dict(
                job_id="a",
                parse_status="ok",
                status="ambiguous",
                applies=None,
                item_id="target",
            )
        )
        + "\n"
        + '{"job_id":'
    )
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    result = inspect_run(run)
    assert result["generation"]["completed"] == result["generation"]["planned"] == 1
    assert (
        result["judgment"]["completed"] == 1
        and result["judgment"]["resolved"] == 0
        and result["judgment"]["ambiguous"] == 1
    )
    assert result["judgment"]["incomplete_tail"]
    assert "1/2" in format_status(result) and "ambiguous" in format_status(result)
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}


def test_pipeline_reports_real_phase_and_attempt_progress_to_callback(tmp_path):
    from feature_space_pilot.engine import demo_config, run_pipeline

    events = []
    result = run_pipeline(
        demo_config(tmp_path), "tasks", progress=events.append, max_jobs=1
    )
    assert result["status"] == "partial"
    assert any(e["event"] == "start" for e in events)
    assert any(
        e["event"] == "attempt_start" and e["stage"] == "generation" for e in events
    )
    assert any(
        e["event"] == "attempt_finished" and e["parse_status"] == "ok" for e in events
    )
    assert events[-1]["event"] == "finished"

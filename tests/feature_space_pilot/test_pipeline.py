import copy
import json
from pathlib import Path
import pytest
from feature_space_pilot.engine import preflight, run_pipeline, demo_config
from feature_space_pilot.clients import FakeClient, HostedClient
from feature_space_pilot.report import generate_report, verify_manifest


def test_dry_run_no_writes_or_calls_and_live_requires_sources_prices(tmp_path):
    c = demo_config(tmp_path)
    before = set(tmp_path.rglob("*"))
    p = preflight(c, "all")
    assert p["generation_jobs"] == 2250 and p["demo"] is True
    assert set(tmp_path.rglob("*")) == before
    with pytest.raises(ValueError, match="source|synthetic|model|price"):
        run_pipeline(c, "human", execute_live=True)
    assert set(tmp_path.rglob("*")) == before


def test_full_offline_run_manifest_matrix_reports_and_exact_resume(tmp_path):
    c = demo_config(tmp_path)
    c["replicates"] = 2
    client = FakeClient()
    result = run_pipeline(c, "all", client=client)
    out = Path(result["output_dir"])
    assert result["status"] == "complete" and result["demo"] is True
    assert result["planned_generation_jobs"] == 450
    calls = client.calls
    assert calls > 450
    assert verify_manifest(out)["status"] == "complete"
    report = generate_report(out)
    assert "SYNTHETIC DEMONSTRATION" in report
    compositions = json.loads((out / "compositionality_audit.json").read_text())
    assert len(compositions) == 6
    assert all(x["confirmed_interactions"] == [] for x in compositions)
    assert all(x["component_calls"] == x["direct_calls"] for x in compositions)
    import csv

    rows = list(csv.reader((out / "binary_matrix.csv").open()))
    assert len(rows) == 76
    assert {v for row in rows[1:] for v in row[1:]} == {"0", "1"}
    second = run_pipeline(c, "all", client=client)
    assert second["run_id"] == result["run_id"] and client.calls == calls
    for name in [
        "task_level_feature_audit.csv",
        "human_norm_comparison.json",
        "embedding_rdm_comparison.json",
        "cost_report.json",
        "generation_jobs.jsonl",
        "novel_feature_audit.csv",
    ]:
        assert (out / name).is_file()
    (out / "binary_matrix.csv").write_text("stale")
    with pytest.raises(ValueError, match="hash"):
        verify_manifest(out)


def test_interrupted_resume_and_charged_parse_retry(tmp_path):
    c = demo_config(tmp_path)
    c["replicates"] = 2
    c["demo_human_count"] = 3
    client = FakeClient(fail_once=True)
    result = run_pipeline(c, "human", client=client, max_jobs=2)
    assert result["status"] == "partial"
    result = run_pipeline(c, "human", client=client)
    assert result["status"] == "complete"
    records = [
        json.loads(x)
        for x in Path(result["output_dir"], "generations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert sum(r["parse_status"] == "error" for r in records) == 1
    assert len({r["job_id"] for r in records if r["parse_status"] == "ok"}) == 18
    assert len(records) == 19


def test_design_changes_get_separate_run_and_ambiguous_cells_stay_partial(tmp_path):
    c = demo_config(tmp_path)
    c["replicates"] = 2
    c["demo_human_count"] = 3
    result = run_pipeline(c, "human", client=FakeClient(ambiguous=True))
    assert result["status"] == "needs_adjudication"
    assert not Path(result["output_dir"], "binary_matrix.csv").exists()
    assert Path(result["output_dir"], "judgment_audit.jsonl").is_file()
    changed = copy.deepcopy(c)
    changed["seed"] += 1
    assert (
        preflight(c, "human")["design_hash"]
        != preflight(changed, "human")["design_hash"]
    )


def test_hosted_adapter_does_not_retry_or_cache_and_rejects_missing_usage(monkeypatch):
    client = HostedClient(
        dict(
            provider="openai_compatible",
            base_url="https://api.example.org/v1",
            api_key_env="PILOT_TEST_KEY",
            model_id="frontier-fixed",
            supports_seed=False,
            supports_temperature=False,
            supports_structured_outputs=True,
            max_output_tokens=500,
        )
    )
    monkeypatch.setenv("PILOT_TEST_KEY", "not-a-real-key")
    captured = []

    def transport(payload):
        captured.append(payload)
        return dict(
            model="frontier-fixed",
            choices=[dict(message=dict(content="{}"), finish_reason="stop")],
            usage=dict(prompt_tokens=10, completion_tokens=4),
        )

    client.transport = transport
    for _ in range(2):
        raw, usage = client.complete(
            system="s", user="u", schema={"type": "object"}, seed=123, temperature=0.8
        )
        assert usage["input_tokens"] == 10
    assert (
        len(captured) == 2
        and "seed" not in captured[0]
        and "temperature" not in captured[0]
    )
    assert captured[0]["response_format"]["json_schema"]["strict"] is True


def test_resume_after_crash_appends_journal_beyond_partial_manifest(
    tmp_path, monkeypatch
):
    import feature_space_pilot.engine as engine

    c = demo_config(tmp_path)
    c["replicates"] = 2
    c["demo_human_count"] = 3
    first = run_pipeline(c, "human", client=FakeClient(), max_jobs=2)
    original = engine.append_jsonl

    def crash(path, record):
        original(path, record)
        if Path(path).name == "generations.jsonl":
            raise KeyboardInterrupt("simulated power loss")

    monkeypatch.setattr(engine, "append_jsonl", crash)
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(c, "human", client=FakeClient())
    monkeypatch.setattr(engine, "append_jsonl", original)
    result = run_pipeline(c, "human", client=FakeClient())
    assert result["status"] == "complete" and result["run_id"] == first["run_id"]


def test_missing_provider_usage_halts_and_keeps_liability(tmp_path):
    class MissingUsage(FakeClient):
        def complete(self, **kw):
            raw, usage = super().complete(**kw)
            return raw, dict(usage, input_tokens=None)

    c = demo_config(tmp_path)
    c["demo_human_count"] = 3
    c["replicates"] = 2
    with pytest.raises(ValueError, match="usage"):
        run_pipeline(c, "human", client=MissingUsage())
    events = [
        json.loads(s)
        for s in (tmp_path / "artifacts/feature_space_pilot/budget_demo.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(events) == 1 and events[0]["event"] == "reserve"


def test_preflight_includes_existing_cumulative_liabilities(tmp_path):
    from feature_space_pilot.budget import Ledger

    c = demo_config(tmp_path)
    c["replicates"] = 2
    c["demo_human_count"] = 3
    c["api_cap_usd"] = 100
    for role in ["generator", "judge"]:
        c[role]["prices"]["input_per_million"] = 2
        c[role]["prices"]["output_per_million"] = 8
    path = tmp_path / "artifacts/feature_space_pilot/budget_demo.jsonl"
    with Ledger(path, cap=100, stage_caps=c["stage_caps"]) as ledger:
        ledger.reserve("prior-unknown-call", "generation", 90)
    result = preflight(c, "human")
    assert any("cumulative" in b for b in result["blockers"])
    assert result["accounted_usd"] == 90


def test_inline_secret_is_rejected_before_artifact_creation(tmp_path):
    c = demo_config(tmp_path)
    c["generator"]["api_key"] = "sensitive-sentinel"
    with pytest.raises(ValueError, match="environment"):
        run_pipeline(c, "tasks", client=FakeClient())
    assert not (tmp_path / "artifacts").exists()


def test_reviewed_interactions_can_be_added_without_repeating_calls(tmp_path):
    c = demo_config(tmp_path)
    c["replicates"] = 2
    client = FakeClient()
    first = run_pipeline(c, "tasks", client=client)
    out = Path(first["output_dir"])
    rs = json.loads((out / "compositionality_audit.json").read_text())
    chosen = next(r for r in rs if r["interaction_candidates"])
    fid = chosen["interaction_candidates"][0]
    calls = client.calls
    c["relational_audit"] = {
        chosen["item_id"]: {
            fid: dict(
                confirmed=True,
                reviewer="reviewer-1",
                note="Describes task-specific deadline conflict",
            )
        }
    }
    second = run_pipeline(c, "tasks", client=client)
    rs = json.loads((out / "compositionality_audit.json").read_text())
    assert client.calls == calls and first["run_id"] == second["run_id"]
    assert (
        fid
        in next(r for r in rs if r["item_id"] == chosen["item_id"])[
            "confirmed_interactions"
        ]
    )
    assert (out / "manifest_history.jsonl").is_file()


def test_pinned_aligned_embeddings_compare_both_feature_geometries(tmp_path):
    from feature_space_pilot.engine import selected_items
    from feature_space_pilot.io import digest, file_hash

    c = demo_config(tmp_path)
    c["replicates"] = 2
    client = FakeClient()
    first = run_pipeline(c, "all", client=client)
    calls = client.calls
    items = selected_items(c, "all")
    vocabulary = sorted({p for i in items for p in FakeClient.properties(i["text"])})
    artifact = dict(
        model_id="synthetic-embedding",
        revision="fixture-1",
        source_url="synthetic:test",
        items=[
            dict(
                item_id=i["item_id"],
                text_hash=digest(i["text"]),
                vector=[int(p in FakeClient.properties(i["text"])) for p in vocabulary],
            )
            for i in items
        ],
    )
    path = tmp_path / "embeddings.json"
    path.write_text(json.dumps(artifact))
    c.update(embedding_file=str(path), embedding_sha256=file_hash(path))
    result = run_pipeline(c, "all", client=client)
    assert client.calls == calls
    comparison = json.loads(
        Path(result["output_dir"], "embedding_rdm_comparison.json").read_text()
    )
    assert comparison["generated_comparison"]["available"] is True
    assert comparison["human_comparisons"][0]["available"] is True


def test_demo_reference_is_materialized_read_only_and_hashed(tmp_path):
    from feature_space_pilot.io import file_hash

    c = demo_config(tmp_path)
    c["replicates"] = 2
    c["demo_human_count"] = 3
    result = run_pipeline(c, "human", client=FakeClient())
    out = Path(result["output_dir"])
    ref = out / "synthetic_human_reference.csv"
    assert ref.is_file() and not ref.stat().st_mode & 0o222
    assert result["source_hashes"]["synthetic_demo"] == file_hash(ref)
    registry = json.loads((out / "frozen_source_registry.json").read_text())
    assert registry["sources"][0]["kind"] == "synthetic"


def test_completed_resume_preflight_reserves_only_remaining_work(tmp_path):
    c = demo_config(tmp_path)
    c["replicates"] = 2
    c["demo_human_count"] = 3
    for role in ["generator", "judge"]:
        c[role]["prices"]["input_per_million"] = 2
        c[role]["prices"]["output_per_million"] = 8
    c["api_cap_usd"] = preflight(c, "human")["worst_case_usd"]
    client = FakeClient()
    first = run_pipeline(c, "human", client=client)
    calls = client.calls
    result = preflight(c, "human")
    assert result["remaining_worst_case_usd"] == 0 and not result["blockers"]
    run_pipeline(c, "human", client=client)
    assert client.calls == calls


def test_opt_in_smoke_freezes_one_per_class_and_one_task_situation_pair(tmp_path):
    from feature_space_pilot.engine import smoke_config, selected_items

    c = smoke_config(demo_config(tmp_path))
    items = selected_items(c, "human")
    assert len(items) == 8
    assert {i["target_type"] for i in items} >= {
        "concept",
        "event",
        "object",
        "situation",
        "situated_task",
    }
    assert preflight(c, "human")["generation_jobs"] == 48
    result = run_pipeline(c, "human", client=FakeClient())
    assert result["status"] == "complete" and result["smoke"] is True


def test_manifest_exposes_code_hashes_and_unfrozen_scope_upper_bound(tmp_path):
    c = demo_config(tmp_path)
    c["replicates"] = 2
    c["demo_human_count"] = 3
    result = run_pipeline(c, "human", client=FakeClient(), max_jobs=1)
    assert "engine.py" in result["source_code_hashes"]
    assert result["judgment_cells_upper_bound"] == 600
    assert result["judgment_plan_status"] == "awaiting_candidate_bank"

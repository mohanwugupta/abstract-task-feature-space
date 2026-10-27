import json
from pathlib import Path
import pytest

from feature_space_pilot.clients import FakeClient
from feature_space_pilot.engine import demo_config, run_pipeline
from feature_space_pilot.budget import Ledger
from test_batch import FakeBatchAPI


def interrupted_run(tmp_path):
    class Interrupted(FakeClient):
        judgments = 0

        def complete(self, **kwargs):
            if "applies" in kwargs["schema"]["properties"]:
                self.judgments += 1
                if self.judgments == 2:
                    raise KeyboardInterrupt()
            return super().complete(**kwargs)

    c = demo_config(tmp_path)
    c["candidate_cap"] = 2
    for role in ["generator", "judge"]:
        c[role]["model_id"] = "gpt-5-mini-2025-08-07"
        c[role]["prices"]["model_id"] = c[role]["model_id"]
        c[role]["token_counting"] = dict(
            method="tiktoken_payload_plus_margin_v1", framing_margin_tokens=512
        )
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(c, "tasks", client=Interrupted())
    return next((tmp_path / "artifacts/feature_space_pilot").glob("demo_tasks_*"))


def advance(out, api):
    from feature_space_pilot.continuation import continuation_plan, continue_batch

    plan = continuation_plan(out)
    return continue_batch(out, api=api, preflight_hash=plan["preflight_hash"])


def test_continuation_preserves_scientific_inputs_and_paid_receipts_then_finishes(
    tmp_path,
):
    from feature_space_pilot.report import verify_manifest
    from feature_space_pilot.engine import _read

    out = interrupted_run(tmp_path)
    names = [
        "frozen_config.json",
        "preflight.json",
        "generation_jobs.jsonl",
        "generations.jsonl",
        "feature_bank.csv",
        "judgment_jobs.jsonl",
        "frozen_prompts.json",
        "stimuli.jsonl",
    ]
    frozen = {name: (out / name).read_bytes() for name in names}
    prior = (out / "judgments.jsonl").read_bytes()
    api = FakeBatchAPI()
    first = advance(out, api)
    assert first["status"] == "batch_pending"
    submitted = [
        json.loads(line)
        for b in api.batches.values()
        for line in api.files[b["input_file_id"]].splitlines()
    ]
    assert len(submitted) == 29
    assert all(row["body"]["model"] == "gpt-5-mini-2025-08-07" for row in submitted)
    assert advance(out, api)["status"] == "complete"
    assert (out / "judgments.jsonl").read_bytes().startswith(prior)
    assert len(_read(out / "judgments.jsonl")) == 30
    assert frozen == {name: (out / name).read_bytes() for name in names}
    assert verify_manifest(out)["execution_amendment_hash"]
    assert advance(out, api)["status"] == "complete" and api.creates == 1


def test_continuation_detects_frozen_input_or_old_journal_tampering(tmp_path):
    from feature_space_pilot.continuation import continuation_plan

    out = interrupted_run(tmp_path)
    advance(out, FakeBatchAPI())
    path = out / "judgments.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"evidence":', b'"evidencX":', 1))
    with pytest.raises(ValueError, match="checkpoint|hash|prefix"):
        continuation_plan(out)


def test_continuation_cannot_overlap_original_spending_controller(tmp_path):
    out = interrupted_run(tmp_path)
    api = FakeBatchAPI()
    c = json.loads((out / "frozen_config.json").read_text())
    with Ledger(
        out.parent / "budget_demo.jsonl",
        cap=c["api_cap_usd"],
        stage_caps=c["stage_caps"],
    ):
        with pytest.raises(ValueError, match="Another process"):
            advance(out, api)
    assert api.creates == 0 and not (out / "execution_amendment.json").exists()


def test_continuation_rejects_execution_code_changes_after_submission(
    tmp_path, monkeypatch
):
    import feature_space_pilot.continuation as module

    out = interrupted_run(tmp_path)
    advance(out, FakeBatchAPI())
    original = module.preflight

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        result["source_code_hashes"]["batch.py"] = "0" * 64
        return result

    monkeypatch.setattr(module, "preflight", changed)
    with pytest.raises(ValueError, match="Execution code"):
        module.continuation_plan(out)


def test_watch_reconnects_after_transient_error_using_fresh_preflight(monkeypatch):
    import feature_space_pilot.continuation as module
    from feature_space_pilot.batch import BatchTransientError

    calls, sleeps, events = [], [], []

    def attempt(out, **kwargs):
        calls.append(kwargs["preflight_hash"])
        if len(calls) == 1:
            raise BatchTransientError("network disconnected")
        return dict(status="batch_pending" if len(calls) == 2 else "needs_adjudication")

    monkeypatch.setattr(module, "continue_batch", attempt)
    monkeypatch.setattr(
        module, "continuation_plan", lambda out: dict(preflight_hash="fresh")
    )
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    result = module.watch_continuation(
        "fixture",
        preflight_hash="first",
        watch=True,
        poll_seconds=5,
        progress=events.append,
    )
    assert result["status"] == "needs_adjudication"
    assert calls == ["first", "fresh", "fresh"] and sleeps == [5, 5]
    assert any(e["event"] == "collector_retry" for e in events)


def test_collector_repair_keeps_original_amendment_and_scientific_pins(
    tmp_path, monkeypatch
):
    import feature_space_pilot.continuation as module

    out = interrupted_run(tmp_path)
    advance(out, FakeBatchAPI())
    old = json.loads((out / "execution_amendment.json").read_text())
    original = module.preflight

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        result["source_code_hashes"]["batch.py"] = "1" * 64
        return result

    monkeypatch.setattr(module, "preflight", changed)
    plan = module.collector_repair_plan(out)
    module.repair_collector(
        out,
        reason="User requested network recovery fix",
        preflight_hash=plan["preflight_hash"],
    )
    new = json.loads((out / "execution_amendment.json").read_text())
    for field in [
        "config",
        "frozen_hashes",
        "original_judgment_checkpoint",
        "original_design_hash",
    ]:
        assert new[field] == old[field]
    assert json.loads((out / "execution_amendment_history.jsonl").read_text()) == old
    assert new["collector_repairs"][-1]["changed_files"] == ["batch.py"]
    assert module.continuation_plan(out)["pending_judgments"] == 29


def test_collector_repair_cannot_authorize_scientific_code_changes(
    tmp_path, monkeypatch
):
    import feature_space_pilot.continuation as module

    out = interrupted_run(tmp_path)
    advance(out, FakeBatchAPI())
    original = module.preflight

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        result["source_code_hashes"]["judge.py"] = "2" * 64
        return result

    monkeypatch.setattr(module, "preflight", changed)
    with pytest.raises(ValueError, match="Execution code"):
        module.collector_repair_plan(out)
    assert not (out / "execution_amendment_history.jsonl").exists()

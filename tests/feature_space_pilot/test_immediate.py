import copy
import json
from pathlib import Path
import pytest
from feature_space_pilot.io import digest, canonical


def fixture(tmp):
    from feature_space_pilot.full.run import load_settings
    from feature_space_pilot.execution import execution_config
    from feature_space_pilot.full.reuse import cell_job

    c = execution_config(load_settings("configs/feature_space_full.json"), "online")
    c.update(root=str(tmp), demo=True)
    item = dict(
        item_id="pencil",
        text="the pencil",
        language="en",
        content_hash=digest("the pencil"),
    )
    feature = dict(
        feature_id="f_" + digest("requires accuracy")[:20], text="requires accuracy"
    )
    job = cell_job(item, feature, c["judge"])
    return c, item, feature, job


def test_immediate_cost_profile_preserves_payload_and_rejects_tampering(tmp_path):
    from feature_space_pilot.immediate.costs import input_bound
    from feature_space_pilot.full.budget import profile_inputs
    from feature_space_pilot.full.reuse import model_identity

    c, i, f, j = fixture(tmp_path)
    base = profile_inputs([i], c["judge"], lambda p: dict(input_tokens=300))
    profile = dict(
        version="provider_item_feature_offsets_v1",
        model_identity=model_identity(c["judge"]),
        base_profile=base,
        feature_counts={f["feature_id"]: dict(delta_tokens=18)},
        boundary_reserve_tokens=8,
        validation=[dict(residual_tokens=0)],
    )
    profile["profile_hash"] = digest(profile)
    assert input_bound(i, f, c["judge"], profile)["input_token_bound"] == 326
    profile["feature_counts"][f["feature_id"]]["delta_tokens"] = 0
    with pytest.raises(ValueError, match="profile"):
        input_bound(i, f, c["judge"], profile)


def test_immediate_concurrency_keeps_cap_and_never_replays_durable_or_unknown_requests(
    tmp_path,
):
    from feature_space_pilot.immediate.execution import ImmediateExecutor
    from feature_space_pilot.full.budget import IndexedLedger

    c, i, f, j = fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    calls = []

    def transport(body):
        calls.append(body)
        return (
            dict(
                status="completed",
                output=[
                    dict(
                        type="message",
                        content=[
                            dict(
                                type="output_text",
                                text=canonical(
                                    dict(
                                        item_id=i["item_id"],
                                        feature_id=f["feature_id"],
                                        applies=1,
                                        status="ok",
                                        evidence="Synthetic fixture",
                                    )
                                ),
                            )
                        ],
                    )
                ],
                usage=dict(input_tokens=100, output_tokens=40),
            ),
            {},
        )

    with IndexedLedger(
        tmp_path / "ledger.jsonl",
        cap=100,
        stage_caps=c["stage_caps"],
        budget_policy="shared_total_only",
    ) as ledger:
        engine = ImmediateExecutor(c, out, ledger, transport=transport, workers=2)
        rows = engine.wave([dict(job=j, attempt=0, input_token_bound=330)])
        assert (
            len(rows) == 1
            and rows[0]["applies"] == 1
            and rows[0]["execution_mode"] == "online"
        )
        settled = ledger.total
        restarted = ImmediateExecutor(c, out, ledger, transport=transport, workers=2)
        assert restarted.wave([dict(job=j, attempt=0, input_token_bound=330)]) == []
        assert len(calls) == 1 and ledger.total == settled
        # A durable dispatch with no receipt must be held, never replayed.
        from feature_space_pilot.io import append_jsonl

        ghost = dict(
            job_id="unacknowledged",
            attempt=0,
            ticket="held",
            item_id="pencil",
            feature_id=f["feature_id"],
        )
        ghost["record_hash"] = digest(ghost)
        append_jsonl(out / "dispatches.jsonl", ghost)
        restarted = ImmediateExecutor(c, out, ledger, transport=transport, workers=2)
        assert "unacknowledged" in restarted.held_job_ids


def test_immediate_budget_fails_before_transport(tmp_path):
    from feature_space_pilot.immediate.execution import ImmediateExecutor
    from feature_space_pilot.full.budget import IndexedLedger

    c, i, f, j = fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    calls = []
    with IndexedLedger(
        tmp_path / "ledger.jsonl", cap=0.0000001, stage_caps={"judgment": 0.0000001}
    ) as ledger:
        engine = ImmediateExecutor(
            c, out, ledger, transport=lambda b: calls.append(b), workers=2
        )
        with pytest.raises(ValueError, match="budget"):
            engine.wave([dict(job=j, attempt=0, input_token_bound=330)])
    assert not calls


def test_unknown_immediate_usage_stays_reserved_and_unresolved(tmp_path):
    from feature_space_pilot.immediate.execution import ImmediateExecutor
    from feature_space_pilot.full.budget import IndexedLedger

    c, i, f, j = fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    def transport(body):
        raise TimeoutError("Synthetic unknown acknowledgement")

    with IndexedLedger(
        tmp_path / "ledger.jsonl",
        cap=100,
        stage_caps=c["stage_caps"],
        budget_policy="shared_total_only",
    ) as ledger:
        e = ImmediateExecutor(c, out, ledger, transport=transport, workers=2)
        rows = e.wave([dict(job=j, attempt=0, input_token_bound=330)])
        assert rows[0]["parse_status"] == "error" and rows[0]["usage_unknown"]
        assert ledger.total > 0 and all(
            not r.get("settled") for r in ledger._balances().values()
        )
        assert j["job_id"] in e.held_job_ids


def test_immediate_wave_deduplicates_and_refuses_third_attempt(tmp_path):
    from feature_space_pilot.immediate.execution import ImmediateExecutor
    from feature_space_pilot.full.budget import IndexedLedger

    c, i, f, j = fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    calls = []

    def transport(body):
        calls.append(body)
        return (
            dict(
                status="completed",
                output=[
                    dict(
                        type="message",
                        content=[
                            dict(
                                type="output_text",
                                text=canonical(
                                    dict(
                                        item_id=i["item_id"],
                                        feature_id=f["feature_id"],
                                        applies=0,
                                        status="ok",
                                        evidence="Synthetic fixture",
                                    )
                                ),
                            )
                        ],
                    )
                ],
                usage=dict(input_tokens=100, output_tokens=40),
            ),
            {},
        )

    with IndexedLedger(
        tmp_path / "ledger.jsonl",
        cap=100,
        stage_caps=c["stage_caps"],
        budget_policy="shared_total_only",
    ) as ledger:
        e = ImmediateExecutor(c, out, ledger, transport=transport, workers=2)
        task = dict(job=j, attempt=0, input_token_bound=330)
        assert len(e.wave([task, task])) == 1
        with pytest.raises(ValueError, match="attempt"):
            e.wave([dict(task, attempt=2)])
    assert len(calls) == 1


def test_cancelled_batch_reuses_partial_results_and_only_requeues_proven_unexecuted(
    tmp_path,
):
    from feature_space_pilot.immediate.migration import CollectOnlySession
    from feature_space_pilot.full.execution import FullSession
    from feature_space_pilot.full.budget import IndexedLedger, profile_inputs
    from test_batch import FakeBatchAPI

    c, i, f, j = fixture(tmp_path)
    from feature_space_pilot.execution import execution_config

    c = execution_config(c, "batch")
    c.update(batch_max_requests=10, batch_max_in_flight=2)
    second = dict(
        i, item_id="ruler", text="the ruler", content_hash=digest("the ruler")
    )
    profile = profile_inputs([i, second], c["judge"], lambda p: dict(input_tokens=250))

    class API(FakeBatchAPI):
        def retrieve(self, bid):
            b = super().retrieve(bid)
            rows = self.files[b["output_file_id"]].splitlines()
            self.files[b["output_file_id"]] = rows[0] + "\n"
            return dict(
                b,
                status="cancelled",
                request_counts=dict(total=2, completed=1, failed=0),
            )

    api = API()
    out = tmp_path / "batch"
    out.mkdir()
    with IndexedLedger(
        tmp_path / "ledger.jsonl",
        cap=100,
        stage_caps=c["stage_caps"],
        budget_policy="shared_total_only",
    ) as ledger:
        s = FullSession(c, out, ledger, api=api)
        s.enqueue_cell(i, f, profile)
        s.enqueue_cell(second, f, profile)
        s.flush("judgment")
        migrating = CollectOnlySession(c, out, ledger, api=api)
        migrating.refresh()
        from feature_space_pilot.engine import _read

        rows = _read(out / "judgments.jsonl")
        assert len(rows) == 2 and sum(r["parse_status"] == "ok" for r in rows) == 1
        cancelled = next(r for r in rows if r["parse_status"] != "ok")
        assert (
            cancelled["known_not_executed"] and cancelled["usage"]["input_tokens"] == 0
        )
        assert ledger._balances()[cancelled["budget_ticket"]]["settled"]
        assert api.creates == 1


def test_scheduler_holds_ambiguity_unknown_and_inflight_but_retries_known_failures():
    from feature_space_pilot.immediate.run import EvidenceIndex

    rows = [
        dict(
            item_id="a",
            feature_id="yes",
            attempt=0,
            parse_status="ok",
            status="ok",
            applies=1,
        ),
        dict(
            item_id="a",
            feature_id="unclear",
            attempt=0,
            parse_status="ok",
            status="ambiguous",
            applies=None,
        ),
        dict(
            item_id="a",
            feature_id="bad",
            attempt=0,
            parse_status="error",
            usage=dict(input_tokens=5, output_tokens=5),
        ),
        dict(
            item_id="a", feature_id="unknown", attempt=0, parse_status="error", usage={}
        ),
    ]
    index = EvidenceIndex(rows, pending={("a", "pending")})
    assert index.kind(("a", "yes")) == "resolved"
    assert index.kind(("a", "unclear")) == "review"
    assert index.kind(("a", "bad")) == "retry"
    assert index.kind(("a", "unknown")) == "held"
    assert index.kind(("a", "pending")) == "pending"
    assert index.kind(("a", "new")) == "primary"


def test_immediate_plan_reuses_existing_same_payload_batch_results(tmp_path):
    from test_full_pipeline import fixture_config
    from feature_space_pilot.full.run import make_plan, preflight, advance
    from feature_space_pilot.immediate.run import build_plan
    from test_batch import FakeBatchAPI
    from feature_space_pilot.full.reuse import model_identity
    from feature_space_pilot.io import write_json

    c, profile = fixture_config(tmp_path)
    api = FakeBatchAPI()
    for _ in range(2):
        p = make_plan(c, profile=profile)
        result = advance(
            p, api=api, preflight_hash=preflight(p)["preflight_hash"], render=False
        )
    source = Path(p["output_dir"])
    bank = json.loads((source / "frozen_global_bank.json").read_text())
    measured = dict(
        version="provider_item_feature_offsets_v1",
        model_identity=model_identity(c["judge"]),
        bank_inventory_hash=bank["inventory_hash"],
        reference_item="a",
        base_profile=profile,
        feature_counts={
            f["feature_id"]: dict(delta_tokens=20) for f in bank["features"]
        },
        boundary_reserve_tokens=8,
        validation=[dict(residual_tokens=0)],
    )
    measured["profile_hash"] = digest(measured)
    path = tmp_path / "counts.json"
    write_json(path, measured)
    plan = build_plan(source, path, allow_synthetic=True)
    assert plan["check"]["live_ready"] and plan["check"]["new_primary_calls"] == 0
    assert len(plan["reuse"]["reused"]) == 2 and plan["state"]["complete"]
    assert plan["config"]["judge"]["pricing_mode"] == "online"
    assert plan["config"]["judge"]["prices"]["input_per_million"] == 0.1


def test_immediate_driver_completes_frozen_scope_and_resumes_without_paid_replays(
    tmp_path,
):
    from test_full_pipeline import fixture_config
    from feature_space_pilot.full.run import make_plan, preflight, advance
    from feature_space_pilot.immediate.run import build_plan, run
    from feature_space_pilot.full.reuse import model_identity
    from feature_space_pilot.io import write_json

    c, profile = fixture_config(tmp_path)
    p = make_plan(c, profile=profile)
    advance(p, preflight_hash=preflight(p)["preflight_hash"], render=False)
    source = Path(p["output_dir"])
    bank = p["bank"]
    measured = dict(
        version="provider_item_feature_offsets_v1",
        model_identity=model_identity(c["judge"]),
        bank_inventory_hash=bank["inventory_hash"],
        reference_item="a",
        base_profile=profile,
        feature_counts={
            f["feature_id"]: dict(delta_tokens=20) for f in bank["features"]
        },
        boundary_reserve_tokens=8,
        validation=[dict(residual_tokens=0)],
    )
    measured["profile_hash"] = digest(measured)
    path = tmp_path / "counts.json"
    write_json(path, measured)
    calls = []

    def transport(body):
        calls.append(body)
        r = json.loads(body["input"][1]["content"])
        return (
            dict(
                status="completed",
                output=[
                    dict(
                        type="message",
                        content=[
                            dict(
                                type="output_text",
                                text=canonical(
                                    dict(
                                        item_id=r["item_id"],
                                        feature_id=r["feature_id"],
                                        applies=1,
                                        status="ok",
                                        evidence="Synthetic driver fixture",
                                    )
                                ),
                            )
                        ],
                    )
                ],
                usage=dict(input_tokens=100, output_tokens=40),
            ),
            {},
        )

    plan = build_plan(source, path, allow_synthetic=True, workers=2)
    result = run(
        plan,
        transport=transport,
        preflight_hash=plan["check"]["preflight_hash"],
        max_waves=1,
    )
    assert (
        result["matrix_complete"] and result["resolved_cells"] == 2 and len(calls) == 2
    )
    resumed = build_plan(source, path, allow_synthetic=True, workers=2)
    assert resumed["check"]["new_primary_calls"] == 0
    run(
        resumed,
        transport=transport,
        preflight_hash=resumed["check"]["preflight_hash"],
        max_waves=1,
    )
    assert len(calls) == 2


def test_unlocated_immediate_reservation_is_held_before_any_new_transport(tmp_path):
    from feature_space_pilot.immediate.execution import ImmediateExecutor
    from feature_space_pilot.full.budget import IndexedLedger

    c, i, f, j = fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    calls = []
    with IndexedLedger(
        tmp_path / "ledger.jsonl",
        cap=100,
        stage_caps=c["stage_caps"],
        budget_policy="shared_total_only",
    ) as ledger:
        ledger.reserve(j["job_id"], "judgment", 0.001)
        engine = ImmediateExecutor(c, out, ledger, transport=lambda b: calls.append(b))
        assert engine.wave([dict(job=j, attempt=0, input_token_bound=330)]) == []
    assert not calls

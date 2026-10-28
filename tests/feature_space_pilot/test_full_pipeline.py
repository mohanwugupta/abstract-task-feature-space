import copy
import json
from pathlib import Path
import pytest


def fixture_config(tmp):
    from test_full_sources import source_fixture
    from feature_space_pilot.engine import load_config
    from feature_space_pilot.full.budget import profile_inputs

    a = source_fixture(tmp, "a")
    b = source_fixture(tmp, "b")
    c = load_config("configs/feature_space_pilot.json")
    c.update(root=str(tmp), demo=True, batch_max_requests=100, batch_max_in_flight=2)
    c["equivalence"] = dict(c["judge"], max_output_tokens=2000)
    c["full_options"] = dict(
        source_runs=[str(a), str(b)],
        expected_items=2,
        artifact_root="full",
        feature_cap=None,
        semantic_reviews_file=None,
        relational_reviews_file=None,
        proposal_neighbors=8,
        equivalence_pairs_per_request=5,
        semantic_matching_dir=None,
    )
    items = [json.loads((p / "stimuli.jsonl").read_text()) for p in [a, b]]
    return c, profile_inputs(items, c["judge"], lambda p: dict(input_tokens=250))


def test_full_offline_prepare_and_two_batch_rounds_preserve_sources_and_finish_matrix(
    tmp_path,
):
    from feature_space_pilot.full.run import make_plan, preflight, advance
    from test_batch import FakeBatchAPI

    c, profile = fixture_config(tmp_path)
    sources = [Path(p) for p in c["full_options"]["source_runs"]]
    before = {str(p): {f.name: f.read_bytes() for f in p.iterdir()} for p in sources}
    plan = make_plan(c, profile=profile)
    assert not Path(plan["output_dir"]).exists()
    check = preflight(plan)
    assert check["live_ready"] and check["applicability"]["required_cells"] == 2
    prepared = advance(plan, preflight_hash=check["preflight_hash"], render=False)
    assert prepared["resolved_cells"] == 0
    api = FakeBatchAPI()
    plan = make_plan(c, profile=profile)
    a = advance(
        plan, api=api, preflight_hash=preflight(plan)["preflight_hash"], render=False
    )
    assert a["status"] == "batch_pending" and api.creates == 1
    plan = make_plan(c, profile=profile)
    b = advance(
        plan, api=api, preflight_hash=preflight(plan)["preflight_hash"], render=False
    )
    assert (
        b["status"] == "complete" and b["matrix_complete"] and b["resolved_cells"] == 2
    )
    assert b["demo"] is True and b["human_reviewed"] is False
    plan = make_plan(c, profile=profile)
    advance(
        plan, api=api, preflight_hash=preflight(plan)["preflight_hash"], render=False
    )
    assert api.creates == 1
    assert {
        str(p): {f.name: f.read_bytes() for f in p.iterdir()} for p in sources
    } == before


def test_full_budget_or_source_blockers_prevent_all_new_paid_jobs(tmp_path):
    from feature_space_pilot.full.run import make_plan, preflight, advance
    from test_batch import FakeBatchAPI

    c, profile = fixture_config(tmp_path)
    c["api_cap_usd"] = 0.000001
    c["stage_caps"] = {"judgment": 0.000001, "adjudication": 0}
    plan = make_plan(c, profile=profile)
    api = FakeBatchAPI()
    assert preflight(plan)["blockers"]
    with pytest.raises(ValueError, match="budget|allocation"):
        advance(
            plan,
            api=api,
            preflight_hash=preflight(plan)["preflight_hash"],
            render=False,
        )
    assert api.creates == 0


def test_full_live_execution_rejects_demo_or_injected_clients_before_reservation(
    tmp_path,
):
    from feature_space_pilot.full.run import make_plan, preflight, advance
    from test_batch import FakeBatchAPI

    c, profile = fixture_config(tmp_path)
    plan = make_plan(c, profile=profile)
    api = FakeBatchAPI()
    with pytest.raises(ValueError, match="empirical|synthetic|injected"):
        advance(
            plan,
            execute_live=True,
            api=api,
            preflight_hash=preflight(plan)["preflight_hash"],
            render=False,
        )
    assert api.creates == 0


def test_another_full_project_reuses_compatible_global_receipts_without_new_calls(
    tmp_path,
):
    from feature_space_pilot.full.run import make_plan, preflight, advance
    from test_batch import FakeBatchAPI

    c, profile = fixture_config(tmp_path)
    api = FakeBatchAPI()
    for _ in range(2):
        plan = make_plan(c, profile=profile)
        advance(
            plan,
            api=api,
            preflight_hash=preflight(plan)["preflight_hash"],
            render=False,
        )
    assert api.creates == 1

    second = copy.deepcopy(c)
    second["full_options"]["report_tag"] = "another full project"
    plan = make_plan(second, profile=profile)
    assert (
        len(plan["reuse"]["reused"]) == 2
        and plan["applicability_cost"]["new_primary_calls"] == 0
    )
    advance(
        plan, api=api, preflight_hash=preflight(plan)["preflight_hash"], render=False
    )
    assert api.creates == 1


def test_staged_execution_attempts_all_primary_cells_before_retrying_invalids(tmp_path):
    from feature_space_pilot.full.run import make_plan, preflight, advance
    from test_batch import FakeBatchAPI

    c, profile = fixture_config(tmp_path)
    c.update(batch_max_requests=1, batch_max_in_flight=1)
    c["full_options"]["retry_policy"] = "staged_primary_then_costed_invalid_retries"

    class API(FakeBatchAPI):
        def retrieve(self, bid):
            b = super().retrieve(bid)
            if bid == "batch-1":
                rows = [
                    json.loads(l) for l in self.files[b["output_file_id"]].splitlines()
                ]
                for r in rows:
                    r["response"]["body"]["status"] = "incomplete"
                self.files[b["output_file_id"]] = (
                    "\n".join(json.dumps(r) for r in rows) + "\n"
                )
            return b

    api = API()
    for _ in range(2):
        p = make_plan(c, profile=profile)
        advance(p, api=api, preflight_hash=preflight(p)["preflight_hash"], render=False)
    batch = api.batches["batch-2"]
    row = json.loads(api.files[batch["input_file_id"]].strip())
    assert json.loads(row["body"]["input"][1]["content"])["item_id"] == "b"
    p = make_plan(c, profile=profile)
    result = advance(
        p, api=api, preflight_hash=preflight(p)["preflight_hash"], render=False
    )
    assert (
        result["status"] == "ready"
    ), "Known funded retries must remain runnable after final primary collection"
    for _ in range(2):
        p = make_plan(c, profile=profile)
        result = advance(
            p, api=api, preflight_hash=preflight(p)["preflight_hash"], render=False
        )
    assert result["matrix_complete"] and api.creates == 3


def test_unlocated_equivalence_liability_blocks_phase_even_if_attempt_limit_reached(
    tmp_path,
):
    from feature_space_pilot.full.run import _phase_cost
    from feature_space_pilot.full.budget import IndexedLedger

    c, _ = fixture_config(tmp_path)
    job = dict(job_id="previous-pair", pairs=[])
    with IndexedLedger(
        tmp_path / "ledger.jsonl", cap=100, stage_caps=c["stage_caps"]
    ) as ledger:
        for _ in range(c["max_attempts"]):
            ledger.reserve(job["job_id"], "adjudication", 0.001)
        cost = _phase_cost([job], c, ledger, [], dict(tasks={}), {}, True)
    assert cost["blockers"] and not cost["finished"]
    assert cost["unlocated_prior_requests"] == 1


def test_provisional_bank_snapshots_preserve_added_origins_without_overwriting(
    tmp_path,
):
    from feature_space_pilot.full.run import make_plan, freeze_inputs

    c, profile = fixture_config(tmp_path)
    plan = make_plan(c, profile=profile)
    plan["source"]["generations_complete"] = False
    _, first = freeze_inputs(plan)
    before = (Path(first) / "bank.json").read_bytes()
    changed = copy.deepcopy(plan)
    changed["bank"]["features"][0]["n_origins"] += 1
    _, second = freeze_inputs(changed)
    assert first != second and (Path(first) / "bank.json").read_bytes() == before


def test_overnight_checkpoint_runs_all_cells_despite_explicit_missing_generation(
    tmp_path,
):
    from test_full_sources import source_fixture
    from feature_space_pilot.full.sources import collect_sources, available_checkpoint
    from feature_space_pilot.full.run import make_plan, preflight, advance
    from feature_space_pilot.io import write_json
    from test_batch import FakeBatchAPI

    c, profile = fixture_config(tmp_path)
    incomplete = source_fixture(tmp_path, "incomplete", complete=False)
    c["full_options"]["source_runs"][1] = str(incomplete)
    source = collect_sources(
        c["full_options"]["source_runs"], allow_synthetic=True, expected_items=2
    )
    from feature_space_pilot.full.budget import profile_inputs

    profile = profile_inputs(
        source["items"], c["judge"], lambda p: dict(input_tokens=250)
    )
    cp = tmp_path / "available.json"
    write_json(cp, available_checkpoint(source))
    c["full_options"].update(
        generation_policy="frozen_available_checkpoint",
        generation_checkpoint_file=str(cp),
    )
    plan = make_plan(c, profile=profile)
    assert preflight(plan)["live_ready"] and not plan["source"]["generations_complete"]
    api = FakeBatchAPI()
    for _ in range(2):
        plan = make_plan(c, profile=profile)
        result = advance(
            plan,
            api=api,
            preflight_hash=preflight(plan)["preflight_hash"],
            render=False,
        )
    assert result["status"] == "complete" and result["matrix_complete"]
    assert (
        result["source_generations_complete"] is False
        and result["generation_scope_closed"] is True
    )
    assert result["missing_generation_jobs"] == 1 and not result["scientific_complete"]
    report = Path(result["report_dir"])
    assert (report / "missing_generation_jobs.csv").exists()
    summary = json.loads((report / "summary.json").read_text())
    assert (
        summary["generation_scope_closed"] and summary["missing_generation_jobs"] == 1
    )


def test_shared_total_budget_policy_removes_stage_partitions_but_keeps_100_hard_cap(
    tmp_path,
):
    from feature_space_pilot.full.run import load_settings
    from feature_space_pilot.full.budget import IndexedLedger

    options = json.loads(Path("configs/feature_space_full.json").read_text())
    options["model_config"] = str(Path("configs/feature_space_pilot.json").resolve())
    options["stage_budget_policy"] = "shared_total_only"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(options))
    c = load_settings(path)
    assert c["api_cap_usd"] == 100 and all(v == 100 for v in c["stage_caps"].values())
    with IndexedLedger(
        tmp_path / "ledger.jsonl",
        cap=100,
        stage_caps=c["stage_caps"],
        budget_policy="shared_total_only",
    ) as ledger:
        ledger.reserve("full-primary", "judgment", 70)
        with pytest.raises(ValueError, match="budget"):
            ledger.reserve("over-cap", "adjudication", 31)
        assert ledger.total == 70

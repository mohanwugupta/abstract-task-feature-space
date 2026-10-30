"""Selective recovery must not repeat completed judgments or rewrite history."""

import copy
import json
from pathlib import Path
import pytest


def source_fixture(tmp_path):
    from feature_space_pilot.io import write_json, write_csv, file_hash, digest

    source = tmp_path / "source"
    (source / "human").mkdir(parents=True)
    items = [
        dict(
            item_id=i,
            text="the " + i,
            language="en",
            target_type="concept",
            content_hash=digest(i),
            source_dataset="vinson_2008",
        )
        for i in ["ascent", "glow"]
    ]
    bank = dict(
        experiment="human",
        features=[
            dict(feature_id=f, text=t)
            for f, t in [
                ("hf_a", "involves upward movement"),
                ("hf_b", "reduces willingness to eat"),
            ]
        ],
    )
    cells = [
        dict(
            item_id="ascent",
            feature_id="hf_a",
            status="reused_binary",
            applies=1,
            source_record_hashes='["old-a"]',
        ),
        dict(
            item_id="ascent",
            feature_id="hf_b",
            status="ambiguous",
            applies="?",
            source_record_hashes='["old-b"]',
        ),
        dict(
            item_id="glow",
            feature_id="hf_a",
            status="usage_unknown",
            applies="?",
            source_record_hashes='["old-c"]',
        ),
        dict(
            item_id="glow",
            feature_id="hf_b",
            status="invalid",
            applies="?",
            source_record_hashes='["old-d"]',
        ),
    ]
    write_json(source / "human/stimuli.json", items)
    write_json(source / "human/human_feature_bank.json", bank)
    write_csv(source / "human/applicability_cells.csv", cells, list(cells[0]))
    write_json(
        source / "source_checkpoint.json", dict(immutable_files={}, journal_prefixes=[])
    )
    write_json(
        source / "manifest.json",
        dict(
            identity=dict(demo=True),
            artifacts={
                str(p.relative_to(source)): file_hash(p)
                for p in source.rglob("*")
                if p.is_file()
            },
        ),
    )
    return source, items, bank, cells


def test_retry_selection_is_exactly_ambiguous_and_human_only(tmp_path):
    from feature_space_pilot.binary_retry import select_ambiguous

    _, items, bank, cells = source_fixture(tmp_path)
    before = copy.deepcopy(cells)
    selected = select_ambiguous(items, bank, cells)
    assert [(r["item_id"], r["feature_id"]) for r in selected] == [("ascent", "hf_b")]
    assert cells == before
    with pytest.raises(ValueError, match="human"):
        select_ambiguous(items, dict(bank, experiment="task"), cells)
    with pytest.raises(ValueError, match="Duplicate"):
        select_ambiguous(items, bank, cells + [cells[1]])
    malformed = copy.deepcopy(cells)
    malformed[1]["applies"] = 0
    with pytest.raises(ValueError, match="binary"):
        select_ambiguous(items, bank, malformed)


def test_overlay_preserves_existing_binary_and_labels_mixed_rubric_provenance(tmp_path):
    from feature_space_pilot.binary_retry import merge_cells, select_ambiguous
    from feature_space_pilot.full.reuse import cell_job
    from feature_space_pilot.io import canonical, digest

    _, items, bank, cells = source_fixture(tmp_path)
    settings = dict(
        model_id="fixture",
        supports_structured_outputs=True,
        max_output_tokens=400,
        rubric_version="binary_v2",
    )
    job = cell_job(items[0], bank["features"][1], settings)
    value = dict(
        item_id="ascent",
        feature_id="hf_b",
        applies=0,
        status="ok",
        evidence="Not characteristic of ascent.",
    )
    record = dict(
        **value,
        job_id=job["job_id"],
        parse_status="ok",
        rubric_version="binary_v2",
        prompt_hash=job["prompt_hash"],
        payload_sha256=job["payload_sha256"],
        judge_model="fixture",
        raw_response=canonical(value),
        response_hash=digest(canonical(value)),
        attempt=0
    )
    record["record_hash"] = digest(record)
    out = merge_cells(cells, [job], [record])
    bykey = {(r["item_id"], r["feature_id"]): r for r in out}
    assert bykey["ascent", "hf_a"]["applies"] == 1
    assert bykey["ascent", "hf_b"]["applies"] == 0
    assert bykey["ascent", "hf_b"]["effective_rubric"] == "binary_v2"
    assert bykey["ascent", "hf_a"]["effective_rubric"] == "binary_v1"
    assert bykey["glow", "hf_a"]["applies"] == "?"
    assert cells[1]["applies"] == "?"
    wrong = dict(record, rubric_version="binary_v1")
    wrong["record_hash"] = digest(
        {k: v for k, v in wrong.items() if k != "record_hash"}
    )
    with pytest.raises(ValueError, match="rubric"):
        merge_cells(cells, [job], [wrong])
    with pytest.raises(ValueError, match="eligible"):
        merge_cells(cells, [dict(job, item_id="ascent", feature_id="hf_a")], [record])


def test_preparation_is_offline_immutable_and_batch_resume_never_duplicates(
    tmp_path, monkeypatch
):
    from feature_space_pilot.binary_retry import prepare, measure_inputs, preflight, run
    from feature_space_pilot.engine import demo_config
    from feature_space_pilot.io import file_hash
    from test_batch import FakeBatchAPI
    import socket

    source, *_ = source_fixture(tmp_path)
    originals = {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()}
    c = demo_config(tmp_path)
    c.update(
        api_cap_usd=100,
        max_attempts=2,
        batch_max_requests=1,
        stage_caps={s: 100 for s in c["stage_caps"]},
        full_options=dict(stage_budget_policy="shared_total_only"),
    )
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: pytest.fail("Unexpected network")
    )
    out = prepare(source, c, tmp_path / "retry")
    assert prepare(source, c, tmp_path / "retry") == out
    assert not preflight(out)["live_ready"]
    measure_inputs(out, transport=lambda p: dict(input_tokens=100))
    check = preflight(out)
    assert check["live_ready"] and check["new_requests"] == 1
    assert check["max_attempts"] == 2 and check["budget_cap_usd"] == 100
    api = FakeBatchAPI()
    state = run(out, preflight_hash=check["preflight_hash"], api=api)
    assert state["status"] == "batch_pending" and api.creates == 1
    state = run(out, collect_only=True, api=api)
    assert state["status"] == "complete" and state["resolved_retry_cells"] == 1
    before = (out / "applicability/judgments.jsonl").read_bytes()
    check = preflight(out)
    run(out, preflight_hash=check["preflight_hash"], api=api)
    assert (
        api.creates == 1
        and (out / "applicability/judgments.jsonl").read_bytes() == before
    )
    assert originals == {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()}
    assert not (source / "budget_live.jsonl").exists()


def retry_fixture(tmp_path):
    from feature_space_pilot.binary_retry import prepare, measure_inputs
    from feature_space_pilot.engine import demo_config

    source, *_ = source_fixture(tmp_path)
    c = demo_config(tmp_path)
    c.update(
        api_cap_usd=100,
        max_attempts=2,
        batch_max_requests=1,
        stage_caps={s: 100 for s in c["stage_caps"]},
        full_options=dict(stage_budget_policy="shared_total_only"),
    )
    for role in ["generator", "judge"]:
        c[role]["prices"].update(input_per_million=1, output_per_million=1)
    out = prepare(source, c, tmp_path / "retry")
    measure_inputs(out, transport=lambda p: dict(input_tokens=100))
    return out


def test_budget_blocked_before_any_selective_batch_submission(tmp_path):
    from feature_space_pilot.binary_retry import preflight, run
    from feature_space_pilot.full.run import ledger_for
    from test_batch import FakeBatchAPI

    out = retry_fixture(tmp_path)
    c = json.loads((out / "frozen_config.json").read_text())
    with ledger_for(c) as ledger:
        ledger.reserve("prior-paid-work", "judgment", 99.9999)
    check = preflight(out)
    assert not check["live_ready"] and any("$100" in b for b in check["blockers"])
    api = FakeBatchAPI()
    with pytest.raises(ValueError, match=r"\$100"):
        run(out, preflight_hash=check["preflight_hash"], api=api)
    assert api.creates == 0


def test_unknown_batch_usage_is_held_without_automatic_resubmission(tmp_path):
    from feature_space_pilot.binary_retry import preflight, run
    from test_batch import FakeBatchAPI

    out = retry_fixture(tmp_path)

    class API(FakeBatchAPI):
        def retrieve(self, bid):
            batch = super().retrieve(bid)
            rows = [
                json.loads(line)
                for line in self.files[batch["output_file_id"]].splitlines()
            ]
            for row in rows:
                row["response"]["body"]["usage"] = {}
            self.files[batch["output_file_id"]] = (
                "\n".join(json.dumps(r) for r in rows) + "\n"
            )
            return batch

    api = API()
    check = preflight(out)
    run(out, preflight_hash=check["preflight_hash"], api=api)
    run(out, collect_only=True, api=api)
    check = preflight(out)
    assert check["states"].get("held") == 1 and check["invalid_retry_requests"] == 0
    run(out, preflight_hash=check["preflight_hash"], api=api)
    assert api.creates == 1


def test_token_profile_cannot_be_changed_to_understate_cost(tmp_path):
    from feature_space_pilot.binary_retry import preflight

    out = retry_fixture(tmp_path)
    p = out / "provider_input_profile.json"
    profile = json.loads(p.read_text())
    profile["items"]["ascent"]["input_tokens"] = 0
    p.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="profile"):
        preflight(out)

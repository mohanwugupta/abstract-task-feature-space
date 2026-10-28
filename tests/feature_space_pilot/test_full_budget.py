from pathlib import Path
import copy
import pytest
from feature_space_pilot.io import digest


def config(tmp):
    from feature_space_pilot.engine import load_config

    c = load_config("configs/feature_space_pilot.json")
    c.update(root=str(tmp), demo=True)
    return c


def items():
    return [
        dict(item_id="a", text="short target", language="en", content_hash="a"),
        dict(item_id="b", text="long target " * 30, language="en", content_hash="b"),
    ]


def features():
    return [
        dict(feature_id="f_" + digest(t)[:20], text=t)
        for t in ["requires accuracy", "accuracy is rewarded", "a rare feature"]
    ]


def test_provider_template_profile_is_explicit_and_tampering_is_rejected(tmp_path):
    from feature_space_pilot.full.budget import profile_inputs, input_bound

    c = config(tmp_path)
    seen = []

    def counter(payload):
        seen.append(payload)
        return dict(input_tokens=250)

    profile = profile_inputs(items(), c["judge"], counter)
    assert len(seen) == 2 and all(p["model"] == c["judge"]["model_id"] for p in seen)
    bound = input_bound(items()[0], features()[0], c["judge"], profile)
    assert bound["method"] == "provider_template_plus_feature_byte_bound_v1"
    assert bound["input_token_bound"] >= 250 and bound["exact_provider_count"] is False
    assert bound["measured_template_input_tokens"] == 250
    bad = copy.deepcopy(profile)
    bad["items"]["a"]["input_tokens"] = 1
    with pytest.raises(ValueError, match="profile"):
        input_bound(items()[0], features()[0], c["judge"], bad)


def test_complete_cost_plan_counts_global_scope_and_respects_stage_and_total_caps(
    tmp_path,
):
    from feature_space_pilot.full.budget import cost_plan, profile_inputs, IndexedLedger

    c = config(tmp_path)
    profile = profile_inputs(items(), c["judge"], lambda p: dict(input_tokens=250))
    with IndexedLedger(
        tmp_path / "ledger.jsonl", cap=100, stage_caps=c["stage_caps"]
    ) as ledger:
        empty = dict(reused=[], unresolved_prior=[], external_pending={})
        plan = cost_plan(
            items(),
            features(),
            c,
            ledger,
            empty,
            profile=profile,
            generations_complete=True,
        )
        assert (
            plan["required_cells"] == 6
            and plan["new_primary_calls"] == 6
            and plan["remaining_attempts_upper_bound"] == 12
        )
        assert plan["output_tokens_per_attempt"] == 400 and not plan["blockers"]
        ledger.reserve("previous", "judgment", 64.9999)
        blocked = cost_plan(
            items(),
            features(),
            c,
            ledger,
            empty,
            profile=profile,
            generations_complete=True,
        )
        assert any("judgment" in b for b in blocked["blockers"])
        assert ledger.total == 64.9999


def test_unfinished_sources_or_missing_measured_counts_block_paid_matrix(tmp_path):
    from feature_space_pilot.full.budget import cost_plan, IndexedLedger

    c = config(tmp_path)
    with IndexedLedger(
        tmp_path / "ledger.jsonl", cap=100, stage_caps=c["stage_caps"]
    ) as ledger:
        plan = cost_plan(
            items(),
            features(),
            c,
            ledger,
            dict(reused=[], unresolved_prior=[], external_pending={}),
            profile=None,
            generations_complete=False,
        )
    assert not plan["live_ready"] and plan["bank_provisional"]
    assert any("generation" in b for b in plan["blockers"]) and any(
        "provider" in b for b in plan["blockers"]
    )


def test_indexed_ledger_keeps_original_durable_limits_and_settlements(tmp_path):
    from feature_space_pilot.full.budget import IndexedLedger
    from feature_space_pilot.budget import Ledger

    c = config(tmp_path)
    prices = c["judge"]["prices"]
    path = tmp_path / "ledger.jsonl"
    with IndexedLedger(path, cap=100, stage_caps=c["stage_caps"]) as ledger:
        tickets = [ledger.reserve("j" + str(n), "judgment", 0.01) for n in range(30)]
        for t in tickets[:15]:
            ledger.settle(t, dict(input_tokens=100, output_tokens=100), prices)
        expected = ledger.total
        with pytest.raises(ValueError, match="settled"):
            ledger.settle(tickets[0], dict(input_tokens=1, output_tokens=1), prices)
        assert ledger.stage_total("judgment") == expected
    with Ledger(path, cap=100, stage_caps=c["stage_caps"]) as reference:
        assert reference.total == pytest.approx(expected)
    with IndexedLedger(path, cap=100, stage_caps=c["stage_caps"]) as resumed:
        assert (
            resumed.total == pytest.approx(expected) and len(resumed._balances()) == 30
        )


def test_staged_budget_funds_every_primary_cell_then_blocks_unaffordable_retries(
    tmp_path,
):
    from feature_space_pilot.full.budget import (
        cost_plan,
        profile_inputs,
        IndexedLedger,
        input_bound,
    )
    from feature_space_pilot.budget import estimate_cost

    c = config(tmp_path)
    c["api_cap_usd"] = 0.001
    c["stage_caps"] = {"judgment": 0.001, "adjudication": 0}
    c["full_options"] = {"retry_policy": "staged_primary_then_costed_invalid_retries"}
    profile = profile_inputs(items(), c["judge"], lambda p: dict(input_tokens=250))
    reuse = dict(reused=[], unresolved_prior=[], external_pending={})
    with IndexedLedger(
        tmp_path / "ledger.jsonl", cap=0.001, stage_caps=c["stage_caps"]
    ) as ledger:
        plan = cost_plan(
            items(),
            features(),
            c,
            ledger,
            reuse,
            profile=profile,
            generations_complete=True,
        )
        assert (
            plan["authorized_stage"] == "primary_pass"
            and plan["new_primary_calls"] == 6
        )
        assert (
            plan["live_ready"] and plan["remaining_completion_upper_bound_usd"] > 0.001
        )
        assert plan["authorized_stage_upper_bound_usd"] < 0.001
        failed = []
        for item in items():
            for feature in features():
                n = input_bound(item, feature, c["judge"], profile)["input_token_bound"]
                price = estimate_cost(c["judge"]["prices"], n, 400)
                ticket = ledger.reserve(
                    item["item_id"] + feature["feature_id"], "judgment", price
                )
                ledger.settle(
                    ticket,
                    dict(input_tokens=n, output_tokens=400),
                    c["judge"]["prices"],
                )
                failed.append(
                    dict(
                        item_id=item["item_id"],
                        feature_id=feature["feature_id"],
                        parse_status="error",
                        attempt=0,
                    )
                )
        retry = cost_plan(
            items(),
            features(),
            c,
            ledger,
            reuse,
            profile=profile,
            generations_complete=True,
            current_records=failed,
        )
        assert (
            retry["authorized_stage"] == "invalid_retries"
            and retry["new_primary_calls"] == 0
        )
        assert not retry["live_ready"] and retry["known_invalid_retry_calls"] == 6


def test_staged_budget_collects_pending_first_pass_before_any_retry(tmp_path):
    from feature_space_pilot.full.budget import cost_plan, profile_inputs, IndexedLedger

    c = config(tmp_path)
    c["full_options"] = {"retry_policy": "staged_primary_then_costed_invalid_retries"}
    profile = profile_inputs(items(), c["judge"], lambda p: dict(input_tokens=250))
    pending = {
        str(n): dict(item_id=i["item_id"], feature_id=f["feature_id"])
        for n, (i, f) in enumerate((i, f) for i in items() for f in features())
    }
    with IndexedLedger(
        tmp_path / "ledger.jsonl", cap=100, stage_caps=c["stage_caps"]
    ) as ledger:
        plan = cost_plan(
            items(),
            features(),
            c,
            ledger,
            dict(reused=[], unresolved_prior=[], external_pending=pending),
            profile=profile,
            generations_complete=True,
        )
        assert (
            plan["authorized_stage"] == "collect_pending"
            and plan["authorized_stage_upper_bound_usd"] == 0
        )


def test_closed_available_checkpoint_can_fund_full_scope_without_fabricating_generation_completion(
    tmp_path,
):
    from feature_space_pilot.full.budget import cost_plan, profile_inputs, IndexedLedger

    c = config(tmp_path)
    profile = profile_inputs(items(), c["judge"], lambda p: dict(input_tokens=250))
    with IndexedLedger(
        tmp_path / "ledger.jsonl", cap=100, stage_caps=c["stage_caps"]
    ) as ledger:
        result = cost_plan(
            items(),
            features(),
            c,
            ledger,
            dict(reused=[], unresolved_prior=[], external_pending={}),
            profile=profile,
            generations_complete=False,
            source_scope_closed=True,
        )
    assert result["live_ready"] and result["bank_provisional"] is False
    assert (
        result["source_generations_complete"] is False
        and result["source_scope_closed"] is True
    )

import copy
from pathlib import Path
import pytest

from feature_space_pilot.engine import load_config
from feature_space_pilot.clients import HostedClient, FakeClient, generation_schema
from feature_space_pilot.budget import Ledger

ROOT = Path(__file__).resolve().parents[2]


def test_counter_covers_exact_transport_payload_schema_and_utf8_offline(monkeypatch):
    from feature_space_pilot.token_accounting import count_payload
    from feature_space_pilot.clients import build_payload
    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("Offline count attempted network access"),
    )
    settings = load_config(ROOT / "configs/feature_space_pilot.json")["generator"]
    assert settings["model_id"] == "gpt-5-mini-2025-08-07"
    payload = build_payload(
        settings,
        system="Diagnostic",
        user="creativity / Kreativität",
        schema=generation_schema("x", 4),
    )
    counted = count_payload(payload, settings)
    assert counted["encoding"] == "o200k_base"
    assert (
        counted["payload_tokens"] > 0
        and counted["input_token_bound"] == counted["payload_tokens"] + 512
    )
    assert counted["payload_bytes"] > counted["payload_tokens"]
    assert counted["request_sha256"]
    client = HostedClient(settings)
    seen = []
    client.transport = lambda p: seen.append(p) or dict(
        choices=[dict(message=dict(content="{}"), finish_reason="stop")],
        usage=dict(prompt_tokens=5, completion_tokens=3),
    )
    client.complete(
        system="Diagnostic",
        user="creativity / Kreativität",
        schema=generation_schema("x", 4),
    )
    assert seen == [payload]
    changed = dict(payload, response_format={})
    assert (
        count_payload(changed, settings)["payload_tokens"] < counted["payload_tokens"]
    )


def test_oversize_rejected_before_reservation_or_transport(tmp_path):
    from feature_space_pilot.engine import _call

    config = load_config(ROOT / "configs/feature_space_pilot.json")
    settings = dict(config["generator"], max_input_tokens=1)
    prompt = dict(system="A request", user="{}")
    client = FakeClient()
    with Ledger(
        tmp_path / "budget.jsonl", cap=1, stage_caps={"generation": 1}
    ) as ledger:
        with pytest.raises(ValueError, match="input.token|input-token"):
            _call(
                client,
                prompt,
                generation_schema("x", 4),
                settings,
                ledger,
                "job",
                "generation",
                tmp_path / "journal.jsonl",
                {},
                lambda raw: {},
                1,
                0,
            )
        assert ledger.events == [] and client.calls == 0
        assert not (tmp_path / "budget.jsonl").exists()
    hosted = HostedClient(settings)
    hosted.transport = lambda p: pytest.fail("Oversize payload reached provider")
    with pytest.raises(ValueError, match="input.token|input-token"):
        hosted.complete(system="A request", user="{}", schema=generation_schema("x", 4))


def test_actual_prompt_audit_includes_all_75_targets_and_long_finance_payload():
    from feature_space_pilot.token_accounting import audit_prompts

    result = audit_prompts(load_config(ROOT / "configs/feature_space_pilot.json"))
    assert result["n_items"] == 75 and result["generation"]["n_templates"] == 225
    assert result["judgment"]["n_templates"] == 75
    assert result["largest_generation"]["item_id"].startswith("finance_")
    assert result["generation"]["max_input_token_bound"] <= 7500
    assert result["judgment"]["max_input_token_bound_with_feature_reserve"] <= 7500
    assert result["declared_output_tokens"] == dict(generation=700, judgment=400)
    assert result["byte_lengths_are_token_counts"] is False


def test_combined_75_by_150_plan_preserves_prior_spend_and_fits_100(tmp_path):
    from feature_space_pilot.preflight import campaign_preflight

    config = load_config(ROOT / "configs/feature_space_pilot.json")
    config["artifact_root"] = str(tmp_path / "artifacts")
    path = tmp_path / "artifacts/budget_live.jsonl"
    with Ledger(path, cap=100, stage_caps=config["stage_caps"]) as ledger:
        ticket = ledger.reserve("prior-diagnostic", "generation", 0.02)
        oldprices = dict(
            config["generator"]["prices"], input_per_million=2, output_per_million=10
        )
        ledger.settle(ticket, dict(input_tokens=602, output_tokens=161), oldprices)
    before = path.read_bytes()
    result = campaign_preflight(config)
    assert (
        result["generation_jobs"] == 675
        and result["judgment_cells_upper_bound"] == 11250
    )
    assert result["worst_case_usd"] == pytest.approx(64.60875)
    assert result["accounted_usd"] == pytest.approx(0.002814)
    assert result["total_liability_usd"] == pytest.approx(64.611564)
    assert result["live_ready"] and not result["blockers"]
    assert sum(config["stage_caps"].values()) <= 100
    assert path.read_bytes() == before
    expensive = copy.deepcopy(config)
    expensive["api_cap_usd"] = 50
    blocked = campaign_preflight(expensive)
    assert not blocked["live_ready"] and any(
        "budget" in b.lower() for b in blocked["blockers"]
    )


def test_stage_allocations_cannot_exceed_cumulative_cap():
    with pytest.raises(ValueError, match="stage|Stage"):
        Ledger(
            "not-created.jsonl", cap=100, stage_caps=dict(generation=70, judgment=70)
        )


def test_individual_stage_cannot_ignore_combined_campaign_budget():
    from feature_space_pilot.engine import preflight

    config = load_config(ROOT / "configs/feature_space_pilot.json")
    config["api_cap_usd"] = 60
    config["stage_caps"] = dict(generation=5, judgment=55)
    result = preflight(config, "human")
    assert result["worst_case_usd"] < 60
    assert not result["live_ready"]
    assert any("Combined human and task" in b for b in result["blockers"])


def test_provider_input_envelope_breach_halts_even_when_total_cost_fits(tmp_path):
    from feature_space_pilot.clients import build_payload
    from feature_space_pilot.token_accounting import count_payload
    from feature_space_pilot.engine import _call

    config = load_config(ROOT / "configs/feature_space_pilot.json")
    settings = config["judge"]
    schema = generation_schema("x", 4)
    bound = count_payload(
        build_payload(settings, system="Fixture", user="{}", schema=schema), settings
    )["input_token_bound"]

    class MiscountedClient:
        calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            return "{}", dict(
                input_tokens=bound + 1, output_tokens=0, finish_reason="stop"
            )

    client = MiscountedClient()
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(judgment=1)
    ) as ledger:
        with pytest.raises(ValueError, match="token envelope"):
            _call(
                client,
                dict(system="Fixture", user="{}"),
                schema,
                settings,
                ledger,
                "job",
                "judgment",
                tmp_path / "journal.jsonl",
                {},
                lambda raw: {},
                2,
                0,
            )
        assert client.calls == 1 and ledger.report()["unsettled_reservations"] == 0
        assert ledger.total > 0

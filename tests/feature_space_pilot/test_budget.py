import pytest
from feature_space_pilot.budget import Ledger, estimate_cost, validate_prices

PRICES = dict(
    model_id="fake",
    revision="fake-1",
    input_per_million=2,
    output_per_million=8,
    verified_at="2026-10-26",
    source_url="https://example.org/prices",
)


def test_budget_missing_prices_and_cap_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="price"):
        validate_prices({})
    with pytest.raises(ValueError, match="500"):
        Ledger(tmp_path / "ledger.jsonl", cap=501)
    assert estimate_cost(PRICES, 1000, 1000) == 0.01
    with Ledger(
        tmp_path / "ledger.jsonl", cap=0.02, stage_caps={"generation": 0.02}
    ) as ledger:
        ticket = ledger.reserve("j", "generation", 0.015)
        with pytest.raises(ValueError, match="budget"):
            ledger.reserve("j2", "generation", 0.01)
        ledger.settle(ticket, dict(input_tokens=1000, output_tokens=1000), PRICES)
        assert ledger.total == 0.01
    with Ledger(
        tmp_path / "ledger.jsonl", cap=0.02, stage_caps={"generation": 0.02}
    ) as ledger:
        assert ledger.total == 0.01
        ledger.reserve("j2", "generation", 0.01)
    with Ledger(
        tmp_path / "ledger.jsonl", cap=0.02, stage_caps={"generation": 0.02}
    ) as ledger:
        assert ledger.total == 0.02  # interrupted call remains fully reserved
        with pytest.raises(ValueError, match="budget"):
            ledger.reserve("j3", "generation", 0.001)


def test_missing_or_excess_usage_keeps_reservation(tmp_path):
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps={"generation": 1}
    ) as ledger:
        ticket = ledger.reserve("j", "generation", 0.01)
        with pytest.raises(ValueError, match="usage"):
            ledger.settle(ticket, {}, PRICES)
        assert ledger.total == 0.01
        with pytest.raises(ValueError, match="reservation"):
            ledger.settle(ticket, dict(input_tokens=100000, output_tokens=0), PRICES)
        assert ledger.total >= 0.2

from pathlib import Path
import pytest
from feature_space_pilot.engine import load_config
from feature_space_pilot.clients import FakeClient

ROOT = Path(__file__).resolve().parents[2]


def test_calibration_is_explicit_single_attempt_and_uses_shared_ledger(
    tmp_path, monkeypatch
):
    from feature_space_pilot.calibrate import calibrate
    import feature_space_pilot.calibrate as module

    c = load_config(ROOT / "configs/feature_space_pilot.json")
    c["artifact_root"] = str(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="execute.live|explicit"):
        calibrate(c, role="judge")
    assert not (tmp_path / "artifacts").exists()
    monkeypatch.setattr(module, "load_api_credentials", lambda config: None)
    fake = FakeClient()
    result = calibrate(c, role="judge", execute_live=True, client=fake)
    assert result["passed"] and fake.calls == 1
    assert result["usage"]["input_tokens"] == 80 and result["input_token_bound"] >= 80
    assert result["scientific_result"] is False
    assert result["ledger"]["unsettled_reservations"] == 0
    assert (tmp_path / "artifacts/budget_live.jsonl").is_file()
    assert Path(result["output_dir"]).is_dir()

import json
import os
from pathlib import Path

import pytest

from feature_space_pilot.engine import demo_config, preflight, run_pipeline


@pytest.fixture(autouse=True)
def isolate_test_credentials(monkeypatch):
    # Ensure environment mutations made by the loader are restored after each test.
    monkeypatch.setenv("PILOT_FAKE_OPENAI_KEY", "")
    monkeypatch.setenv("PILOT_FAKE_JUDGE_KEY", "")


def live_config(tmp_path):
    c = demo_config(tmp_path)
    c["demo"] = False
    for role in ["generator", "judge"]:
        c[role].update(
            provider="openai_compatible",
            base_url="https://api.example.test/v1",
            api_key_env="PILOT_FAKE_OPENAI_KEY",
            capabilities_verified_at="2026-10-26",
            capabilities_source="https://example.test/offline-fixture",
        )
    return c


def test_env_file_loads_only_named_credentials_and_exported_values_win(
    tmp_path, monkeypatch
):
    from feature_space_pilot.environment import load_api_credentials

    c = live_config(tmp_path)
    c["judge"]["api_key_env"] = "PILOT_FAKE_JUDGE_KEY"
    monkeypatch.setenv("PILOT_FAKE_OPENAI_KEY", "exported-fixture")
    monkeypatch.delenv("PILOT_FAKE_JUDGE_KEY", raising=False)
    monkeypatch.delenv("PILOT_UNRELATED_VALUE", raising=False)
    (tmp_path / ".env").write_text(
        "# local credentials\n"
        "PILOT_FAKE_OPENAI_KEY=from-file-fixture\n"
        'export PILOT_FAKE_JUDGE_KEY="quoted-fixture" # comment\n'
        "PILOT_UNRELATED_VALUE=do-not-load\n"
    )
    load_api_credentials(c)
    assert os.environ["PILOT_FAKE_OPENAI_KEY"] == "exported-fixture"
    assert os.environ["PILOT_FAKE_JUDGE_KEY"] == "quoted-fixture"
    assert "PILOT_UNRELATED_VALUE" not in os.environ


def test_env_loading_is_literal_and_errors_do_not_echo_values(tmp_path, monkeypatch):
    from feature_space_pilot.environment import load_api_credentials

    c = live_config(tmp_path)
    monkeypatch.delenv("PILOT_FAKE_OPENAI_KEY", raising=False)
    marker = tmp_path / "must-not-exist"
    literal = f"$(touch {marker})"
    (tmp_path / ".env").write_text(f"PILOT_FAKE_OPENAI_KEY='{literal}'\n")
    load_api_credentials(c)
    assert os.environ["PILOT_FAKE_OPENAI_KEY"] == literal
    assert not marker.exists()
    monkeypatch.delenv("PILOT_FAKE_OPENAI_KEY")
    (tmp_path / ".env").write_text(
        'PILOT_FAKE_OPENAI_KEY="secret-marker-no-closing-quote\n'
    )
    with pytest.raises(ValueError) as error:
        load_api_credentials(c)
    assert ".env" in str(error.value)
    assert "secret-marker" not in str(error.value)


def test_missing_key_fails_before_creating_artifacts_or_reserving_budget(
    tmp_path, monkeypatch
):
    c = live_config(tmp_path)
    monkeypatch.delenv("PILOT_FAKE_OPENAI_KEY", raising=False)
    plan = preflight(c, "tasks")
    assert not plan["blockers"]
    with pytest.raises(ValueError, match=r"PILOT_FAKE_OPENAI_KEY.*\.env"):
        run_pipeline(
            c,
            "tasks",
            execute_live=True,
            preflight_hash=plan["preflight_hash"],
            max_jobs=1,
        )
    assert not (tmp_path / "artifacts").exists()


def test_live_entrypoint_loads_project_env_without_leaking_key(tmp_path, monkeypatch):
    import feature_space_pilot.engine as engine
    from feature_space_pilot.clients import FakeClient

    c = live_config(tmp_path)
    monkeypatch.delenv("PILOT_FAKE_OPENAI_KEY", raising=False)
    secret = "offline-secret-sentinel-never-a-real-key"
    (tmp_path / ".env").write_text(f"PILOT_FAKE_OPENAI_KEY={secret}\n")
    different_cwd = tmp_path / "unrelated-directory"
    different_cwd.mkdir()
    monkeypatch.chdir(different_cwd)

    def fake_hosted(settings):
        assert os.environ[settings["api_key_env"]] == secret
        return FakeClient()

    monkeypatch.setattr(engine, "HostedClient", fake_hosted)
    plan = preflight(c, "tasks")
    result = run_pipeline(
        c, "tasks", execute_live=True, preflight_hash=plan["preflight_hash"], max_jobs=1
    )
    assert result["status"] == "partial"
    assert result["completed_generation_jobs"] == 1
    for path in Path(result["output_dir"]).iterdir():
        if path.is_file():
            assert secret not in path.read_text(), path.name


def test_offline_run_does_not_read_env_or_load_credentials(tmp_path, monkeypatch):
    import feature_space_pilot.engine as engine

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline execution must not load credentials")

    monkeypatch.setattr(engine, "load_api_credentials", forbidden)
    (tmp_path / ".env").write_text('PILOT_FAKE_OPENAI_KEY="malformed\n')
    c = demo_config(tmp_path)
    assert preflight(c, "tasks")["generation_jobs"] == 135
    result = run_pipeline(c, "tasks", max_jobs=1)
    assert result["status"] == "partial"

import json
from pathlib import Path

from feature_space_pilot.clients import HostedClient


def test_user_selected_luna_low_default_without_unsupported_sampling():
    config = json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/feature_space_pilot.json"
        ).read_text()
    )
    assert config["replicates"] == 3 and config["api_cap_usd"] == 100
    for role in ["generator", "judge"]:
        model = config[role]
        assert model["model_id"] == "gpt-6-luna"
        assert model["revision"] == "gpt-6-luna"
        assert model["reasoning_effort"] == "low"
        assert model["supports_structured_outputs"] is True
        assert (
            model["supports_seed"] is False and model["supports_temperature"] is False
        )
        assert model["prices"]["input_per_million"] == 0.1
        assert model["prices"]["output_per_million"] == 0.5
        client = HostedClient(model)
        captured = []

        def transport(payload):
            captured.append(payload)
            return dict(
                model=model["model_id"],
                choices=[dict(message=dict(content="{}"), finish_reason="stop")],
                usage=dict(prompt_tokens=5, completion_tokens=3),
            )

        client.count_transport = lambda payload: dict(input_tokens=5)
        client.transport = transport
        client.complete(
            system="Fixture",
            user="Fixture",
            schema={"type": "object"},
            seed=42,
            temperature=0.8,
        )
        assert captured[0]["reasoning"]["effort"] == "low"
        assert "temperature" not in captured[0] and "seed" not in captured[0]

"""Explicit one-call diagnostic for configured token/JSON envelopes; never a study."""

from __future__ import annotations
import uuid

from .engine import resolve, selected_items, preflight, _call, _read
from .environment import load_api_credentials
from .clients import HostedClient, generation_schema, judgment_schema
from .prompts import generation_prompt, parse_generation
from .judge import judge_prompt, parse_judgment
from .budget import Ledger, estimate_cost
from .io import write_json


def calibrate(config, *, role="judge", execute_live=False, client=None):
    from .execution import execution_config

    config = execution_config(config, "online")
    if not execute_live:
        raise ValueError("Calibration requires explicit --execute-live")
    if role not in {"generator", "judge"}:
        raise ValueError("Unknown calibration role")
    plan = preflight(config, "tasks")
    if not plan["live_ready"]:
        raise ValueError(
            "Calibration blocked by preflight: " + "; ".join(plan["blockers"])
        )
    settings = config[role]
    ceiling = estimate_cost(
        settings["prices"], settings["max_input_tokens"], settings["max_output_tokens"]
    )
    if ceiling > 0.01:
        raise ValueError("One-call calibration ceiling exceeds $0.01")
    load_api_credentials(config)
    root = resolve(config, config.get("artifact_root", "artifacts/feature_space_pilot"))
    output = root / ("calibration_" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    item = next(i for i in selected_items(config, "tasks") if i["item_id"] == "poem_T")
    if role == "generator":
        prompt = generation_prompt(item, "P1", live=True)
        schema = generation_schema(item["item_id"], prompt["cap"])
        parser = lambda raw: parse_generation(raw, item["item_id"], cap=prompt["cap"])
    else:
        feature = dict(
            feature_id="calibration_feature", text="requires eight lines of verse"
        )
        prompt = judge_prompt(item, feature)
        schema = judgment_schema(item["item_id"], feature["feature_id"])
        parser = lambda raw: parse_judgment(raw, item["item_id"], feature["feature_id"])
    path = output / "attempt.jsonl"
    with Ledger(
        root / "budget_live.jsonl",
        cap=config["api_cap_usd"],
        stage_caps=config["stage_caps"],
    ) as ledger:
        result = _call(
            client or HostedClient(settings),
            prompt,
            schema,
            settings,
            ledger,
            "calibration-" + output.name,
            "generation" if role == "generator" else "judgment",
            path,
            dict(item_id=item["item_id"]),
            parser,
            1,
            0,
        )
        attempts = _read(path)
        last = attempts[-1]
        report = dict(
            passed=result is not None,
            scientific_result=False,
            role=role,
            model=settings["model_id"],
            reasoning_effort=settings.get("reasoning_effort"),
            max_calls=1,
            output_dir=str(output),
            usage=last["usage"],
            input_token_bound=last["token_budget"]["input_token_bound"],
            token_budget=last["token_budget"],
            output_token_ceiling=settings["max_output_tokens"],
            error=last["error"],
            ledger=ledger.report(),
        )
    write_json(output / "result.json", report)
    return report


def main():
    import argparse, json
    from .engine import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/feature_space_pilot.json")
    parser.add_argument("--role", choices=["generator", "judge"], default="judge")
    parser.add_argument("--execute-live", action="store_true")
    args = parser.parse_args()
    try:
        result = calibrate(
            load_config(args.config), role=args.role, execute_live=args.execute_live
        )
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            parser.exit(1)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Calibration blocked: {error}\n")


if __name__ == "__main__":
    main()

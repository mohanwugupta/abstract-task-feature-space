"""Read-only source/model/cost preflight; never creates artifacts or calls a model."""

import argparse
import json
from .engine import load_config, preflight, demo_config


def campaign_preflight(config):
    """One cumulative allowance for separate human/task outputs and their retries."""
    from .io import digest

    if sum(config["stage_caps"].values()) > config["api_cap_usd"] + 1e-12:
        return dict(
            live_ready=False,
            blockers=["Stage budget allocations exceed cumulative budget"],
        )
    plans = {
        stage: preflight(config, stage, campaign_check=False)
        for stage in ["human", "tasks"]
    }
    accounted = plans["human"]["accounted_usd"]
    worst = sum(p["worst_case_usd"] for p in plans.values())
    remaining = sum(p["remaining_worst_case_usd"] for p in plans.values())
    blockers = sorted({b for p in plans.values() for b in p["blockers"]})
    if accounted + remaining > config["api_cap_usd"] + 1e-12:
        blockers.append("Combined remaining plan exceeds cumulative budget")
    for stage in ["generation", "judgment"]:
        from .budget import Ledger
        from .engine import resolve
        from .io import read_jsonl

        path = (
            resolve(
                config, config.get("artifact_root", "artifacts/feature_space_pilot")
            )
            / "budget_live.jsonl"
        )
        ledger = Ledger(
            path, cap=config["api_cap_usd"], stage_caps=config["stage_caps"]
        )
        ledger.events = read_jsonl(path) if path.exists() else []
        needed = sum(
            p["estimates"][stage]["remaining_worst_case_usd"] for p in plans.values()
        )
        if ledger.stage_total(stage) + needed > config["stage_caps"][stage] + 1e-12:
            blockers.append(f"Combined {stage} plan exceeds stage budget")
    return dict(
        stages=plans,
        generation_jobs=sum(p["generation_jobs"] for p in plans.values()),
        judgment_cells_upper_bound=sum(
            p["judgment_cells_upper_bound"] for p in plans.values()
        ),
        worst_case_usd=worst,
        remaining_worst_case_usd=remaining,
        accounted_usd=accounted,
        total_liability_usd=accounted + remaining,
        budget_cap_usd=config["api_cap_usd"],
        remaining_after_plan_usd=config["api_cap_usd"] - accounted - remaining,
        preflight_hash=digest([p["preflight_hash"] for p in plans.values()]),
        blockers=sorted(set(blockers)),
        live_ready=not config.get("demo") and not blockers,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/feature_space_pilot.json")
    p.add_argument("--stage", choices=["human", "tasks", "all"])
    p.add_argument("--demo", action="store_true")
    p.add_argument(
        "--campaign",
        action="store_true",
        help="Combined cumulative human+task budget and scientific readiness",
    )
    a = p.parse_args()
    c = load_config(a.config)
    if a.demo:
        c = demo_config(c["root"])
    try:
        if a.campaign:
            result = campaign_preflight(c)
        elif a.stage:
            result = preflight(c, a.stage)
        else:
            result = {s: preflight(c, s) for s in ["human", "tasks"]}
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, OSError, KeyError) as e:
        p.exit(2, f"Preflight failed: {e}\n")


if __name__ == "__main__":
    main()

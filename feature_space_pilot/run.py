"""Run offline demos or explicitly preflighted hosted frontier API stages."""

import argparse
import json
from .engine import demo_config, load_config, preflight, run_pipeline, smoke_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/feature_space_pilot.json")
    p.add_argument("--stage", required=True, choices=["human", "tasks", "all"])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Synthetic offline fixtures, never a human benchmark",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Frozen eight-target source-stratified smoke; use --stage human",
    )
    p.add_argument("--preflight-hash", help="Exact hash printed by the current dry-run")
    p.add_argument("--output-root", help="Offline demo workspace root")
    p.add_argument(
        "--max-jobs",
        type=int,
        help="Stop after this many pending generation jobs and write a partial manifest",
    )
    a = p.parse_args()
    c = load_config(a.config)
    if a.demo:
        c = demo_config(a.output_root or c["root"])
    elif a.output_root:
        p.error("--output-root is only supported for offline demos")
    if a.max_jobs is not None and a.max_jobs < 1:
        p.error("--max-jobs must be positive")
    try:
        if a.smoke:
            if a.stage != "human":
                p.error("--smoke requires --stage human")
            c = smoke_config(c)
        if a.dry_run:
            result = preflight(c, a.stage)
        else:
            result = run_pipeline(
                c,
                a.stage,
                execute_live=a.execute_live,
                preflight_hash=a.preflight_hash,
                max_jobs=a.max_jobs,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as e:
        p.exit(2, f"Pilot blocked: {e}\n")


if __name__ == "__main__":
    main()

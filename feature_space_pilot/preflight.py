"""Read-only source/model/cost preflight; never creates artifacts or calls a model."""

import argparse
import json
from .engine import load_config, preflight, demo_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/feature_space_pilot.json")
    p.add_argument("--stage", choices=["human", "tasks", "all"])
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()
    c = load_config(a.config)
    if a.demo:
        c = demo_config(c["root"])
    try:
        result = (
            preflight(c, a.stage)
            if a.stage
            else {s: preflight(c, s) for s in ["human", "tasks"]}
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, OSError, KeyError) as e:
        p.exit(2, f"Preflight failed: {e}\n")


if __name__ == "__main__":
    main()

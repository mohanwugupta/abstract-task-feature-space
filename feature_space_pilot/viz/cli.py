"""Render a single frozen run offline. Never calls the generation/API pipeline."""

import argparse
import json
from pathlib import Path
from .load import load_run
from .derive import human_sources
from .exports import export_run, ExportConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-features", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    try:
        run = load_run(args.run_dir)
        if args.dry_run or args.inventory:
            result = dict(
                run_id=run.manifest["run_id"],
                demo=run.demo,
                status=run.manifest["status"],
                n_items=len(run.items),
                n_features=len(run.features),
                n_resolved=len(run.judgments),
                manifest_sha256=run.manifest_hash,
                human_sources=[
                    {k: s.get(k) for k in ["source_dataset", "available", "reason"]}
                    for s in human_sources(run)
                ],
                no_api_calls=True,
            )
        else:
            result = export_run(
                args.run_dir,
                args.output_dir,
                config=ExportConfig(
                    args.max_features, args.dpi, args.seed, args.cluster
                ),
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Visualization blocked: {error}\n")


if __name__ == "__main__":
    main()

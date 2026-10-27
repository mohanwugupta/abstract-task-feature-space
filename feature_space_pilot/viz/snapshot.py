"""Copy a verified sealed checkpoint for analysis while its collector keeps running."""

from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import tempfile
import time
from datetime import datetime, timezone
from ..io import write_json
from ..report import verify_manifest


def snapshot_run(run_dir, output_root=None):
    source = Path(run_dir).expanduser().resolve()
    root = (
        Path(output_root).expanduser().resolve()
        if output_root
        else source.parent.parent / "feature_space_analysis_snapshots"
    )
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Analysis snapshots must be outside the source run")
    for attempt in range(3):
        try:
            raw = (source / "manifest.json").read_bytes()
            manifest = verify_manifest(source)
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", manifest["run_id"]):
                raise ValueError("Invalid source run identity")
            if raw != (source / "manifest.json").read_bytes():
                raise ValueError("Source checkpoint changed while copying")
            checksum = hashlib.sha256(raw).hexdigest()
            parent = root / manifest["run_id"]
            parent.mkdir(parents=True, exist_ok=True)
            target = parent / checksum[:16]
            if target.exists():
                verify_manifest(target)
                if (target / "manifest.json").read_bytes() != raw:
                    raise ValueError("Snapshot identity collision")
                return str(target)
            with tempfile.TemporaryDirectory(
                prefix=".snapshot-", dir=parent
            ) as temporary:
                stage = Path(temporary) / "run"
                stage.mkdir()
                for name in manifest["artifacts"]:
                    if Path(name).name != name:
                        raise ValueError("Source artifact must be a top-level file")
                    shutil.copyfile(source / name, stage / name)
                (stage / "manifest.json").write_bytes(raw)
                verify_manifest(stage)
                if raw != (source / "manifest.json").read_bytes():
                    raise ValueError("Source checkpoint changed while copying")
                stage.rename(target)
            write_json(
                parent / (checksum[:16] + ".snapshot.json"),
                dict(
                    source_run=str(source),
                    source_manifest_sha256=checksum,
                    snapshot=str(target),
                    captured_utc=datetime.now(timezone.utc).isoformat(),
                    policy="Exact verified checkpoint copy; original manifest and receipts never resealed or edited",
                ),
            )
            return str(target)
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                dict(snapshot_dir=snapshot_run(args.run_dir, args.output_root)),
                indent=2,
            )
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Snapshot blocked: {error}\n")


if __name__ == "__main__":
    main()

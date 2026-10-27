"""Read-only progress snapshots, including journals of an already-running process."""

from __future__ import annotations
import json
from pathlib import Path
import time


def _rows(path):
    if not path.exists():
        return [], False
    data = path.read_bytes()
    lines = data.splitlines()
    rows = []
    tail = False
    for n, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeError):
            if n == len(lines) - 1 and not data.endswith(b"\n"):
                tail = True
            else:
                raise ValueError(f"Invalid journal line in {path.name}") from None
    return rows, tail


def inspect_run(path):
    path = Path(path).expanduser().resolve()
    if not (path / "generation_jobs.jsonl").is_file():
        raise ValueError("Select a run directory with generation_jobs.jsonl")
    config = (
        json.loads((path / "frozen_config.json").read_text())
        if (path / "frozen_config.json").exists()
        else {}
    )
    result = dict(
        run_id=path.name,
        run_dir=str(path),
        approximate_live_snapshot=True,
        execution_mode=config.get("execution_mode", "online"),
        generator_model=config.get("generator", {}).get("model_id"),
        judge_model=config.get("judge", {}).get("model_id"),
    )
    for stage, journal in [
        ("generation", "generations.jsonl"),
        ("judgment", "judgments.jsonl"),
    ]:
        jobs, _ = _rows(path / (stage + "_jobs.jsonl"))
        records, tail = _rows(path / journal)
        successful = {r["job_id"]: r for r in records if r.get("parse_status") == "ok"}
        resolved = sum(
            r.get("status") == "ok"
            and type(r.get("applies")) is int
            and r["applies"] in (0, 1)
            for r in successful.values()
        )
        file = path / journal
        result[stage] = dict(
            planned=len(jobs),
            completed=len(successful),
            attempts=len(records),
            errors=sum(r.get("parse_status") != "ok" for r in records),
            resolved=resolved,
            ambiguous=sum(r.get("status") == "ambiguous" for r in successful.values()),
            incomplete_tail=tail,
            last_item=records[-1].get("item_id") if records else None,
            seconds_since_update=(
                max(0, time.time() - file.stat().st_mtime) if file.exists() else None
            ),
        )
    ledger_root = Path(config.get("artifact_root", "artifacts/feature_space_pilot"))
    if not ledger_root.is_absolute():
        ledger_root = Path(config.get("root", path.parents[2])) / ledger_root
    events, tail = _rows(
        ledger_root
        / ("budget_demo.jsonl" if config.get("demo") else "budget_live.jsonl")
    )
    tickets = {}
    for event in events:
        if event["event"] == "reserve":
            tickets[event["ticket"]] = dict(event, settled=False)
        elif event["event"] in {"settle", "overage"} and event["ticket"] in tickets:
            tickets[event["ticket"]].update(amount=event["amount"], settled=True)
    total = sum(t["amount"] for t in tickets.values())
    result["budget"] = dict(
        accounted_usd=total,
        cap_usd=config.get("api_cap_usd"),
        unsettled_reservations=sum(not t["settled"] for t in tickets.values()),
        incomplete_tail=tail,
    )
    manifest = path / "manifest.json"
    result["status"] = (
        json.loads(manifest.read_text()).get("status")
        if manifest.exists()
        else "running_or_interrupted"
    )
    batches = path / "batch_state.json"
    amendment = path / "execution_amendment.json"
    if amendment.exists():
        execution = json.loads(amendment.read_text())["config"]
        result["original_execution_mode"] = result["execution_mode"]
        result["execution_mode"] = "mixed_online_batch"
        result["judge_model"] = execution["judge"]["model_id"]
    if batches.exists():
        state = json.loads(batches.read_text())
        result["batches"] = [
            dict(
                id=b.get("id"),
                stage=b["stage"],
                status=b["status"],
                request_counts=b.get("request_counts"),
            )
            for b in state.get("batches", [])
        ]
    return result


def format_status(result):
    g, j = result["generation"], result["judgment"]
    budget = result["budget"]
    percent = lambda r: (
        f"{100*r['completed']/r['planned']:.1f}%" if r["planned"] else "not planned yet"
    )
    lines = [
        f"Run: {result['run_id']} ({result['status']})",
        f"Mode: {result['execution_mode']}; generator={result['generator_model'] or 'unspecified'}, judge={result['judge_model'] or 'unspecified'}",
        f"Generation: {g['completed']}/{g['planned']} ({percent(g)}); {g['attempts']} attempts, {g['errors']} errors",
        f"Judgments: {j['completed']}/{j['planned']} ({percent(j)}); {j['resolved']} resolved 0/1, {j['ambiguous']} ambiguous; {j['errors']} errors",
        f"Budget accounted: ${budget['accounted_usd']:.4f}; {budget['unsettled_reservations']} outstanding reservation(s)",
    ]
    current = j if j["planned"] else g
    if current["last_item"]:
        lines.append(
            f"Latest item: {current['last_item']}; last journal update {current['seconds_since_update']:.0f}s ago"
        )
    if g["incomplete_tail"] or j["incomplete_tail"]:
        lines.append(
            "A partially written final line will be included on the next refresh."
        )
    for batch in result.get("batches", []):
        lines.append(
            f"Batch {batch['id'] or '(not submitted)'}: {batch['stage']} / {batch['status']} / {batch.get('request_counts')}"
        )
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        if args.interval < 1:
            raise ValueError("Watch interval must be at least one second")
        path = args.run_dir
        if not path:
            root = Path("artifacts/feature_space_pilot")
            choices = [
                p for p in root.glob("live_*") if (p / "generation_jobs.jsonl").exists()
            ]
            if not choices:
                raise ValueError("No live runs found; supply --run-dir")
            path = max(
                choices, key=lambda p: (p / "generation_jobs.jsonl").stat().st_mtime
            )
        while True:
            result = inspect_run(path)
            print(
                json.dumps(result, indent=2) if args.json else format_status(result),
                flush=True,
            )
            if not args.watch:
                break
            print("", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Status unavailable: {error}\n")


if __name__ == "__main__":
    main()

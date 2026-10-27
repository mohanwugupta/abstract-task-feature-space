"""Continue frozen, generation-complete runs through Batch without changing job IDs."""

from __future__ import annotations
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .bank import consolidate
from .batch import BatchSession, BatchTransientError, BatchCreationUncertain
from .budget import Ledger, estimate_cost, validate_prices
from .clients import judgment_schema
from .engine import (
    _read,
    _resume_manifest,
    _finish_run,
    _write_judgment_outputs,
    preflight,
    resolve,
    selected_items,
)
from .environment import load_api_credentials
from .execution import execution_config
from .io import append_jsonl, canonical, digest, file_hash, read_jsonl, write_json
from .judge import judge_prompt, RUBRIC
from .planner import plan_generation
from .progress import console_reporter, emit
from .prompts import DIRECTORY

FROZEN = (
    "frozen_config.json",
    "preflight.json",
    "stimuli.jsonl",
    "generation_jobs.jsonl",
    "generations.jsonl",
    "feature_bank.csv",
    "judgment_jobs.jsonl",
    "frozen_prompts.json",
    "frozen_source_registry.json",
    "reconciliation_fixture.json",
)


def _execution(c):
    result = copy.deepcopy(c)
    catalog = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/feature_space_pilot_mini_online.json"
        ).read_text()
    )
    for role in ("generator", "judge"):
        s = result[role]
        if c.get("demo"):
            s["batch_prices"] = copy.deepcopy(s["prices"])
        elif not s.get("batch_prices"):
            prices = catalog[role]["batch_prices"]
            if (s["model_id"], s["revision"]) != (
                prices["model_id"],
                prices["revision"],
            ):
                raise ValueError(
                    "Verified Batch prices for the frozen model are required; no model substitution"
                )
            s["batch_prices"] = copy.deepcopy(prices)
        validate_prices(s["batch_prices"])
        if (s["model_id"], s["revision"]) != (
            s["batch_prices"]["model_id"],
            s["batch_prices"]["revision"],
        ):
            raise ValueError("Batch prices must match the frozen model/revision")
    result.update(
        batch_max_requests=1000,
        batch_max_in_flight=16,
        batch_max_queued_input_tokens=10_000_000,
    )
    return execution_config(result, "batch")


def _context(out, *, allow_collector_update=False):
    out = Path(out).expanduser().resolve()
    c = json.loads((out / "frozen_config.json").read_text())
    expected_parent = resolve(
        c, c.get("artifact_root", "artifacts/feature_space_pilot")
    ).resolve()
    if out.parent != expected_parent:
        raise ValueError(
            "Continuation must retain the original artifact root and cumulative ledger"
        )
    original = json.loads((out / "preflight.json").read_text())
    stage = original["stage"]
    if stage not in {"human", "tasks"}:
        raise ValueError("Continuation requires one frozen human or tasks stage")
    if (out / "manifest.json").exists():
        _resume_manifest(out)
    frozen = {name: file_hash(out / name) for name in FROZEN}
    amendment = None
    if (out / "execution_amendment.json").exists():
        amendment = json.loads((out / "execution_amendment.json").read_text())
        if amendment["amendment_hash"] != digest(
            {k: v for k, v in amendment.items() if k != "amendment_hash"}
        ):
            raise ValueError("Execution amendment hash mismatch")
        if frozen != amendment["frozen_hashes"]:
            raise ValueError("Frozen continuation input hash changed")
        checkpoint = amendment["original_judgment_checkpoint"]
        data = (out / "judgments.jsonl").read_bytes()
        import hashlib

        if (
            len(data) < checkpoint["bytes"]
            or hashlib.sha256(data[: checkpoint["bytes"]]).hexdigest()
            != checkpoint["sha256"]
        ):
            raise ValueError("Original judgment prefix/checkpoint changed")
        execution = amendment["config"]
    else:
        execution = _execution(c)
    current = preflight(c, stage)
    if amendment and amendment["source_code_hashes"] != current["source_code_hashes"]:
        changed = {
            name
            for name in set(amendment["source_code_hashes"])
            | set(current["source_code_hashes"])
            if amendment["source_code_hashes"].get(name)
            != current["source_code_hashes"].get(name)
        }
        if not allow_collector_update or not changed <= {
            "batch.py",
            "continuation.py",
            "status.py",
        }:
            raise ValueError(
                "Execution code changed after handoff; only an explicitly recorded collector repair may update batch.py, continuation.py or status.py"
            )
    if current["blockers"]:
        raise ValueError("; ".join(current["blockers"]))
    for key in ("source_hashes", "input_hashes", "prompt_hashes"):
        if current[key] != original[key]:
            raise ValueError("Frozen scientific source/input/prompt hashes changed")
    for name in (
        "bank.py",
        "judge.py",
        "prompts.py",
        "planner.py",
        "registry.py",
        "human_inputs.py",
        "scientific.py",
    ):
        old = original["source_code_hashes"].get(name)
        if old and file_hash(Path(__file__).parent / name) != old:
            raise ValueError(
                "Scientific code changed; continuation requires the original " + name
            )
    items = read_jsonl(out / "stimuli.jsonl")
    if items != selected_items(c, stage):
        raise ValueError("Frozen stimulus inventory changed")
    prompts = {p.name: p.read_text() for p in sorted(DIRECTORY.glob("*.txt"))}
    if prompts != json.loads((out / "frozen_prompts.json").read_text()):
        raise ValueError("Frozen prompt text changed")
    jobs = plan_generation(
        items,
        c["generator"],
        replicates=c["replicates"],
        seed=c["seed"],
        live=not c.get("demo"),
        protocols=c.get("protocols"),
    )
    if jobs != read_jsonl(out / "generation_jobs.jsonl"):
        raise ValueError("Frozen generation jobs changed")
    gen_records = _read(out / "generations.jsonl")
    generations = [r for r in gen_records if r["parse_status"] == "ok"]
    if len(generations) != len(jobs) or {r["job_id"] for r in generations} != {
        j["job_id"] for j in jobs
    }:
        raise ValueError("All frozen generations must be complete before continuation")
    bank, _, _ = consolidate(
        generations, cap=c["candidate_cap"], merges=c.get("merges", [])
    )
    with (out / "feature_bank.csv").open(newline="") as f:
        bank_rows = list(csv.DictReader(f))
    expected_bank = [
        {
            k: canonical(v) if isinstance(v, (list, dict)) else str(v)
            for k, v in row.items()
        }
        for row in bank
    ]
    if not bank or bank_rows != expected_bank:
        raise ValueError("Frozen feature bank differs from generation inventory")
    judge_jobs = []
    for item in items:
        for feature in bank:
            p = judge_prompt(item, feature)
            judge_jobs.append(
                dict(
                    job_id=digest(
                        [
                            item["content_hash"],
                            feature["feature_id"],
                            p["prompt_hash"],
                            c["judge"],
                        ]
                    ),
                    item_id=item["item_id"],
                    feature_id=feature["feature_id"],
                    **p,
                )
            )
    if judge_jobs != read_jsonl(out / "judgment_jobs.jsonl"):
        raise ValueError("Frozen judgment scope or prompt changed")
    records = _read(out / "judgments.jsonl")
    ids = {j["job_id"] for j in judge_jobs}
    successful = [r["job_id"] for r in records if r["parse_status"] == "ok"]
    if any(r["job_id"] not in ids for r in records) or len(successful) != len(
        set(successful)
    ):
        raise ValueError("Unknown or duplicate successful frozen judgment")
    ledger_path = expected_parent / (
        "budget_demo.jsonl" if c.get("demo") else "budget_live.jsonl"
    )
    ledger = Ledger(ledger_path, cap=c["api_cap_usd"], stage_caps=c["stage_caps"])
    ledger.events = read_jsonl(ledger_path)
    balances = ledger._balances()
    for record in gen_records + records:
        if balances.get(record["budget_ticket"], {}).get("job_id") != record["job_id"]:
            raise ValueError(
                "Primary receipt does not match original ledger reservation"
            )
    used = {}
    for event in ledger.events:
        if event["event"] == "reserve":
            used[event["job_id"]] = used.get(event["job_id"], 0) + 1
    pending = [j for j in judge_jobs if j["job_id"] not in successful]
    attempts = sum(
        max(0, c["max_attempts"] - used.get(j["job_id"], 0)) for j in pending
    )
    settings = execution["judge"]
    liability = attempts * estimate_cost(
        settings["prices"], settings["max_input_tokens"], settings["max_output_tokens"]
    )
    if (
        ledger.total + liability > ledger.cap + 1e-12
        or ledger.stage_total("judgment") + liability
        > ledger.stage_caps["judgment"] + 1e-12
    ):
        raise ValueError(
            "Continuation liability exceeds original cumulative/stage budget"
        )
    plan = dict(
        run_id=out.name,
        output_dir=str(out),
        execution_mode="batch",
        model_id=settings["model_id"],
        completed_generation_jobs=len(generations),
        completed_judgments=len(successful),
        planned_judgments=len(judge_jobs),
        pending_judgments=len(pending),
        remaining_attempts_upper_bound=attempts,
        remaining_worst_case_usd=liability,
        accounted_usd=ledger.total,
        budget_cap_usd=ledger.cap,
        blockers=[],
        live_ready=not c.get("demo"),
        batch_max_requests=execution["batch_max_requests"],
        batch_max_in_flight=execution["batch_max_in_flight"],
        batch_max_queued_input_tokens=execution["batch_max_queued_input_tokens"],
    )
    plan["preflight_hash"] = digest(
        [
            frozen,
            digest(execution),
            current["source_code_hashes"],
            file_hash(out / "judgments.jsonl"),
            digest(ledger.events),
        ]
    )
    return (
        out,
        c,
        original,
        execution,
        amendment,
        frozen,
        items,
        jobs,
        generations,
        bank,
        judge_jobs,
        records,
        ledger_path,
        current,
        plan,
    )


def continuation_plan(out):
    """Read-only planning; never loads credentials, locks the ledger or calls APIs."""
    return _context(out)[-1]


def collector_repair_plan(out):
    context = _context(out, allow_collector_update=True)
    amendment, current, plan = context[4], context[-2], context[-1]
    if amendment is None:
        raise ValueError("Collector repair requires an existing execution amendment")
    changed = sorted(
        name
        for name in set(amendment["source_code_hashes"])
        | set(current["source_code_hashes"])
        if amendment["source_code_hashes"].get(name)
        != current["source_code_hashes"].get(name)
    )
    return dict(plan, changed_collector_files=changed)


def repair_collector(out, *, reason, preflight_hash):
    """Audited local code amendment; never changes scientific inputs or submits work."""
    if not reason or not reason.strip():
        raise ValueError("Record the explicit reason for the collector repair")
    context = _context(out, allow_collector_update=True)
    (
        out,
        c,
        original,
        execution,
        amendment,
        frozen,
        items,
        jobs,
        generations,
        bank,
        judge_jobs,
        records,
        ledger_path,
        current,
        plan,
    ) = context
    if amendment is None or preflight_hash != plan["preflight_hash"]:
        raise ValueError("Obtain a fresh collector-repair dry-run hash")
    changed = collector_repair_plan(out)["changed_collector_files"]
    with Ledger(
        ledger_path, cap=c["api_cap_usd"], stage_caps=c["stage_caps"]
    ) as ledger:
        if collector_repair_plan(out)["preflight_hash"] != preflight_hash:
            raise ValueError("Collector repair inputs changed; obtain a fresh dry run")
        if not changed:
            return dict(status="unchanged", changed_collector_files=[])
        history = out / "execution_amendment_history.jsonl"
        prior = read_jsonl(history) if history.exists() else []
        if not prior or prior[-1] != amendment:
            append_jsonl(history, amendment)
        updated = copy.deepcopy(amendment)
        updated.setdefault("collector_repairs", []).append(
            dict(
                reason=reason.strip(),
                date_utc=datetime.now(timezone.utc).isoformat(),
                changed_files=changed,
                previous_amendment_hash=amendment["amendment_hash"],
            )
        )
        updated["source_code_hashes"] = current["source_code_hashes"]
        updated.pop("amendment_hash", None)
        updated["amendment_hash"] = digest(updated)
        if (out / "manifest.json").exists():
            append_jsonl(out / "manifest_history.jsonl", _resume_manifest(out))
        write_json(out / "execution_amendment.json", updated)
        judgments = [r for r in records if r["parse_status"] == "ok"]
        judgments, status = _write_judgment_outputs(c, out, items, bank, judgments)
        state = json.loads((out / "batch_state.json").read_text())
        if any(not b.get("ingested") for b in state["batches"]):
            status = "batch_pending"
        result = _finish_run(
            c=c,
            out=out,
            plan=original,
            stage=original["stage"],
            jobs=jobs,
            items=items,
            generations=generations,
            bank=bank,
            judgments=judgments,
            ledger=ledger,
            execute_live=not c.get("demo"),
            evaluation_config=c,
            source_hashes=original["source_hashes"],
            status=status,
            execution_amendment=updated,
        )
        return dict(
            status=result["status"],
            changed_collector_files=changed,
            execution_amendment_hash=updated["amendment_hash"],
        )


def watch_continuation(
    out, *, preflight_hash, watch=False, poll_seconds=30, progress=None
):
    """Reconnect safely; uncertain POSTs are reconciled, never automatically replayed."""
    while True:
        try:
            if preflight_hash is None:
                preflight_hash = continuation_plan(out)["preflight_hash"]
            result = continue_batch(
                out, execute_live=True, preflight_hash=preflight_hash, progress=progress
            )
        except (BatchTransientError, BatchCreationUncertain):
            if not watch:
                raise
            emit(
                progress,
                "collector_retry",
                message="Temporary connection failure or uncertain acknowledgement; retaining reservations and retrying reconciliation after the next poll",
            )
        else:
            if not watch or result["status"] != "batch_pending":
                return result
        emit(
            progress,
            "waiting",
            message=f"Waiting {poll_seconds:g}s; reconnection is automatic, Ctrl+C detaches without cancelling provider work",
        )
        preflight_hash = None
        time.sleep(poll_seconds)


def continue_batch(
    out, *, execute_live=False, preflight_hash=None, api=None, progress=None
):
    context = _context(out)
    (
        out,
        c,
        original,
        execution,
        amendment,
        frozen,
        items,
        jobs,
        generations,
        bank,
        judge_jobs,
        records,
        ledger_path,
        current,
        plan,
    ) = context
    if preflight_hash != plan["preflight_hash"]:
        raise ValueError("Obtain a fresh continuation dry-run preflight hash")
    if not c.get("demo"):
        if not execute_live or api is not None:
            raise ValueError(
                "Live continuation requires explicit execution; injected clients prohibited"
            )
        load_api_credentials(execution)
    elif execute_live or api is None:
        raise ValueError("Offline continuation requires an injected fixture API")
    with Ledger(
        ledger_path, cap=c["api_cap_usd"], stage_caps=c["stage_caps"]
    ) as ledger:
        if continuation_plan(out)["preflight_hash"] != preflight_hash:
            raise ValueError(
                "Continuation inputs or balance changed; obtain a fresh dry-run hash"
            )
        if (out / "manifest.json").exists():
            old = _resume_manifest(out)
            if old["status"] == "complete":
                return dict(old, output_dir=str(out))
            append_jsonl(out / "manifest_history.jsonl", old)
        if amendment is None:
            data = (out / "judgments.jsonl").read_bytes()
            amendment = dict(
                version="batch_continuation_v1",
                date_utc=datetime.now(timezone.utc).isoformat(),
                original_design_hash=original["design_hash"],
                frozen_hashes=frozen,
                original_judgment_checkpoint=dict(
                    bytes=len(data), sha256=file_hash(out / "judgments.jsonl")
                ),
                completed_judgments_at_handoff=plan["completed_judgments"],
                config=execution,
                source_code_hashes=current["source_code_hashes"],
                policy="Transport/pricing/scheduling amendment only; retain model, prompts, bank, original job IDs, paid receipts, maximum attempts and cumulative ledger",
            )
            amendment["amendment_hash"] = digest(amendment)
            write_json(out / "execution_amendment.json", amendment)
        session = BatchSession(execution, out, ledger, api=api, progress=progress)
        session.refresh()
        records = _read(out / "judgments.jsonl")
        done = {r["job_id"] for r in records if r["parse_status"] == "ok"}
        emit(
            progress,
            "stage_progress",
            stage="judgment",
            completed=len(done),
            planned=len(judge_jobs),
        )
        for job in judge_jobs:
            if job["job_id"] in done:
                continue
            session.enqueue(
                prompt=job,
                schema=judgment_schema(job["item_id"], job["feature_id"]),
                settings=execution["judge"],
                job_id=job["job_id"],
                stage="judgment",
                journal_name="judgments.jsonl",
                record_base=dict(
                    item_id=job["item_id"],
                    feature_id=job["feature_id"],
                    rubric_version=RUBRIC,
                    judge_model=c["judge"]["model_id"],
                    prompt_hash=job["prompt_hash"],
                ),
            )
        session.flush("judgment")
        judgments = [
            r for r in _read(out / "judgments.jsonl") if r["parse_status"] == "ok"
        ]
        judgments, status = _write_judgment_outputs(c, out, items, bank, judgments)
        if session.active:
            status = "batch_pending"
        return _finish_run(
            c=c,
            out=out,
            plan=original,
            stage=original["stage"],
            jobs=jobs,
            items=items,
            generations=generations,
            bank=bank,
            judgments=judgments,
            ledger=ledger,
            execute_live=execute_live,
            evaluation_config=c,
            source_hashes=original["source_hashes"],
            status=status,
            progress=progress,
            execution_amendment=amendment,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    parser.add_argument("--preflight-hash")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--repair-collector",
        metavar="REASON",
        help="Explicitly record a collector-only code update, preserving scientific pins",
    )
    args = parser.parse_args()
    try:
        if args.poll_seconds < 5:
            raise ValueError("Polling interval must be at least five seconds")
        if args.repair_collector:
            result = (
                collector_repair_plan(args.run_dir)
                if args.dry_run
                else repair_collector(
                    args.run_dir,
                    reason=args.repair_collector,
                    preflight_hash=args.preflight_hash,
                )
            )
        elif args.dry_run:
            result = continuation_plan(args.run_dir)
        else:
            reporter = None if args.quiet else console_reporter()
            result = watch_continuation(
                args.run_dir,
                preflight_hash=args.preflight_hash,
                watch=args.watch,
                poll_seconds=args.poll_seconds,
                progress=reporter,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        parser.exit(
            130,
            "Detached; provider Batch jobs, receipts and reservations are retained.\n",
        )
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Continuation blocked: {error}\n")


if __name__ == "__main__":
    main()

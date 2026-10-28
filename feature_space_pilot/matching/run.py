"""Preflight, submit and collect separate automated semantic-matching analyses."""

from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import time

from ..batch import OpenAIBatchAPI, BatchTransientError, BatchCreationUncertain
from ..budget import Ledger, estimate_cost, validate_prices
from ..clients import build_payload
from ..engine import load_config as load_model_config, _read
from ..environment import load_api_credentials
from ..io import canonical, digest, file_hash, read_jsonl, write_json
from ..progress import console_reporter, emit
from ..references import load_reference, validate_item_alignment
from ..token_accounting import validate_payload, provider_count_payload
from ..viz.load import load_run
from .protocol import (
    build_inventory,
    build_jobs,
    payload_for_job,
    response_schema,
    RUBRIC,
    VERSION,
)
from .batch import MatchingBatchSession
from .analysis import write_analysis


def load_settings(path):
    path = Path(path).resolve()
    options = json.loads(path.read_text())
    c = load_model_config(path.parent / options["model_config"])
    settings = copy.deepcopy(c["judge"])
    settings.update(
        max_input_tokens=options["max_input_tokens"],
        max_output_tokens=options["max_output_tokens"],
    )
    if (
        settings["pricing_mode"] != "batch"
        or settings["api_format"] != "responses"
        or not settings.get("supports_structured_outputs")
    ):
        raise ValueError("Semantic matching requires verified Batch Responses settings")
    validate_prices(settings["prices"])
    if (
        settings["prices"]["model_id"] != settings["model_id"]
        or settings["prices"]["revision"] != settings["revision"]
    ):
        raise ValueError("Semantic grader model and prices differ")
    c["generator"] = copy.deepcopy(settings)
    c["judge"] = settings
    c.update(
        max_attempts=options["max_attempts"],
        batch_max_requests=options["batch_max_requests"],
        batch_max_in_flight=options["batch_max_in_flight"],
        batch_max_queued_input_tokens=options["batch_max_queued_input_tokens"],
        semantic_options=options,
    )
    if type(c["max_attempts"]) is not int or not 1 <= c["max_attempts"] <= 2:
        raise ValueError("Semantic matching allows at most two attempts")
    return c


def _ledger(c):
    path = Path(c.get("artifact_root", "artifacts/feature_space_pilot"))
    if not path.is_absolute():
        path = Path(c["root"]) / path
    return Ledger(
        path / ("budget_demo.jsonl" if c.get("demo") else "budget_live.jsonl"),
        cap=c["api_cap_usd"],
        stage_caps=c["stage_caps"],
    )


def make_design(inventory, source, c):
    """Pure frozen job design; fixture sources remain ineligible for live execution."""
    options = c["semantic_options"]
    jobs = build_jobs(inventory, chunk_size=options["propositions_per_request"])
    code = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    code.update(
        {
            "support/" + name: file_hash(Path(__file__).parents[1] / name)
            for name in [
                "batch.py",
                "budget.py",
                "clients.py",
                "engine.py",
                "environment.py",
                "io.py",
                "prompts.py",
                "references.py",
                "token_accounting.py",
            ]
        }
    )
    identity_config = {k: v for k, v in c.items() if k not in {"root", "artifact_root"}}
    design = digest(
        [
            VERSION,
            inventory,
            source["source_manifest_sha256"],
            source.get("reference_sha256"),
            identity_config,
            code,
            RUBRIC,
        ]
    )
    for job in jobs:
        job["job_id"] = "semantic_" + digest([design, job["job_id"]])
    root = Path(options["analysis_root"])
    if not root.is_absolute():
        root = Path(c["root"]) / root
    out = root / ("semantic_" + design[:16])
    if source.get("snapshot_dir"):
        original = Path(source["snapshot_dir"]).resolve()
        if (
            out.resolve() == original
            or out.resolve().is_relative_to(original)
            or original.is_relative_to(out.resolve())
        ):
            raise ValueError("Matching analysis must be outside its source snapshot")
    s = c["judge"]
    ceiling = (
        len(jobs)
        * c["max_attempts"]
        * estimate_cost(s["prices"], s["max_input_tokens"], s["max_output_tokens"])
    )
    if ceiling > options["completion_budget_usd"] + 1e-12:
        raise ValueError(
            "Semantic completion envelope exceeds its declared analysis budget"
        )
    for job in jobs:
        validate_payload(
            build_payload(
                s,
                system=RUBRIC,
                user=canonical(payload_for_job(job)),
                schema=response_schema(job),
            ),
            s,
        )
    return dict(
        analysis_id=out.name,
        design_hash=design,
        output_dir=str(out.resolve()),
        inventory=inventory,
        source=source,
        config=c,
        source_code_hashes=code,
        jobs=jobs,
        completion_ceiling_usd=ceiling,
        rubric=RUBRIC,
    )


def make_plan(source_run, config):
    run = load_run(source_run)
    if run.demo:
        raise ValueError(
            "Synthetic source runs are not empirical semantic-recovery data"
        )
    spec = next(
        (
            s
            for s in run.sources
            if s.get("dataset_id") == "vinson_2008"
            and s.get("verified")
            and s.get("kind") == "human"
        ),
        None,
    )
    if spec is None:
        raise ValueError("Verified Vinson human feature-code source required")
    path = Path(spec["path"])
    if not path.is_absolute():
        path = Path(run.config["root"]) / path
    reference = load_reference(dict(spec, path=str(path)), live=True)
    for item in run.items.values():
        if item.get("source_dataset") == "vinson_2008":
            validate_item_alignment(item, reference)
    inventory = build_inventory(
        run.items,
        run.generations,
        reference["rows"],
        comparison_cap=config["semantic_options"]["comparison_cap"],
    )
    source = dict(
        snapshot_dir=str(run.path),
        source_manifest_sha256=run.manifest_hash,
        source_run_id=run.manifest["run_id"],
        reference_path=str(path),
        reference_sha256=spec["sha256"],
        source_verified=True,
        generator_models=run.manifest["model_snapshots"],
        excluded_sources=[
            s.get("dataset_id")
            for s in run.sources
            if s.get("dataset_id") != "vinson_2008"
        ],
    )
    return make_design(inventory, source, config)


def preflight(plan):
    c = plan["config"]
    ledger = _ledger(c)
    ledger.events = read_jsonl(ledger.path) if ledger.path.exists() else []
    ledger._balances()
    records = _read(Path(plan["output_dir"]) / "judgments.jsonl")
    complete = {r["job_id"] for r in records if r["parse_status"] == "ok"}
    used = {}
    for event in ledger.events:
        if event["event"] == "reserve":
            used[event["job_id"]] = used.get(event["job_id"], 0) + 1
    attempts = sum(
        max(0, c["max_attempts"] - used.get(j["job_id"], 0))
        for j in plan["jobs"]
        if j["job_id"] not in complete
    )
    s = c["judge"]
    remaining = attempts * estimate_cost(
        s["prices"], s["max_input_tokens"], s["max_output_tokens"]
    )
    blockers = []
    if (
        ledger.total + remaining > c["api_cap_usd"] + 1e-12
        or ledger.stage_total("adjudication") + remaining
        > c["stage_caps"]["adjudication"] + 1e-12
    ):
        blockers.append("Shared cumulative/adjudication budget would be exceeded")
    return dict(
        analysis_id=plan["analysis_id"],
        output_dir=plan["output_dir"],
        source_run_id=plan["source"]["source_run_id"],
        model_id=s["model_id"],
        reasoning_effort=s.get("reasoning_effort"),
        execution_mode="batch",
        human_reviewed=False,
        targets=len(plan["inventory"]["items"]),
        propositions=len(plan["inventory"]["propositions"]),
        study_requests=sum(not j["is_control"] for j in plan["jobs"]),
        control_requests=sum(j["is_control"] for j in plan["jobs"]),
        completed_requests=len(complete),
        remaining_attempts_upper_bound=attempts,
        completion_ceiling_usd=plan["completion_ceiling_usd"],
        remaining_worst_case_usd=remaining,
        analysis_budget_usd=c["semantic_options"]["completion_budget_usd"],
        cumulative_accounted_usd=ledger.total,
        cumulative_max_liability_usd=ledger.total + remaining,
        blockers=blockers,
        preflight_hash=digest(
            [
                plan["design_hash"],
                digest(ledger.events),
                sorted(complete),
                remaining,
                blockers,
            ]
        ),
    )


def _freeze(plan):
    out = Path(plan["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    values = {
        "config.json": plan["config"],
        "source_provenance.json": plan["source"],
        "inventory.json": plan["inventory"],
        "jobs.json": plan["jobs"],
        "rubric.json": dict(version=VERSION, text=RUBRIC),
        "code_hashes.json": plan["source_code_hashes"],
    }
    old = (
        json.loads((out / "analysis_manifest.json").read_text())
        if (out / "analysis_manifest.json").exists()
        else None
    )
    cache_path = out / "token_counts.json"
    if (
        old
        and cache_path.exists()
        and file_hash(cache_path) != old["artifacts"].get("token_counts.json")
    ):
        raise ValueError("Frozen semantic provider counts changed")
    for name, value in values.items():
        path = out / name
        if path.exists():
            if json.loads(path.read_text()) != value or (
                old and file_hash(path) != old["frozen_hashes"][name]
            ):
                raise ValueError("Frozen semantic analysis input changed: " + name)
        else:
            if old:
                raise ValueError("Missing frozen semantic input: " + name)
            write_json(path, value)
    return {name: file_hash(out / name) for name in values}


def _payload(job, settings):
    return build_payload(
        settings,
        system=RUBRIC,
        user=canonical(payload_for_job(job)),
        schema=response_schema(job),
    )


def _counts(plan, api, progress):
    out = Path(plan["output_dir"])
    path = out / "token_counts.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    pending = {}
    settings = plan["config"]["judge"]
    for job in plan["jobs"]:
        payload = _payload(job, settings)
        preliminary = validate_payload(payload, settings)
        key = preliminary["request_sha256"]
        if key not in cache:
            pending[key] = payload
        else:
            count = cache[key]
            if (
                count.get("request_sha256") != key
                or count.get("exact_provider_count") is not True
                or type(count.get("payload_tokens")) is not int
                or count["payload_tokens"] < 0
                or count.get("input_token_bound")
                != count["payload_tokens"]
                + settings["token_counting"]["framing_margin_tokens"]
                or count["input_token_bound"] > settings["max_input_tokens"]
            ):
                raise ValueError("Invalid cached provider token count")
    if not pending:
        return cache
    workers = plan["config"]["semantic_options"]["count_workers"]
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Count workers must be 1–8")
    emit(
        progress,
        "token_count",
        message=f"Counting exact provider inputs for {len(pending)} matching requests with {workers} workers",
    )
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {
        pool.submit(provider_count_payload, payload, settings, api.count): key
        for key, payload in pending.items()
    }
    try:
        for n, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            cache[key] = future.result()
            if n % 25 == 0 or n == len(pending):
                write_json(path, cache)
                emit(
                    progress,
                    "token_count",
                    message=f"Provider counts: {n}/{len(pending)} prepared",
                )
    finally:
        for f in futures:
            f.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        write_json(path, cache)
    return cache


def advance(plan, *, execute_live=False, preflight_hash=None, api=None, progress=None):
    c = plan["config"]
    check = preflight(plan)
    if check["blockers"]:
        raise ValueError("; ".join(check["blockers"]))
    if check["preflight_hash"] != preflight_hash:
        raise ValueError("Obtain a fresh semantic matching dry-run hash")
    if not c.get("demo"):
        if (
            not execute_live
            or api is not None
            or not plan["source"].get("source_verified")
        ):
            raise ValueError(
                "Live semantic matching requires verified sources and explicit execution; injected clients prohibited"
            )
        load_api_credentials(c)
    elif execute_live or api is None:
        raise ValueError("Offline semantic matching requires an injected fixture API")
    out = Path(plan["output_dir"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / (plan["analysis_id"] + ".lock")).open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another semantic collector owns this analysis") from None
        frozen = _freeze(plan)
        api = api or OpenAIBatchAPI(c["judge"])
        cache = _counts(plan, api, progress)
        with _ledger(c) as ledger:
            # Exact counts may take time; recheck the current shared allowance.
            fresh = preflight(plan)
            if fresh["blockers"]:
                raise ValueError("; ".join(fresh["blockers"]))
            session = MatchingBatchSession(c, out, ledger, api=api, progress=progress)
            session.state.setdefault("token_counts", {}).update(cache)
            session.refresh()
            records = _read(out / "judgments.jsonl")
            done = {r["job_id"] for r in records if r["parse_status"] == "ok"}
            emit(
                progress,
                "stage_progress",
                stage="semantic matching",
                completed=len(done),
                planned=len(plan["jobs"]),
            )
            for job in plan["jobs"]:
                if job["job_id"] not in done:
                    session.enqueue_job(job)
            session.flush("adjudication")
            records = _read(out / "judgments.jsonl")
            analysis = write_analysis(
                out,
                plan["inventory"],
                plan["jobs"],
                records,
                source_id=plan["source"]["source_run_id"],
                model_id=c["judge"]["model_id"],
                synthetic=bool(c.get("demo")),
            )
            done = {r["job_id"] for r in records if r["parse_status"] == "ok"}
            status = (
                "batch_pending"
                if session.active
                else (
                    "complete"
                    if len(done) == len(plan["jobs"])
                    and analysis["control_checks"]["passed"]
                    else "needs_review"
                )
            )
            ticket_ids = {r["budget_ticket"] for r in records} | {
                t["ticket"] for t in session.state["tasks"].values() if t.get("ticket")
            }
            balances = ledger._balances()
            own_cost = sum(balances[t]["amount"] for t in ticket_ids)
            manifest = dict(
                version=VERSION,
                analysis_id=plan["analysis_id"],
                status=status,
                source=plan["source"],
                source_code_hashes=plan["source_code_hashes"],
                grader=c["judge"],
                human_reviewed=False,
                demo=bool(c.get("demo")),
                billed_execution=execute_live,
                completed_requests=len(done),
                planned_requests=len(plan["jobs"]),
                scored_propositions=analysis["n_scored_propositions"],
                planned_propositions=analysis["n_propositions"],
                control_checks=analysis["control_checks"],
                analysis_accounted_usd=own_cost,
                cumulative_accounted_usd=ledger.total,
                completion_ceiling_usd=plan["completion_ceiling_usd"],
                frozen_hashes=frozen,
                artifacts={
                    p.name: file_hash(p)
                    for p in sorted(out.iterdir())
                    if p.is_file() and p.name != "analysis_manifest.json"
                },
            )
            write_json(out / "analysis_manifest.json", manifest)
            emit(
                progress,
                "finished",
                message=f"Semantic matching: {status}; {len(done)}/{len(plan['jobs'])} requests collected; {analysis['n_scored_propositions']}/{analysis['n_propositions']} propositions scored; accounted ${own_cost:.4f}",
            )
            return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument(
        "--config", default="configs/feature_space_semantic_matching.json"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--analysis-dir")
    parser.add_argument("--preflight-hash")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=120)
    args = parser.parse_args()
    try:
        if args.status:
            if not args.analysis_dir:
                raise ValueError("Status requires --analysis-dir")
            result = json.loads(
                (Path(args.analysis_dir) / "analysis_manifest.json").read_text()
            )
            print(
                json.dumps(
                    {
                        k: result.get(k)
                        for k in [
                            "analysis_id",
                            "status",
                            "completed_requests",
                            "planned_requests",
                            "scored_propositions",
                            "planned_propositions",
                            "analysis_accounted_usd",
                            "control_checks",
                        ]
                    },
                    indent=2,
                )
            )
            return
        if not args.run_dir:
            raise ValueError("Supply an immutable --run-dir analysis snapshot")
        if args.poll_seconds < 5:
            raise ValueError("Polling interval must be at least five seconds")
        c = load_settings(args.config)
        plan = make_plan(args.run_dir, c)
        if args.dry_run:
            result = preflight(plan)
        else:
            expected = plan["analysis_id"]
            reporter = console_reporter()
            token = args.preflight_hash
            while True:
                try:
                    if token is None:
                        plan = make_plan(args.run_dir, c)
                        if plan["analysis_id"] != expected:
                            raise ValueError(
                                "Frozen matcher design/code changed; do not silently start a new analysis"
                            )
                        token = preflight(plan)["preflight_hash"]
                    result = advance(
                        plan, execute_live=True, preflight_hash=token, progress=reporter
                    )
                except (BatchTransientError, BatchCreationUncertain):
                    if not args.watch:
                        raise
                    emit(
                        reporter,
                        "collector_retry",
                        message="Temporary semantic transport failure; retained work and reservations will be reconciled",
                    )
                except ValueError as problem:
                    if (
                        not args.watch
                        or str(problem)
                        != "Another process owns the cumulative budget ledger"
                    ):
                        raise
                    emit(
                        reporter,
                        "collector_retry",
                        message="Shared budget controller is busy; semantic collection will retry",
                    )
                else:
                    if not args.watch or result["status"] != "batch_pending":
                        break
                emit(
                    reporter,
                    "waiting",
                    message=f"Waiting {args.poll_seconds:g}s; reconnects automatically, Ctrl+C detaches without cancelling Batch work",
                )
                token = None
                time.sleep(args.poll_seconds)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        parser.exit(130, "Detached; semantic Batch work and reservations retained.\n")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Semantic matching blocked: {error}\n")


if __name__ == "__main__":
    main()

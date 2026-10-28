"""Run independent baselines without changing active scientific studies."""

import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import time
from ..batch import BatchCreationUncertain, BatchTransientError
from ..budget import Ledger, estimate_cost
from ..engine import load_config
from ..environment import load_api_credentials
from ..io import digest, file_hash, read_jsonl, write_json
from ..matching.protocol import parse_response
from ..progress import console_reporter, emit
from .core import VERSION, embedding_texts, build_pairs, cosine_scores
from .embedding import (
    settings,
    make_jobs,
    embedding_api,
    EmbeddingSession,
    read_receipts,
    load_vectors,
)
from .report import write_report, evidence_hash


def load_snapshot(path):
    path = Path(path).resolve()
    manifest_path = path / "analysis_manifest.json"
    raw = manifest_path.read_bytes()
    m = json.loads(raw)
    values = {}
    for name in ["inventory.json", "jobs.json", "judgments.jsonl"]:
        p = path / name
        if not p.exists():
            if name == "judgments.jsonl" and name not in m["artifacts"]:
                values[name] = []
                continue
            raise ValueError("Missing matching snapshot input: " + name)
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != m["artifacts"].get(name):
            raise ValueError("Matching snapshot busy or hash mismatch: " + name)
        if name.endswith(".jsonl"):
            records = [json.loads(line) for line in data.splitlines() if line.strip()]
            for r in records:
                if r.get("record_hash") != digest(
                    {k: v for k, v in r.items() if k != "record_hash"}
                ) or r.get("response_hash") != digest(r.get("raw_response")):
                    raise ValueError("Matching snapshot journal hash mismatch")
            values[name] = records
        else:
            values[name] = json.loads(data)
    if manifest_path.read_bytes() != raw:
        raise ValueError("Matching snapshot busy; manifest changed")
    jobs = {j["job_id"]: j for j in values["jobs.json"]}
    decisions = {}
    for r in values["judgments.jsonl"]:
        if r.get("parse_status") != "ok":
            continue
        job = jobs.get(r["job_id"])
        if job is None:
            raise ValueError("Unknown matching snapshot job")
        parsed = parse_response(r["raw_response"], job)
        if job.get("is_control"):
            continue
        for row in parsed["results"]:
            key = job["item_id"], row["proposition_id"]
            if key in decisions:
                raise ValueError("Duplicate matching snapshot grade")
            decisions[key] = row
    return dict(
        path=str(path),
        manifest=m,
        manifest_hash=hashlib.sha256(raw).hexdigest(),
        inventory=values["inventory.json"],
        records=values["judgments.jsonl"],
        decisions=decisions,
    )


def ledger_for(c):
    root = Path(c["root"]) / c.get("artifact_root", "artifacts/feature_space_pilot")
    return Ledger(
        root / ("budget_demo.jsonl" if c.get("demo") else "budget_live.jsonl"),
        cap=c["api_cap_usd"],
        stage_caps=c["stage_caps"],
    )


def make_plan(source, c):
    c = copy.deepcopy(c)
    c.setdefault("baseline_embedding", settings())
    s = c["baseline_embedding"]
    snap = load_snapshot(source)
    inv = snap["inventory"]
    texts = embedding_texts(inv)
    jobs = make_jobs(texts, s)
    code = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    code.update(
        {
            "support/" + n: file_hash(Path(__file__).parents[1] / n)
            for n in ["batch.py", "budget.py", "io.py", "environment.py", "prompts.py"]
        }
    )
    identity = dict(
        version=VERSION,
        inventory_sha256=digest(inv),
        source_analysis_id=snap["manifest"]["analysis_id"],
        source_grader=snap["manifest"]["grader"],
        settings=s,
        code_hashes=code,
        demo=bool(c.get("demo")),
    )
    design = digest(identity)
    for j in jobs:
        j["job_id"] = "baseline_embedding_" + digest([design, j["job_id"]])
    out = (
        Path(c["root"])
        / "artifacts/feature_space_baselines"
        / ("baseline_" + design[:16])
    )
    if out.resolve().is_relative_to(Path(source).resolve()):
        raise ValueError("Baseline output must be separate from matching input")
    ceiling = (
        len(jobs)
        * s["max_attempts"]
        * estimate_cost(s["prices"], s["max_input_tokens"], 0)
    )
    if ceiling > s["completion_budget_usd"] + 1e-12:
        raise ValueError("Embedding completion envelope exceeds baseline allowance")
    return dict(
        analysis_id=out.name,
        output_dir=str(out),
        config=c,
        identity=identity,
        design_hash=design,
        source_dir=str(Path(source).resolve()),
        inventory=inv,
        texts=texts,
        jobs=jobs,
        completion_ceiling_usd=ceiling,
    )


def preflight(plan):
    c = plan["config"]
    ledger = ledger_for(c)
    ledger.events = read_jsonl(ledger.path) if ledger.path.exists() else []
    ledger._balances()
    used = {}
    for e in ledger.events:
        if e["event"] == "reserve":
            used[e["job_id"]] = used.get(e["job_id"], 0) + 1
    records = read_receipts(Path(plan["output_dir"]) / "embeddings.jsonl")
    done = {r["job_id"] for r in records if r["parse_status"] == "ok"}
    s = c["baseline_embedding"]
    remaining = sum(
        max(0, s["max_attempts"] - used.get(j["job_id"], 0))
        for j in plan["jobs"]
        if j["job_id"] not in done
    ) * estimate_cost(s["prices"], s["max_input_tokens"], 0)
    blockers = []
    if (
        ledger.total + remaining > ledger.cap + 1e-12
        or ledger.stage_total("embeddings") + remaining
        > ledger.stage_caps["embeddings"] + 1e-12
    ):
        blockers.append("Shared cumulative/embedding budget would be exceeded")
    return dict(
        analysis_id=plan["analysis_id"],
        output_dir=plan["output_dir"],
        model_id=s["model_id"],
        execution_mode="batch",
        embedding_texts=len(plan["texts"]),
        embedding_requests=len(plan["jobs"]),
        completion_ceiling_usd=plan["completion_ceiling_usd"],
        remaining_worst_case_usd=remaining,
        baseline_allowance_usd=s["completion_budget_usd"],
        cumulative_accounted_usd=ledger.total,
        blockers=blockers,
        preflight_hash=digest(
            [plan["design_hash"], ledger.events, sorted(done), remaining, blockers]
        ),
    )


def freeze(plan):
    out = Path(plan["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    old = (
        json.loads((out / "baseline_manifest.json").read_text())
        if (out / "baseline_manifest.json").exists()
        else None
    )
    hashes = {}
    for name, value in [
        ("inventory.json", plan["inventory"]),
        ("embedding_jobs.json", plan["jobs"]),
        ("embedding_texts.json", plan["texts"]),
        ("frozen_config.json", plan["config"]),
        ("identity.json", plan["identity"]),
    ]:
        p = out / name
        if p.exists():
            if (
                json.loads(p.read_text()) != value
                or old
                and file_hash(p) != old["frozen_hashes"][name]
            ):
                raise ValueError("Frozen baseline input changed: " + name)
        elif old:
            raise ValueError("Missing frozen baseline input: " + name)
        else:
            write_json(p, value)
        hashes[name] = file_hash(p)
    return hashes


def advance(
    plan,
    *,
    execute_live=False,
    preflight_hash=None,
    api=None,
    render=True,
    progress=None,
):
    c = plan["config"]
    check = preflight(plan)
    if check["blockers"]:
        raise ValueError("; ".join(check["blockers"]))
    if preflight_hash != check["preflight_hash"]:
        raise ValueError("Obtain a fresh baseline dry-run hash")
    snap = load_snapshot(plan["source_dir"])
    if digest(snap["inventory"]) != plan["identity"]["inventory_sha256"]:
        raise ValueError("Matching inventory changed")
    if (
        snap["manifest"]["analysis_id"] != plan["identity"]["source_analysis_id"]
        or snap["manifest"]["grader"] != plan["identity"]["source_grader"]
    ):
        raise ValueError("Matching analysis identity or grader changed")
    if execute_live:
        if (
            c.get("demo")
            or api is not None
            or snap["manifest"].get("demo")
            or not snap["manifest"]["source"].get("source_verified")
        ):
            raise ValueError(
                "Live embedding execution requires verified empirical source and real API"
            )
        load_api_credentials(c)
    elif api is not None and not c.get("demo"):
        raise ValueError("Injected fixture API prohibited for empirical baseline")
    out = Path(plan["output_dir"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / (plan["analysis_id"] + ".lock")).open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another baseline collector owns this design") from None
        frozen = freeze(plan)
        active = False
        own_cost = 0.0
        if execute_live or api is not None:
            with ledger_for(c) as ledger:
                fresh = preflight(plan)
                if fresh["blockers"]:
                    raise ValueError("; ".join(fresh["blockers"]))
                session = EmbeddingSession(
                    c, out, ledger, api=api or embedding_api(c), progress=progress
                )
                session.refresh()
                done = {
                    r["job_id"]
                    for r in read_receipts(out / "embeddings.jsonl")
                    if r["parse_status"] == "ok"
                }
                for j in plan["jobs"]:
                    if j["job_id"] not in done:
                        session.enqueue_job(j)
                session.flush("embeddings")
                active = session.active
                balances = ledger._balances()
                own_cost = sum(
                    balances[t["ticket"]]["amount"]
                    for t in session.state["tasks"].values()
                    if t.get("ticket")
                )
        elif (out / "baseline_manifest.json").exists():
            own_cost = json.loads((out / "baseline_manifest.json").read_text())[
                "accounted_embedding_usd"
            ]
        vectors = load_vectors(out, plan["jobs"])
        complete = len(vectors) == len(plan["texts"])
        scores = cosine_scores(plan["inventory"], vectors) if complete else None
        rows = build_pairs(plan["inventory"], snap["decisions"], embedding=scores)
        vector_fingerprint = digest(
            [
                (r["job_id"], r["vector_sha256"])
                for r in read_receipts(out / "embeddings.jsonl")
                if r["parse_status"] == "ok"
            ]
        )
        version = digest([evidence_hash(snap), vector_fingerprint, render])[:16]
        report = out / "reports" / version
        summary = write_report(report, rows, plan["inventory"], snap, render=render)
        if active:
            status = "batch_pending"
        elif (
            complete
            and summary["calibration"]["available"]
            and not summary["unscored_pairs"]
            and summary["control_checks"]["passed"]
        ):
            status = "complete"
        elif (
            summary["unscored_pairs"]
            and snap["manifest"].get("status") == "batch_pending"
        ):
            status = "waiting_for_judge"
        elif execute_live or api is not None or summary["unscored_pairs"]:
            status = "needs_review"
        else:
            status = "lexical_only"
        m = dict(
            analysis_id=plan["analysis_id"],
            status=status,
            demo=bool(c.get("demo")),
            human_reviewed=False,
            source_matching_analysis=snap["manifest"]["analysis_id"],
            source_manifest_sha256=summary["source_manifest_sha256"],
            latest_observed_source_manifest_sha256=snap["manifest_hash"],
            embedding_vectors=len(vectors),
            planned_embedding_vectors=len(plan["texts"]),
            planned_embedding_requests=len(plan["jobs"]),
            accounted_embedding_usd=own_cost,
            completion_ceiling_usd=plan["completion_ceiling_usd"],
            report_dir=str(report),
            frozen_hashes=frozen,
            identity=plan["identity"],
            report_artifacts={
                p.name: file_hash(p) for p in report.iterdir() if p.is_file()
            },
        )
        write_json(out / "baseline_manifest.json", m)
        write_json(
            out / "latest_report.json",
            dict(
                report_dir=str(report),
                status=status,
                source_manifest_sha256=summary["source_manifest_sha256"],
            ),
        )
        emit(
            progress,
            "finished",
            message=f"Baselines: {status}; {len(vectors)}/{len(plan['texts'])} embeddings; {len(rows)-summary['unscored_pairs']}/{len(rows)} pairs have Luna scores; accounted ${own_cost:.5f}",
        )
        return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matching-dir", required=True)
    p.add_argument("--config", default="configs/feature_space_pilot.json")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--offline", action="store_true")
    group.add_argument("--execute-live", action="store_true")
    p.add_argument("--preflight-hash")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=120)
    a = p.parse_args()
    if a.poll_seconds < 5:
        p.error("Polling interval must be at least five seconds")
    try:
        c = load_config(a.config)
        plan = make_plan(a.matching_dir, c)
        if a.dry_run:
            print(json.dumps(preflight(plan), indent=2))
            return
        expected = plan["analysis_id"]
        token = a.preflight_hash
        reporter = console_reporter()
        while True:
            try:
                if token is None:
                    plan = make_plan(a.matching_dir, c)
                    if plan["analysis_id"] != expected:
                        raise ValueError(
                            "Baseline code/design changed; refusing a new analysis"
                        )
                    token = preflight(plan)["preflight_hash"]
                result = advance(
                    plan,
                    execute_live=a.execute_live,
                    preflight_hash=token,
                    progress=reporter,
                )
            except (BatchTransientError, BatchCreationUncertain):
                if not a.watch:
                    raise
                emit(
                    reporter,
                    "collector_retry",
                    message="Temporary embedding transport issue; retaining reservations and reconciling next poll",
                )
            except ValueError as problem:
                if not a.watch or not (
                    str(problem)
                    in {
                        "Another process owns the cumulative budget ledger",
                        "Obtain a fresh baseline dry-run hash",
                    }
                    or str(problem).startswith("Matching snapshot busy")
                ):
                    raise
                emit(
                    reporter,
                    "collector_retry",
                    message="Shared ledger or source snapshot busy; waiting for fresh consistent inputs",
                )
            else:
                if not a.watch or result["status"] not in {
                    "batch_pending",
                    "waiting_for_judge",
                }:
                    break
            emit(
                reporter,
                "waiting",
                message=f"Waiting {a.poll_seconds:g}s; existing studies continue independently",
            )
            token = None
            time.sleep(a.poll_seconds)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as problem:
        p.exit(2, f"Baseline blocked: {problem}\n")
    except KeyboardInterrupt:
        p.exit(130, "Local collector detached; submitted Batch work retained.\n")


if __name__ == "__main__":
    main()

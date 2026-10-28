"""Migrate a frozen full bank to concurrent standard-price atomic judgments."""

import argparse
import copy
import fcntl
import json
import signal
import time
from collections import defaultdict, Counter
from pathlib import Path
from ..batch import OpenAIBatchAPI, BatchTransientError
from ..budget import estimate_cost
from ..engine import _read
from ..environment import load_api_credentials
from ..execution import execution_config
from ..io import digest, file_hash, write_json, read_jsonl
from ..progress import console_reporter
from ..full.run import ledger_for, read_ledger, _freeze_json
from ..full.reuse import previous_full_sources, reuse_evidence, cell_job, model_identity
from ..full.matrix import matrix_state
from ..full.sources import collect_sources, apply_checkpoint
from ..full.report import write_full_report
from .costs import MeasuredInputs
from .execution import ImmediateExecutor
from .migration import CollectOnlySession


class EvidenceIndex:
    def __init__(self, rows=(), *, pending=(), held=()):
        self.cells = defaultdict(list)
        self.pending = set(pending)
        self.held = set(held)
        for row in rows:
            self.add(row)

    def add(self, row):
        self.cells[row["item_id"], row["feature_id"]].append(row)

    def kind(self, key):
        rows = self.cells.get(key, [])
        values = {
            r["applies"]
            for r in rows
            if r.get("parse_status") == "ok"
            and r.get("status") == "ok"
            and type(r.get("applies")) is int
            and r["applies"] in (0, 1)
        }
        if len(values) == 1:
            return "resolved"
        if len(values) > 1 or any(
            r.get("parse_status") == "ok" and r.get("status") == "ambiguous"
            for r in rows
        ):
            return "review"
        if key in self.held:
            return "held"
        if key in self.pending:
            return "pending"
        if any(
            not r.get("known_not_executed")
            and (
                r.get("usage_unknown")
                or not all(
                    type(r.get("usage", {}).get(k)) is int
                    for k in ["input_tokens", "output_tokens"]
                )
            )
            for r in rows
        ):
            return "held"
        if not rows:
            return "primary"
        return "retry" if self.attempt(key) < 2 else "exhausted"

    def attempt(self, key):
        return max(
            (r.get("attempt", 0) + 1 for r in self.cells.get(key, [])), default=0
        )


def evidence(reuse, prior_sources, own, out):
    compatible = {r["source_journal_record_hash"] for r in reuse["compatible_attempts"]}
    old = [
        r
        for s in prior_sources
        for r in s["judgments"]
        if r["record_hash"] in compatible
    ]
    received = {r["budget_ticket"] for r in own}
    dispatch = (
        read_jsonl(out / "applicability/dispatches.jsonl")
        if (out / "applicability/dispatches.jsonl").exists()
        else []
    )
    held = {
        (r["item_id"], r["feature_id"]) for r in dispatch if r["ticket"] not in received
    }
    pending = {
        (r["item_id"], r["feature_id"]) for r in reuse["external_pending"].values()
    }
    return EvidenceIndex(old + own, pending=pending, held=held)


def check_cost(items, features, index, meter, c, ledger):
    counts = Counter()
    primary = retry = total_input = 0.0
    attempts = 0
    byitem = []
    bounds = []
    for item in items:
        n = 0
        price = 0.0
        for f in features:
            key = item["item_id"], f["feature_id"]
            kind = index.kind(key)
            counts[kind] += 1
            bound = meter.bound(item, f)["input_token_bound"]
            total_input += bound
            bounds.append(bound)
            usd = estimate_cost(
                c["judge"]["prices"], bound, c["judge"]["max_output_tokens"]
            )
            if kind == "primary":
                primary += usd
                n += 1
                price += usd
                attempts += 2
            elif kind == "retry":
                retry += usd
                attempts += 1
        byitem.append(
            dict(
                item_id=item["item_id"],
                new_primary_calls=n,
                remaining_upper_bound_usd=price,
            )
        )
    stage = (
        "primary_pass"
        if counts["primary"]
        else (
            "collect_pending"
            if counts["pending"]
            else "invalid_retries" if counts["retry"] else "review_only"
        )
    )
    authorized = (
        primary
        if stage == "primary_pass"
        else retry if stage == "invalid_retries" else 0.0
    )
    blockers = []
    if ledger.total + authorized > 100 + 1e-12:
        blockers.append("Entire remaining immediate stage exceeds the shared $100 cap")
    result = dict(
        required_cells=len(items) * len(features),
        eligible_items=len(items),
        consolidated_features=len(features),
        final_consolidated_features=len(features),
        bank_provisional=False,
        new_primary_calls=counts["primary"],
        reused_resolved_cells=counts["resolved"],
        prior_unresolved_cells=counts["review"] + counts["held"],
        external_pending_cells=counts["pending"],
        known_invalid_retry_calls=counts["retry"],
        retry_exhausted_cells=counts["exhausted"],
        retry_policy="staged_primary_then_costed_invalid_retries",
        authorized_stage=stage,
        authorized_stage_upper_bound_usd=authorized,
        first_attempt_upper_bound_usd=primary,
        known_invalid_retry_upper_bound_usd=retry,
        remaining_completion_upper_bound_usd=primary * 2 + retry,
        remaining_attempts_upper_bound=attempts,
        cumulative_accounted_usd=ledger.total,
        stage_cap_usd=100,
        budget_cap_usd=100,
        max_attempts=2,
        output_tokens_per_attempt=400,
        input_bound_range=[min(bounds), max(bounds)],
        serialized_input_bound_total=total_input,
        input_counting="provider_item_feature_offsets_with_cross_item_validation_v1",
        prices=c["judge"]["prices"],
        by_item=byitem,
        blockers=blockers,
        live_ready=not blockers,
        cell_states=dict(counts),
        unsubmitted_primary_funding_protected=True,
    )
    result["preflight_hash"] = digest(
        [result, meter.profile["profile_hash"], digest(ledger.events)]
    )
    return result


def build_plan(
    source,
    profile_path,
    *,
    allow_synthetic=False,
    workers=256,
    max_rps=50.0,
    wave_size=1024,
):
    source = Path(source).resolve()
    m = json.loads((source / "full_manifest.json").read_text())
    if not m.get("bank_frozen"):
        raise ValueError("A frozen consolidated global bank is required")
    if m.get("demo") and not allow_synthetic:
        raise ValueError("Immediate empirical execution cannot use synthetic sources")
    required = [
        "frozen_config.json",
        "stimuli.json",
        "frozen_global_bank.json",
        "literal_binary_prompt.json",
    ]
    for name in required:
        if file_hash(source / name) != m["frozen_hashes"][name]:
            raise ValueError("Frozen source changed: " + name)
    original = json.loads((source / "frozen_config.json").read_text())
    c = execution_config(original, "online")
    if c["api_cap_usd"] != 100 or c["max_attempts"] != 2:
        raise ValueError("Immediate migration preserves the $100/two-attempt limits")
    items = json.loads((source / "stimuli.json").read_text())
    bank = json.loads((source / "frozen_global_bank.json").read_text())
    literal = json.loads((source / "literal_binary_prompt.json").read_text())
    if (
        cell_job(items[0], bank["features"][0], c["judge"])["system"]
        != literal["system"]
    ):
        raise ValueError("Literal binary prompt differs from original")
    profile = json.loads(Path(profile_path).read_text())
    meter = MeasuredInputs(profile, c["judge"])
    if profile["bank_inventory_hash"] != bank["inventory_hash"]:
        raise ValueError("Input profile bank differs")
    identity = dict(
        version="immediate_global_v1",
        source_frozen_hashes={n: m["frozen_hashes"][n] for n in required},
        source_run=str(source),
        judge=c["judge"],
        profile_hash=profile["profile_hash"],
        workers=workers,
        max_rps=max_rps,
        wave_size=wave_size,
        code={p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
        demo=bool(c.get("demo")),
    )
    out = source.parent / ("full_immediate_" + digest(identity)[:16])
    prior_sources = previous_full_sources(source.parent, out)
    reuse = reuse_evidence(prior_sources, items, bank["features"], c["judge"])
    own = _read(out / "applicability/judgments.jsonl")
    index = evidence(reuse, prior_sources, own, out)
    ledger = read_ledger(c)
    check = check_cost(items, bank["features"], index, meter, c, ledger)
    state = matrix_state(items, bank["features"], reuse["reused"] + own)
    return dict(
        source=source,
        source_manifest=m,
        original_config=original,
        config=c,
        items=items,
        bank=bank,
        profile=profile,
        meter=meter,
        identity=identity,
        output_dir=out,
        analysis_id=out.name,
        reuse=reuse,
        prior_sources=prior_sources,
        own=own,
        index=index,
        check=check,
        state=state,
        literal=literal,
    )


def freeze(plan):
    out = plan["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, value in [
        ("identity.json", plan["identity"]),
        ("frozen_config.json", plan["config"]),
        ("stimuli.json", plan["items"]),
        ("frozen_global_bank.json", plan["bank"]),
        ("literal_binary_prompt.json", plan["literal"]),
        ("provider_feature_input_profile.json", plan["profile"]),
    ]:
        hashes[name] = _freeze_json(out / name, value)
    return hashes


def refresh_evidence(plan, own):
    sources = previous_full_sources(plan["source"].parent, plan["output_dir"])
    reuse = reuse_evidence(
        sources, plan["items"], plan["bank"]["features"], plan["config"]["judge"]
    )
    plan.update(
        reuse=reuse,
        prior_sources=sources,
        own=own,
        index=evidence(reuse, sources, own, plan["output_dir"]),
    )


def report(plan, check, *, render=False):
    c = plan["config"]
    source = collect_sources(
        [Path(c["root"]) / p for p in c["full_options"]["source_runs"]],
        allow_synthetic=bool(c.get("demo")),
        expected_items=len(plan["items"]),
    )
    cp = plan["source"] / "generation_checkpoint.json"
    if cp.exists():
        source = apply_checkpoint(source, json.loads(cp.read_text()))
    state = matrix_state(
        plan["items"], plan["bank"]["features"], plan["reuse"]["reused"] + plan["own"]
    )
    plan["state"] = state
    evidence_hash = digest(
        [
            [
                (r.get("record_hash"), r.get("reuse_hash"))
                for r in plan["reuse"]["reused"] + plan["own"]
            ],
            check,
        ]
    )[:16]
    out = plan["output_dir"] / "reports" / evidence_hash
    semdir = c["full_options"].get("semantic_matching_dir")
    semdir = Path(c["root"]) / semdir if semdir else None
    write_full_report(
        out,
        source,
        plan["bank"],
        state,
        c,
        check,
        plan["reuse"],
        profile=plan["profile"]["base_profile"],
        render=render,
        semantic_matching_dir=semdir,
        primary_records=plan["own"],
    )
    write_json(plan["output_dir"] / "latest_report.json", dict(report_dir=str(out)))
    write_json(plan["output_dir"] / "reuse_receipts.json", plan["reuse"]["reused"])
    return out


def save_status(plan, check, status, *, frozen=None, report_dir=None, pace=None):
    out = plan["output_dir"]
    previous = (
        json.loads((out / "full_manifest.json").read_text())
        if (out / "full_manifest.json").exists()
        else {}
    )
    counters = check["cell_states"]
    resolved = counters.get("resolved", 0)
    value = dict(
        version="global_full_immediate_v1",
        analysis_id=plan["analysis_id"],
        status=status,
        demo=bool(plan["config"].get("demo")),
        bank_frozen=True,
        canonical_features=len(plan["bank"]["features"]),
        n_items=len(plan["items"]),
        planned_cells=len(plan["items"]) * len(plan["bank"]["features"]),
        resolved_cells=resolved,
        matrix_complete=resolved == len(plan["items"]) * len(plan["bank"]["features"]),
        scientific_complete=False,
        human_reviewed=False,
        source_full_run=str(plan["source"]),
        generation_scope_closed=True,
        source_generations_complete=plan["source_manifest"][
            "source_generations_complete"
        ],
        missing_generation_jobs=plan["source_manifest"].get(
            "missing_generation_jobs", 0
        ),
        frozen_hashes=frozen or previous.get("frozen_hashes", {}),
        applicability=check,
        report_dir=str(report_dir) if report_dir else previous.get("report_dir"),
        execution_mode="online",
        current_requests_per_second=pace.rate if pace else None,
        max_requests_per_second=pace.ceiling if pace else None,
        updated_at=time.time(),
    )
    write_json(out / "full_manifest.json", value)
    write_json(out / "current_cost_report.json", check)
    return value


def run(
    plan,
    *,
    execute_live=False,
    transport=None,
    preflight_hash=None,
    max_waves=None,
    progress=None,
):
    c = plan["config"]
    out = plan["output_dir"]
    if execute_live and (c.get("demo") or transport is not None):
        raise ValueError(
            "Live immediate execution requires empirical sources and the actual provider"
        )
    if preflight_hash != plan["check"]["preflight_hash"]:
        raise ValueError("Obtain a fresh immediate preflight hash")
    if plan["check"]["blockers"]:
        raise ValueError("; ".join(plan["check"]["blockers"]))
    if execute_live:
        load_api_credentials(c)
    if transport is not None and not c.get("demo"):
        raise ValueError("Injected immediate client requires synthetic configuration")
    frozen = freeze(plan)
    (out / "applicability").mkdir(exist_ok=True)
    import threading

    stopped = threading_event() if execute_live else threading.Event()
    waves = 0
    last_refresh = 0.0
    last_report = 0.0
    last_count = 0
    engine = None
    sourceapi = (
        OpenAIBatchAPI(plan["original_config"]["judge"]) if execute_live else None
    )
    ordered = sorted(
        enumerate(plan["items"]),
        key=lambda x: (x[1].get("target_type") in {"concept", "object", "event"}, x[0]),
    )
    with (out.parent / (out.name + ".lock")).open("a") as lock, (
        plan["source"].parent / (plan["source"].name + ".lock")
    ).open("a") as source_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not execute_live and transport is None:
            directory = report(plan, plan["check"])
            return save_status(
                plan, plan["check"], "prepared", frozen=frozen, report_dir=directory
            )
        stage = None
        cursor = None
        remaining_check = plan["check"]
        last_status = None
        while not stopped.is_set():
            try:
                with ledger_for(c) as ledger:
                    if engine is None:
                        engine = ImmediateExecutor(
                            c,
                            out / "applicability",
                            ledger,
                            transport=transport,
                            workers=plan["identity"]["workers"],
                            max_rps=plan["identity"]["max_rps"],
                            allowed_prior_job_ids=[
                                r["job_id"]
                                for r in plan["reuse"]["compatible_attempts"]
                            ],
                        )
                    engine.ledger = ledger
                    if execute_live and time.monotonic() - last_refresh > 120:
                        if {
                            p.name: file_hash(p)
                            for p in Path(__file__).parent.glob("*.py")
                        } != plan["identity"]["code"]:
                            raise ValueError("Frozen immediate controller code changed")
                        collector = CollectOnlySession(
                            plan["original_config"],
                            plan["source"] / "applicability",
                            ledger,
                            api=sourceapi,
                            progress=progress,
                        )
                        collector.refresh()
                        refresh_evidence(plan, engine.rows)
                        last_refresh = time.monotonic()
                        cursor = None
                    index = plan["index"]
                    if cursor is None:
                        remaining_check = check_cost(
                            plan["items"],
                            plan["bank"]["features"],
                            index,
                            plan["meter"],
                            c,
                            ledger,
                        )
                        if remaining_check["blockers"]:
                            last_status = save_status(
                                plan,
                                remaining_check,
                                "preflight_blocked",
                                frozen=frozen,
                                pace=engine.pace,
                            )
                            break
                        stage = remaining_check["authorized_stage"]
                        if stage in {"collect_pending", "review_only"}:
                            last_status = save_status(
                                plan,
                                remaining_check,
                                (
                                    "waiting_for_batch_cancellation"
                                    if stage == "collect_pending"
                                    else "needs_review"
                                ),
                                frozen=frozen,
                                pace=engine.pace,
                            )
                            if stage == "review_only":
                                break
                            cursor = None
                        else:
                            cursor = (
                                (i, f)
                                for _, i in ordered
                                for f in plan["bank"]["features"]
                            )
                    tasks = []
                    if cursor is not None:
                        for item, feature in cursor:
                            key = item["item_id"], feature["feature_id"]
                            kind = index.kind(key)
                            if kind != (
                                "primary" if stage == "primary_pass" else "retry"
                            ):
                                continue
                            j = cell_job(item, feature, c["judge"])
                            if j["job_id"] in engine.held_job_ids:
                                index.held.add(key)
                                continue
                            tasks.append(
                                dict(
                                    job=j,
                                    attempt=index.attempt(key),
                                    **plan["meter"].bound(item, feature),
                                )
                            )
                            if len(tasks) >= plan["identity"]["wave_size"]:
                                break
                    if tasks:
                        rows = engine.wave(tasks)
                        for row in rows:
                            index.add(row)
                        plan["own"] = engine.rows
                        waves += 1
                        remaining_check["new_primary_calls"] = max(
                            0,
                            remaining_check["new_primary_calls"]
                            - sum(r["attempt"] == 0 for r in rows),
                        )
                        remaining_check["cumulative_accounted_usd"] = ledger.total
                        admitted = sum(
                            estimate_cost(
                                c["judge"]["prices"],
                                t["input_token_bound"],
                                c["judge"]["max_output_tokens"],
                            )
                            for t in tasks
                        )
                        if stage == "primary_pass":
                            remaining_check["first_attempt_upper_bound_usd"] = max(
                                0.0,
                                remaining_check["first_attempt_upper_bound_usd"]
                                - admitted,
                            )
                            remaining_check["authorized_stage_upper_bound_usd"] = (
                                remaining_check["first_attempt_upper_bound_usd"]
                            )
                        else:
                            remaining_check["known_invalid_retry_upper_bound_usd"] = (
                                max(
                                    0.0,
                                    remaining_check[
                                        "known_invalid_retry_upper_bound_usd"
                                    ]
                                    - admitted,
                                )
                            )
                            remaining_check["authorized_stage_upper_bound_usd"] = (
                                remaining_check["known_invalid_retry_upper_bound_usd"]
                            )
                        # Counters are updated from the index for meaningful live progress.
                        counts = Counter(
                            index.kind((i["item_id"], f["feature_id"]))
                            for i in plan["items"]
                            for f in plan["bank"]["features"]
                        )
                        remaining_check["cell_states"] = dict(counts)
                        last_status = save_status(
                            plan,
                            remaining_check,
                            "running",
                            frozen=frozen,
                            pace=engine.pace,
                        )
                        print(
                            f"Immediate: {len(engine.rows):,} own receipts; {counts['resolved']:,}/{len(plan['items'])*len(plan['bank']['features']):,} resolved; {counts['primary']:,} unattempted; {engine.pace.rate:.2f} req/s; ${ledger.total:.4f} accounted",
                            flush=True,
                        )
                        if engine.fatal_error:
                            raise ValueError(engine.fatal_error)
                    elif cursor is not None:
                        cursor = None
            except ValueError as problem:
                if str(problem) != "Another process owns the cumulative budget ledger":
                    raise
                time.sleep(1)
                continue
            if max_waves is not None and waves >= max_waves:
                break
            if (
                len(engine.rows) - last_count >= 25000
                or time.monotonic() - last_report > 900
            ):
                directory = report(plan, remaining_check, render=False)
                last_report = time.monotonic()
                last_count = len(engine.rows)
                last_status = save_status(
                    plan,
                    remaining_check,
                    last_status["status"],
                    frozen=frozen,
                    report_dir=directory,
                    pace=engine.pace,
                )
            if not tasks and cursor is None and stage == "collect_pending":
                time.sleep(20)
        if engine:
            plan["own"] = engine.rows
        with ledger_for(c) as ledger:
            final = check_cost(
                plan["items"],
                plan["bank"]["features"],
                plan["index"],
                plan["meter"],
                c,
                ledger,
            )
        directory = report(plan, final, render=False)
        status = (
            "complete"
            if plan["state"]["complete"]
            else (
                "needs_review"
                if final["authorized_stage"] == "review_only"
                else last_status["status"] if last_status else "stopped"
            )
        )
        return save_status(
            plan,
            final,
            status,
            frozen=frozen,
            report_dir=directory,
            pace=engine.pace if engine else None,
        )


def threading_event():
    import threading

    event = threading.Event()
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: event.set())
    return event


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-full-run", required=True)
    p.add_argument(
        "--input-profile",
        default="artifacts/feature_space_immediate/feature_input_profile.json",
    )
    p.add_argument("--workers", type=int, default=256)
    p.add_argument("--wave-size", type=int, default=1024)
    p.add_argument("--max-rps", type=float, default=50.0)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    p.add_argument("--preflight-hash")
    p.add_argument("--max-waves", type=int)
    a = p.parse_args()
    if (
        not 1 <= a.workers <= 256
        or not 0 < a.max_rps <= 100
        or not 1 <= a.wave_size <= 4096
    ):
        p.error("Workers 1–256 and max-rps >0 to100 are required")
    try:
        plan = build_plan(
            a.source_full_run,
            a.input_profile,
            workers=a.workers,
            max_rps=a.max_rps,
            wave_size=a.wave_size,
        )
        if a.dry_run:
            print(
                json.dumps(
                    dict(
                        analysis_id=plan["analysis_id"],
                        output_dir=str(plan["output_dir"]),
                        **plan["check"],
                    ),
                    indent=2,
                )
            )
            return
        token = a.preflight_hash or (
            plan["check"]["preflight_hash"] if a.prepare else None
        )
        result = run(
            plan,
            execute_live=a.execute_live,
            preflight_hash=token,
            max_waves=a.max_waves,
            progress=console_reporter(),
        )
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as problem:
        p.exit(2, "Immediate pilot blocked: " + str(problem) + "\n")


if __name__ == "__main__":
    main()

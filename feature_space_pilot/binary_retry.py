"""Prepare and explicitly execute only historical human abstentions under binary v2."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import fcntl
import json
from pathlib import Path
import time
from .io import canonical, digest, file_hash, write_csv, write_json, read_jsonl
from .judge import RUBRIC, parse_judgment, rubric_for
from .engine import load_config, _read
from .environment import load_api_credentials
from .execution import execution_config
from .batch import OpenAIBatchAPI
from .budget import estimate_cost
from .full.run import ledger_for, read_ledger, _freeze_json
from .full.budget import profile_inputs, input_bound
from .full.execution import FullSession
from .full.reuse import cell_job
from .experiments.banks import experiment_of
from .experiments.evidence import verify_receipt
from .experiments.run import verify_export

VERSION = "selected_human_abstentions_binary_v2_v1"


def select_ambiguous(items, bank, cells):
    if bank.get("experiment") != "human" or any(
        experiment_of(i) != "human" for i in items
    ):
        raise ValueError("Retry scope must contain only the human experiment")
    ids = {i["item_id"] for i in items}
    features = {f["feature_id"] for f in bank["features"]}
    seen = set()
    selected = []
    for row in cells:
        key = row["item_id"], row["feature_id"]
        if key in seen:
            raise ValueError("Duplicate retry-scope cell")
        seen.add(key)
        if key[0] not in ids or key[1] not in features:
            raise ValueError("Cell outside human feature scope")
        if row["status"] != "ambiguous":
            continue
        if row.get("applies") not in ("?", None, ""):
            raise ValueError("Ambiguous cell already has a binary value")
        hashes = row.get("source_record_hashes", [])
        hashes = json.loads(hashes) if isinstance(hashes, str) else hashes
        if not hashes:
            raise ValueError("Ambiguous source evidence is required")
        selected.append(
            dict(
                item_id=key[0],
                feature_id=key[1],
                legacy_status="ambiguous",
                legacy_rubric="binary_v1",
                legacy_source_record_hashes=sorted(set(hashes)),
            )
        )
    return selected


def _validated_records(jobs, records):
    byid = {j["job_id"]: j for j in jobs}
    for record in records:
        job = byid.get(record["job_id"])
        if job is None:
            raise ValueError("Retry record outside eligible job scope")
        if record.get("rubric_version") != RUBRIC:
            raise ValueError("Retry rubric mismatch")
        for name in ["item_id", "feature_id", "prompt_hash", "payload_sha256"]:
            if record.get(name) != job.get(name):
                raise ValueError("Retry request identity mismatch: " + name)
        if job.get("judge_model") and record.get("judge_model") != job["judge_model"]:
            raise ValueError("Retry model identity mismatch")
        verify_receipt(record)
        if record.get("parse_status") == "ok":
            parsed = parse_judgment(
                record["raw_response"],
                record["item_id"],
                record["feature_id"],
                rubric_version=RUBRIC,
            )
            if any(
                record.get(k) != parsed[k] for k in ["applies", "status", "evidence"]
            ):
                raise ValueError("Retry parsed response mismatch")


def merge_cells(cells, jobs, records):
    bykey = {(r["item_id"], r["feature_id"]): r for r in cells}
    if len(bykey) != len(cells):
        raise ValueError("Duplicate source cell")
    eligible = {(j["item_id"], j["feature_id"]) for j in jobs}
    if any(key not in bykey or bykey[key]["status"] != "ambiguous" for key in eligible):
        raise ValueError(
            "Retry job is not eligible: only original ambiguous cells may change"
        )
    _validated_records(jobs, records)
    fresh = defaultdict(list)
    for r in records:
        if r.get("parse_status") == "ok":
            fresh[r["item_id"], r["feature_id"]].append(r)
    result = []
    for original in cells:
        key = original["item_id"], original["feature_id"]
        row = dict(
            original,
            effective_rubric="binary_v1",
            retry_record_hashes=[],
            combined_policy="preserved_binary_v1_plus_selected_binary_v2",
        )
        values = {r["applies"] for r in fresh[key]}
        if len(values) > 1:
            row.update(
                applies="?", status="conflicting_binary_v2", effective_rubric=RUBRIC
            )
        elif len(values) == 1:
            row.update(
                applies=next(iter(values)),
                status="binary_v2_resolved",
                effective_rubric=RUBRIC,
            )
        row["retry_record_hashes"] = sorted({r["record_hash"] for r in fresh[key]})
        result.append(row)
    return result


def _cells(source):
    with (source / "human/applicability_cells.csv").open() as f:
        return list(csv.DictReader(f))


def prepare(source, config, output_root):
    source, output_root = Path(source).resolve(), Path(output_root).resolve()
    verify_export(source)
    manifest = json.loads((source / "manifest.json").read_text())
    c = deepcopy(config)
    if bool(manifest["identity"].get("demo")) != bool(c.get("demo")):
        raise ValueError("Synthetic and empirical retry sources must stay separate")
    if c["api_cap_usd"] != 100 or c["max_attempts"] != 2:
        raise ValueError("Preserve the cumulative $100 and two-attempt limits")
    if c.get("demo"):
        for role in ["judge", "generator"]:
            c[role].setdefault("batch_prices", deepcopy(c[role]["prices"]))
    c = execution_config(c, "batch")
    c["judge"]["rubric_version"] = RUBRIC
    c["stage_caps"] = {k: 100 for k in c["stage_caps"]}
    c.setdefault("full_options", {})["stage_budget_policy"] = "shared_total_only"
    c.update(
        batch_max_requests=min(c.get("batch_max_requests", 1000), 2000),
        batch_max_in_flight=min(c.get("batch_max_in_flight", 8), 8),
        batch_max_queued_input_tokens=min(
            c.get("batch_max_queued_input_tokens", 8_000_000), 8_000_000
        ),
    )
    protected = {source}
    provenance = json.loads((source / "source_checkpoint.json").read_text())
    protected.update(
        Path(provenance[k]).resolve()
        for k in ["pooled_run", "generation_run"]
        if provenance.get(k)
    )
    protected.update(
        Path(r["path"]).resolve() for r in provenance.get("original_sources", [])
    )
    if any(output_root == p or output_root.is_relative_to(p) for p in protected):
        raise ValueError("Retry outputs must be separate from original sources")
    items = json.loads((source / "human/stimuli.json").read_text())
    bank = json.loads((source / "human/human_feature_bank.json").read_text())
    selected = select_ambiguous(items, bank, _cells(source))
    base = Path(__file__).parent
    dependencies = [
        Path(__file__),
        base / "judge.py",
        base / "clients.py",
        base / "batch.py",
        base / "full/execution.py",
        base / "full/budget.py",
        base / "full/reuse.py",
        base / "prompts/binary_v2.txt",
        base / "budget.py",
        base / "engine.py",
        base / "full/run.py",
        base / "io.py",
        base / "execution.py",
        base / "environment.py",
        base / "token_accounting.py",
        base / "experiments/evidence.py",
        base / "experiments/sources.py",
        base / "experiments/run.py",
    ]
    identity = dict(
        version=VERSION,
        source=str(source),
        source_manifest_sha256=file_hash(source / "manifest.json"),
        config=c,
        selection_sha256=digest(selected),
        code={str(p): file_hash(p) for p in dependencies},
    )
    out = output_root / ("retry_" + digest(identity)[:16])
    if (out / "prepared_manifest.json").exists():
        verify(out)
        return out
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Preserve incomplete retry export and choose another output root"
        )
    out.mkdir(parents=True)
    write_json(out / "identity.json", identity)
    write_json(out / "frozen_config.json", c)
    write_json(out / "stimuli.json", items)
    write_json(out / "human_feature_bank.json", bank)
    im = {i["item_id"]: i for i in items}
    fm = {f["feature_id"]: f for f in bank["features"]}
    with (out / "jobs.jsonl").open("w") as f:
        for row in selected:
            job = cell_job(im[row["item_id"]], fm[row["feature_id"]], c["judge"])
            record = {
                k: job[k]
                for k in [
                    "job_id",
                    "item_id",
                    "feature_id",
                    "rubric_version",
                    "prompt_hash",
                    "payload_sha256",
                ]
            }
            record.update(row, judge_model=c["judge"]["model_id"])
            f.write(canonical(record) + "\n")
    statuses = Counter(r["status"] for r in _cells(source))
    write_json(
        out / "selection_summary.json",
        dict(
            eligible_cells=len(selected),
            source_statuses=dict(statuses),
            primary_scope="human_only",
            task_features_included=False,
            resolved_cells_repeated=0,
            original_errors_included=False,
            mixed_rubric_output=True,
            api_calls=0,
        ),
    )
    write_json(
        out / "prepared_manifest.json",
        dict(
            identity=identity,
            artifacts={
                str(p.relative_to(out)): file_hash(p)
                for p in out.rglob("*")
                if p.is_file()
            },
        ),
    )
    write_json(output_root / "latest.json", dict(run_dir=str(out), api_calls=0))
    return out


def verify(out, *, check_code=False):
    out = Path(out).resolve()
    m = json.loads((out / "prepared_manifest.json").read_text())
    for name, h in m["artifacts"].items():
        if file_hash(out / name) != h:
            raise ValueError("Frozen retry artifact changed: " + name)
    source = Path(m["identity"]["source"])
    if file_hash(source / "manifest.json") != m["identity"]["source_manifest_sha256"]:
        raise ValueError("Original source manifest changed")
    verify_export(source)
    if check_code and any(file_hash(p) != h for p, h in m["identity"]["code"].items()):
        raise ValueError("Frozen retry code changed; prepare a new version")
    return dict(verified=True, frozen_artifacts=len(m["artifacts"]), api_calls=0)


def _load(out):
    out = Path(out).resolve()
    verify(out)
    c = json.loads((out / "frozen_config.json").read_text())
    jobs = read_jsonl(out / "jobs.jsonl")
    if rubric_for(c["judge"]) != RUBRIC:
        raise ValueError("Selective retry requires binary v2")
    items = json.loads((out / "stimuli.json").read_text())
    bank = json.loads((out / "human_feature_bank.json").read_text())
    profile_path = out / "provider_input_profile.json"
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else None
    return out, c, jobs, items, bank, profile


def measure_inputs(out, *, transport=None):
    out, c, jobs, items, bank, existing = _load(out)
    if existing is not None:
        return existing
    if transport is not None and not c.get("demo"):
        raise ValueError("Injected token counts require synthetic fixtures")
    if transport is None:
        load_api_credentials(c)
        transport = OpenAIBatchAPI(c["judge"]).count
    selected_ids = {j["item_id"] for j in jobs}
    profile = profile_inputs(
        [i for i in items if i["item_id"] in selected_ids], c["judge"], transport
    )
    _freeze_json(out / "provider_input_profile.json", profile)
    return profile


def _check(out, c, jobs, items, bank, profile, ledger):
    records = _read(out / "applicability/judgments.jsonl")
    _validated_records(jobs, records)
    grouped = defaultdict(list)
    for row in records:
        grouped[row["job_id"]].append(row)
    state_path = out / "applicability/batch_state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else dict(tasks={}, batches=[])
    )
    if state_path.exists():
        if state.get("state_hash") != digest(
            {k: v for k, v in state.items() if k != "state_hash"}
        ):
            raise ValueError("Retry batch state changed")
        if state["config_hash"] != digest(c):
            raise ValueError("Retry batch config changed")
    active = {t["job_id"] for t in state["tasks"].values() if not t.get("recorded")}
    allowed = {j["job_id"] for j in jobs}
    if not active <= allowed:
        raise ValueError("Active batch contains ineligible requests")
    reservations = Counter(
        e["job_id"] for e in ledger.events if e["event"] == "reserve"
    )
    im = {i["item_id"]: i for i in items}
    fm = {f["feature_id"]: f for f in bank["features"]}
    counts = Counter()
    cost = 0.0
    eligible = []
    bounds = []
    for job in jobs:
        jid = job["job_id"]
        rows = grouped[jid]
        values = {r["applies"] for r in rows if r["parse_status"] == "ok"}
        used = max(
            reservations[jid], max((r.get("attempt", 0) + 1 for r in rows), default=0)
        )
        if len(values) == 1:
            counts["resolved"] += 1
            continue
        if len(values) > 1:
            counts["conflicting"] += 1
            continue
        if jid in active:
            counts["pending"] += 1
            if used < c["max_attempts"]:
                bound = input_bound(
                    im[job["item_id"]], fm[job["feature_id"]], c["judge"], profile
                )["input_token_bound"]
                cost += (c["max_attempts"] - used) * estimate_cost(
                    c["judge"]["prices"], bound, c["judge"]["max_output_tokens"]
                )
            continue
        uncertain = any(
            not r.get("known_not_executed")
            and (
                r.get("usage_unknown")
                or not all(
                    type(r.get("usage", {}).get(k)) is int
                    for k in ["input_tokens", "output_tokens"]
                )
            )
            for r in rows
        )
        if uncertain or (reservations[jid] and not rows):
            counts["held"] += 1
            continue
        if used >= c["max_attempts"]:
            counts["exhausted"] += 1
            continue
        b = input_bound(im[job["item_id"]], fm[job["feature_id"]], c["judge"], profile)[
            "input_token_bound"
        ]
        if b > c["judge"]["max_input_tokens"]:
            raise ValueError("Retry input exceeds token envelope")
        bounds.append(b)
        counts["primary" if used == 0 else "retry"] += 1
        cost += (c["max_attempts"] - used) * estimate_cost(
            c["judge"]["prices"], b, c["judge"]["max_output_tokens"]
        )
        eligible.append(jid)
    blockers = []
    if profile is None and eligible:
        blockers.append(
            "Provider-measured binary-v2 input profile required before paid execution"
        )
    if ledger.total + cost > 100 + 1e-12:
        blockers.append(
            "Entire remaining two-attempt retry stage exceeds the cumulative $100 cap"
        )
    result = dict(
        version=VERSION,
        new_requests=counts["primary"],
        invalid_retry_requests=counts["retry"],
        selection_count=len(jobs),
        states=dict(counts),
        max_attempts=2,
        budget_cap_usd=100,
        cumulative_accounted_usd=ledger.total,
        remaining_attempts_upper_bound_usd=cost,
        cumulative_completion_ceiling_usd=ledger.total + cost,
        input_bound_range=[min(bounds), max(bounds)] if bounds else [],
        input_counting=(
            "provider_template_plus_feature_byte_bound_v1"
            if profile
            else "unmeasured_offline_bound"
        ),
        execution_mode="batch",
        rubric_version=RUBRIC,
        prices=c["judge"]["prices"],
        blockers=blockers,
        live_ready=not blockers,
        eligible_job_ids_sha256=digest(eligible),
    )
    result["preflight_hash"] = digest(
        [
            result,
            digest(ledger.events),
            digest(profile),
            file_hash(out / "prepared_manifest.json"),
        ]
    )
    return result, eligible, records


def preflight(out):
    out, c, jobs, items, bank, profile = _load(out)
    result, _, _ = _check(out, c, jobs, items, bank, profile, read_ledger(c))
    return result


class CollectOnlySession(FullSession):
    def _submit_prepared(self, batch):
        raise ValueError("Collection cannot submit paid jobs; use explicit execution")


def run(out, *, execute_live=False, preflight_hash=None, collect_only=False, api=None):
    out, c, jobs, items, bank, profile = _load(out)
    if api is not None and not c.get("demo"):
        raise ValueError("Injected provider requires a synthetic fixture")
    if execute_live and (c.get("demo") or api is not None):
        raise ValueError("Paid execution requires empirical inputs and actual provider")
    if not collect_only and not execute_live and api is None:
        raise ValueError("Paid execution requires explicit --execute-live")
    verify(out, check_code=True)
    if api is None:
        load_api_credentials(c)
    with (out / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ledger_for(c) as ledger:
            initial, eligible, _ = _check(out, c, jobs, items, bank, profile, ledger)
            if not collect_only:
                if preflight_hash != initial["preflight_hash"]:
                    raise ValueError("Obtain a fresh selective retry preflight hash")
                if initial["blockers"]:
                    raise ValueError("; ".join(initial["blockers"]))
            directory = out / "applicability"
            directory.mkdir(exist_ok=True)
            session = (CollectOnlySession if collect_only else FullSession)(
                c, directory, ledger, api=api
            )
            session.refresh()
            current, eligible, _ = _check(out, c, jobs, items, bank, profile, ledger)
            if not collect_only:
                if current["blockers"]:
                    raise ValueError("; ".join(current["blockers"]))
                eligible = set(eligible)
                im = {i["item_id"]: i for i in items}
                fm = {f["feature_id"]: f for f in bank["features"]}
                for job in jobs:
                    if job["job_id"] not in eligible:
                        continue
                    if not session.enqueue_cell(
                        im[job["item_id"]], fm[job["feature_id"]], profile
                    ):
                        break
                session.flush("judgment")
            check, _, records = _check(out, c, jobs, items, bank, profile, ledger)
    status = (
        "complete"
        if check["states"].get("resolved", 0) == len(jobs)
        else (
            "batch_pending"
            if session.active
            else (
                "ready"
                if check["new_requests"] or check["invalid_retry_requests"]
                else "needs_reconciliation"
            )
        )
    )
    result = dict(
        status=status,
        resolved_retry_cells=check["states"].get("resolved", 0),
        selected_retry_cells=len(jobs),
        mixed_rubric=True,
        cost_report=check,
        new_task_requests=0,
    )
    source = Path(json.loads((out / "identity.json").read_text())["source"])
    merged = merge_cells(_cells(source), jobs, records)
    fields = (
        list(merged[0])
        if merged
        else ["item_id", "feature_id", "applies", "status", "effective_rubric"]
    )
    write_csv(out / "effective_cells.csv", merged, fields)
    bykey = {(r["item_id"], r["feature_id"]): r["applies"] for r in merged}
    fids = [f["feature_id"] for f in bank["features"]]
    write_csv(
        out / "effective_matrix_partial.csv",
        [
            dict(
                item_id=i["item_id"], **{fid: bykey[i["item_id"], fid] for fid in fids}
            )
            for i in items
        ],
        ["item_id"] + fids,
    )
    result["effective_definitive_cells"] = sum(
        type(r["applies"]) is int or str(r["applies"]) in {"0", "1"} for r in merged
    )
    write_json(out / "run_state.json", result)
    verify(out)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-export",
        default="artifacts/feature_space_human_refresh/checkpoints/separated_99f0a371fc62c20a",
    )
    p.add_argument("--config", default="configs/feature_space_pilot.json")
    p.add_argument("--output-root", default="artifacts/feature_space_binary_retry")
    p.add_argument("--run-dir")
    p.add_argument("--preflight-hash")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=30)
    mode = p.add_mutually_exclusive_group(required=True)
    for flag in [
        "prepare",
        "dry-run",
        "measure-inputs",
        "execute-live",
        "collect",
        "verify",
    ]:
        mode.add_argument("--" + flag, action="store_true")
    a = p.parse_args()
    if a.poll_seconds < 5:
        p.error("Polling interval must be at least 5 seconds")
    if a.prepare:
        out = prepare(a.source_export, load_config(a.config), a.output_root)
        print(json.dumps(dict(run_dir=str(out), **preflight(out)), indent=2))
        return
    out = Path(
        a.run_dir
        or json.loads((Path(a.output_root) / "latest.json").read_text())["run_dir"]
    )
    if a.verify:
        print(json.dumps(verify(out), indent=2))
        return
    if a.measure_inputs:
        measure_inputs(out)
    if a.dry_run or a.measure_inputs:
        print(json.dumps(preflight(out), indent=2))
        return
    result = run(
        out,
        execute_live=a.execute_live,
        collect_only=a.collect,
        preflight_hash=a.preflight_hash,
    )
    print(json.dumps(result, indent=2), flush=True)
    while a.watch and (
        result["status"] == "batch_pending"
        or (a.execute_live and result["status"] == "ready")
    ):
        time.sleep(a.poll_seconds)
        check = preflight(out)
        result = run(
            out,
            execute_live=a.execute_live,
            collect_only=a.collect,
            preflight_hash=check["preflight_hash"],
        )
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

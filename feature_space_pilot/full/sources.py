"""Read only immutable inputs and sealed journal prefixes from baseline runs."""

import csv
import copy
import hashlib
import io
import json
from pathlib import Path
from ..io import digest
from ..prompts import parse_generation

FILES = [
    "frozen_config.json",
    "stimuli.jsonl",
    "generation_jobs.jsonl",
    "generations.jsonl",
    "feature_bank.csv",
    "judgment_jobs.jsonl",
    "judgments.jsonl",
    "execution_amendment.json",
    "evaluation_inputs.json",
    "frozen_source_registry.json",
]


def _jsonl(data, name):
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if name in {"generations.jsonl", "judgments.jsonl"}:
        for r in rows:
            if r.get("record_hash") != digest(
                {k: v for k, v in r.items() if k != "record_hash"}
            ) or r.get("response_hash") != digest(r.get("raw_response")):
                raise ValueError("Source journal hash mismatch: " + name)
    return rows


def load_source(path, *, allow_synthetic=False):
    path = Path(path).resolve()
    raw = (path / "manifest.json").read_bytes()
    m = json.loads(raw)
    if m.get("demo") and not allow_synthetic:
        raise ValueError("Expanded live pilot cannot use synthetic sources")
    content = {}
    hashes = {}
    blobs = {}
    for name in FILES:
        p = path / name
        expected = m["artifacts"].get(name)
        if expected is None:
            content[name] = [] if name.endswith((".jsonl", ".csv")) else {}
            continue
        if not p.exists():
            raise ValueError("Missing source artifact: " + name)
        data = p.read_bytes()
        checkpoint = m.get("journal_checkpoints", {}).get(name)
        if checkpoint:
            data = data[: checkpoint["bytes"]]
            expected = checkpoint["sha256"]
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError("Source artifact hash mismatch: " + name)
        hashes[name] = actual
        blobs[name] = data.decode("utf-8")
        if name.endswith(".jsonl"):
            content[name] = _jsonl(data, name)
        elif name.endswith(".csv"):
            content[name] = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        else:
            content[name] = json.loads(data)
    for name in ["frozen_config.json", "stimuli.jsonl", "generation_jobs.jsonl"]:
        if not hashes.get(name):
            raise ValueError("Missing required frozen source input: " + name)
    items = {i["item_id"]: i for i in content["stimuli.jsonl"]}
    jobs = {j["job_id"]: j for j in content["generation_jobs.jsonl"]}
    if len(items) != len(content["stimuli.jsonl"]) or len(jobs) != len(
        content["generation_jobs.jsonl"]
    ):
        raise ValueError("Duplicate source item/job identity")
    successful = {}
    generations = []
    model = content["frozen_config.json"]["generator"]["model_id"]
    for record in content["generations.jsonl"]:
        j = jobs.get(record["job_id"])
        if (
            j is None
            or record["item_id"] != j["item_id"]
            or record["item_id"] not in items
        ):
            raise ValueError("Unknown generation source item/job")
        if record.get("parse_status") == "ok":
            parsed = parse_generation(
                record["raw_response"], record["item_id"], cap=j.get("cap")
            )
            if parsed["features"] != record["features"]:
                raise ValueError("Parsed source features changed")
            if record["job_id"] in successful:
                raise ValueError("Duplicate accepted source generation")
            successful[record["job_id"]] = record
        generations.append(
            dict(record, source_run_id=m["run_id"], generation_model=model)
        )
    state_path = path / "batch_state.json"
    pending = []
    if state_path.exists():
        state = json.loads(state_path.read_bytes())
        if state.get("state_hash") != digest(
            {k: v for k, v in state.items() if k != "state_hash"}
        ):
            raise ValueError("Source Batch state hash mismatch")
        for b in state["batches"]:
            if b.get("ingested"):
                continue
            for cid in b["task_ids"]:
                t = state["tasks"][cid]
                if not t.get("recorded"):
                    pending.append(
                        dict(
                            job_id=t["job_id"],
                            stage=t["stage"],
                            batch_id=b.get("id"),
                            status=b["status"],
                            ticket=t.get("ticket"),
                            item_id=t.get("record_base", {}).get("item_id"),
                            feature_id=t.get("record_base", {}).get("feature_id"),
                            body=t["body"],
                        )
                    )
    return dict(
        path=str(path),
        manifest=m,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        config=content["frozen_config.json"],
        items=list(items.values()),
        jobs=list(jobs.values()),
        generations=generations,
        judgments=content["judgments.jsonl"],
        judgment_jobs=content["judgment_jobs.jsonl"],
        baseline_features=content["feature_bank.csv"],
        evaluation_inputs=content["evaluation_inputs.json"],
        execution_amendment=content["execution_amendment.json"],
        sources=content["frozen_source_registry.json"],
        input_hashes=hashes,
        frozen_blobs=blobs,
        pending=pending,
        generations_complete=len(successful)
        == len(jobs)
        == m["planned_generation_jobs"],
    )


def collect_sources(paths, *, allow_synthetic=False, expected_items=75):
    snapshots = []
    seen_runs = {}
    items = {}
    generations = []
    records = set()
    coverage = {}
    for path in paths:
        s = load_source(path, allow_synthetic=allow_synthetic)
        rid = s["manifest"]["run_id"]
        if rid in seen_runs:
            if s["input_hashes"] != seen_runs[rid]["input_hashes"]:
                raise ValueError("Conflicting source aliases")
            continue
        seen_runs[rid] = s
        snapshots.append(s)
        for item in s["items"]:
            iid = item["item_id"]
            if iid in items and items[iid] != item:
                raise ValueError("Conflicting global item identity: " + iid)
            items[iid] = item
            coverage.setdefault(iid, dict(planned=0, completed=0, failed_attempts=0))
        for job in s["jobs"]:
            coverage[job["item_id"]]["planned"] += 1
        for r in s["generations"]:
            key = r["job_id"], r.get("attempt", 0)
            if key in records:
                raise ValueError("Duplicate generation attempt across sources")
            records.add(key)
            generations.append(r)
            coverage[r["item_id"]][
                "completed" if r["parse_status"] == "ok" else "failed_attempts"
            ] += 1
    if len(items) != expected_items:
        raise ValueError(
            f"Global item scope must contain exactly {expected_items} items, found {len(items)}"
        )
    return dict(
        snapshots=snapshots,
        items=list(items.values()),
        generations=generations,
        coverage=coverage,
        generations_complete=all(s["generations_complete"] for s in snapshots),
        source_fingerprint=digest(
            [(s["manifest"]["run_id"], s["input_hashes"]) for s in snapshots]
        ),
        generation_fingerprint=digest([r["record_hash"] for r in generations]),
    )


def _frozen_inputs(source):
    return {
        s["manifest"]["run_id"]: {
            k: v
            for k, v in s["input_hashes"].items()
            if k in {"frozen_config.json", "stimuli.jsonl", "generation_jobs.jsonl"}
        }
        for s in source["snapshots"]
    }


def available_checkpoint(source):
    """Close the observed discovery scope without inventing successful samples."""
    successful = {
        r["job_id"] for r in source["generations"] if r["parse_status"] == "ok"
    }
    missing = [
        dict(source_run_id=s["manifest"]["run_id"], **j)
        for s in source["snapshots"]
        for j in s["jobs"]
        if j["job_id"] not in successful
    ]
    value = dict(
        version="available_generation_checkpoint_v1",
        items=source["items"],
        generations=source["generations"],
        coverage=source["coverage"],
        generation_fingerprint=source["generation_fingerprint"],
        generations_complete=source["generations_complete"],
        source_completion={
            s["manifest"]["run_id"]: s["generations_complete"]
            for s in source["snapshots"]
        },
        frozen_source_inputs=_frozen_inputs(source),
        missing_generation_jobs=missing,
        policy="User-authorized available-generation checkpoint; retain all accepted features and all 75 items; late generations remain in original studies and require a new vocabulary version",
    )
    value["checkpoint_hash"] = digest(value)
    return value


def apply_checkpoint(source, checkpoint):
    if checkpoint.get(
        "version"
    ) != "available_generation_checkpoint_v1" or checkpoint.get(
        "checkpoint_hash"
    ) != digest(
        {k: v for k, v in checkpoint.items() if k != "checkpoint_hash"}
    ):
        raise ValueError("Generation checkpoint hash mismatch")
    if checkpoint["items"] != source["items"] or checkpoint[
        "frozen_source_inputs"
    ] != _frozen_inputs(source):
        raise ValueError("Generation checkpoint source identity mismatch")
    live = {r["record_hash"]: r for r in source["generations"]}
    if any(live.get(r["record_hash"]) != r for r in checkpoint["generations"]):
        raise ValueError(
            "Generation checkpoint receipt absent or changed in verified sources"
        )
    result = copy.deepcopy(source)
    for k in [
        "generations",
        "coverage",
        "generation_fingerprint",
        "generations_complete",
        "missing_generation_jobs",
    ]:
        result[k] = copy.deepcopy(checkpoint[k])
    for s in result["snapshots"]:
        rid = s["manifest"]["run_id"]
        s["generations"] = [
            r for r in result["generations"] if r["source_run_id"] == rid
        ]
        s["generations_complete"] = checkpoint["source_completion"][rid]
    result.update(
        generation_scope_closed=True,
        generation_checkpoint_hash=checkpoint["checkpoint_hash"],
        generation_policy=checkpoint["policy"],
    )
    return result

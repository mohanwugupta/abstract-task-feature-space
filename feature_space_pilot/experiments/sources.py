"""Read frozen inputs and sealed prefixes while inference continues independently."""

import hashlib
import json
import csv
from pathlib import Path
from ..io import digest, file_hash
from ..prompts import parse_generation
from .banks import experiment_of
from .evidence import verify_receipt


def journal_prefix(path, *, parse=True):
    path = Path(path).resolve()
    with path.open("rb") as f:
        # Bound the read at open-time length; a concurrent append is next checkpoint.
        data = f.read(path.stat().st_size)
    end = data.rfind(b"\n") + 1
    data = data[:end]
    checkpoint = dict(
        path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
    )
    return (
        [json.loads(line) for line in data.splitlines() if line.strip()]
        if parse
        else []
    ), checkpoint


def verify_preserved(provenance):
    for name, expected in provenance["immutable_files"].items():
        if file_hash(name) != expected:
            raise ValueError("Original immutable input changed: " + name)
    for checkpoint in provenance["journal_prefixes"]:
        with Path(checkpoint["path"]).open("rb") as f:
            data = f.read(checkpoint["bytes"])
        if (
            len(data) != checkpoint["bytes"]
            or hashlib.sha256(data).hexdigest() != checkpoint["sha256"]
        ):
            raise ValueError("Original journal prefix changed: " + checkpoint["path"])
    return dict(
        verified=True,
        immutable_files=len(provenance["immutable_files"]),
        journal_prefixes=len(provenance["journal_prefixes"]),
        active_append_growth_allowed=True,
        source_writes=0,
        api_calls=0,
    )


def _frozen(root, provenance):
    m = json.loads((root / "full_manifest.json").read_text())
    for name, h in m["frozen_hashes"].items():
        path = root / name
        if file_hash(path) != h:
            raise ValueError("Original frozen hash mismatch: " + str(path))
        provenance["immutable_files"][str(path.resolve())] = h
    return m


def _matching(path, provenance):
    """Evaluation-only snapshot; generation never receives human norm labels."""
    from ..matching.protocol import parse_response

    root = Path(path).resolve()
    p = root / "analysis_manifest.json"
    manifest = json.loads(p.read_text())
    provenance["immutable_files"][str(p)] = file_hash(p)
    values = {}
    for name in ("inventory.json", "jobs.json", "judgments.jsonl"):
        q = root / name
        expected = manifest["artifacts"][name]
        data = q.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Semantic matching snapshot hash mismatch: " + name)
        provenance["immutable_files"][str(q)] = expected
        values[name] = (
            [json.loads(line) for line in data.splitlines() if line.strip()]
            if name.endswith(".jsonl")
            else json.loads(data)
        )
    jobs = {j["job_id"]: j for j in values["jobs.json"]}
    for r in values["judgments.jsonl"]:
        verify_receipt(r)
        if r.get("parse_status") == "ok":
            parsed = parse_response(r["raw_response"], jobs[r["job_id"]])
            if parsed["results"] != r["results"]:
                raise ValueError("Semantic matching parsed evidence changed")
    return dict(
        inventory=values["inventory.json"],
        records=values["judgments.jsonl"],
        manifest=manifest,
        source_path=str(root),
    )


def load_inputs(pooled_run):
    root = Path(pooled_run).resolve()
    provenance = dict(
        immutable_files={}, journal_prefixes=[], source_writes=0, api_calls=0
    )
    current = _frozen(root, provenance)
    c = json.loads((root / "frozen_config.json").read_text())
    generation_root = Path(current.get("source_full_run") or root).resolve()
    if generation_root != root:
        _frozen(generation_root, provenance)
    checkpoint = json.loads(
        (generation_root / "generation_checkpoint.json").read_text()
    )
    if checkpoint["checkpoint_hash"] != digest(
        {k: v for k, v in checkpoint.items() if k != "checkpoint_hash"}
    ):
        raise ValueError("Generation checkpoint hash mismatch")
    items = json.loads((root / "stimuli.json").read_text())
    if items != checkpoint["items"]:
        raise ValueError("Experiment stimuli differ from frozen generation checkpoint")
    pooled = json.loads((root / "frozen_global_bank.json").read_text())
    jobs, source_jobs, source_experiments = [], {}, {}
    original_sources = []
    for name in c["full_options"]["source_runs"]:
        p = (Path(c["root"]) / name).resolve()
        m = json.loads((p / "manifest.json").read_text())
        rid = m["run_id"]
        expected = checkpoint["frozen_source_inputs"][rid]
        content = {}
        for filename, h in expected.items():
            q = p / filename
            if file_hash(q) != h:
                raise ValueError("Generation source frozen hash mismatch: " + str(q))
            provenance["immutable_files"][str(q)] = h
            content[filename] = (
                json.loads(q.read_text())
                if filename.endswith(".json")
                else [
                    json.loads(line)
                    for line in q.read_text().splitlines()
                    if line.strip()
                ]
            )
        js = content["generation_jobs.jsonl"]
        source_jobs[rid] = {j["job_id"]: j for j in js}
        source_items = content["stimuli.jsonl"]
        kinds = {experiment_of(i) for i in source_items}
        if len(kinds) != 1:
            raise ValueError("Original generation source mixes experimental stimuli")
        source_experiments[rid] = kinds.pop()
        source_item_ids = {i["item_id"] for i in source_items}
        if any(j["item_id"] not in source_item_ids for j in js):
            raise ValueError(
                "Generation job provenance differs from its source stimuli"
            )
        jobs.extend(dict(j, source_run_id=rid) for j in js)
        original_sources.append(
            dict(
                path=str(p),
                run_id=rid,
                experiment=source_experiments[rid],
                model=content["frozen_config.json"]["generator"]["model_id"],
            )
        )
    itemmap = {i["item_id"]: i for i in items}
    for g in checkpoint["generations"]:
        rid = g["source_run_id"]
        job = source_jobs[rid].get(g["job_id"])
        if (
            job is None
            or any(
                job.get(k) != g.get(k)
                for k in ("item_id", "prompt_family", "replicate")
            )
            or experiment_of(itemmap[g["item_id"]]) != source_experiments[rid]
        ):
            raise ValueError("Generation experiment provenance mismatch")
        original = {
            k: v
            for k, v in g.items()
            if k not in {"record_hash", "source_run_id", "generation_model"}
        }
        if (
            digest(original) != g["record_hash"]
            or digest(g.get("raw_response")) != g["response_hash"]
        ):
            raise ValueError("Original generation receipt hash mismatch")
        if g.get("parse_status") == "ok":
            parsed = parse_generation(
                g["raw_response"], g["item_id"], cap=job.get("cap")
            )
            if parsed["features"] != g["features"]:
                raise ValueError("Generation phrase provenance changed")
    # Only projects with the same frozen pooled vocabulary qualify as old-ID sources.
    candidates = {root, generation_root}
    for mpath in root.parent.glob("full_*/full_manifest.json"):
        m = json.loads(mpath.read_text())
        if (
            m.get("demo", False) == current.get("demo", False)
            and m.get("frozen_hashes", {}).get("frozen_global_bank.json")
            == current["frozen_hashes"]["frozen_global_bank.json"]
        ):
            candidates.add(mpath.parent.resolve())
    evidence = []
    for p in sorted(candidates):
        m = _frozen(p, provenance)
        settings = json.loads((p / "frozen_config.json").read_text())["judge"]
        source_items = json.loads((p / "stimuli.json").read_text())
        source_bank = json.loads((p / "frozen_global_bank.json").read_text())
        literal = json.loads((p / "literal_binary_prompt.json").read_text())["system"]
        records = []
        journal = p / "applicability/judgments.jsonl"
        if journal.exists():
            records, prefix = journal_prefix(journal)
            provenance["journal_prefixes"].append(prefix)
        reused = p / "reuse_receipts.json"
        if reused.exists():
            provenance["immutable_files"][str(reused)] = file_hash(reused)
            for r in json.loads(reused.read_text()):
                if (
                    digest({k: v for k, v in r.items() if k != "reuse_hash"})
                    != r["reuse_hash"]
                ):
                    raise ValueError("Original reuse receipt hash mismatch")
                records.append(r["source_record"])
        evidence.append(
            dict(
                source_id=m["analysis_id"],
                items=source_items,
                features=source_bank["features"],
                settings=settings,
                literal_system=literal,
                records=records,
            )
        )
    matching_path = c["full_options"].get("semantic_matching_dir")
    semantic = (
        _matching(Path(c["root"]) / matching_path, provenance)
        if matching_path
        else None
    )
    if semantic and not {i["item_id"] for i in semantic["inventory"]["items"]} <= {
        i["item_id"] for i in items if experiment_of(i) == "human"
    }:
        raise ValueError("Semantic matching contains non-benchmark items")
    category_reference = []
    if c.get("human_abstract_summary"):
        q = (Path(c["root"]) / c["human_abstract_summary"]).resolve()
        if file_hash(q) != c["human_abstract_summary_sha256"]:
            raise ValueError("Aggregate category reference hash mismatch")
        provenance["immutable_files"][str(q)] = file_hash(q)
        with q.open(encoding="utf-8-sig", newline="") as f:
            category_reference = list(csv.DictReader(f))
    ledger = Path(c["root"]) / "artifacts/feature_space_pilot/budget_live.jsonl"
    if ledger.exists():
        _, prefix = journal_prefix(ledger, parse=False)
        provenance["journal_prefixes"].append(prefix)
    provenance.update(
        pooled_run=str(root),
        generation_run=str(generation_root),
        generation_checkpoint_hash=checkpoint["checkpoint_hash"],
        original_sources=original_sources,
    )
    verify_preserved(provenance)
    return dict(
        items=items,
        generations=checkpoint["generations"],
        generation_jobs=jobs,
        generation_checkpoint=checkpoint,
        pooled_bank=pooled,
        evidence_sources=evidence,
        config=c,
        semantic_snapshot=semantic,
        category_reference=category_reference,
        provenance=provenance,
    )

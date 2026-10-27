"""Preflight and recoverable pipeline. Importing this module performs no I/O or API calls."""

from __future__ import annotations
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from .io import (
    append_jsonl,
    canonical,
    digest,
    file_hash,
    read_jsonl,
    write_csv,
    write_json,
)
from .registry import load_items, validate_items
from .fixtures import task_items, demo_human_items, reconciliation_fixture
from .planner import plan_generation, pending_jobs
from .prompts import DIRECTORY, parse_generation
from .clients import FakeClient, HostedClient, generation_schema, judgment_schema
from .bank import consolidate
from .judge import RUBRIC, judge_prompt, parse_judgment, build_matrix
from .budget import Ledger, estimate_cost, validate_prices

STAGES = ("human", "tasks", "all")


def demo_config(root):
    model = dict(
        provider="fake",
        model_id="fake-frontier-v1",
        revision="fixture_v1",
        supports_seed=True,
        supports_temperature=True,
        supports_structured_outputs=True,
        max_output_tokens=600,
        max_input_tokens=30000,
        temperature=0.8,
        prices=dict(
            model_id="fake-frontier-v1",
            revision="fixture_v1",
            input_per_million=0,
            output_per_million=0,
            verified_at="2026-10-26",
            source_url="synthetic:zero-billing",
        ),
    )
    return dict(
        schema_version="pilot_v1",
        root=str(root),
        demo=True,
        replicates=10,
        seed=42,
        candidate_cap=200,
        max_attempts=2,
        api_cap_usd=500,
        stage_caps=dict(
            generation=120, judgment=220, adjudication=80, embeddings=30, reserve=50
        ),
        generator=copy.deepcopy(model),
        judge=copy.deepcopy(model),
        sources=[],
        protocols={},
        human_target_count=60,
        rate_limit_seconds=0,
        judgment_audit_fraction=0.05,
    )


def load_config(path):
    path = Path(path).resolve()
    c = json.loads(path.read_text())
    c["root"] = str((path.parent / Path(c.get("root", ".."))).resolve())
    return c


def resolve(c, path):
    p = Path(path)
    return p if p.is_absolute() else Path(c["root"]) / p


def selected_items(c, stage):
    human = []
    tasks = []
    if stage in {"human", "all"}:
        if c.get("demo"):
            human = demo_human_items()[: c.get("demo_human_count", 60)]
        elif c.get("human_stimuli"):
            human = load_items(resolve(c, c["human_stimuli"]))
    if stage in {"tasks", "all"} or c.get("smoke"):
        tasks = (
            load_items(resolve(c, c["task_stimuli"]))
            if c.get("task_stimuli")
            else task_items()
        )
    items = human + tasks
    if c.get("smoke_item_ids"):
        items = [i for i in items if i["item_id"] in c["smoke_item_ids"]]
    return validate_items(items)


def smoke_config(c):
    """Predeclared smoke amendment; never lowers source or model verification gates."""
    full = selected_items(c, "all")
    ids = []
    for typ in ["concept", "event", "object"]:
        choices = sorted(i["item_id"] for i in full if i["target_type"] == typ)
        if not choices:
            raise ValueError(
                f"Smoke requires a verified/frozen human item in class {typ}"
            )
        ids.append(choices[0])
    ids += ["poem_domain", "poem_type", "poem_T", "S_deadline", "poem_S_deadline"]
    if not set(ids) <= {i["item_id"] for i in full}:
        raise ValueError("Smoke task/situation fixture is missing")
    result = copy.deepcopy(c)
    result.update(
        smoke=True,
        smoke_item_ids=ids,
        replicates=2,
        candidate_cap=10,
        human_target_count=3,
        scope_amendment="smoke_v1: one per human class plus one task/situation pair and hierarchy parents; no scientific scale-up claim",
    )
    return result


def _source_specs(c):
    if c.get("source_registry"):
        return json.loads(resolve(c, c["source_registry"]).read_text())["sources"]
    return c.get("sources", [])


def preflight(c, stage):
    if stage not in STAGES:
        raise ValueError("Unknown pilot stage")
    for role in ["generator", "judge"]:
        if {"api_key", "authorization", "access_token", "secret"} & c[role].keys():
            raise ValueError(
                "Configure credentials through an environment variable name only"
            )
    if not 0 < c["api_cap_usd"] <= 500:
        raise ValueError("API cap must be >0 and <=500")
    if not 1 <= c["candidate_cap"] <= 200:
        raise ValueError("Candidate cap must be 1–200; amend design to increase it")
    if type(c["replicates"]) is not int or c["replicates"] < 2 or c["replicates"] % 2:
        raise ValueError(
            "Even positive replicate count >=2 required for matched component budgets"
        )
    if type(c["max_attempts"]) is not int or not 1 <= c["max_attempts"] <= 5:
        raise ValueError("max_attempts must be 1–5")
    items = selected_items(c, stage)
    blockers = []
    warnings = []
    demo = bool(c.get("demo"))
    if not demo and stage == "all":
        blockers.append("Live human and task stages must use separate frozen outputs")
    human = [x for x in items if x["target_type"] in {"concept", "event", "object"}]
    refs = []
    source_hashes = {}
    if not demo and stage == "human":
        from .references import load_reference

        if len(human) != c.get("human_target_count", 60):
            blockers.append("Human source sample is not frozen at the declared size")
        for spec in _source_specs(c):
            if spec.get("primary", True):
                try:
                    spec = dict(
                        spec, path=str(resolve(c, spec.get("path", "__missing__")))
                    )
                    ref = load_reference(spec, live=True)
                    refs.append(ref)
                    source_hashes[spec["dataset_id"]] = spec["sha256"]
                except (ValueError, OSError, KeyError) as e:
                    blockers.append(
                        f'Human source gate ({spec.get("dataset_id","unknown")}): {spec.get("blocker", str(e))}'
                    )
        by_source = {r["dataset_id"]: r for r in refs}
        for item in human:
            ref = by_source.get(item["source_dataset"])
            if ref is None:
                blockers.append(
                    f'Missing verified human source for {item["source_dataset"]}'
                )
                continue
            if (
                item["language"] != ref["language"]
                and item.get("translation_status") != "reviewed"
            ):
                blockers.append("Human stimulus requires reviewed translation")
            if item["text"] not in {r["item"] for r in ref["rows"]} and item.get(
                "source_text"
            ) not in {r["item"] for r in ref["rows"]}:
                blockers.append(
                    f'Human stimulus absent from verified reference: {item["item_id"]}'
                )
        if not refs:
            blockers.append("No verified human source data")
    # Freeze both the complete literal template set and each constructed request.
    prompt_hashes = {p.name: file_hash(p) for p in sorted(DIRECTORY.glob("*.txt"))}
    settings = c["generator"]
    jobs = plan_generation(
        items,
        settings,
        replicates=c["replicates"],
        seed=c["seed"],
        protocols=c.get("protocols"),
    )
    if not demo:
        for item in human:
            from .prompts import generation_prompt

            try:
                generation_prompt(item, "P0", live=True, protocols=c.get("protocols"))
            except ValueError as e:
                blockers.append(str(e))
    count = len(jobs)
    if not demo and stage == "human" and len(human) != c.get("human_target_count", 60):
        count = c.get("human_target_count", 60) * 3 * c["replicates"]
    scope_count = (
        c.get("human_target_count", 60)
        if not demo and stage == "human" and not c.get("smoke")
        else len(items)
    )
    judgment_cells = scope_count * c["candidate_cap"]
    estimate = {}
    deviations = []
    for role, key, calls in [
        ("generator", "generation", count),
        ("judge", "judgment", judgment_cells),
    ]:
        m = c[role]
        try:
            if not m.get("model_id") or not m.get("revision"):
                raise ValueError(
                    "Pinned hosted frontier model ID and revision required"
                )
            prices = validate_prices(m.get("prices") or {})
            if (
                prices["model_id"] != m["model_id"]
                or prices["revision"] != m["revision"]
            ):
                raise ValueError("Model-specific price identity mismatch")
            if not m.get("supports_structured_outputs"):
                raise ValueError(
                    "Verified strict structured output capability required"
                )
            if not demo and m.get("provider") != "openai_compatible":
                raise ValueError(
                    "Select a hosted OpenAI-compatible API provider adapter"
                )
            if not demo and (
                not m.get("capabilities_verified_at")
                or not m.get("capabilities_source")
            ):
                raise ValueError(
                    "Verified provider capability documentation is required"
                )
            input_limit = m["max_input_tokens"]
            output_limit = m["max_output_tokens"]
            if (
                type(input_limit) is not int
                or input_limit < 1
                or type(output_limit) is not int
                or output_limit < 1
            ):
                raise ValueError("Positive input/output token limits required")
            worst = (
                estimate_cost(prices, input_limit, output_limit)
                * calls
                * c["max_attempts"]
            )
            estimate[key] = dict(
                calls=calls,
                attempts_with_retries=calls * c["max_attempts"],
                input_tokens_per_attempt=input_limit,
                output_tokens_per_attempt=output_limit,
                worst_case_usd=worst,
                prices=prices,
            )
            if worst > c["stage_caps"][key]:
                blockers.append(
                    f"{key} worst-case estimate exceeds stage budget; freeze a smaller bank/design"
                )
        except (ValueError, KeyError) as e:
            blockers.append(f"{role} model/price gate: {e}")
        for capability in ["seed", "temperature"]:
            if not m.get("supports_" + capability):
                deviations.append(
                    f"{role}: {capability} omitted by frozen provider capability policy"
                )
    total = sum(e["worst_case_usd"] for e in estimate.values())
    if total > c["api_cap_usd"]:
        blockers.append("Worst-case plan exceeds cumulative budget")
    ledger_path = resolve(
        c, c.get("artifact_root", "artifacts/feature_space_pilot")
    ) / ("budget_demo.jsonl" if demo else "budget_live.jsonl")
    prior = Ledger(ledger_path, cap=c["api_cap_usd"], stage_caps=c["stage_caps"])
    if ledger_path.exists():
        prior.events = read_jsonl(ledger_path)
    if not demo and stage == "human" and len(human) == 60:
        expected = {"concept": 24, "event": 24, "object": 12}
        if {t: sum(x["target_type"] == t for x in human) for t in expected} != expected:
            blockers.append(
                "Human class quotas differ from frozen 24/24/12 design; documented amendment required"
            )
    source_code = {
        p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    design_config = {
        k: v
        for k, v in c.items()
        if k
        not in {
            "root",
            "adjudications",
            "relational_audit",
            "embedding_file",
            "embedding_sha256",
            "category_coding_file",
            "category_coding_sha256",
        }
    }
    design = digest(
        dict(
            config=design_config,
            stage=stage,
            item_hashes=[x["content_hash"] for x in items],
            prompts=prompt_hashes,
            source_hashes=source_hashes,
            code=source_code,
            fixture_hash=digest(reconciliation_fixture()),
        )
    )
    run_id = ("demo_" if demo else "live_") + stage + "_" + design[:16]
    run_dir = (
        resolve(c, c.get("artifact_root", "artifacts/feature_space_pilot")) / run_id
    )
    remaining_total = 0
    for key, e in estimate.items():
        path = run_dir / (
            "generations.jsonl" if key == "generation" else "judgments.jsonl"
        )
        records = _read(path)
        complete = {r["job_id"] for r in records if r.get("parse_status") == "ok"}
        if key == "generation":
            ids = [j["job_id"] for j in jobs] if len(jobs) == count else None
        else:
            jpath = run_dir / "judgment_jobs.jsonl"
            ids = [j["job_id"] for j in read_jsonl(jpath)] if jpath.exists() else None
        if ids is None:
            attempts = e["attempts_with_retries"]
        else:
            used = {}
            for event in prior.events:
                if event["event"] == "reserve":
                    used[event["job_id"]] = used.get(event["job_id"], 0) + 1
            attempts = sum(
                max(0, c["max_attempts"] - used.get(j, 0))
                for j in ids
                if j not in complete
            )
        remaining = (
            estimate_cost(
                e["prices"],
                e["input_tokens_per_attempt"],
                e["output_tokens_per_attempt"],
            )
            * attempts
        )
        e.update(
            remaining_attempts_upper_bound=attempts, remaining_worst_case_usd=remaining
        )
        remaining_total += remaining
        if prior.stage_total(key) + remaining > c["stage_caps"][key] + 1e-12:
            blockers.append(
                f"{key} remaining plan plus prior cumulative liabilities exceeds stage budget"
            )
    if prior.total + remaining_total > c["api_cap_usd"] + 1e-12:
        blockers.append(
            "Remaining plan plus prior cumulative liabilities exceeds budget"
        )
    return dict(
        stage=stage,
        demo=demo,
        design_hash=design,
        preflight_hash=digest([design, remaining_total, prior.total, blockers]),
        selected_item_count=len(items),
        selected_item_ids=[x["item_id"] for x in items],
        generation_jobs=count,
        actual_plannable_generation_jobs=len(jobs),
        judgment_cells_upper_bound=judgment_cells,
        estimates=estimate,
        worst_case_usd=total if len(estimate) == 2 else None,
        remaining_worst_case_usd=remaining_total if len(estimate) == 2 else None,
        budget_cap_usd=c["api_cap_usd"],
        blockers=sorted(set(blockers)),
        warnings=warnings,
        accounted_usd=prior.total,
        remaining_usd=c["api_cap_usd"] - prior.total,
        capability_deviations=deviations,
        prompt_hashes=prompt_hashes,
        source_hashes=source_hashes,
        source_code_hashes=source_code,
        live_ready=not demo and not blockers,
    )


def _read(path):
    records = read_jsonl(path) if path.exists() else []
    for r in records:
        content = {k: v for k, v in r.items() if k != "record_hash"}
        if r.get("record_hash") != digest(content) or r.get("response_hash") != digest(
            r.get("raw_response")
        ):
            raise ValueError(f"Journal record hash mismatch: {path.name}")
    return records


def _resume_manifest(out):
    from .report import verify_manifest

    m = json.loads((out / "manifest.json").read_text())
    if m["status"] == "complete":
        return verify_manifest(out)
    # Reports are regenerated after interrupted writes; frozen inputs and all
    # previously checkpointed journal bytes must still be identical.
    frozen = {
        "stimuli.jsonl",
        "generation_jobs.jsonl",
        "frozen_config.json",
        "frozen_prompts.json",
        "preflight.json",
        "reconciliation_fixture.json",
        "judgment_jobs.jsonl",
    }
    for name in frozen & m["artifacts"].keys():
        if file_hash(out / name) != m["artifacts"][name]:
            raise ValueError(f"Frozen artifact hash changed: {name}")
    for name, checkpoint in m.get("journal_checkpoints", {}).items():
        data = (out / name).read_bytes()
        if (
            len(data) < checkpoint["bytes"]
            or hashlib.sha256(data[: checkpoint["bytes"]]).hexdigest()
            != checkpoint["sha256"]
        ):
            raise ValueError(f"Journal checkpoint hash mismatch: {name}")
    return m


def evaluation_inputs_hash(c):
    return digest(
        {
            k: c.get(k)
            for k in [
                "adjudications",
                "relational_audit",
                "embedding_file",
                "embedding_sha256",
                "category_coding_file",
                "category_coding_sha256",
            ]
        }
    )


def _call(
    client,
    prompt,
    schema,
    settings,
    ledger,
    job_id,
    stage,
    path,
    record_base,
    parser,
    max_attempts,
    delay,
):
    # Every provider attempt is reserved first; no provider retry is hidden in the adapter.
    attempts = sum(
        e["event"] == "reserve" and e.get("job_id") == job_id for e in ledger.events
    )
    for attempt in range(attempts, max_attempts):
        input_bound = (
            len(canonical([prompt["system"], prompt["user"], schema]).encode()) + 2048
        )
        if input_bound > settings["max_input_tokens"]:
            raise ValueError("Request exceeds frozen input-token ceiling")
        amount = estimate_cost(
            settings["prices"], input_bound, settings["max_output_tokens"]
        )
        ticket = ledger.reserve(job_id, stage, amount)
        started = time.monotonic()
        raw = ""
        usage = {}
        parsed = None
        error = None
        try:
            raw, usage = client.complete(
                system=prompt["system"],
                user=prompt["user"],
                schema=schema,
                seed=record_base.get("seed"),
                temperature=settings.get("temperature", 0),
            )
        except Exception as e:
            # Do not log provider error strings: they may contain credentials or request bodies.
            error = f"Provider exception: {type(e).__name__}; usage unknown, reservation retained"
        fatal = False
        if error is None:
            try:
                ledger.settle(ticket, usage, settings["prices"])
            except ValueError as e:
                error = str(e)
                fatal = True
            if error is None:
                try:
                    if usage.get("finish_reason") not in {"stop", "end_turn"}:
                        raise ValueError("Truncated/refused response")
                    parsed = parser(raw)
                except (ValueError, KeyError, TypeError) as e:
                    error = str(e)
        record = dict(
            record_base,
            job_id=job_id,
            attempt=attempt,
            raw_response=raw,
            response_hash=digest(raw),
            parse_status="error" if error else "ok",
            error=error,
            usage=usage,
            latency_seconds=round(time.monotonic() - started, 6),
            budget_ticket=ticket,
            **(parsed or {}),
        )
        record["record_hash"] = digest(record)
        append_jsonl(path, record)
        if fatal:
            raise ValueError(error)
        if parsed is not None:
            return record
        if delay:
            time.sleep(min(delay, 30))
    return None


def run_pipeline(
    c, stage, *, client=None, execute_live=False, preflight_hash=None, max_jobs=None
):
    plan = preflight(c, stage)
    if execute_live:
        if c.get("demo"):
            raise ValueError("synthetic sources/models cannot execute live")
        if plan["blockers"]:
            raise ValueError("; ".join(plan["blockers"]))
        if preflight_hash != plan["preflight_hash"]:
            raise ValueError("Run --dry-run and supply its matching --preflight-hash")
        if client is not None:
            raise ValueError("Injected/fake client is prohibited during live execution")
    elif not c.get("demo"):
        raise ValueError(
            "Paid execution requires --execute-live and a matching dry-run preflight hash"
        )
    elif client is not None and not isinstance(client, FakeClient):
        raise ValueError(
            "Offline demo accepts only FakeClient; use execute-live for paid adapters"
        )
    if plan["blockers"]:
        raise ValueError("; ".join(plan["blockers"]))
    items = selected_items(c, stage)
    jobs = plan_generation(
        items,
        c["generator"],
        replicates=c["replicates"],
        seed=c["seed"],
        live=execute_live,
        protocols=c.get("protocols"),
    )
    run_id = (
        ("demo_" if c.get("demo") else "live_") + stage + "_" + plan["design_hash"][:16]
    )
    out = resolve(c, c.get("artifact_root", "artifacts/feature_space_pilot")) / run_id
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = resolve(
        c, c.get("artifact_root", "artifacts/feature_space_pilot")
    ) / ("budget_demo.jsonl" if c.get("demo") else "budget_live.jsonl")
    with Ledger(
        ledger_path, cap=c["api_cap_usd"], stage_caps=c["stage_caps"]
    ) as ledger:
        if (out / "manifest.json").exists():
            old = _resume_manifest(out)
            if old["design_hash"] != plan["design_hash"]:
                raise ValueError("Frozen design changed")
            if old["status"] == "complete" and old.get(
                "evaluation_inputs_hash"
            ) == evaluation_inputs_hash(c):
                return dict(old, output_dir=str(out))
            append_jsonl(out / "manifest_history.jsonl", old)
        # These are reproducible frozen inputs, not mutable historical results.
        for name, records in [
            ("stimuli.jsonl", items),
            ("generation_jobs.jsonl", jobs),
        ]:
            text = "".join(canonical(r) + "\n" for r in records)
            p = out / name
            if p.exists() and p.read_text() != text:
                raise ValueError(f"Frozen input changed: {name}")
            if not p.exists():
                p.write_text(text)
        if not (out / "frozen_config.json").exists():
            write_json(out / "frozen_config.json", c)
        if not (out / "preflight.json").exists():
            write_json(out / "preflight.json", plan)
        write_json(
            out / "evaluation_inputs.json",
            {
                k: c.get(k)
                for k in [
                    "adjudications",
                    "relational_audit",
                    "embedding_file",
                    "embedding_sha256",
                    "category_coding_file",
                    "category_coding_sha256",
                ]
            },
        )
        write_json(out / "reconciliation_fixture.json", reconciliation_fixture())
        write_json(
            out / "frozen_prompts.json",
            {p.name: p.read_text() for p in sorted(DIRECTORY.glob("*.txt"))},
        )
        evaluation_config = dict(c)
        frozen_sources = _source_specs(c)
        source_hashes = dict(plan["source_hashes"])
        if c.get("demo") and any(
            i["target_type"] in {"concept", "event", "object"} for i in items
        ):
            ref_path = out / "synthetic_human_reference.csv"
            if not ref_path.exists():
                rows = [
                    dict(
                        item=i["text"],
                        feature=f,
                        category=i["target_type"],
                        frequency=1,
                        language="en",
                    )
                    for i in items
                    if i["target_type"] in {"concept", "event", "object"}
                    for f in FakeClient.properties(i["text"])
                ]
                write_csv(
                    ref_path,
                    rows,
                    ["item", "feature", "category", "frequency", "language"],
                )
                ref_path.chmod(0o444)
            spec = dict(
                dataset_id="synthetic_demo",
                kind="synthetic",
                verified=False,
                path=str(ref_path),
                sha256=file_hash(ref_path),
                language="en",
                data_level="item_features",
                columns={
                    k: k
                    for k in ["item", "feature", "category", "frequency", "language"]
                },
                participants="synthetic fixtures; no people",
                instructions="Synthetic test rule, not participant instructions",
                license="Synthetic engineering fixture",
                url="synthetic:fixture_v1",
                codebook="feature_space_pilot.clients.FakeClient.properties",
            )
            evaluation_config["_evaluation_reference_spec"] = spec
            frozen_sources = [spec]
            source_hashes["synthetic_demo"] = spec["sha256"]
        write_json(out / "frozen_source_registry.json", dict(sources=frozen_sources))
        gen_client = client or (
            FakeClient() if c.get("demo") else HostedClient(c["generator"])
        )
        judge_client = client or (
            FakeClient() if c.get("demo") else HostedClient(c["judge"])
        )
        gen_path = out / "generations.jsonl"
        done = _read(gen_path)
        processed = 0
        for job in pending_jobs(jobs, done):
            if max_jobs is not None and processed >= max_jobs:
                break
            parser = lambda raw, j=job: parse_generation(
                raw, j["item_id"], cap=j["cap"]
            )
            _call(
                gen_client,
                job,
                generation_schema(job["item_id"], job["cap"]),
                c["generator"],
                ledger,
                job["job_id"],
                "generation",
                gen_path,
                {
                    k: job[k]
                    for k in [
                        "item_id",
                        "prompt_family",
                        "prompt_version",
                        "prompt_hash",
                        "model_id",
                        "model_revision",
                        "replicate",
                        "seed",
                    ]
                },
                parser,
                c["max_attempts"],
                c.get("rate_limit_seconds", 0),
            )
            processed += 1
            if c.get("rate_limit_seconds"):
                time.sleep(min(c["rate_limit_seconds"], 30))
        generations = [r for r in _read(gen_path) if r["parse_status"] == "ok"]
        if len({r["job_id"] for r in generations}) != len(generations):
            raise ValueError("Duplicate successful generation jobs")
        status = "partial"
        bank = []
        judgments = []
        if len(generations) == len(jobs):
            bank, candidates, review = consolidate(
                generations, cap=c["candidate_cap"], merges=c.get("merges", [])
            )
            if not bank:
                raise ValueError(
                    "No candidate features; empty generation inventory requires review"
                )
            for name, rows, fields in [
                (
                    "feature_bank.csv",
                    bank,
                    [
                        "feature_id",
                        "text",
                        "cluster_id",
                        "inventory_hash",
                        "merge_decisions",
                    ],
                ),
                (
                    "feature_candidates.csv",
                    candidates,
                    [
                        "phrase",
                        "normalized",
                        "job_id",
                        "origin_item",
                        "prompt_family",
                        "replicate",
                        "raw_feature_index",
                        "feature_id",
                        "selected",
                        "normalization",
                    ],
                ),
                (
                    "merge_review.csv",
                    review,
                    ["left", "right", "decision", "reviewer", "reason"],
                ),
            ]:
                write_csv(out / name, rows, fields)
            judge_path = out / "judgments.jsonl"
            judge_records = _read(judge_path)
            seen = {r["job_id"] for r in judge_records if r["parse_status"] == "ok"}
            judge_jobs = []
            for item in items:
                for feature in bank:
                    p = judge_prompt(item, feature)
                    job_id = digest(
                        [
                            item["content_hash"],
                            feature["feature_id"],
                            p["prompt_hash"],
                            c["judge"],
                        ]
                    )
                    judge_jobs.append(
                        dict(
                            job_id=job_id,
                            item_id=item["item_id"],
                            feature_id=feature["feature_id"],
                            **p,
                        )
                    )
            p = out / "judgment_jobs.jsonl"
            text = "".join(canonical(j) + "\n" for j in judge_jobs)
            if p.exists() and p.read_text() != text:
                raise ValueError("Frozen judgment scope changed")
            if not p.exists():
                p.write_text(text)
            for job in judge_jobs:
                if job["job_id"] in seen:
                    continue
                parser = lambda raw, j=job: parse_judgment(
                    raw, j["item_id"], j["feature_id"]
                )
                result = _call(
                    judge_client,
                    job,
                    judgment_schema(job["item_id"], job["feature_id"]),
                    c["judge"],
                    ledger,
                    job["job_id"],
                    "judgment",
                    judge_path,
                    dict(
                        item_id=job["item_id"],
                        feature_id=job["feature_id"],
                        rubric_version=RUBRIC,
                        judge_model=c["judge"]["model_id"],
                        prompt_hash=job["prompt_hash"],
                    ),
                    parser,
                    c["max_attempts"],
                    c.get("rate_limit_seconds", 0),
                )
                if result and result.get("status") == "ambiguous":
                    append_jsonl(out / "judgment_audit.jsonl", result)
                if c.get("rate_limit_seconds"):
                    time.sleep(min(c["rate_limit_seconds"], 30))
            judgments = [r for r in _read(judge_path) if r["parse_status"] == "ok"]
            # Human adjudications resolve ambiguous/missing cells, never overwrite a primary judgment silently.
            by_cell = {(r["item_id"], r["feature_id"]): r for r in judgments}
            for a in c.get("adjudications", []):
                if not a.get("reviewer") or not a.get("evidence"):
                    raise ValueError("Named adjudication reviewer/evidence required")
                key = (a["item_id"], a["feature_id"])
                validated = parse_judgment(
                    canonical(
                        {
                            k: a[k]
                            for k in [
                                "item_id",
                                "feature_id",
                                "applies",
                                "status",
                                "evidence",
                            ]
                        }
                    ),
                    *key,
                )
                prior = by_cell.get(key)
                if (
                    prior
                    and prior["status"] == "ok"
                    and prior["applies"] != a["applies"]
                ):
                    raise ValueError(
                        "Conflicting resolved judgment requires a new rubric/design"
                    )
                by_cell[key] = dict(
                    validated,
                    rubric_version=RUBRIC,
                    judge_model="human:" + a["reviewer"],
                    prompt_hash="manual_adjudication",
                )
            judgments = list(by_cell.values())
            resolved = [r for r in judgments if r["status"] == "ok"]
            fields = [
                "item_id",
                "feature_id",
                "rubric_version",
                "judge_model",
                "applies",
                "evidence",
                "prompt_hash",
            ]
            write_csv(
                out / "binary_judgments.csv",
                [{k: r[k] for k in fields} for r in resolved],
                fields,
            )
            scope = dict(
                item_ids=[x["item_id"] for x in items],
                feature_ids=[f["feature_id"] for f in bank],
                excluded_combinations=[],
                policy="complete_selected_stage_cross_product",
            )
            write_json(out / "matrix_scope.json", scope)
            try:
                matrix = build_matrix(
                    scope["item_ids"], scope["feature_ids"], judgments
                )
            except ValueError:
                status = "needs_adjudication"
            else:
                write_csv(
                    out / "binary_matrix.csv",
                    [
                        dict(item_id=i, **dict(zip(scope["feature_ids"], row)))
                        for i, row in zip(scope["item_ids"], matrix)
                    ],
                    ["item_id"] + scope["feature_ids"],
                )
                status = "complete"
        all_run_attempts = _read(gen_path) + _read(out / "judgments.jsonl")
        write_json(
            out / "cost_report.json",
            dict(
                ledger.report(),
                billed_execution=execute_live,
                run_input_tokens=sum(
                    r["usage"].get("input_tokens", 0) or 0 for r in all_run_attempts
                ),
                run_output_tokens=sum(
                    r["usage"].get("output_tokens", 0) or 0 for r in all_run_attempts
                ),
                run_provider_attempts=len(all_run_attempts),
                ledger_path=str(ledger_path),
                billing_note="USD estimates based on provider token usage; provider invoice remains external authority",
                demo_billed_usd=0 if c.get("demo") else None,
            ),
        )
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=c["root"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        manifest = dict(
            schema_version="pilot_v1",
            run_id=run_id,
            design_hash=plan["design_hash"],
            stage=stage,
            demo=bool(c.get("demo")),
            smoke=bool(c.get("smoke")),
            status=status,
            planned_generation_jobs=len(jobs),
            completed_generation_jobs=len(generations),
            planned_judgments=len(items) * len(bank),
            resolved_judgments=sum(r.get("status") == "ok" for r in judgments),
            judgment_cells_upper_bound=plan["judgment_cells_upper_bound"],
            judgment_plan_status="frozen" if bank else "awaiting_candidate_bank",
            source_code_hashes=plan["source_code_hashes"],
            model_snapshots={"generator": c["generator"], "judge": c["judge"]},
            seed=c["seed"],
            prompt_hashes=plan["prompt_hashes"],
            source_hashes=source_hashes,
            config_hash=digest(c),
            evaluation_inputs_hash=evaluation_inputs_hash(c),
            selected_item_ids=[x["item_id"] for x in items],
            git_commit=commit,
            fixture_hash=digest(reconciliation_fixture()),
            candidate_inventory_hash=bank[0]["inventory_hash"] if bank else None,
            journal_checkpoints={
                p.name: dict(bytes=p.stat().st_size, sha256=file_hash(p))
                for p in [out / "generations.jsonl", out / "judgments.jsonl"]
                if p.exists()
            },
            artifacts={},
            limitations=(
                ["Synthetic engineering demonstration; no scientific results"]
                if c.get("demo")
                else []
            ),
        )
        write_json(out / "manifest.json", manifest)
        from .report import write_reports, seal_manifest

        write_reports(out, evaluation_config, generations, items, bank, judgments)
        manifest = seal_manifest(out)
        return dict(manifest, output_dir=str(out))

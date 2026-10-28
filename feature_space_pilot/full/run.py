"""Prepare or execute the uncapped global pilot without modifying baseline studies."""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import json
from pathlib import Path
import time
from ..batch import OpenAIBatchAPI, BatchTransientError, BatchCreationUncertain
from ..budget import estimate_cost
from ..engine import load_config, _read
from ..environment import load_api_credentials
from ..io import (
    canonical,
    digest,
    file_hash,
    read_jsonl,
    write_json,
    write_csv,
    append_jsonl,
)
from ..progress import console_reporter, emit
from .sources import collect_sources, apply_checkpoint
from .bank import build_bank, propose_equivalences
from .reuse import reuse_evidence, cell_job, previous_full_sources
from .matrix import matrix_state
from .budget import (
    IndexedLedger,
    cost_plan,
    profile_inputs,
    validate_profile,
    template_payload,
)
from .execution import FullSession
from .equivalence import (
    build_jobs as equivalence_jobs,
    reviews_from_records,
    EquivalenceSession,
    payload as equivalence_payload,
    token_budget,
)
from .report import write_full_report


def load_settings(path):
    path = Path(path).resolve()
    options = json.loads(path.read_text())
    if options.get("feature_cap") is not None:
        raise ValueError("Expanded pilot forbids an arbitrary feature cap")
    c = load_config(path.parent / options["model_config"])
    c["full_options"] = options
    for k in [
        "max_attempts",
        "batch_max_requests",
        "batch_max_in_flight",
        "batch_max_queued_input_tokens",
    ]:
        c[k] = options[k]
    if c["api_cap_usd"] != 100 or sum(c["stage_caps"].values()) > 100:
        raise ValueError("Full pilot must preserve the cumulative $100 cap")
    budget_policy = options.get("stage_budget_policy", "original_allocations")
    if budget_policy == "shared_total_only":
        c["stage_caps"] = {stage: c["api_cap_usd"] for stage in c["stage_caps"]}
    elif budget_policy != "original_allocations":
        raise ValueError("Unknown stage budget policy")
    c["equivalence"] = dict(
        c["judge"], max_output_tokens=options["equivalence_output_tokens"]
    )
    return c


def resolve(c, path):
    p = Path(path)
    return p if p.is_absolute() else Path(c["root"]) / p


def source_ready(source):
    return bool(source.get("generation_scope_closed", source["generations_complete"]))


def ledger_for(c):
    return IndexedLedger(
        resolve(c, c.get("artifact_root", "artifacts/feature_space_pilot"))
        / ("budget_demo.jsonl" if c.get("demo") else "budget_live.jsonl"),
        cap=c["api_cap_usd"],
        stage_caps=c["stage_caps"],
        budget_policy=c.get("full_options", {}).get(
            "stage_budget_policy", "original_allocations"
        ),
    )


def read_ledger(c):
    ledger = ledger_for(c)
    ledger.events = read_jsonl(ledger.path) if ledger.path.exists() else []
    ledger._balances()
    return ledger


def _reviews(c):
    path = c["full_options"].get("semantic_reviews_file")
    if not path:
        return [], None
    path = resolve(c, path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    return rows, file_hash(path)


def _state(path):
    if not path.exists():
        return dict(tasks={}, batches=[])
    s = json.loads(path.read_text())
    if s.get("state_hash") != digest({k: v for k, v in s.items() if k != "state_hash"}):
        raise ValueError("Full Batch state hash mismatch")
    return s


def _effective_reviews(automatic, manual):
    result = {tuple(sorted((r["left"], r["right"]))): dict(r) for r in automatic}
    for r in manual:
        if r.get("decision") == "pending":
            continue
        key = tuple(sorted((r["left"], r["right"])))
        if key in result:
            r = dict(r, supersedes_automated_review=digest(result[key]))
        result[key] = r
    return list(result.values())


def _phase_cost(jobs, c, ledger, records, state, counts, source_complete):
    s = c["equivalence"]
    done = {r["job_id"] for r in records if r["parse_status"] == "ok"}
    used = CounterLike(ledger.events)
    remaining = 0.0
    attempts = missing_counts = exhausted = 0
    active = {t["job_id"] for t in state["tasks"].values() if not t.get("recorded")}
    located = active | {r["job_id"] for r in records}
    unlocated = sum(
        used.get(j["job_id"], 0) > 0 and j["job_id"] not in located for j in jobs
    )
    for j in jobs:
        if j["job_id"] in done:
            continue
        n = max(0, c["max_attempts"] - used.get(j["job_id"], 0))
        attempts += n
        budget = token_budget(j, s, counts)
        remaining += n * estimate_cost(
            s["prices"], budget["input_token_bound"], s["max_output_tokens"]
        )
        if n and not budget["exact_provider_count"]:
            missing_counts += 1
        if not n and j["job_id"] not in active:
            exhausted += 1
    blockers = []
    if unlocated:
        blockers.append(
            f"{unlocated} equivalence requests have prior reservations without located receipts or Batch state; reconcile before proceeding"
        )
    if not source_complete:
        blockers.append(
            "All source generation jobs must be complete before semantic assistance"
        )
    if missing_counts:
        blockers.append("Provider input counts for equivalence requests are missing")
    if ledger.total + remaining > ledger.cap + 1e-12:
        blockers.append("Equivalence phase exceeds the cumulative budget")
    if (
        ledger.stage_total("adjudication") + remaining
        > ledger.stage_caps.get("adjudication", 0) + 1e-12
    ):
        blockers.append("Equivalence phase exceeds the adjudication-stage allocation")
    finished = (
        not active
        and not unlocated
        and all(
            j["job_id"] in done or used.get(j["job_id"], 0) >= c["max_attempts"]
            for j in jobs
        )
    )
    return dict(
        planned_requests=len(jobs),
        completed_requests=len(done),
        active_requests=len(active),
        remaining_attempts_upper_bound=attempts,
        remaining_upper_bound_usd=remaining,
        cumulative_completion_upper_bound_usd=ledger.total + remaining,
        missing_provider_counts=missing_counts,
        exhausted_requests=exhausted,
        unlocated_prior_requests=unlocated,
        finished=finished,
        blockers=blockers,
        live_ready=not blockers,
    )


def CounterLike(events):
    result = {}
    for e in events:
        if e["event"] == "reserve":
            result[e["job_id"]] = result.get(e["job_id"], 0) + 1
    return result


def make_plan(c, *, profile=None):
    options = c["full_options"]
    if options.get("feature_cap") is not None:
        raise ValueError("No arbitrary full-bank feature cap is permitted")
    source = collect_sources(
        [resolve(c, p) for p in options["source_runs"]],
        allow_synthetic=bool(c.get("demo")),
        expected_items=options["expected_items"],
    )
    generation_policy = options.get("generation_policy", "all_planned_successful")
    if generation_policy == "frozen_available_checkpoint":
        checkpoint_path = resolve(c, options["generation_checkpoint_file"])
        source = apply_checkpoint(source, json.loads(checkpoint_path.read_text()))
    elif generation_policy != "all_planned_successful":
        raise ValueError("Unknown source generation policy")
    manual, review_hash = _reviews(c)
    base = build_bank(source["generations"], reviews=manual)
    proposals = propose_equivalences(
        base, neighbors=options.get("proposal_neighbors", 8)
    )
    reviewed = {
        tuple(
            sorted(
                (
                    r.get("submitted_left", r["left"]),
                    r.get("submitted_right", r["right"]),
                )
            )
        ): r["decision"]
        for r in base["review_decisions"]
    }
    for p in proposals:
        p["decision"] = reviewed.get(tuple(sorted((p["left"], p["right"]))), "pending")
    jobs = equivalence_jobs(
        base,
        [p for p in proposals if p["decision"] == "pending"],
        pairs_per_job=options.get("equivalence_pairs_per_request", 5),
    )
    if profile is None and options.get("provider_input_profile"):
        path = resolve(c, options["provider_input_profile"])
        if path.exists():
            profile = json.loads(path.read_text())
    if profile is not None:
        validate_profile(profile, c["judge"])
    code = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    for name in [
        "batch.py",
        "budget.py",
        "clients.py",
        "engine.py",
        "judge.py",
        "prompts.py",
        "io.py",
        "report.py",
        "metrics.py",
        "composition.py",
        "environment.py",
        "references.py",
        "source_adapters.py",
        "human_inputs.py",
        "token_accounting.py",
    ]:
        code["support/" + name] = file_hash(Path(__file__).parents[1] / name)
    code["binary_prompt"] = file_hash(
        Path(__file__).parents[1] / "prompts/binary_v1.txt"
    )
    frozen_sources = [
        dict(
            run_id=s["manifest"]["run_id"],
            inputs={
                k: v
                for k, v in s["input_hashes"].items()
                if k in {"frozen_config.json", "stimuli.jsonl", "generation_jobs.jsonl"}
            },
        )
        for s in source["snapshots"]
    ]
    identity = dict(
        version="global_full_v1",
        source_frozen_inputs=frozen_sources,
        judge=c["judge"],
        equivalence=c["equivalence"],
        max_attempts=c["max_attempts"],
        review_hash=review_hash,
        profile_hash=profile.get("profile_hash") if profile else None,
        code_hashes=code,
        feature_cap=None,
        options=options,
        demo=bool(c.get("demo")),
        generation_checkpoint_hash=source.get("generation_checkpoint_hash"),
    )
    design = digest(identity)
    out = resolve(c, options["artifact_root"]) / ("full_" + design[:16])
    for j in jobs:
        j["job_id"] = "equivalence_" + digest([j["job_id"], c["equivalence"]])
    eqout = out / "equivalence"
    eqrecords = _read(eqout / "equivalences.jsonl")
    automatic = reviews_from_records(eqrecords)
    bank = build_bank(
        source["generations"], reviews=_effective_reviews(automatic, manual)
    )
    automatic_decisions = {
        tuple(sorted((r["left"], r["right"]))): r["decision"] for r in automatic
    }
    for p in proposals:
        key = tuple(sorted((p["left"], p["right"])))
        decision = automatic_decisions.get(key)
        if decision and p["decision"] == "pending":
            p["decision"] = decision
    counts_path = eqout / "provider_input_counts.json"
    count_wrapper = (
        json.loads(counts_path.read_text()) if counts_path.exists() else None
    )
    if count_wrapper:
        if count_wrapper.get("hash") != digest(
            {k: v for k, v in count_wrapper.items() if k != "hash"}
        ) or count_wrapper["settings_hash"] != digest(c["equivalence"]):
            raise ValueError("Equivalence provider count cache changed")
    counts = count_wrapper["counts"] if count_wrapper else {}
    ledger = read_ledger(c)
    eqstate = _state(eqout / "batch_state.json")
    eqcost = _phase_cost(
        jobs, c, ledger, eqrecords, eqstate, counts, source_ready(source)
    )
    reuse_sources = source["snapshots"] + previous_full_sources(
        resolve(c, options["artifact_root"]), out
    )
    reuse = reuse_evidence(reuse_sources, source["items"], bank["features"], c["judge"])
    held_path = out / "applicability/unlocated_prior_requests.json"
    if held_path.exists():
        resolved = {(r["item_id"], r["feature_id"]) for r in reuse["reused"]}
        reuse["unresolved_prior"].extend(
            r
            for r in json.loads(held_path.read_text())
            if (r["item_id"], r["feature_id"]) not in resolved
        )
    primary = _read(out / "applicability/judgments.jsonl")
    ownstate = _state(out / "applicability/batch_state.json")
    matrix = matrix_state(source["items"], bank["features"], reuse["reused"] + primary)
    fullcost = cost_plan(
        source["items"],
        bank["features"],
        c,
        ledger,
        reuse,
        profile=profile,
        generations_complete=source["generations_complete"],
        source_scope_closed=source_ready(source),
        current_records=primary,
        active_tasks=list(ownstate["tasks"].values()),
    )
    if not eqcost["finished"]:
        fullcost["bank_provisional"] = True
        fullcost["final_consolidated_features"] = None
        fullcost["blockers"].append(
            "Semantic assistance is unfinished; final canonical bank is not frozen"
        )
        fullcost["live_ready"] = False
    return dict(
        config=c,
        identity=identity,
        design_hash=design,
        analysis_id=out.name,
        output_dir=str(out),
        source=source,
        base_bank=base,
        bank=bank,
        proposals=proposals,
        equivalence_jobs=jobs,
        equivalence_records=eqrecords,
        equivalence_counts=counts,
        equivalence_cost=eqcost,
        profile=profile,
        reuse=reuse,
        state=matrix,
        primary_records=primary,
        primary_active=any(not b.get("ingested") for b in ownstate["batches"]),
        applicability_cost=fullcost,
        ledger_events_hash=digest(ledger.events),
    )


def preflight(plan, *, stage="all"):
    phase = (
        ("equivalence" if not plan["equivalence_cost"]["finished"] else "applicability")
        if stage == "all"
        else stage
    )
    cost = (
        plan["equivalence_cost"]
        if phase == "equivalence"
        else plan["applicability_cost"]
    )
    result = dict(
        analysis_id=plan["analysis_id"],
        output_dir=plan["output_dir"],
        requested_stage=stage,
        active_phase=phase,
        source_generations_complete=plan["source"]["generations_complete"],
        generation_scope_closed=source_ready(plan["source"]),
        missing_generation_jobs=len(plan["source"].get("missing_generation_jobs", [])),
        feature_cap=None,
        bank_inventory_hash=plan["bank"]["inventory_hash"],
        equivalence=plan["equivalence_cost"],
        applicability=plan["applicability_cost"],
        blockers=cost["blockers"],
        live_ready=not cost["blockers"],
    )
    result["preflight_hash"] = digest(
        [
            plan["design_hash"],
            plan["source"]["generation_fingerprint"],
            plan["ledger_events_hash"],
            [
                r.get("record_hash", r.get("reuse_hash"))
                for r in plan["primary_records"] + plan["reuse"]["reused"]
            ],
            [r["record_hash"] for r in plan["equivalence_records"]],
            result,
        ]
    )
    return result


def _freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Frozen full-pilot input changed: " + path.name)
    else:
        write_json(path, value)
    return file_hash(path)


def freeze_inputs(plan):
    out = Path(plan["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    c = plan["config"]
    source = plan["source"]
    frozen = {}
    for name, value in [
        ("identity.json", plan["identity"]),
        ("frozen_config.json", c),
        ("stimuli.json", source["items"]),
        ("source_frozen_inputs.json", plan["identity"]["source_frozen_inputs"]),
    ]:
        frozen[name] = _freeze_json(out / name, value)
    frozen["literal_binary_prompt.json"] = _freeze_json(
        out / "literal_binary_prompt.json",
        dict(
            system=template_payload(source["items"][0], c["judge"])["input"][0][
                "content"
            ],
            rubric_version="binary_v1",
        ),
    )
    if plan["profile"] is not None:
        frozen["provider_input_profile.json"] = _freeze_json(
            out / "provider_input_profile.json", plan["profile"]
        )
    if source_ready(source):
        frozen["closed_generations.json"] = _freeze_json(
            out / "closed_generations.json", source["generations"]
        )
        frozen["equivalence_jobs.json"] = _freeze_json(
            out / "equivalence_jobs.json", plan["equivalence_jobs"]
        )
        if source.get("generation_checkpoint_hash"):
            checkpoint = json.loads(
                resolve(c, c["full_options"]["generation_checkpoint_file"]).read_text()
            )
            frozen["generation_checkpoint.json"] = _freeze_json(
                out / "generation_checkpoint.json", checkpoint
            )
    # Inventory identity describes columns; snapshot identity also includes origins
    # and review evidence that may grow without changing the vocabulary.
    bankdir = out / "banks" / digest(plan["bank"])[:16]
    bankdir.mkdir(parents=True, exist_ok=True)
    if not (bankdir / "bank.json").exists():
        write_json(bankdir / "bank.json", plan["bank"])
        write_csv(
            bankdir / "feature_bank.csv",
            plan["bank"]["features"],
            [
                "feature_id",
                "text",
                "equivalence_keys",
                "source_phrases",
                "n_origins",
                "human_reviewed",
                "inventory_hash",
            ],
        )
        write_csv(
            bankdir / "feature_origins.csv",
            plan["bank"]["origins"],
            list(plan["bank"]["origins"][0]) if plan["bank"]["origins"] else ["status"],
        )
        write_csv(
            bankdir / "feature_exclusions.csv",
            plan["bank"]["exclusions"],
            (
                list(plan["bank"]["exclusions"][0])
                if plan["bank"]["exclusions"]
                else ["status"]
            ),
        )
    elif json.loads((bankdir / "bank.json").read_text()) != plan["bank"]:
        raise ValueError("Frozen canonical bank artifact changed")
    if source_ready(source) and plan["equivalence_cost"]["finished"]:
        frozen["frozen_global_bank.json"] = _freeze_json(
            out / "frozen_global_bank.json", plan["bank"]
        )
        frozen["global_matrix_scope.json"] = _freeze_json(
            out / "global_matrix_scope.json",
            dict(
                item_ids=[i["item_id"] for i in source["items"]],
                feature_ids=[f["feature_id"] for f in plan["bank"]["features"]],
                excluded_combinations=[],
                policy="complete_frozen_item_shared_vocabulary_no_feature_cap",
            ),
        )
    return frozen, str(bankdir)


def _report(plan, *, render=True):
    c = plan["config"]
    out = Path(plan["output_dir"])
    source = plan["source"]
    relational = {}
    if c["full_options"].get("relational_reviews_file"):
        relational = json.loads(
            resolve(c, c["full_options"]["relational_reviews_file"]).read_text()
        )
    semdir = c["full_options"].get("semantic_matching_dir")
    semdir = resolve(c, semdir) if semdir and resolve(c, semdir).exists() else None
    semantic_checkpoint = None
    if semdir:
        m = json.loads((semdir / "analysis_manifest.json").read_text())
        semantic_checkpoint = m["artifacts"].get("judgments.jsonl")
    evidence = digest(
        [
            plan["bank"]["inventory_hash"],
            source["generation_fingerprint"],
            [
                (r.get("record_hash"), r.get("reuse_hash"))
                for r in plan["primary_records"] + plan["reuse"]["reused"]
            ],
            [(p["left_id"], p["right_id"], p["decision"]) for p in plan["proposals"]],
            semantic_checkpoint,
            relational,
            render,
        ]
    )[:16]
    reportdir = out / "reports" / evidence
    summary = write_full_report(
        reportdir,
        source,
        plan["bank"],
        plan["state"],
        c,
        plan["applicability_cost"],
        plan["reuse"],
        profile=plan["profile"],
        proposals=plan["proposals"],
        render=render,
        relational_reviews=relational,
        semantic_matching_dir=semdir,
        primary_records=plan["primary_records"],
    )
    return str(reportdir), summary


def advance(
    plan,
    *,
    execute_live=False,
    preflight_hash=None,
    api=None,
    render=True,
    stage="all",
    progress=None,
):
    check = preflight(plan, stage=stage)
    c = plan["config"]
    requested = execute_live or api is not None
    if preflight_hash != check["preflight_hash"]:
        raise ValueError("Obtain a fresh full-pilot preflight hash")
    if execute_live:
        if (
            c.get("demo")
            or api is not None
            or any(s["manifest"].get("demo") for s in plan["source"]["snapshots"])
        ):
            raise ValueError(
                "Live full pilot requires empirical sources; synthetic or injected clients prohibited"
            )
        load_api_credentials(c)
    elif api is not None and not c.get("demo"):
        raise ValueError(
            "Injected fixture API is prohibited for empirical full execution"
        )
    if requested and check["blockers"]:
        raise ValueError("; ".join(check["blockers"]))
    out = Path(plan["output_dir"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / (plan["analysis_id"] + ".lock")).open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "Another full-pilot controller owns this project"
            ) from None
        frozen, bankdir = freeze_inputs(plan)
        if requested:
            with ledger_for(c) as ledger:
                fresh = make_plan(c, profile=plan["profile"])
                if fresh["design_hash"] != plan["design_hash"]:
                    raise ValueError("Full scientific design changed")
                current = preflight(fresh, stage=stage)
                if current["preflight_hash"] != preflight_hash:
                    raise ValueError("Obtain a fresh full-pilot preflight hash")
                if current["blockers"]:
                    raise ValueError("; ".join(current["blockers"]))
                if current["active_phase"] == "equivalence":
                    phaseout = out / "equivalence"
                    phaseout.mkdir(exist_ok=True)
                    session = EquivalenceSession(
                        c,
                        phaseout,
                        ledger,
                        api=api or OpenAIBatchAPI(c["equivalence"]),
                        progress=progress,
                    )
                    session.refresh()
                    for job in fresh["equivalence_jobs"]:
                        if not session.enqueue_pair_job(
                            job, fresh["equivalence_counts"]
                        ):
                            break
                    session.flush("adjudication")
                    if session.unlocated_prior_jobs:
                        write_json(
                            phaseout / "unlocated_prior_requests.json",
                            list(session.unlocated_prior_jobs.values()),
                        )
                else:
                    phaseout = out / "applicability"
                    phaseout.mkdir(exist_ok=True)
                    session = FullSession(
                        c,
                        phaseout,
                        ledger,
                        api=api or OpenAIBatchAPI(c["judge"]),
                        progress=progress,
                        prior_attempts=fresh["reuse"]["compatible_attempts"],
                    )
                    session.refresh()
                    current_records = _read(phaseout / "judgments.jsonl")
                    current_state = matrix_state(
                        fresh["source"]["items"],
                        fresh["bank"]["features"],
                        fresh["reuse"]["reused"] + current_records,
                    )
                    prior = {
                        (r["item_id"], r["feature_id"])
                        for r in fresh["reuse"]["unresolved_prior"]
                    }
                    externally_pending = {
                        (r.get("item_id"), r.get("feature_id"))
                        for r in fresh["reuse"]["external_pending"].values()
                    }
                    # Scheduling favors case-study rows, but every frozen cell stays in scope.
                    ordered = sorted(
                        enumerate(fresh["source"]["items"]),
                        key=lambda x: (
                            x[1].get("target_type") in {"concept", "event", "object"},
                            x[0],
                        ),
                    )
                    funding_stage = fresh["applicability_cost"]["authorized_stage"]
                    if funding_stage in {"collect_pending", "review_only"}:
                        ordered = []
                    attempt_policy = (
                        "primary"
                        if funding_stage == "primary_pass"
                        else "retry" if funding_stage == "invalid_retries" else "all"
                    )
                    filled = False
                    for n, item in ordered:
                        for k, feature in enumerate(fresh["bank"]["features"]):
                            key = item["item_id"], feature["feature_id"]
                            if (
                                current_state["matrix"][n, k] >= 0
                                or key in prior
                                or key in externally_pending
                            ):
                                continue
                            if not session.enqueue_cell(
                                item,
                                feature,
                                fresh["profile"],
                                attempt_policy=attempt_policy,
                            ):
                                filled = True
                                break
                        if filled:
                            break
                    session.flush("judgment")
                    if session.unlocated_prior_jobs:
                        write_json(
                            phaseout / "unlocated_prior_requests.json",
                            list(session.unlocated_prior_jobs.values()),
                        )
        post = make_plan(c, profile=plan["profile"])
        if post["design_hash"] != plan["design_hash"]:
            raise ValueError("Full project identity changed during execution")
        frozen, bankdir = freeze_inputs(post)
        reportdir, summary = _report(post, render=render)
        state_paths = [
            out / "equivalence/batch_state.json",
            out / "applicability/batch_state.json",
        ]
        active = any(
            any(not b.get("ingested") for b in _state(p)["batches"])
            for p in state_paths
        )
        nextcheck = preflight(post, stage=stage)
        if active:
            status = "batch_pending"
        elif (
            post["state"]["complete"]
            and source_ready(post["source"])
            and post["equivalence_cost"]["finished"]
        ):
            status = "complete"
        elif not source_ready(post["source"]):
            status = "waiting_for_sources"
        elif nextcheck["blockers"]:
            status = "preflight_blocked"
        elif (
            requested
            and nextcheck["active_phase"] == "applicability"
            and post["state"]["unresolved"]
            and nextcheck["applicability"]["authorized_stage"] == "review_only"
        ):
            status = "needs_review"
        else:
            status = "ready"
        matrix_name = (
            "binary_matrix.csv"
            if post["state"]["complete"]
            else "applicability_matrix_partial.csv"
        )
        m = dict(
            version="global_full_v1",
            analysis_id=post["analysis_id"],
            status=status,
            demo=bool(c.get("demo")),
            human_reviewed=False,
            source_generations_complete=post["source"]["generations_complete"],
            generation_scope_closed=source_ready(post["source"]),
            missing_generation_jobs=len(
                post["source"].get("missing_generation_jobs", [])
            ),
            generation_fingerprint=post["source"]["generation_fingerprint"],
            bank_inventory_hash=post["bank"]["inventory_hash"],
            bank_dir=bankdir,
            bank_frozen=source_ready(post["source"])
            and post["equivalence_cost"]["finished"],
            canonical_features=len(post["bank"]["features"]),
            n_items=len(post["source"]["items"]),
            feature_cap=None,
            planned_cells=post["state"]["planned_cells"],
            resolved_cells=post["state"]["resolved_cells"],
            matrix_complete=post["state"]["complete"],
            matrix_file=str(Path(reportdir) / matrix_name),
            report_dir=reportdir,
            equivalence=post["equivalence_cost"],
            applicability=post["applicability_cost"],
            frozen_hashes=frozen,
            identity=post["identity"],
            source_provenance=[
                dict(
                    run_id=s["manifest"]["run_id"],
                    path=s["path"],
                    manifest_sha256=s["manifest_sha256"],
                    input_hashes=s["input_hashes"],
                )
                for s in post["source"]["snapshots"]
            ],
            report_manifest_sha256=file_hash(Path(reportdir) / "report_manifest.json"),
            scientific_completion_requires_manual_reviews=True,
            scientific_complete=False,
            scientific_status=(
                "synthetic_engineering_fixture"
                if c.get("demo")
                else (
                    "matrix_complete_pending_scientific_reviews"
                    if post["state"]["complete"]
                    else "incomplete_data_or_judgments"
                )
            ),
        )
        write_json(out / "full_manifest.json", m)
        write_json(out / "reuse_receipts.json", post["reuse"]["reused"])
        write_json(
            out / "current_cost_report.json",
            dict(
                equivalence=post["equivalence_cost"],
                applicability=post["applicability_cost"],
            ),
        )
        write_json(
            out / "latest_report.json",
            dict(
                report_dir=reportdir,
                status=status,
                matrix_complete=m["matrix_complete"],
            ),
        )
        emit(
            progress,
            "finished",
            message=f"Full pilot: {status}; {m['canonical_features']} uncapped features, {m['resolved_cells']}/{m['planned_cells']} cells; report {reportdir}",
        )
        return m


def measure_inputs(c, *, stage="all", progress=None):
    if c.get("demo"):
        raise ValueError("Provider input audit requires empirical source configuration")
    plan = make_plan(c)
    load_api_credentials(c)
    if plan["profile"] is None:
        emit(
            progress,
            "token_count",
            message="Measuring 75 frozen atomic templates; no model-completion jobs submitted",
        )
        profile = profile_inputs(
            plan["source"]["items"], c["judge"], OpenAIBatchAPI(c["judge"]).count
        )
        path = resolve(c, c["full_options"]["provider_input_profile"])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, profile)
        plan = make_plan(c)
    if stage != "applicability" and source_ready(plan["source"]):
        out = Path(plan["output_dir"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with (out.parent / (plan["analysis_id"] + ".lock")).open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            phase = out / "equivalence"
            phase.mkdir(parents=True, exist_ok=True)
            counts = dict(plan["equivalence_counts"])
            api = OpenAIBatchAPI(c["equivalence"])
            pending = {}
            for job in plan["equivalence_jobs"]:
                body = equivalence_payload(job, c["equivalence"])
                key = digest(body)
                if key not in counts:
                    pending[key] = body

            def count(pair):
                key, body = pair
                result = api.count(
                    {
                        k: v
                        for k, v in body.items()
                        if k in {"model", "input", "text", "reasoning", "instructions"}
                    }
                )
                n = result.get("input_tokens")
                if type(n) is not int or n < 0:
                    raise ValueError("Invalid provider equivalence input count")
                return key, dict(request_sha256=key, input_tokens=n)

            def save():
                wrapper = dict(settings_hash=digest(c["equivalence"]), counts=counts)
                wrapper["hash"] = digest(wrapper)
                write_json(phase / "provider_input_counts.json", wrapper)

            emit(
                progress,
                "token_count",
                message=f"Measuring {len(pending)} complete-bank equivalence requests; no completion jobs submitted",
            )
            try:
                with ThreadPoolExecutor(max_workers=6) as pool:
                    futures = [pool.submit(count, pair) for pair in pending.items()]
                    for n, future in enumerate(as_completed(futures), 1):
                        key, value = future.result()
                        counts[key] = value
                        if n % 50 == 0 or n == len(futures):
                            save()
                            emit(
                                progress,
                                "token_count",
                                message=f"Equivalence input counts {n}/{len(futures)}",
                            )
            finally:
                save()
    return make_plan(c)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/feature_space_full.json")
    p.add_argument(
        "--stage", choices=["all", "equivalence", "applicability"], default="all"
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--count-inputs", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    p.add_argument("--preflight-hash")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=120)
    p.add_argument(
        "--auto-count-inputs",
        action="store_true",
        help="In explicit live watch mode, audit missing provider counts before any completion submission",
    )
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args()
    if a.poll_seconds < 5:
        p.error("Poll interval must be at least five seconds")
    if a.execute_live and not a.preflight_hash:
        p.error("Explicit execution requires a fresh --preflight-hash")
    try:
        c = load_settings(a.config)
        reporter = console_reporter()
        plan = make_plan(c)
        expected = plan["analysis_id"]
        if a.count_inputs:
            plan = measure_inputs(c, stage=a.stage, progress=reporter)
            print(json.dumps(preflight(plan, stage=a.stage), indent=2))
            return
        if a.dry_run:
            print(json.dumps(preflight(plan, stage=a.stage), indent=2))
            return
        token = a.preflight_hash
        while True:
            try:
                plan = make_plan(c)
                if plan["analysis_id"] != expected:
                    raise ValueError(
                        "Frozen full-pilot code/config/source-input design changed"
                    )
                check = preflight(plan, stage=a.stage)
                if a.prepare:
                    result = advance(
                        plan,
                        preflight_hash=check["preflight_hash"],
                        render=not a.no_figures,
                        stage=a.stage,
                        progress=reporter,
                    )
                elif check["blockers"]:
                    # Save a concrete reviewable plan; never submit an affordable subset.
                    result = advance(
                        plan,
                        preflight_hash=check["preflight_hash"],
                        render=not a.no_figures,
                        stage=a.stage,
                        progress=reporter,
                    )
                    permanent_budget = any(
                        "budget" in b or "allocation" in b for b in check["blockers"]
                    )
                    if permanent_budget or not a.watch:
                        break
                    if a.auto_count_inputs and source_ready(plan["source"]):
                        counted = measure_inputs(c, stage=a.stage, progress=reporter)
                        if counted["analysis_id"] != expected:
                            raise ValueError(
                                "Input-profile design changed; obtain a new full preflight"
                            )
                        plan = counted
                        token = None
                        continue
                else:
                    token = token or check["preflight_hash"]
                    result = advance(
                        plan,
                        execute_live=True,
                        preflight_hash=token,
                        render=not a.no_figures,
                        stage=a.stage,
                        progress=reporter,
                    )
            except (BatchTransientError, BatchCreationUncertain):
                if not a.watch:
                    raise
                emit(
                    reporter,
                    "collector_retry",
                    message="Temporary transport/uncertain acknowledgement; retaining all reservations and reconciling later",
                )
            except ValueError as problem:
                transient = (
                    str(problem)
                    in {
                        "Another process owns the cumulative budget ledger",
                        "Obtain a fresh full-pilot preflight hash",
                    }
                    or str(problem).startswith("Matching snapshot busy")
                    or str(problem).startswith("Missing required frozen source input")
                )
                if not a.watch or not transient:
                    raise
                emit(
                    reporter,
                    "collector_retry",
                    message="Source checkpoint or shared ledger is busy; will retry without submitting duplicate work",
                )
            else:
                if not a.watch or result["status"] in {"complete", "needs_review"}:
                    break
                if a.execute_live and result["status"] == "preflight_blocked":
                    break
            emit(
                reporter,
                "waiting",
                message=f"Waiting {a.poll_seconds:g}s; existing collectors continue unchanged",
            )
            token = None
            time.sleep(a.poll_seconds)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as problem:
        p.exit(2, f"Full pilot blocked: {problem}\n")
    except KeyboardInterrupt:
        p.exit(
            130,
            "Local full-pilot collector detached; existing provider work retained.\n",
        )


if __name__ == "__main__":
    main()

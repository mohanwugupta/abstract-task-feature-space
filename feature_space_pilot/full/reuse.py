"""Reuse only the same literal atomic query under compatible request configuration."""

from collections import Counter, defaultdict
import json
import hashlib
from pathlib import Path
from ..clients import build_payload, judgment_schema
from ..io import digest, file_hash
from ..judge import judge_prompt, parse_judgment, rubric_for

POLICY = "same_item_content_feature_text_rubric_revision_and_literal_request_v1"


def cell_job(item, feature, settings):
    version = rubric_for(settings)
    p = judge_prompt(item, feature, rubric_version=version)
    body = build_payload(
        settings,
        system=p["system"],
        user=p["user"],
        schema=judgment_schema(
            item["item_id"], feature["feature_id"], rubric_version=version
        ),
    )
    return dict(
        job_id=digest(
            [item["content_hash"], feature["feature_id"], p["prompt_hash"], settings]
        ),
        item_id=item["item_id"],
        feature_id=feature["feature_id"],
        **p,
        payload=body,
        payload_sha256=digest(body)
    )


def model_identity(s):
    return {
        k: s.get(k)
        for k in [
            "provider",
            "base_url",
            "model_id",
            "revision",
            "api_format",
            "reasoning_effort",
            "max_output_tokens",
        ]
    }


def reuse_evidence(sources, items, features, settings):
    itemmap = {i["item_id"]: i for i in items}
    featuremap = {f["feature_id"]: f for f in features}
    reused = {}
    unresolved = []
    pending = {}
    rejected = Counter()
    conflicts = set()
    compatible_attempts = {}
    for source in sources:
        olditems = {i["item_id"]: i for i in source["items"]}
        oldfeatures = {f["feature_id"]: f for f in source["baseline_features"]}
        oldjobs = {j["job_id"]: j for j in source["judgment_jobs"]}
        oldsettings = source["config"]["judge"]
        compatible_model = model_identity(oldsettings) == model_identity(settings)
        for record in source["judgments"]:
            if not compatible_model:
                rejected["model_configuration_mismatch"] += 1
                continue
            iid, fid = record["item_id"], record["feature_id"]
            item = itemmap.get(iid)
            feature = featuremap.get(fid)
            if not item or not feature:
                rejected["not_in_canonical_scope"] += 1
                continue
            olditem = olditems.get(iid)
            oldfeature = oldfeatures.get(fid)
            if not olditem or any(
                item.get(k) != olditem.get(k)
                for k in ["content_hash", "text", "language"]
            ):
                rejected["item_identity_mismatch"] += 1
                continue
            if not oldfeature or feature["text"] != oldfeature["text"]:
                rejected["feature_meaning_or_text_mismatch"] += 1
                continue
            oldjob = oldjobs.get(record["job_id"])
            expected = cell_job(item, feature, settings)
            if (
                not oldjob
                or oldjob["system"] != expected["system"]
                or oldjob["user"] != expected["user"]
                or record.get("prompt_hash") != expected["prompt_hash"]
                or record.get("rubric_version") != rubric_for(settings)
            ):
                rejected["literal_prompt_or_rubric_mismatch"] += 1
                continue
            body = build_payload(
                oldsettings,
                system=oldjob["system"],
                user=oldjob["user"],
                schema=judgment_schema(
                    iid, fid, rubric_version=rubric_for(oldsettings)
                ),
            )
            if body != expected["payload"]:
                rejected["request_payload_mismatch"] += 1
                continue
            compatible_attempts[(expected["job_id"], record.get("attempt", 0))] = dict(
                item_id=iid,
                feature_id=fid,
                job_id=expected["job_id"],
                attempt=record.get("attempt", 0),
                source_run_id=source["manifest"]["run_id"],
                source_job_id=record["job_id"],
                source_journal_record_hash=record["record_hash"],
            )
            if record.get("parse_status") != "ok":
                rejected["invalid_prior_response"] += 1
                continue
            parsed = parse_judgment(
                record["raw_response"],
                iid,
                fid,
                rubric_version=record["rubric_version"],
            )
            if parsed["status"] != "ok":
                unresolved.append(
                    dict(
                        item_id=iid,
                        feature_id=fid,
                        job_id=expected["job_id"],
                        reason="prior_ambiguous",
                        source_run_id=source["manifest"]["run_id"],
                    )
                )
                continue
            key = iid, fid
            r = dict(
                parsed,
                item_id=iid,
                feature_id=fid,
                job_id=expected["job_id"],
                parse_status="ok",
                reuse_policy=POLICY,
                source_run_id=source["manifest"]["run_id"],
                source_manifest_sha256=source["manifest_sha256"],
                source_journal_record_hash=record["record_hash"],
                source_job_id=record["job_id"],
                prompt_hash=expected["prompt_hash"],
                payload_sha256=expected["payload_sha256"],
                judge_model=settings["model_id"],
                source_record=record,
            )
            r["reuse_hash"] = digest(r)
            if key in reused and reused[key]["applies"] != r["applies"]:
                conflicts.add(key)
            else:
                reused[key] = r
        for task in source["pending"]:
            if task["stage"] != "judgment":
                continue
            iid, fid = task.get("item_id"), task.get("feature_id")
            if iid not in itemmap or fid not in featuremap:
                continue
            j = cell_job(itemmap[iid], featuremap[fid], settings)
            if (
                task["job_id"] == j["job_id"]
                or compatible_model
                and task["body"] == j["payload"]
            ):
                pending[j["job_id"]] = dict(
                    task, source_run_id=source["manifest"]["run_id"]
                )
    for key in conflicts:
        reused.pop(key, None)
        unresolved.append(
            dict(
                item_id=key[0],
                feature_id=key[1],
                reason="conflicting_prior_binary_evidence",
            )
        )
    resolved = set(reused)
    unresolved = [
        r for r in unresolved if (r["item_id"], r["feature_id"]) not in resolved
    ]
    return dict(
        reused=[reused[k] for k in sorted(reused)],
        unresolved_prior=unresolved,
        external_pending=pending,
        rejected_counts=dict(rejected),
        policy=POLICY,
        compatible_attempts=list(compatible_attempts.values()),
    )


def previous_full_sources(root, current):
    """Discover compatible evidence in earlier full projects, never mutate them."""
    from ..engine import _read

    result = []
    root = Path(root).resolve()
    current = Path(current).resolve()
    for manifest_path in root.glob("full_*/full_manifest.json"):
        p = manifest_path.parent
        if p.resolve() == current:
            continue
        m = json.loads(manifest_path.read_text())
        if not m.get("bank_frozen"):
            continue
        names = [
            "frozen_config.json",
            "stimuli.json",
            "frozen_global_bank.json",
            "literal_binary_prompt.json",
        ]
        if not all(n in m.get("frozen_hashes", {}) and (p / n).exists() for n in names):
            continue
        for n in names:
            if file_hash(p / n) != m["frozen_hashes"][n]:
                raise ValueError("Prior full source frozen hash mismatch: " + n)
        c = json.loads((p / "frozen_config.json").read_text())
        items = json.loads((p / "stimuli.json").read_text())
        bank = json.loads((p / "frozen_global_bank.json").read_text())
        literal = json.loads((p / "literal_binary_prompt.json").read_text())["system"]
        im = {i["item_id"]: i for i in items}
        fm = {f["feature_id"]: f for f in bank["features"]}
        path = p / "applicability/judgments.jsonl"
        records = _read(path)
        if (p / "reuse_receipts.json").exists():
            for r in json.loads((p / "reuse_receipts.json").read_text()):
                if r.get("reuse_hash") != digest(
                    {k: v for k, v in r.items() if k != "reuse_hash"}
                ):
                    raise ValueError("Prior reuse receipt hash mismatch")
                records.append(r["source_record"])
        jobs = []
        for r in records:
            if r["item_id"] not in im or r["feature_id"] not in fm:
                raise ValueError("Prior full receipt outside frozen scope")
            j = cell_job(im[r["item_id"]], fm[r["feature_id"]], c["judge"])
            j.update(job_id=r["job_id"], system=literal)
            j["prompt_hash"] = digest([literal, j["user"]])
            body = build_payload(
                c["judge"],
                system=literal,
                user=j["user"],
                schema=judgment_schema(
                    r["item_id"], r["feature_id"], rubric_version=rubric_for(c["judge"])
                ),
            )
            if r.get("payload_sha256") and digest(body) != r["payload_sha256"]:
                raise ValueError("Prior full literal request reconstruction changed")
            jobs.append(j)
        pending = []
        state_path = p / "applicability/batch_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("state_hash") != digest(
                {k: v for k, v in state.items() if k != "state_hash"}
            ):
                raise ValueError("Prior full Batch hash mismatch")
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
                                body=t["body"],
                                item_id=t["record_base"].get("item_id"),
                                feature_id=t["record_base"].get("feature_id"),
                                batch_id=b.get("id"),
                                ticket=t.get("ticket"),
                                status=b["status"],
                            )
                        )
        result.append(
            dict(
                manifest=dict(run_id=m["analysis_id"], stage="global"),
                manifest_sha256=file_hash(manifest_path),
                config=c,
                items=items,
                baseline_features=bank["features"],
                judgment_jobs=jobs,
                judgments=records,
                pending=pending,
            )
        )
    return result

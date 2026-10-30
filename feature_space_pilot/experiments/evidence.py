"""Re-index existing judgments without issuing, canceling or changing any request."""

from collections import Counter, defaultdict
from ..io import digest
from ..judge import parse_judgment, rubric_for
from ..full.reuse import cell_job, model_identity
from ..full.matrix import matrix_state


def verify_receipt(record):
    if record.get("record_hash") != digest(
        {k: v for k, v in record.items() if k != "record_hash"}
    ):
        raise ValueError("Judgment receipt hash mismatch")
    if record.get("response_hash") != digest(record.get("raw_response")):
        raise ValueError("Judgment response hash mismatch")


def reconstruct_matrix(items, bank, sources, settings):
    im = {i["item_id"]: i for i in items}
    mappings = defaultdict(list)
    for m in bank["old_to_new"]:
        mappings[m["old_feature_id"]].append(m)
    accepted, rejected = [], []
    local_records = []
    cell_receipts = defaultdict(list)
    seen = set()
    for source in sources:
        olditems = {i["item_id"]: i for i in source["items"]}
        oldfeatures = {f["feature_id"]: f for f in source["features"]}
        request_cache = {}
        model_ok = model_identity(source["settings"]) == model_identity(settings)
        for r in source["records"]:
            iid, old_id = r["item_id"], r["feature_id"]
            if iid not in im or old_id not in mappings:
                continue
            verify_receipt(r)
            item, olditem, oldfeature = (
                im[iid],
                olditems.get(iid),
                oldfeatures.get(old_id),
            )
            reason = None
            if not model_ok:
                reason = "model_configuration_mismatch"
            elif olditem is None or any(
                item.get(k) != olditem.get(k)
                for k in ("text", "language", "content_hash")
            ):
                reason = "item_identity_mismatch"
            elif oldfeature is None:
                reason = "missing_source_proposition"
            elif r.get("judge_model") != settings["model_id"] or r.get(
                "rubric_version"
            ) != rubric_for(settings):
                reason = "model_or_rubric_mismatch"
            else:
                key = iid, old_id
                if key not in request_cache:
                    request_cache[key] = cell_job(item, oldfeature, settings)
                expected = request_cache[key]
                if (
                    source["literal_system"] != expected["system"]
                    or r.get("prompt_hash") != expected["prompt_hash"]
                ):
                    reason = "literal_prompt_identity_mismatch"
                elif r.get("payload_sha256") != expected["payload_sha256"]:
                    reason = "request_payload_mismatch"
            for m in mappings[old_id]:
                fid = m["feature_id"]
                unique = iid, fid, r["record_hash"]
                if unique in seen:
                    continue
                seen.add(unique)
                fail = reason
                if not fail and (
                    oldfeature["text"] != m["old_proposition"]
                    or not m["reusable_meaning"]
                ):
                    fail = "proposition_meaning_requires_review"
                link = dict(
                    item_id=iid,
                    feature_id=fid,
                    old_feature_id=old_id,
                    source_id=source["source_id"],
                    source_record_hash=r["record_hash"],
                    original_prompt_hash=r.get("prompt_hash"),
                    original_payload_sha256=r.get("payload_sha256"),
                    meaning_proof=m["proof"],
                )
                if fail:
                    rejected.append(dict(link, reason=fail))
                    continue
                status, value = "invalid", None
                if r.get("usage_unknown"):
                    status = "usage_unknown"
                elif r.get("parse_status") == "ok":
                    parsed = parse_judgment(
                        r["raw_response"],
                        iid,
                        old_id,
                        rubric_version=r["rubric_version"],
                    )
                    if any(
                        parsed[k] != r.get(k) for k in ("applies", "status", "evidence")
                    ):
                        raise ValueError(
                            "Parsed receipt differs from original raw response"
                        )
                    status, value = parsed["status"], parsed["applies"]
                link.update(
                    status=status,
                    applies=value,
                    evidence=r.get("evidence"),
                    judge_model=r.get("judge_model"),
                    rubric_version=r.get("rubric_version"),
                )
                accepted.append(link)
                cell_receipts[iid, fid].append(link)
                local_records.append(
                    dict(
                        item_id=iid,
                        feature_id=fid,
                        applies=value,
                        status=status,
                        parse_status="ok" if status in {"ok", "ambiguous"} else "error",
                    )
                )
    state = matrix_state(items, bank["features"], local_records)
    reasons = {
        (r["item_id"], r["feature_id"]): r["reason"] for r in state["unresolved"]
    }
    rejected_cells = defaultdict(set)
    for r in rejected:
        rejected_cells[r["item_id"], r["feature_id"]].add(r["reason"])
    audit = []
    for n, item in enumerate(items):
        for k, feature in enumerate(bank["features"]):
            key = item["item_id"], feature["feature_id"]
            receipts = cell_receipts.get(key, [])
            value = int(state["matrix"][n, k])
            status = "reused_binary" if value >= 0 else reasons[key]
            if status == "invalid" and any(
                r["status"] == "usage_unknown" for r in receipts
            ):
                status = "usage_unknown"
            if status == "missing" and rejected_cells[key]:
                status = "incompatible_evidence"
            audit.append(
                dict(
                    item_id=key[0],
                    feature_id=key[1],
                    applies=value if value >= 0 else "?",
                    status=status,
                    source_record_hashes=sorted(
                        {r["source_record_hash"] for r in receipts}
                    ),
                    old_feature_ids=sorted({r["old_feature_id"] for r in receipts}),
                    rejection_reasons=sorted(rejected_cells[key]),
                )
            )
    counts = Counter(r["status"] for r in audit)
    return dict(
        state=state,
        cell_audit=audit,
        reused_receipts=accepted,
        rejected_receipts=rejected,
        audit_summary=dict(
            n_items=len(items),
            n_features=len(bank["features"]),
            planned_cells=state["planned_cells"],
            reused_binary_cells=state["resolved_cells"],
            unresolved_cells=len(state["unresolved"]),
            cell_statuses=dict(counts),
            compatible_receipts=len(accepted),
            rejected_receipts=len(rejected),
            rejected_reasons=dict(Counter(r["reason"] for r in rejected)),
            api_calls=0,
            missing_is_zero=False,
            reuse_policy="Verify the original item, old proposition, literal prompt, rubric, decoding/model payload and receipt; reindex only with an explicit meaning proof. Original IDs remain in the audit.",
        ),
    )

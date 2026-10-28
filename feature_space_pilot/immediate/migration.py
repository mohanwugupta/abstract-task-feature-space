"""Collect partial Batch output during migration without resubmitting any batch."""

import json
from ..full.execution import FullSession
from ..io import append_jsonl, digest


class CachedAPI:
    def __init__(self, api, batch, files):
        self.api = api
        self.batch = batch
        self.files = files

    def retrieve(self, bid):
        return self.batch if bid == self.batch["id"] else self.api.retrieve(bid)

    def content(self, fid):
        return self.files[fid] if fid in self.files else self.api.content(fid)

    def __getattr__(self, name):
        return getattr(self.api, name)


class CollectOnlySession(FullSession):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cancel_proofs = {}

    def _submit_prepared(self, batch):
        raise ValueError(
            "Migration collector cannot submit Batch jobs; reconcile unacknowledged state first"
        )

    def _refresh_one(self, batch):
        if batch.get("ingested"):
            return
        if not batch.get("id"):
            raise ValueError("Cannot migrate unacknowledged Batch work automatically")
        api = self.api
        remote = api.retrieve(batch["id"])
        files = {}
        if remote["status"] == "cancelled":
            rows = []
            for name in ["output_file_id", "error_file_id"]:
                if remote.get(name):
                    content = api.content(remote[name])
                    files[remote[name]] = content
                    rows.extend(
                        json.loads(l) for l in content.splitlines() if l.strip()
                    )
            ids = [r.get("custom_id") for r in rows]
            if len(ids) != len(set(ids)) or not set(ids) <= set(batch["task_ids"]):
                raise ValueError("Invalid cancelled Batch output IDs")
            proven = {
                r["custom_id"]
                for r in rows
                if not r.get("response")
                and (r.get("error") or {}).get("code") == "batch_cancelled"
            }
            counts = remote.get("request_counts") or {}
            paid_or_failed_rows = [r for r in rows if r.get("custom_id") not in proven]
            if (
                all(type(counts.get(k)) is int for k in ["completed", "failed"])
                and len(paid_or_failed_rows) == counts["completed"] + counts["failed"]
            ):
                proven.update(set(batch["task_ids"]) - set(ids))
            proof = dict(
                batch_id=batch["id"],
                status="cancelled",
                request_counts=counts,
                output_hashes={k: digest(v) for k, v in files.items()},
                known_not_executed=sorted(proven),
            )
            proof["record_hash"] = digest(proof)
            append_jsonl(self.out / "cancellation_proofs.jsonl", proof)
            self.cancel_proofs.update({cid: proof["record_hash"] for cid in proven})
        self.api = CachedAPI(api, remote, files)
        try:
            super()._refresh_one(batch)
        finally:
            self.api = api

    def _ingest(self, batch, row):
        cid = row["custom_id"]
        if cid in self.cancel_proofs and not row.get("response"):
            # Zero usage is derived from terminal cancellation plus complete output
            # reconciliation. It is explicitly labeled, not a model response.
            self.state["tasks"][cid]["record_base"].update(
                known_not_executed=True,
                cancellation_proof_hash=self.cancel_proofs[cid],
                usage_provenance="zero_inference_from_reconciled_terminal_cancellation",
            )
            row = dict(
                row,
                response=dict(
                    status_code=499,
                    body=dict(
                        status="cancelled", usage=dict(input_tokens=0, output_tokens=0)
                    ),
                ),
            )
        super()._ingest(batch, row)

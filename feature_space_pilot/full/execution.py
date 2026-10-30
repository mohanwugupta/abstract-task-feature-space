"""Atomic global cells using the existing durable Batch transport and binary parser."""

from collections import Counter
from ..batch import BatchSession, BatchCreationUncertain
from ..budget import estimate_cost
from ..clients import judgment_schema
from ..engine import _read
from ..io import append_jsonl, digest, file_hash, read_jsonl
from .budget import input_bound
from .reuse import cell_job
from ..progress import emit


class FullSession(BatchSession):
    journal_name = "judgments.jsonl"

    def __init__(self, *a, **k):
        self.allowed_prior_job_ids = set(k.pop("allowed_prior_job_ids", ()))
        prior_attempts = list(k.pop("prior_attempts", ()))
        self.allowed_prior_job_ids.update(r["job_id"] for r in prior_attempts)
        super().__init__(*a, **k)
        records = _read(self.out / self.journal_name)
        self.completed = {r["job_id"] for r in records if r["parse_status"] == "ok"}
        self.owned_job_ids = {r["job_id"] for r in records}
        self.unlocated_prior_jobs = {}
        self._journals[self.journal_name] = {
            (r["job_id"], r["attempt"]): r for r in records
        }
        self.attempts = Counter(
            e["job_id"] for e in self.ledger.events if e["event"] == "reserve"
        )
        for r in records + prior_attempts:
            self.attempts[r["job_id"]] = max(
                self.attempts[r["job_id"]], r.get("attempt", 0) + 1
            )
        self.reservations = {}
        for e in self.ledger.events:
            if e["event"] == "reserve":
                self.reservations.setdefault(e["job_id"], []).append(e)
        self.active_jobs = {
            t["job_id"] for t in self.state["tasks"].values() if not t.get("recorded")
        }

    def prior_is_located(self, jid, item_id=None, feature_id=None):
        if (
            self.attempts[jid]
            and jid not in self.owned_job_ids
            and jid not in self.allowed_prior_job_ids
            and jid not in self.active_jobs
        ):
            self.unlocated_prior_jobs[jid] = dict(
                job_id=jid,
                item_id=item_id,
                feature_id=feature_id,
                reason="prior_request_receipt_not_located",
            )
            return False
        return True

    def enqueue_cell(self, item, feature, profile, *, attempt_policy="all"):
        if len(self.pending) >= self.chunk_size * self._slots():
            return False
        j = cell_job(item, feature, self.config["judge"])
        jid = j["job_id"]
        if jid in self.completed or jid in self.active_jobs:
            return True
        if not self.prior_is_located(jid, item["item_id"], feature["feature_id"]):
            return True
        attempt = self.attempts[jid]
        if (
            attempt_policy == "primary"
            and attempt > 0
            or attempt_policy == "retry"
            and attempt == 0
        ):
            return True
        if attempt >= self.config["max_attempts"]:
            return True
        counted = input_bound(item, feature, self.config["judge"], profile)
        if counted["input_token_bound"] > self.config["judge"]["max_input_tokens"]:
            raise ValueError("Global request exceeds measured input envelope")
        if (
            self._queued_tokens() + counted["input_token_bound"]
            > self.max_queued_tokens
        ):
            return False
        cid = digest([jid, attempt])
        self.pending[cid] = dict(
            custom_id=cid,
            job_id=jid,
            attempt=attempt,
            stage="judgment",
            journal_name="judgments.jsonl",
            record_base=dict(
                item_id=item["item_id"],
                feature_id=feature["feature_id"],
                rubric_version=j["rubric_version"],
                judge_model=self.config["judge"]["model_id"],
                prompt_hash=j["prompt_hash"],
                payload_sha256=j["payload_sha256"],
            ),
            schema=judgment_schema(
                item["item_id"],
                feature["feature_id"],
                rubric_version=j["rubric_version"],
            ),
            body=j["payload"],
            token_budget=counted,
            prices=self.config["judge"]["prices"],
            max_output_tokens=self.config["judge"]["max_output_tokens"],
        )
        self.active_jobs.add(jid)
        return True

    def _submit_prepared(self, batch):
        if file_hash(self.out / batch["input_name"]) != batch["input_sha256"]:
            raise ValueError("Frozen full Batch input changed")
        for cid in batch["task_ids"]:
            task = self.state["tasks"][cid]
            if task.get("ticket"):
                continue
            prior = self.reservations.get(task["job_id"], [])
            expected = estimate_cost(
                task["prices"],
                task["token_budget"]["input_token_bound"],
                task["max_output_tokens"],
            )
            if len(prior) > task["attempt"]:
                e = prior[task["attempt"]]
                if e["stage"] != task["stage"] or abs(e["amount"] - expected) > 1e-12:
                    raise ValueError("Conflicting interrupted full reservation")
                task["ticket"] = e["ticket"]
            else:
                task["ticket"] = self.ledger.reserve(
                    task["job_id"], task["stage"], expected
                )
                event = self.ledger.events[-1]
                self.reservations.setdefault(task["job_id"], []).append(event)
                self.attempts[task["job_id"]] += 1
        self._save()
        api = self._api(batch["stage"])
        uploaded = api.upload(self.out / batch["input_name"])
        batch.update(input_file_id=uploaded["id"], status="creating")
        self._save()
        try:
            created = api.create(
                uploaded["id"],
                dict(
                    pilot_batch_key=batch["key"],
                    run_id=self.out.name,
                    stage=batch["stage"],
                    full_project=self.out.parent.name,
                ),
            )
        except Exception:
            raise BatchCreationUncertain(
                "Full Batch acknowledgement uncertain; retain reservations and reconcile metadata without resubmitting"
            ) from None
        batch.update(id=created["id"], status=created["status"])
        self._save()
        emit(
            self.progress,
            "batch_submitted",
            message=f"Submitted {created['id']}: {batch['stage']}, {len(batch['task_ids'])} atomic requests",
        )

    def refresh(self):
        super().refresh()
        archive_path = self.out / "completed_batches.jsonl"
        archive = read_jsonl(archive_path) if archive_path.exists() else []
        for r in archive:
            if r.get("archive_hash") != digest(
                {k: v for k, v in r.items() if k != "archive_hash"}
            ):
                raise ValueError("Completed global Batch archive hash mismatch")
        archived = {r["key"] for r in archive}
        changed = False
        for b in self.state["batches"]:
            if not b.get("ingested"):
                continue
            if not all(
                self.state["tasks"][cid].get("recorded") for cid in b["task_ids"]
            ):
                raise ValueError("Cannot compact Batch before every receipt is durable")
            if b["key"] not in archived:
                r = {
                    k: b.get(k)
                    for k in [
                        "key",
                        "id",
                        "stage",
                        "status",
                        "input_name",
                        "input_sha256",
                        "input_file_id",
                        "request_counts",
                    ]
                }
                r["request_count"] = len(b["task_ids"])
                r["budget_tickets"] = [
                    self.state["tasks"][cid]["ticket"] for cid in b["task_ids"]
                ]
                r["provider_artifacts"] = {
                    p.name: file_hash(p)
                    for p in self.out.glob("batch_*_" + b["key"] + ".jsonl")
                }
                r["archive_hash"] = digest(r)
                append_jsonl(archive_path, r)
                archived.add(b["key"])
            for cid in b["task_ids"]:
                self.state["tasks"].pop(cid)
            changed = True
        if changed:
            self.state["batches"] = [
                b for b in self.state["batches"] if not b.get("ingested")
            ]
            self._save()
        self.completed = {
            r["job_id"]
            for r in self._journals.get(self.journal_name, {}).values()
            if r["parse_status"] == "ok"
        }
        self.owned_job_ids = {
            r["job_id"] for r in self._journals.get(self.journal_name, {}).values()
        }
        self.active_jobs = {
            t["job_id"] for t in self.state["tasks"].values() if not t.get("recorded")
        }

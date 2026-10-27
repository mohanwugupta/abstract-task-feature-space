"""Durable asynchronous OpenAI Batch queue and exact-ID result ingestion.

Every request is pre-reserved in the shared ledger. An uncertain create is never
blindly retried; resume reconciles its metadata key with the provider list.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import re
import time
import uuid
import urllib.request
import urllib.error

from .io import canonical, digest, file_hash, write_json, append_jsonl
from .clients import build_payload
from .budget import estimate_cost
from .token_accounting import validate_payload
from .prompts import parse_generation
from .judge import parse_judgment
from .progress import emit

TERMINAL = {"completed", "failed", "expired", "cancelled"}


class OpenAIBatchAPI:
    def __init__(self, settings):
        self.settings = settings
        self.endpoint = (
            "/v1/responses"
            if settings.get("api_format") == "responses"
            else "/v1/chat/completions"
        )
        if settings["base_url"].rstrip("/") != "https://api.openai.com/v1":
            raise ValueError(
                "Batch transport is verified only for the OpenAI API origin"
            )

    def _request(
        self, method, path, body=None, content_type="application/json", raw=False
    ):
        key = os.environ.get(self.settings["api_key_env"])
        if not key:
            raise ValueError("Missing configured API key for explicit Batch operation")
        request = urllib.request.Request(
            self.settings["base_url"].rstrip("/") + path,
            data=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": content_type},
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.get("timeout_seconds", 90)
            ) as response:
                data = response.read()
        except urllib.error.HTTPError as error:
            raise ValueError(
                f"Batch API HTTP {error.code}; no automatic request retry"
            ) from None
        return data.decode("utf-8") if raw else json.loads(data)

    @staticmethod
    def _id(value):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Invalid provider file/batch ID")
        return value

    def upload(self, path):
        boundary = "pilot-" + uuid.uuid4().hex
        data = Path(path).read_bytes()
        if len(data) > 200_000_000:
            raise ValueError("Batch input exceeds 200 MB")
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="pilot.jsonl"\r\nContent-Type: application/jsonl\r\n\r\n'.encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )
        return self._request(
            "POST", "/files", body, "multipart/form-data; boundary=" + boundary
        )

    def create(self, file_id, metadata):
        return self._request(
            "POST",
            "/batches",
            canonical(
                dict(
                    input_file_id=self._id(file_id),
                    endpoint=self.endpoint,
                    completion_window="24h",
                    metadata=metadata,
                )
            ).encode(),
        )

    def count(self, payload):
        return self._request(
            "POST", "/responses/input_tokens", canonical(payload).encode()
        )

    def retrieve(self, batch_id):
        return self._request("GET", "/batches/" + self._id(batch_id))

    def content(self, file_id):
        return self._request(
            "GET", "/files/" + self._id(file_id) + "/content", raw=True
        )

    def list(self):
        batches = []
        after = None
        for _ in range(100):
            page = self._request(
                "GET",
                "/batches?limit=100" + ("&after=" + self._id(after) if after else ""),
            )
            rows = page.get("data", [])
            batches.extend(rows)
            if not page.get("has_more"):
                return batches
            if not rows:
                raise ValueError("Invalid provider pagination while reconciling batch")
            after = rows[-1]["id"]
        raise ValueError(
            "Batch reconciliation exceeded list limit; inspect provider before resubmitting"
        )


class BatchStageClient:
    def __init__(self, session):
        self.session = session


class BatchSession:
    def __init__(self, config, out, ledger, *, api=None, progress=None):
        self.config = config
        self.out = Path(out)
        self.ledger = ledger
        self.progress = progress
        self.api = api
        self.apis = (
            {}
            if api
            else {
                stage: OpenAIBatchAPI(config[role])
                for stage, role in [("generation", "generator"), ("judgment", "judge")]
            }
        )
        self.path = self.out / "batch_state.json"
        self.max_in_flight = config.get("batch_max_in_flight", 1)
        self.max_queued_tokens = config.get("batch_max_queued_input_tokens", 10_000_000)
        self.chunk_size = config.get("batch_max_requests", 1000)
        if type(self.max_in_flight) is not int or not 1 <= self.max_in_flight <= 100:
            raise ValueError("Batch in-flight chunk limit must be 1–100")
        if type(self.max_queued_tokens) is not int or self.max_queued_tokens < 1:
            raise ValueError("Batch queued-input-token limit must be positive")
        if type(self.chunk_size) is not int or not 1 <= self.chunk_size <= 50_000:
            raise ValueError("Batch size must be 1–50000")
        self.pending = {}
        self._journals = {}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get("state_hash") != digest(
                {k: v for k, v in self.state.items() if k != "state_hash"}
            ):
                raise ValueError("Batch state hash mismatch")
            if self.state["config_hash"] != digest(config):
                raise ValueError("Frozen Batch configuration changed")
        else:
            self.state = dict(
                version="batch_v1", config_hash=digest(config), tasks={}, batches=[]
            )

    def _api(self, stage):
        return self.api or self.apis[stage]

    def _save(self):
        content = {k: v for k, v in self.state.items() if k != "state_hash"}
        self.state["state_hash"] = digest(content)
        write_json(self.path, self.state)

    def _slots(self):
        return self.max_in_flight - sum(
            not b.get("ingested") for b in self.state["batches"]
        )

    def _queued_tokens(self):
        active = {
            cid
            for b in self.state["batches"]
            if not b.get("ingested")
            for cid in b["task_ids"]
        }
        return sum(
            self.state["tasks"][cid]["token_budget"]["input_token_bound"]
            for cid in active
        ) + sum(t["token_budget"]["input_token_bound"] for t in self.pending.values())

    def enqueue(
        self, *, prompt, schema, settings, job_id, stage, journal_name, record_base
    ):
        if journal_name not in {"generations.jsonl", "judgments.jsonl"}:
            raise ValueError("Invalid Batch journal path")
        if len(self.pending) >= self.chunk_size * self._slots():
            return None
        if any(
            t["job_id"] == job_id and not t.get("recorded")
            for t in self.state["tasks"].values()
        ):
            return None
        attempts = sum(
            e["event"] == "reserve" and e.get("job_id") == job_id
            for e in self.ledger.events
        )
        if attempts >= self.config["max_attempts"]:
            return None
        payload = build_payload(
            settings,
            system=prompt["system"],
            user=prompt["user"],
            schema=schema,
            seed=record_base.get("seed"),
            temperature=settings.get("temperature", 0),
        )
        counted = validate_payload(payload, settings)
        if counted.get("provider_count_required"):
            from .token_accounting import provider_count_payload

            cache = self.state.setdefault("token_counts", {})
            key = counted["request_sha256"]
            if key not in cache:
                emit(
                    self.progress,
                    "token_count",
                    message=f"Counting provider input tokens for {stage} {record_base.get('item_id','')}",
                )
                cache[key] = provider_count_payload(
                    payload, settings, self._api(stage).count
                )
            counted = cache[key]
        if counted["input_token_bound"] > self.max_queued_tokens:
            raise ValueError("One request exceeds Batch queued-input-token window")
        if (
            self._queued_tokens() + counted["input_token_bound"]
            > self.max_queued_tokens
        ):
            return None
        cid = digest([job_id, attempts])
        self.pending[cid] = dict(
            custom_id=cid,
            job_id=job_id,
            attempt=attempts,
            stage=stage,
            journal_name=journal_name,
            record_base=record_base,
            schema=schema,
            body=payload,
            token_budget=counted,
            prices=settings["prices"],
            max_output_tokens=settings["max_output_tokens"],
        )
        return None

    def flush(self, stage):
        # Submit independent chunks up to both the chunk and input-token windows.
        while self._slots() > 0 and any(
            t["stage"] == stage for t in self.pending.values()
        ):
            self._flush_one(stage)

    def _flush_one(self, stage):
        queued = [t for t in self.pending.values() if t["stage"] == stage]
        if not queued:
            return
        limit = self.config.get("batch_max_requests", 1000)
        if type(limit) is not int or not 1 <= limit <= 50_000:
            raise ValueError("Batch size must be 1–50000")
        selected = []
        byte_count = 0
        for task in queued[:limit]:
            line = (
                canonical(
                    dict(
                        custom_id=task["custom_id"],
                        method="POST",
                        url=(
                            "/v1/responses"
                            if "input" in task["body"]
                            else "/v1/chat/completions"
                        ),
                        body=task["body"],
                    )
                )
                + "\n"
            )
            if selected and byte_count + len(line.encode()) > 190_000_000:
                break
            if len(line.encode()) > 190_000_000:
                raise ValueError("Individual Batch request exceeds input file limit")
            selected.append(task)
            byte_count += len(line.encode())
        amount = sum(
            estimate_cost(
                t["prices"],
                t["token_budget"]["input_token_bound"],
                t["max_output_tokens"],
            )
            for t in selected
        )
        if (
            self.ledger.total + amount > self.ledger.cap + 1e-12
            or self.ledger.stage_total(stage) + amount
            > self.ledger.stage_caps[stage] + 1e-12
        ):
            raise ValueError("Batch reservations exceed cumulative or stage budget")
        key = uuid.uuid4().hex
        filename = "batch_input_" + key + ".jsonl"
        (self.out / filename).write_text(
            "".join(
                canonical(
                    dict(
                        custom_id=t["custom_id"],
                        method="POST",
                        url=(
                            "/v1/responses"
                            if "input" in t["body"]
                            else "/v1/chat/completions"
                        ),
                        body=t["body"],
                    )
                )
                + "\n"
                for t in selected
            ),
            encoding="utf-8",
        )
        batch = dict(
            key=key,
            id=None,
            stage=stage,
            status="preparing",
            task_ids=[t["custom_id"] for t in selected],
            input_name=filename,
            input_sha256=file_hash(self.out / filename),
            input_file_id=None,
            ingested=False,
        )
        self.state["batches"].append(batch)
        for task in selected:
            self.state["tasks"][task["custom_id"]] = dict(task, ticket=None)
        self._save()
        self._submit_prepared(batch)
        for task in selected:
            self.pending.pop(task["custom_id"], None)

    def _submit_prepared(self, batch):
        if file_hash(self.out / batch["input_name"]) != batch["input_sha256"]:
            raise ValueError("Frozen Batch input file changed")
        for cid in batch["task_ids"]:
            task = self.state["tasks"][cid]
            if task.get("ticket"):
                continue
            prior = [
                e
                for e in self.ledger.events
                if e["event"] == "reserve" and e.get("job_id") == task["job_id"]
            ]
            if len(prior) > task["attempt"]:
                reservation = prior[task["attempt"]]
                expected = estimate_cost(
                    task["prices"],
                    task["token_budget"]["input_token_bound"],
                    task["max_output_tokens"],
                )
                if (
                    reservation["stage"] != task["stage"]
                    or abs(reservation["amount"] - expected) > 1e-12
                ):
                    raise ValueError("Conflicting interrupted Batch reservation")
                task["ticket"] = reservation["ticket"]
            else:
                task["ticket"] = self.ledger.reserve(
                    task["job_id"],
                    task["stage"],
                    estimate_cost(
                        task["prices"],
                        task["token_budget"]["input_token_bound"],
                        task["max_output_tokens"],
                    ),
                )
        self._save()
        api = self._api(batch["stage"])
        uploaded = api.upload(self.out / batch["input_name"])
        batch["input_file_id"] = uploaded["id"]
        batch["status"] = "creating"
        self._save()
        metadata = dict(
            pilot_batch_key=batch["key"], run_id=self.out.name, stage=batch["stage"]
        )
        try:
            created = api.create(uploaded["id"], metadata)
        except Exception:
            raise ValueError(
                "Batch creation acknowledgement uncertain; reservations retained. Resume reconciles metadata; do not manually resubmit."
            ) from None
        batch.update(id=created["id"], status=created["status"])
        self._save()
        emit(
            self.progress,
            "batch_submitted",
            message=f"Submitted {created['id']}: {batch['stage']}, {len(batch['task_ids'])} requests; async window 24h",
        )

    def refresh(self):
        for batch in self.state["batches"]:
            if batch.get("ingested"):
                continue
            api = self._api(batch["stage"])
            if not batch.get("id"):
                if batch["status"] == "preparing":
                    self._submit_prepared(batch)
                else:
                    if batch["status"] != "creating":
                        raise ValueError("Invalid interrupted Batch state")
                    matches = [
                        b
                        for b in api.list()
                        if b.get("metadata", {}).get("pilot_batch_key") == batch["key"]
                        and b.get("input_file_id") == batch["input_file_id"]
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            "Uncertain Batch creation; no unique provider match. Reservations retained; do not resubmit."
                        )
                    batch.update(id=matches[0]["id"], status=matches[0]["status"])
                    self._save()
            remote = api.retrieve(batch["id"])
            batch.update(
                status=remote["status"], request_counts=remote.get("request_counts")
            )
            self._save()
            emit(
                self.progress,
                "batch_status",
                message=f"Batch {batch['id']}: {batch['status']} {batch.get('request_counts')}",
            )
            if batch["status"] not in TERMINAL:
                continue
            rows = []
            for field in ["output_file_id", "error_file_id"]:
                if remote.get(field):
                    content = api.content(remote[field])
                    name = "batch_" + field + "_" + batch["key"] + ".jsonl"
                    (self.out / name).write_text(content, encoding="utf-8")
                    rows.extend(
                        json.loads(line)
                        for line in content.splitlines()
                        if line.strip()
                    )
            ids = [r.get("custom_id") for r in rows]
            if len(ids) != len(set(ids)) or any(
                i not in batch["task_ids"] for i in ids
            ):
                raise ValueError(
                    "Unknown or duplicate Batch custom_id/request ID; nothing settled"
                )
            if (
                batch["status"] == "failed"
                and not rows
                and (remote.get("request_counts") or {}).get("completed", 0) == 0
            ):
                rows = [
                    dict(
                        custom_id=cid,
                        error=dict(code="batch_validation_failed"),
                        response=None,
                    )
                    for cid in batch["task_ids"]
                ]
            for row in rows:
                self._ingest(batch, row)
            # Missing terminal outputs are unknown usage, never fabricated zero.
            for cid in set(batch["task_ids"]) - {r["custom_id"] for r in rows}:
                self._ingest(
                    batch,
                    dict(
                        custom_id=cid,
                        response=None,
                        error=dict(code="missing_terminal_output"),
                    ),
                )
            batch["ingested"] = True
            self._save()

    def _ingest(self, batch, row):
        from .engine import _read

        task = self.state["tasks"][row["custom_id"]]
        if task.get("recorded"):
            return
        path = self.out / task["journal_name"]
        if task["journal_name"] not in self._journals:
            self._journals[task["journal_name"]] = {
                (r["job_id"], r["attempt"]): r for r in _read(path)
            }
        cache = self._journals[task["journal_name"]]
        match = cache.get((task["job_id"], task["attempt"]))
        existing = [match] if match else []
        if existing:
            if (
                len(existing) != 1
                or existing[0].get("batch_custom_id") != row["custom_id"]
            ):
                raise ValueError("Conflicting Batch attempt journal")
            task["recorded"] = True
            return
        response = row.get("response") or {}
        body = response.get("body") or {}
        from .clients import parse_api_response

        raw, usage = parse_api_response(body)
        error = None
        parsed = None
        fatal = False
        no_execution = (row.get("error") or {}).get("code") in {
            "batch_expired",
            "batch_validation_failed",
        }
        if no_execution:
            usage.update(input_tokens=0, output_tokens=0)
        if all(
            type(usage.get(k)) is int and usage[k] >= 0
            for k in ["input_tokens", "output_tokens"]
        ):
            prior = self.ledger._balances().get(task["ticket"])
            try:
                if not prior.get("settled"):
                    self.ledger.settle(task["ticket"], usage, task["prices"])
                elif (
                    abs(
                        prior["amount"]
                        - estimate_cost(
                            task["prices"],
                            usage["input_tokens"],
                            usage["output_tokens"],
                        )
                    )
                    > 1e-12
                ):
                    raise ValueError("Conflicting previously settled Batch cost")
                if (
                    usage["input_tokens"] > task["token_budget"]["input_token_bound"]
                    or usage["output_tokens"] > task["max_output_tokens"]
                ):
                    raise ValueError(
                        "Batch usage exceeded token envelope; actual cost recorded"
                    )
            except ValueError as problem:
                error = str(problem)
                fatal = True
        else:
            error = "Batch result usage unknown; full reservation retained"
        if error is None:
            try:
                if (
                    response.get("status_code") != 200
                    or usage["finish_reason"] != "stop"
                ):
                    raise ValueError("Batch request failed or did not finish normally")
                if task["stage"] == "generation":
                    parsed = parse_generation(
                        raw,
                        task["record_base"]["item_id"],
                        cap=task["schema"]["properties"]["features"].get("maxItems"),
                    )
                else:
                    parsed = parse_judgment(
                        raw,
                        task["record_base"]["item_id"],
                        task["record_base"]["feature_id"],
                    )
            except (ValueError, KeyError, TypeError) as problem:
                error = str(problem)
        record = dict(
            task["record_base"],
            job_id=task["job_id"],
            attempt=task["attempt"],
            raw_response=raw,
            response_hash=digest(raw),
            parse_status="error" if error else "ok",
            error=error,
            usage=usage,
            latency_seconds=0,
            budget_ticket=task["ticket"],
            token_budget=task["token_budget"],
            execution_mode="batch",
            batch_id=batch["id"],
            batch_custom_id=row["custom_id"],
            **(parsed or {}),
        )
        record["record_hash"] = digest(record)
        append_jsonl(path, record)
        cache[(task["job_id"], task["attempt"])] = record
        task["recorded"] = True
        if fatal:
            self._save()
            raise ValueError(error)

    @property
    def active(self):
        return any(not b.get("ingested") for b in self.state["batches"]) or bool(
            self.pending
        )

"""Embedding-specific payloads/parser around the existing durable Batch lifecycle."""

from pathlib import Path
import uuid
import numpy as np
from ..batch import BatchSession, OpenAIBatchAPI
from ..budget import estimate_cost
from ..io import append_jsonl, canonical, digest, file_hash, read_jsonl


def read_receipts(path):
    rows = read_jsonl(path) if Path(path).exists() else []
    for row in rows:
        if row.get("record_hash") != digest(
            {k: v for k, v in row.items() if k != "record_hash"}
        ):
            raise ValueError("Embedding journal hash mismatch")
    return rows


def settings():
    return dict(
        model_id="text-embedding-3-small",
        dimensions=1536,
        inputs_per_request=64,
        max_input_tokens=16000,
        max_attempts=2,
        completion_budget_usd=0.05,
        token_margin=512,
        prices=dict(
            model_id="text-embedding-3-small",
            revision="text-embedding-3-small",
            input_per_million=0.01,
            output_per_million=0,
            verified_at="2026-10-27",
            source_url="https://developers.openai.com/api/docs/models/text-embedding-3-small",
            batch_discount_source="https://developers.openai.com/api/docs/guides/batch",
        ),
    )


def make_jobs(texts, s):
    jobs = []
    for offset in range(0, len(texts), s["inputs_per_request"]):
        chunk = texts[offset : offset + s["inputs_per_request"]]
        # UTF-8 bytes conservatively bound ordinary BPE input tokens. This is
        # not a measured tokenizer count and does not download tokenizer tables.
        counts = [len(x["text"].encode("utf-8")) for x in chunk]
        if any(n < 1 or n > 8192 for n in counts) or sum(counts) > 300_000:
            raise ValueError("Embedding input exceeds provider token limits")
        bound = sum(counts) + s["token_margin"]
        if bound > s["max_input_tokens"]:
            raise ValueError("Embedding request exceeds frozen envelope")
        job = dict(
            text_ids=[x["text_id"] for x in chunk],
            body=dict(
                model=s["model_id"],
                input=[x["text"] for x in chunk],
                encoding_format="float",
                dimensions=s["dimensions"],
            ),
            token_budget=dict(
                method="embedding_utf8_byte_upper_bound_v1",
                payload_tokens=None,
                payload_bytes=sum(counts),
                framing_margin_tokens=s["token_margin"],
                input_token_bound=bound,
            ),
        )
        job["job_id"] = "baseline_embedding_" + digest([job, s])
        jobs.append(job)
    return jobs


def embedding_api(c):
    api = OpenAIBatchAPI(c["judge"])
    api.endpoint = "/v1/embeddings"
    return api


class EmbeddingSession(BatchSession):
    def enqueue_job(self, job):
        if any(
            t["job_id"] == job["job_id"] and not t.get("recorded")
            for t in self.state["tasks"].values()
        ):
            return
        attempts = sum(
            e["event"] == "reserve" and e.get("job_id") == job["job_id"]
            for e in self.ledger.events
        )
        if attempts >= self.config["baseline_embedding"]["max_attempts"]:
            return
        if len(self.pending) >= self.chunk_size * self._slots():
            return
        cid = digest([job["job_id"], attempts])
        self.pending[cid] = dict(
            custom_id=cid,
            job_id=job["job_id"],
            attempt=attempts,
            stage="embeddings",
            journal_name="embeddings.jsonl",
            record_base=dict(text_ids=job["text_ids"]),
            body=job["body"],
            token_budget=job["token_budget"],
            prices=self.config["baseline_embedding"]["prices"],
            max_output_tokens=0,
        )

    def _flush_one(self, stage):
        tasks = [t for t in self.pending.values() if t["stage"] == stage][
            : self.chunk_size
        ]
        if not tasks:
            return
        if sum(len(t["body"]["input"]) for t in tasks) > 50_000:
            raise ValueError("Embedding Batch exceeds 50,000 input limit")
        tokens = sum(t["token_budget"]["input_token_bound"] for t in tasks)
        if (
            tokens
            + self._queued_tokens()
            - sum(t["token_budget"]["input_token_bound"] for t in self.pending.values())
            > self.max_queued_tokens
        ):
            raise ValueError("Embedding Batch exceeds queued token window")
        amount = sum(
            estimate_cost(t["prices"], t["token_budget"]["input_token_bound"], 0)
            for t in tasks
        )
        if (
            self.ledger.total + amount > self.ledger.cap + 1e-12
            or self.ledger.stage_total(stage) + amount
            > self.ledger.stage_caps[stage] + 1e-12
        ):
            raise ValueError("Embedding reservations exceed shared budget")
        key = uuid.uuid4().hex
        name = "batch_input_" + key + ".jsonl"
        payload = "".join(
            canonical(
                dict(
                    custom_id=t["custom_id"],
                    method="POST",
                    url="/v1/embeddings",
                    body=t["body"],
                )
            )
            + "\n"
            for t in tasks
        )
        if len(payload.encode()) > 190_000_000:
            raise ValueError("Embedding Batch file too large")
        (self.out / name).write_text(payload)
        batch = dict(
            key=key,
            id=None,
            stage=stage,
            status="preparing",
            task_ids=[t["custom_id"] for t in tasks],
            input_name=name,
            input_sha256=file_hash(self.out / name),
            input_file_id=None,
            ingested=False,
        )
        self.state["batches"].append(batch)
        for t in tasks:
            self.state["tasks"][t["custom_id"]] = dict(t, ticket=None)
        self._save()
        self._submit_prepared(batch)
        for t in tasks:
            self.pending.pop(t["custom_id"], None)

    def _ingest(self, batch, row):
        task = self.state["tasks"][row["custom_id"]]
        if task.get("recorded"):
            return
        path = self.out / "embeddings.jsonl"
        cache = self._journals.setdefault("embeddings.jsonl", None)
        if cache is None:
            cache = {(r["job_id"], r["attempt"]): r for r in read_receipts(path)}
            self._journals["embeddings.jsonl"] = cache
        prior_record = cache.get((task["job_id"], task["attempt"]))
        if prior_record:
            if prior_record["batch_custom_id"] != row["custom_id"]:
                raise ValueError("Conflicting embedding receipt")
            task["recorded"] = True
            return
        response = row.get("response") or {}
        body = response.get("body") or {}
        no_execution = (row.get("error") or {}).get("code") in {
            "batch_expired",
            "batch_validation_failed",
        }
        usage = dict(
            input_tokens=(
                0 if no_execution else (body.get("usage") or {}).get("prompt_tokens")
            ),
            output_tokens=0,
        )
        error = None
        fatal = False
        vectors = None
        if type(usage["input_tokens"]) is not int or usage["input_tokens"] < 0:
            error = "Embedding usage unknown; full reservation retained"
        else:
            prior = self.ledger._balances().get(task["ticket"])
            if not prior:
                raise ValueError("Missing embedding reservation")
            if not prior.get("settled"):
                self.ledger.settle(task["ticket"], usage, task["prices"])
            elif (
                abs(
                    prior["amount"]
                    - estimate_cost(task["prices"], usage["input_tokens"], 0)
                )
                > 1e-12
            ):
                raise ValueError("Conflicting embedding settled usage")
            if usage["input_tokens"] > task["token_budget"]["input_token_bound"]:
                error = "Embedding token envelope exceeded; actual cost recorded"
                fatal = True
        if error is None:
            try:
                if (
                    response.get("status_code") != 200
                    or body.get("model") != task["body"]["model"]
                ):
                    raise ValueError("Embedding request failed or model differs")
                data = body.get("data")
                n = len(task["body"]["input"])
                if not isinstance(data, list) or len(data) != n:
                    raise ValueError("Missing embedding vectors")
                indices = [x.get("index") for x in data]
                if any(type(i) is not int for i in indices) or set(indices) != set(
                    range(n)
                ):
                    raise ValueError("Unknown or duplicate embedding index")
                vectors = np.asarray(
                    [x["embedding"] for x in sorted(data, key=lambda x: x["index"])],
                    dtype=np.float32,
                )
                if (
                    vectors.shape != (n, task["body"]["dimensions"])
                    or not np.isfinite(vectors).all()
                    or np.any(np.linalg.norm(vectors, axis=1) == 0)
                ):
                    raise ValueError("Invalid embedding vector values/dimensions")
            except (ValueError, KeyError, TypeError) as problem:
                error = str(problem)
        vector_name = None
        if error is None:
            vector_name = "vectors_" + row["custom_id"] + ".npy"
            np.save(self.out / vector_name, vectors, allow_pickle=False)
        record = dict(
            task["record_base"],
            job_id=task["job_id"],
            attempt=task["attempt"],
            parse_status="error" if error else "ok",
            error=error,
            usage=usage,
            response_model=body.get("model"),
            response_hash=digest(body),
            budget_ticket=task["ticket"],
            token_budget=task["token_budget"],
            batch_id=batch["id"],
            batch_custom_id=row["custom_id"],
            vector_file=vector_name,
            vector_sha256=file_hash(self.out / vector_name) if vector_name else None,
        )
        record["record_hash"] = digest(record)
        append_jsonl(path, record)
        cache[task["job_id"], task["attempt"]] = record
        task["recorded"] = True
        if fatal:
            self._save()
            raise ValueError(error)


def load_vectors(out, jobs):
    out = Path(out)
    expected = {j["job_id"]: j for j in jobs}
    vectors = {}
    done = set()
    for record in read_receipts(out / "embeddings.jsonl"):
        if record.get("parse_status") != "ok":
            continue
        job = expected.get(record["job_id"])
        if (
            job is None
            or record["job_id"] in done
            or record["text_ids"] != job["text_ids"]
        ):
            raise ValueError("Conflicting embedding output inventory")
        if record["record_hash"] != digest(
            {k: v for k, v in record.items() if k != "record_hash"}
        ):
            raise ValueError("Embedding journal hash mismatch")
        path = out / record["vector_file"]
        if path.parent != out or file_hash(path) != record["vector_sha256"]:
            raise ValueError("Embedding vector file changed")
        a = np.load(path, allow_pickle=False)
        if (
            a.shape != (len(job["text_ids"]), job["body"]["dimensions"])
            or not np.isfinite(a).all()
        ):
            raise ValueError("Invalid saved embedding shape/values")
        for tid, v in zip(job["text_ids"], a):
            vectors[tid] = v
        done.add(record["job_id"])
    return vectors

"""Reuse durable Batch submission/reconciliation with a semantic-response parser."""

from ..batch import BatchSession
from ..budget import estimate_cost
from ..clients import parse_api_response
from ..engine import _read
from ..io import append_jsonl, digest, canonical
from .protocol import parse_response, payload_for_job, response_schema, RUBRIC


class MatchingBatchSession(BatchSession):
    def _api(self, stage):
        return self.api or self.apis["judgment"]

    def enqueue_job(self, job):
        return self.enqueue(
            prompt=dict(
                system=RUBRIC,
                user=canonical(payload_for_job(job)),
            ),
            schema=response_schema(job),
            settings=self.config["judge"],
            job_id=job["job_id"],
            stage="adjudication",
            journal_name="judgments.jsonl",
            record_base=dict(
                item_id=job["item_id"],
                feature_id=job["job_id"],
                is_control=job.get("is_control", False),
                semantic_job=job,
                grader_model=self.config["judge"]["model_id"],
            ),
        )

    def _ingest(self, batch, row):
        task = self.state["tasks"][row["custom_id"]]
        if task.get("recorded"):
            return
        path = self.out / task["journal_name"]
        if task["journal_name"] not in self._journals:
            self._journals[task["journal_name"]] = {
                (r["job_id"], r["attempt"]): r for r in _read(path)
            }
        cache = self._journals[task["journal_name"]]
        prior_record = cache.get((task["job_id"], task["attempt"]))
        if prior_record:
            if prior_record.get("batch_custom_id") != row["custom_id"]:
                raise ValueError("Conflicting semantic attempt receipt")
            task["recorded"] = True
            return
        response = row.get("response") or {}
        raw, usage = parse_api_response(response.get("body") or {})
        error = None
        parsed = None
        fatal = False
        if (row.get("error") or {}).get("code") in {
            "batch_expired",
            "batch_validation_failed",
        }:
            usage.update(input_tokens=0, output_tokens=0)
        if all(
            type(usage.get(k)) is int and usage[k] >= 0
            for k in ["input_tokens", "output_tokens"]
        ):
            prior = self.ledger._balances().get(task["ticket"])
            try:
                if prior is None:
                    raise ValueError("Missing semantic budget reservation")
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
                    raise ValueError("Conflicting settled semantic cost")
                if (
                    usage["input_tokens"] > task["token_budget"]["input_token_bound"]
                    or usage["output_tokens"] > task["max_output_tokens"]
                ):
                    raise ValueError(
                        "Semantic usage exceeded frozen envelope; actual cost recorded"
                    )
            except ValueError as problem:
                error = str(problem)
                fatal = True
        else:
            error = "Semantic usage unknown; full reservation retained"
        if error is None:
            try:
                if (
                    response.get("status_code") != 200
                    or usage.get("finish_reason") != "stop"
                ):
                    raise ValueError("Semantic request failed or truncated")
                parsed = parse_response(raw, task["record_base"]["semantic_job"])
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
            budget_ticket=task["ticket"],
            token_budget=task["token_budget"],
            execution_mode="batch",
            batch_id=batch["id"],
            batch_custom_id=row["custom_id"],
            human_reviewed=False,
            **(parsed or {})
        )
        record["record_hash"] = digest(record)
        append_jsonl(path, record)
        cache[(task["job_id"], task["attempt"])] = record
        task["recorded"] = True
        if fatal:
            self._save()
            raise ValueError(error)

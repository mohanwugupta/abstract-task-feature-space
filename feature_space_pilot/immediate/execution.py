"""One atomic request per cell; concurrent transport, serialized money and receipts."""

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..budget import estimate_cost
from ..clients import parse_api_response
from ..engine import _read
from ..io import append_jsonl, canonical, digest, read_jsonl
from ..judge import parse_judgment


class AdmissionRejected(Exception):
    def __init__(self, code, retry_after=1.0, fatal=False):
        self.code = code
        self.retry_after = max(0.0, retry_after)
        self.fatal = fatal


class StandardTransport:
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, body):
        s = self.settings
        request = urllib.request.Request(
            s["base_url"].rstrip("/") + "/responses",
            data=canonical(body).encode(),
            headers={
                "Authorization": "Bearer " + os.environ[s["api_key_env"]],
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=s.get("timeout_seconds", 90)
            ) as response:
                return json.load(response), {
                    k.lower(): v
                    for k, v in response.headers.items()
                    if k.lower().startswith("x-ratelimit")
                }
        except urllib.error.HTTPError as error:
            if error.code in {400, 401, 403, 404, 422, 429}:
                try:
                    code = json.loads(error.read()).get("error", {}).get("code") or str(
                        error.code
                    )
                except (ValueError, TypeError):
                    code = str(error.code)
                try:
                    delay = float(error.headers.get("Retry-After", 1))
                except (ValueError, TypeError):
                    delay = 1.0
                raise AdmissionRejected(
                    code,
                    delay,
                    fatal=error.code != 429
                    or code in {"insufficient_quota", "billing_hard_limit_reached"},
                ) from None
            raise


class Pace:
    def __init__(self, max_rps=50.0, initial_rps=5.0):
        self.maximum = max_rps
        self.rate = min(max_rps, initial_rps)
        self.ceiling = max_rps
        self.next = 0.0
        self.last_growth = time.monotonic()
        self.lock = threading.Lock()
        self.max_tokens_per_call = 700

    def wait(self):
        with self.lock:
            now = time.monotonic()
            interval = (
                900 if self.rate * self.max_tokens_per_call * 60 >= 1000000 else 60
            )
            if now - self.last_growth >= interval:
                self.rate = min(self.ceiling, self.rate * 1.5)
                self.last_growth = now
            when = max(now, self.next)
            self.next = when + 1 / max(0.1, self.rate)
        if when > now:
            time.sleep(when - now)

    def observe(self, headers, input_bound):
        with self.lock:
            self.max_tokens_per_call = max(self.max_tokens_per_call, input_bound + 400)
            cap = self.maximum
            for name, divisor in [
                ("x-ratelimit-limit-requests", 60),
                ("x-ratelimit-limit-tokens", 60 * self.max_tokens_per_call),
            ]:
                try:
                    cap = min(cap, float(headers[name]) * 0.85 / divisor)
                except (KeyError, ValueError, TypeError):
                    pass
            self.ceiling = max(0.1, cap)
            self.rate = min(self.rate, self.ceiling)

    def backoff(self, seconds):
        with self.lock:
            self.rate = max(0.1, self.rate * 0.5)
            self.next = max(self.next, time.monotonic() + seconds)
            self.last_growth = time.monotonic()


class ImmediateExecutor:
    def __init__(
        self,
        config,
        out,
        ledger,
        *,
        transport=None,
        workers=128,
        max_rps=50.0,
        allowed_prior_job_ids=()
    ):
        self.config = config
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger
        self.transport = transport or StandardTransport(config["judge"])
        self.workers = workers
        self.pace = Pace(max_rps)
        self.rows = _read(out / "judgments.jsonl")
        self.dispatches = (
            read_jsonl(out / "dispatches.jsonl")
            if (out / "dispatches.jsonl").exists()
            else []
        )
        if any(
            r.get("record_hash")
            != digest({k: v for k, v in r.items() if k != "record_hash"})
            for r in self.dispatches
        ):
            raise ValueError("Immediate dispatch record hash mismatch")
        self.finished = {(r["job_id"], r["attempt"]) for r in self.rows}
        received = {r["budget_ticket"] for r in self.rows}
        self.held_job_ids = {
            r["job_id"] for r in self.dispatches if r["ticket"] not in received
        }
        self.held_job_ids.update(
            r["job_id"] for r in self.rows if r.get("usage_unknown")
        )
        located = {r["job_id"] for r in self.rows + self.dispatches} | set(
            allowed_prior_job_ids
        )
        self.held_job_ids.update(
            e["job_id"]
            for e in ledger.events
            if e["event"] == "reserve" and e["job_id"] not in located
        )
        self.audit_lock = threading.Lock()
        self.fatal_error = None

    def _request(self, task):
        started = time.monotonic()
        for n in range(5):
            self.pace.wait()
            try:
                response, headers = self.transport(task["job"]["payload"])
                self.pace.observe(headers, task["input_token_bound"])
                return response, None, False, time.monotonic() - started
            except AdmissionRejected as error:
                rejection = dict(
                    dispatch_id=task["dispatch_id"],
                    job_id=task["job"]["job_id"],
                    network_attempt=n,
                    http_rejection_code=error.code,
                    known_not_executed=True,
                )
                rejection["record_hash"] = digest(rejection)
                with self.audit_lock:
                    append_jsonl(self.out / "transport_rejections.jsonl", rejection)
                if error.fatal or n == 4:
                    return None, error, True, time.monotonic() - started
                self.pace.backoff(error.retry_after + random.uniform(0, 0.25))
            except Exception as error:
                return None, type(error).__name__, False, time.monotonic() - started

    def wave(self, tasks):
        unique = {}
        for t in tasks:
            if (
                type(t["attempt"]) is not int
                or not 0 <= t["attempt"] < self.config["max_attempts"]
            ):
                raise ValueError("Immediate attempt exceeds frozen maximum")
            key = t["job"]["job_id"], t["attempt"]
            if key not in self.finished and t["job"]["job_id"] not in self.held_job_ids:
                unique[key] = dict(t)
        tasks = list(unique.values())
        prices = self.config["judge"]["prices"]
        limit = self.config["judge"]["max_output_tokens"]
        amount = sum(
            estimate_cost(prices, t["input_token_bound"], limit) for t in tasks
        )
        if (
            self.ledger.total + amount > self.ledger.cap + 1e-12
            or self.ledger.stage_total("judgment") + amount
            > self.ledger.stage_caps["judgment"] + 1e-12
        ):
            raise ValueError("Immediate wave exceeds cumulative budget")
        for task in tasks:
            j = task["job"]
            task["ticket"] = self.ledger.reserve(
                j["job_id"],
                "judgment",
                estimate_cost(prices, task["input_token_bound"], limit),
            )
            dispatch = dict(
                job_id=j["job_id"],
                attempt=task["attempt"],
                item_id=j["item_id"],
                feature_id=j["feature_id"],
                ticket=task["ticket"],
                payload_sha256=j["payload_sha256"],
                input_token_bound=task["input_token_bound"],
                execution_mode="online",
                dispatched_at=time.time(),
            )
            dispatch["record_hash"] = digest(dispatch)
            task["dispatch_id"] = dispatch["record_hash"]
            append_jsonl(self.out / "dispatches.jsonl", dispatch)
        result = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._request, t): t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                j = task["job"]
                response, error, not_executed, latency = future.result()
                raw, usage = parse_api_response(response or {})
                if not_executed:
                    usage.update(input_tokens=0, output_tokens=0)
                known = all(
                    type(usage.get(k)) is int and usage[k] >= 0
                    for k in ["input_tokens", "output_tokens"]
                )
                parsed = None
                message = None
                fatal = False
                if known:
                    try:
                        self.ledger.settle(task["ticket"], usage, prices)
                    except ValueError as problem:
                        message = str(problem)
                        fatal = True
                    if (
                        usage["input_tokens"] > task["input_token_bound"]
                        or usage["output_tokens"] > limit
                    ):
                        message = "Provider usage exceeded immediate envelope; actual cost recorded"
                        fatal = True
                else:
                    message = (
                        "Immediate usage unknown; reservation retained and request held"
                    )
                if error:
                    message = (
                        "Provider rejected admission"
                        if not_executed
                        else "Provider acknowledgement unknown: " + str(error)
                    )
                    if isinstance(error, AdmissionRejected) and error.fatal:
                        fatal = True
                if message is None:
                    try:
                        if usage.get("finish_reason") != "stop":
                            raise ValueError("Immediate response failed or truncated")
                        parsed = parse_judgment(
                            raw,
                            j["item_id"],
                            j["feature_id"],
                            rubric_version=j["rubric_version"],
                        )
                    except (ValueError, TypeError, KeyError) as problem:
                        message = str(problem)
                row = dict(
                    item_id=j["item_id"],
                    feature_id=j["feature_id"],
                    job_id=j["job_id"],
                    attempt=task["attempt"],
                    rubric_version=j["rubric_version"],
                    judge_model=self.config["judge"]["model_id"],
                    prompt_hash=j["prompt_hash"],
                    payload_sha256=j["payload_sha256"],
                    raw_response=raw,
                    response_hash=digest(raw),
                    parse_status="error" if message else "ok",
                    error=message,
                    usage=usage,
                    usage_unknown=not known,
                    known_not_executed=not_executed,
                    budget_ticket=task["ticket"],
                    token_budget=dict(input_token_bound=task["input_token_bound"]),
                    execution_mode="online",
                    latency_seconds=latency,
                )
                row.update(parsed or {})
                row["record_hash"] = digest(row)
                append_jsonl(self.out / "judgments.jsonl", row)
                self.rows.append(row)
                result.append(row)
                self.finished.add((j["job_id"], task["attempt"]))
                if not known:
                    self.held_job_ids.add(j["job_id"])
                if fatal:
                    self.fatal_error = message
        return result

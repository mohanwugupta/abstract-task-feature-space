"""Durable cumulative USD ledger. Unknown usage retains the full reservation."""

from __future__ import annotations
import fcntl
import math
import uuid
from pathlib import Path
from .io import append_jsonl, read_jsonl


def validate_prices(prices):
    for k in [
        "model_id",
        "revision",
        "input_per_million",
        "output_per_million",
        "verified_at",
        "source_url",
    ]:
        if k not in prices or prices[k] is None or prices[k] == "":
            raise ValueError(f"Missing verified model price field: {k}")
    for k in ["input_per_million", "output_per_million"]:
        if (
            type(prices[k]) not in (int, float)
            or not math.isfinite(prices[k])
            or prices[k] < 0
        ):
            raise ValueError("Invalid model price")
    return prices


def estimate_cost(prices, input_tokens, output_tokens):
    validate_prices(prices)
    return (
        prices["input_per_million"] * input_tokens
        + prices["output_per_million"] * output_tokens
    ) / 1_000_000


class Ledger:
    def __init__(self, path, *, cap=500, stage_caps=None):
        if not math.isfinite(cap) or not 0 < cap <= 500:
            raise ValueError("Pilot cap must be >0 and <=500 USD")
        self.path = Path(path)
        self.cap = cap
        self.stage_caps = stage_caps or {
            "generation": 120,
            "judgment": 220,
            "adjudication": 80,
            "embeddings": 30,
            "reserve": 50,
        }
        if any(not math.isfinite(v) or v < 0 for v in self.stage_caps.values()):
            raise ValueError("Invalid stage budget")
        if sum(self.stage_caps.values()) > cap + 1e-12:
            raise ValueError("Stage budget allocations exceed cumulative cap")
        self.events = []
        self.lock = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("Another process owns the cumulative budget ledger")
        try:
            self.events = read_jsonl(self.path) if self.path.exists() else []
            self._balances()  # validate before spending
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        if self.lock:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None

    def _balances(self):
        tickets = {}
        for e in self.events:
            if e["event"] == "reserve":
                if e["ticket"] in tickets:
                    raise ValueError("Duplicate ledger reservation")
                tickets[e["ticket"]] = dict(e)
            elif e["event"] in {"settle", "overage"}:
                if e["ticket"] not in tickets:
                    raise ValueError("Unknown ledger reservation")
                tickets[e["ticket"]]["amount"] = e["amount"]
                tickets[e["ticket"]]["settled"] = True
            else:
                raise ValueError("Corrupt budget ledger event")
        return tickets

    @property
    def total(self):
        return sum(t["amount"] for t in self._balances().values())

    def stage_total(self, stage):
        return sum(
            t["amount"] for t in self._balances().values() if t["stage"] == stage
        )

    def _append(self, event):
        if self.lock is None:
            raise ValueError("Ledger must be locked")
        append_jsonl(self.path, event)
        self.events.append(event)

    def reserve(self, job_id, stage, amount):
        if stage not in self.stage_caps or not math.isfinite(amount) or amount < 0:
            raise ValueError("Invalid budget reservation")
        if (
            self.total + amount > self.cap + 1e-12
            or self.stage_total(stage) + amount > self.stage_caps[stage] + 1e-12
        ):
            raise ValueError("Cumulative or stage budget exceeded")
        ticket = uuid.uuid4().hex
        self._append(
            dict(
                event="reserve",
                ticket=ticket,
                job_id=job_id,
                stage=stage,
                amount=amount,
            )
        )
        return ticket

    def settle(self, ticket, usage, prices):
        prior = self._balances().get(ticket)
        if not prior or prior.get("settled"):
            raise ValueError("Unknown or settled reservation")
        if any(
            type(usage.get(k)) is not int or usage[k] < 0
            for k in ["input_tokens", "output_tokens"]
        ):
            raise ValueError(
                "Missing/invalid provider token usage; full reservation retained"
            )
        amount = estimate_cost(prices, usage["input_tokens"], usage["output_tokens"])
        over = amount > prior["amount"] + 1e-12
        self._append(
            dict(
                event="overage" if over else "settle",
                ticket=ticket,
                amount=amount,
                usage=usage,
                model_id=prices["model_id"],
            )
        )
        if over:
            raise ValueError(
                "Provider usage exceeded reservation; actual cost recorded, execution halted"
            )
        return amount

    def report(self):
        tickets = self._balances()
        return dict(
            accounted_usd=self.total,
            remaining_usd=self.cap - self.total,
            unsettled_reservations=sum(not t.get("settled") for t in tickets.values()),
            stages={s: self.stage_total(s) for s in self.stage_caps},
            input_tokens=sum(
                e.get("usage", {}).get("input_tokens", 0) for e in self.events
            ),
            output_tokens=sum(
                e.get("usage", {}).get("output_tokens", 0) for e in self.events
            ),
        )

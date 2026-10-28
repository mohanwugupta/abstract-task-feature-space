"""Measured template bounds and scalable access to the unchanged cumulative ledger."""

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from ..budget import Ledger, estimate_cost
from ..clients import build_payload, judgment_schema
from ..io import canonical, digest
from ..judge import judge_prompt
from .reuse import model_identity

PROFILE_VERSION = "atomic_provider_template_v1"


class IndexedLedger(Ledger):
    """Same durable events/guards, cached totals while this process owns the lock."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._index = None
        self._indexed_count = -1

    def __enter__(self):
        self._index = None
        self._indexed_count = -1
        return super().__enter__()

    def _balances(self):
        if self._index is None or self._indexed_count != len(self.events):
            self._index = Ledger._balances(self)
            self._indexed_count = len(self.events)
            self._total = sum(x["amount"] for x in self._index.values())
            self._stages = Counter()
            for x in self._index.values():
                self._stages[x["stage"]] += x["amount"]
        return self._index

    @property
    def total(self):
        self._balances()
        return self._total

    def stage_total(self, stage):
        self._balances()
        return self._stages[stage]

    def _append(self, event):
        index = self._balances()
        kind = event["event"]
        ticket = event["ticket"]
        if kind == "reserve":
            if ticket in index:
                raise ValueError("Duplicate reservation")
            stage = event["stage"]
            delta = event["amount"]
        else:
            if ticket not in index:
                raise ValueError("Unknown reservation")
            stage = index[ticket]["stage"]
            delta = event["amount"] - index[ticket]["amount"]
        super()._append(event)
        if kind == "reserve":
            index[ticket] = dict(event)
        else:
            index[ticket].update(amount=event["amount"], settled=True)
        self._indexed_count = len(self.events)
        self._total += delta
        self._stages[stage] += delta


def template_payload(item, settings):
    feature = dict(feature_id="f_" + "0" * 20, text="")
    p = judge_prompt(item, feature)
    return build_payload(
        settings,
        system=p["system"],
        user=p["user"],
        schema=judgment_schema(item["item_id"], feature["feature_id"]),
    )


def profile_inputs(items, settings, transport, *, workers=6):
    def count(item):
        payload = template_payload(item, settings)
        request = {
            k: v
            for k, v in payload.items()
            if k in {"model", "input", "text", "reasoning", "instructions"}
        }
        result = transport(request)
        n = result.get("input_tokens")
        if type(n) is not int or n < 0:
            raise ValueError("Invalid provider template input count")
        return item["item_id"], dict(
            input_tokens=n,
            baseline_payload_sha256=digest(payload),
            item_content_hash=item["content_hash"],
        )

    with ThreadPoolExecutor(max_workers=min(8, max(1, workers))) as pool:
        counts = dict(pool.map(count, items))
    profile = dict(
        version=PROFILE_VERSION,
        model_identity=model_identity(settings),
        items=counts,
        method="Provider counts for actual item/rubric/schema with empty proposition; per-feature byte additions bounded separately",
        endpoint="/v1/responses/input_tokens",
    )
    profile["profile_hash"] = digest(profile)
    return profile


def validate_profile(profile, settings):
    if (
        profile.get("profile_hash")
        != digest({k: v for k, v in profile.items() if k != "profile_hash"})
        or profile.get("version") != PROFILE_VERSION
        or profile.get("model_identity") != model_identity(settings)
    ):
        raise ValueError("Provider input profile changed or model mismatch")


def _bound(item, feature, settings, entry):
    extra = (
        len(canonical(feature["text"]).encode())
        + 2 * len(feature["feature_id"].encode())
        + 128
    )
    if entry is not None:
        baseline = entry["input_tokens"]
        method = "provider_template_plus_feature_byte_bound_v1"
    else:
        baseline = len(canonical(template_payload(item, settings)).encode())
        method = "offline_utf8_payload_upper_bound_not_measured_tokens"
    return dict(
        method=method,
        input_token_bound=baseline + extra,
        measured_template_input_tokens=entry["input_tokens"] if entry else None,
        per_feature_addition_bound=extra,
        exact_provider_count=False,
        planning_basis=(
            "Measured fixed template plus escaped feature/ID byte upper bound and 128-token boundary reserve"
            if entry
            else "UTF-8 byte upper bound, not a measured model token count"
        ),
    )


def input_bound(item, feature, settings, profile):
    entry = None
    if profile is not None:
        validate_profile(profile, settings)
        entry = profile["items"].get(item["item_id"])
        if (
            not entry
            or entry["item_content_hash"] != item["content_hash"]
            or entry["baseline_payload_sha256"]
            != digest(template_payload(item, settings))
        ):
            raise ValueError("Provider input profile item/prompt mismatch")
    return _bound(item, feature, settings, entry)


def cost_plan(
    items,
    features,
    c,
    ledger,
    reuse,
    *,
    profile=None,
    generations_complete=False,
    current_records=(),
    active_tasks=(),
):
    s = c["judge"]
    prices = s["prices"]
    attempt_limit = c.get("max_attempts", 2)
    if profile is not None:
        validate_profile(profile, s)
    resolved = {(r["item_id"], r["feature_id"]) for r in reuse["reused"]}
    resolved.update(
        (r["item_id"], r["feature_id"])
        for r in current_records
        if r.get("parse_status") == "ok" and r.get("status") == "ok"
    )
    prior_unresolved = {
        (r["item_id"], r["feature_id"]) for r in reuse["unresolved_prior"]
    }
    prior_unresolved.update(
        (r["item_id"], r["feature_id"])
        for r in current_records
        if r.get("parse_status") == "ok" and r.get("status") == "ambiguous"
    )
    pending = {
        (r.get("item_id"), r.get("feature_id"))
        for r in reuse["external_pending"].values()
    }
    pending.update(
        (t["record_base"]["item_id"], t["record_base"]["feature_id"])
        for t in active_tasks
        if not t.get("recorded")
    )
    pending -= resolved | prior_unresolved
    used = Counter()
    for r in list(reuse.get("compatible_attempts", [])) + list(current_records):
        used[r["item_id"], r["feature_id"]] = max(
            used[r["item_id"], r["feature_id"]], r.get("attempt", 0) + 1
        )
    for t in active_tasks:
        if t.get("ticket"):
            used[t["record_base"]["item_id"], t["record_base"]["feature_id"]] = max(
                used[t["record_base"]["item_id"], t["record_base"]["feature_id"]],
                t.get("attempt", 0) + 1,
            )
    total_input = 0
    remaining_cost = 0.0
    primary_cost = 0.0
    retry_cost = 0.0
    retry_calls = 0
    new_calls = attempts = exhausted = oversized = 0
    min_bound = None
    max_bound = 0
    byitem = []
    for item in items:
        bounds = (
            [input_bound(item, f, s, profile)["input_token_bound"] for f in features]
            if profile is None
            else []
        )
        if profile is not None:
            # Validate a fixed template once per item; feature additions are explicit upper bounds.
            if features:
                input_bound(item, features[0], s, profile)
            entry = profile["items"][item["item_id"]]
            bounds = [_bound(item, f, s, entry)["input_token_bound"] for f in features]
        item_calls = 0
        item_cost = 0.0
        for feature, bound in zip(features, bounds):
            total_input += bound
            max_bound = max(max_bound, bound)
            min_bound = bound if min_bound is None else min(min_bound, bound)
            if bound > s["max_input_tokens"]:
                oversized += 1
            key = item["item_id"], feature["feature_id"]
            if key in resolved or key in prior_unresolved:
                continue
            count = max(0, attempt_limit - max(used[key], int(key in pending)))
            price = estimate_cost(prices, bound, s["max_output_tokens"])
            if not count and key not in pending:
                exhausted += 1
            if key not in pending and count and used[key] == 0:
                new_calls += 1
                primary_cost += price
                item_calls += 1
            elif key not in pending and count and used[key] > 0:
                retry_calls += 1
                retry_cost += count * price
            attempts += count
            remaining_cost += count * price
            item_cost += count * price
        byitem.append(
            dict(
                item_id=item["item_id"],
                new_primary_calls=item_calls,
                remaining_upper_bound_usd=item_cost,
            )
        )
    blockers = []
    if not generations_complete:
        blockers.append(
            "All source generation jobs must be complete before freezing the global bank"
        )
    if profile is None:
        blockers.append(
            "Measured provider input profile is required before paid global judgments"
        )
    if not features:
        blockers.append("Complete feature bank is empty")
    if oversized:
        blockers.append(
            f"{oversized} actual pair bounds exceed the frozen input ceiling"
        )
    retry_policy = c.get("full_options", {}).get(
        "retry_policy", "complete_retry_envelope"
    )
    if retry_policy == "staged_primary_then_costed_invalid_retries":
        authorized_stage = (
            "primary_pass"
            if new_calls
            else (
                "collect_pending"
                if pending
                else "invalid_retries" if retry_calls else "review_only"
            )
        )
        authorized_cost = primary_cost if new_calls else 0.0 if pending else retry_cost
    elif retry_policy == "complete_retry_envelope":
        authorized_stage = "all_configured_attempts"
        authorized_cost = remaining_cost
    else:
        raise ValueError("Unknown full-pilot retry funding policy")
    if ledger.total + authorized_cost > ledger.cap + 1e-12:
        blockers.append(
            "Authorized full-scope stage exceeds the cumulative $100 budget"
        )
    if (
        ledger.stage_total("judgment") + authorized_cost
        > ledger.stage_caps["judgment"] + 1e-12
    ):
        blockers.append(
            "Authorized full-scope stage exceeds the judgment-stage allocation"
        )
    return dict(
        required_cells=len(items) * len(features),
        eligible_items=len(items),
        consolidated_features=len(features),
        final_consolidated_features=len(features) if generations_complete else None,
        bank_provisional=not generations_complete,
        reused_resolved_cells=len(resolved),
        prior_unresolved_cells=len(prior_unresolved),
        external_pending_cells=len(pending),
        retry_exhausted_cells=exhausted,
        new_primary_calls=new_calls,
        known_invalid_retry_calls=retry_calls,
        known_invalid_retry_upper_bound_usd=retry_cost,
        retry_policy=retry_policy,
        authorized_stage=authorized_stage,
        authorized_stage_upper_bound_usd=authorized_cost,
        fully_funds_all_configured_attempts=(
            ledger.total + remaining_cost <= ledger.cap + 1e-12
            and ledger.stage_total("judgment") + remaining_cost
            <= ledger.stage_caps["judgment"] + 1e-12
        ),
        unsubmitted_primary_funding_protected=True,
        remaining_attempts_upper_bound=attempts,
        output_tokens_per_attempt=s["max_output_tokens"],
        input_bound_range=[min_bound, max_bound],
        serialized_input_bound_total=total_input,
        input_counting=(
            "provider_template_plus_feature_byte_bound_v1"
            if profile
            else "offline_byte_upper_bound_not_token_measurement"
        ),
        first_attempt_upper_bound_usd=primary_cost,
        remaining_completion_upper_bound_usd=remaining_cost,
        cumulative_accounted_usd=ledger.total,
        cumulative_completion_upper_bound_usd=ledger.total + remaining_cost,
        stage_accounted_usd=ledger.stage_total("judgment"),
        stage_cap_usd=ledger.stage_caps["judgment"],
        budget_cap_usd=ledger.cap,
        by_item=byitem,
        blockers=blockers,
        live_ready=not blockers,
        prices=prices,
        max_attempts=attempt_limit,
        subset_policy="No feature or item truncation; blocked whole-bank work is reported explicitly",
    )

"""Uncached hosted API adapter and an explicitly synthetic offline test client."""

from __future__ import annotations
import json
import os
import urllib.request
from .io import canonical


def generation_schema(item_id, cap):
    schema = dict(
        type="object",
        additionalProperties=False,
        required=["item_id", "features"],
        properties={
            "item_id": dict(type="string", enum=[item_id]),
            "features": dict(type="array", maxItems=cap, items=dict(type="string")),
        },
    )
    if cap is None:
        schema["properties"]["features"].pop("maxItems")
    return schema


def judgment_schema(item_id, feature_id, *, rubric_version="binary_v2"):
    from .judge import rubric_for, LEGACY_RUBRIC

    rubric_for(dict(rubric_version=rubric_version))
    legacy = rubric_version == LEGACY_RUBRIC
    return dict(
        type="object",
        additionalProperties=False,
        required=["item_id", "feature_id", "applies", "status", "evidence"],
        properties={
            "item_id": dict(type="string", enum=[item_id]),
            "feature_id": dict(type="string", enum=[feature_id]),
            "applies": (
                {"type": ["integer", "null"], "enum": [0, 1, None]}
                if legacy
                else dict(type="integer", enum=[0, 1])
            ),
            "status": dict(
                type="string", enum=["ok", "ambiguous"] if legacy else ["ok"]
            ),
            "evidence": dict(type="string"),
        },
    )


def build_payload(settings, *, system, user, schema, seed=None, temperature=0.8):
    """Single request builder shared by token audit, reservation and transport."""
    s = settings
    payload = dict(
        model=s["model_id"],
        messages=[dict(role="system", content=system), dict(role="user", content=user)],
        max_completion_tokens=s["max_output_tokens"],
        stream=False,
    )
    if s.get("supports_temperature"):
        payload["temperature"] = temperature
    if s.get("supports_seed") and seed is not None:
        payload["seed"] = seed
    if s.get("reasoning_effort"):
        payload["reasoning_effort"] = s["reasoning_effort"]
    if not s.get("supports_structured_outputs"):
        raise ValueError("Verified strict structured-output support is required")
    payload["response_format"] = dict(
        type="json_schema",
        json_schema=dict(name="pilot_response", strict=True, schema=schema),
    )
    if s.get("api_format") == "responses":
        payload = dict(
            model=s["model_id"],
            input=payload["messages"],
            max_output_tokens=s["max_output_tokens"],
            stream=False,
            store=False,
            text=dict(
                format=dict(
                    type="json_schema",
                    name="pilot_response",
                    strict=True,
                    schema=schema,
                )
            ),
        )
        if s.get("reasoning_effort"):
            payload["reasoning"] = dict(effort=s["reasoning_effort"])
    return payload


def parse_api_response(response):
    usage = response.get("usage") or {}
    if "choices" in response:
        choice = (response.get("choices") or [{}])[0]
        raw = (choice.get("message") or {}).get("content") or ""
        reason = choice.get("finish_reason")
        inputs, outputs = usage.get("prompt_tokens"), usage.get("completion_tokens")
    else:
        parts = [
            part
            for message in response.get("output", [])
            for part in message.get("content", [])
        ]
        raw = "".join(
            p.get("text", "") for p in parts if p.get("type") == "output_text"
        )
        reason = (
            "stop"
            if response.get("status") == "completed"
            and not any(p.get("type") == "refusal" for p in parts)
            else response.get("status", "error")
        )
        inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
    return raw, dict(
        input_tokens=inputs,
        output_tokens=outputs,
        response_model=response.get("model"),
        response_id=response.get("id"),
        system_fingerprint=response.get("system_fingerprint"),
        finish_reason=reason,
    )


class HostedClient:
    """OpenAI-compatible HTTPS Chat Completions; retries belong to the budget controller.

    Configure a pinned frontier snapshot and verified provider capabilities. No SDK,
    implicit retries, response cache, vLLM import, cluster dispatch, or key logging.
    """

    def __init__(self, settings):
        if settings.get("pricing_mode") == "batch":
            raise ValueError(
                "Use online pricing for synchronous requests; Batch pricing requires Batch transport"
            )
        self.settings = dict(settings)
        if settings["provider"] != "openai_compatible":
            raise ValueError("Unsupported hosted API provider adapter")
        if not settings["base_url"].startswith("https://"):
            raise ValueError("Hosted API requires HTTPS")
        self.transport = self._transport
        self._count_cache = {}

    def _transport(self, payload):
        key = os.environ.get(self.settings["api_key_env"])
        if not key:
            raise ValueError(
                f'Missing API key environment variable {self.settings["api_key_env"]}'
            )
        request = urllib.request.Request(
            self.settings["base_url"].rstrip("/")
            + (
                "/responses"
                if self.settings.get("api_format") == "responses"
                else "/chat/completions"
            ),
            data=canonical(payload).encode(),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=self.settings.get("timeout_seconds", 90)
        ) as r:
            return json.load(r)

    def count_payload(self, payload):
        from .token_accounting import count_payload, provider_count_payload

        preliminary = count_payload(payload, self.settings)
        if not preliminary.get("provider_count_required"):
            return preliminary
        key = preliminary["request_sha256"]
        if key not in self._count_cache:
            self._count_cache[key] = provider_count_payload(
                payload, self.settings, self.count_transport
            )
        return self._count_cache[key]

    def count_transport(self, payload):
        request = urllib.request.Request(
            self.settings["base_url"].rstrip("/") + "/responses/input_tokens",
            data=canonical(payload).encode(),
            headers={
                "Authorization": "Bearer " + os.environ[self.settings["api_key_env"]],
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=self.settings.get("timeout_seconds", 90)
        ) as response:
            return json.load(response)

    def complete(self, *, system, user, schema, seed=None, temperature=0.8):
        payload = build_payload(
            self.settings,
            system=system,
            user=user,
            schema=schema,
            seed=seed,
            temperature=temperature,
        )
        if self.settings.get("token_counting"):
            from .token_accounting import validate_payload

            preliminary = validate_payload(payload, self.settings)
            if preliminary.get("provider_count_required"):
                counted = self.count_payload(payload)
                if counted["input_token_bound"] > self.settings["max_input_tokens"]:
                    raise ValueError(
                        "Provider count exceeds frozen input-token ceiling"
                    )
        response = self.transport(payload)
        return parse_api_response(response)


class FakeClient:
    """Deterministic synthetic behavior for engineering validation, never a benchmark."""

    def __init__(self, *, fail_once=False, ambiguous=False):
        self.calls = 0
        self.fail_once = fail_once
        self.ambiguous = ambiguous

    @staticmethod
    def properties(text):
        t = text.casefold()
        result = ["can be described with semantic properties"]
        poem = "poem" in t or "creative production" in t
        finance = "reconcil" in t or "business administration" in t
        deadline = "five minutes" in t
        review = "independently reviewed" in t
        if "synthetic concept" in t:
            result += ["describes a synthetic abstract concept"]
        if "synthetic event" in t:
            result += ["describes a synthetic event"]
        if "synthetic object" in t:
            result += ["describes a synthetic object"]
        if poem:
            result += ["involves creative production"]
        if finance:
            result += ["involves financial record analysis"]
        if "eight-line" in t:
            result += [
                "requires eight lines of verse",
                "requires a consistent rhyme scheme",
            ]
        if "INV-020".casefold() in t:
            result += ["requires reconciling twenty invoices"]
        if deadline:
            result += ["has a five-minute deadline"]
        if review:
            result += ["will receive independent quality review"]
        if deadline and finance:
            result += [
                "meeting the deadline can conflict with checking every discrepancy"
            ]
        if deadline and poem:
            result += [
                "meeting the deadline can conflict with revising the rhyme scheme"
            ]
        if review and finance:
            result += ["independent review can reveal financial matching errors"]
        if review and poem:
            result += ["independent review can identify inconsistencies in rhyme"]
        if review and deadline and (poem or finance):
            result += [
                "short deadline can constrain revisions before independent review"
            ]
        return result

    def complete(self, *, system, user, schema, seed=None, temperature=0.8):
        self.calls += 1
        request = json.loads(user)
        usage = dict(
            input_tokens=80,
            output_tokens=40,
            response_model="fake-frontier-v1",
            finish_reason="stop",
        )
        if self.fail_once and self.calls == 1:
            return "{invalid", usage
        if "proposition" in request:
            ambiguous = self.ambiguous
            return (
                canonical(
                    dict(
                        item_id=request["item_id"],
                        feature_id=request["feature_id"],
                        applies=(
                            None
                            if ambiguous
                            else int(
                                request["proposition"]
                                in self.properties(request["text"])
                            )
                        ),
                        status="ambiguous" if ambiguous else "ok",
                        evidence="Synthetic fixture rule; not a model judgment.",
                    )
                ),
                usage,
            )
        props = self.properties(request["text"])
        cap = schema["properties"]["features"].get("maxItems", len(props))
        if seed is not None and len(props) > 2:
            start = seed % len(props)
            props = props[start:] + props[:start]
        return canonical(dict(item_id=request["item_id"], features=props[:cap])), usage

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


def judgment_schema(item_id, feature_id):
    return dict(
        type="object",
        additionalProperties=False,
        required=["item_id", "feature_id", "applies", "status", "evidence"],
        properties={
            "item_id": dict(type="string", enum=[item_id]),
            "feature_id": dict(type="string", enum=[feature_id]),
            "applies": {"type": ["integer", "null"], "enum": [0, 1, None]},
            "status": dict(type="string", enum=["ok", "ambiguous"]),
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
    return payload


class HostedClient:
    """OpenAI-compatible HTTPS Chat Completions; retries belong to the budget controller.

    Configure a pinned frontier snapshot and verified provider capabilities. No SDK,
    implicit retries, response cache, vLLM import, cluster dispatch, or key logging.
    """

    def __init__(self, settings):
        self.settings = dict(settings)
        if settings["provider"] != "openai_compatible":
            raise ValueError("Unsupported hosted API provider adapter")
        if not settings["base_url"].startswith("https://"):
            raise ValueError("Hosted API requires HTTPS")
        self.transport = self._transport

    def _transport(self, payload):
        key = os.environ.get(self.settings["api_key_env"])
        if not key:
            raise ValueError(
                f'Missing API key environment variable {self.settings["api_key_env"]}'
            )
        request = urllib.request.Request(
            self.settings["base_url"].rstrip("/") + "/chat/completions",
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

            validate_payload(payload, self.settings)
        response = self.transport(payload)
        usage = response.get("usage") or {}
        choices = response.get("choices") or []
        choice = choices[0] if choices else {}
        return choice.get("message", {}).get("content") or "", dict(
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            response_model=response.get("model"),
            response_id=response.get("id"),
            system_fingerprint=response.get("system_fingerprint"),
            finish_reason=choice.get("finish_reason"),
        )


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

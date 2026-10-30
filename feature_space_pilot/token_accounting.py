"""Labeled offline planning bounds and provider input counts, with framing reserve."""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import importlib.metadata
import math

from .io import canonical, digest, file_hash

TABLE = (
    Path(__file__).resolve().parents[1]
    / "data/feature_space_pilot/tokenizers/o200k_base.tiktoken"
)
TABLE_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
# OpenAI tiktoken o200k_base pattern, MIT license; see tokenizer attribution.
PATTERN = "|".join(
    [
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)


@lru_cache(maxsize=8)
def _encoding(model):
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    try:
        name = tiktoken.model.encoding_name_for_model(model)
    except KeyError:
        raise ValueError("No verified tokenizer mapping for selected model") from None
    if name != "o200k_base" or file_hash(TABLE) != TABLE_SHA256:
        raise ValueError("Unsupported or modified pinned tokenizer table")
    return tiktoken.Encoding(
        name=name,
        pat_str=PATTERN,
        mergeable_ranks=load_tiktoken_bpe(str(TABLE), expected_hash=TABLE_SHA256),
        special_tokens={"<|endoftext|>": 199999, "<|endofprompt|>": 200018},
    )


def count_payload(payload, settings):
    options = settings.get("token_counting") or {}
    margin = options.get("framing_margin_tokens")
    if type(margin) is not int or margin < 128:
        raise ValueError("Explicit conservative framing/schema margin required")
    if options.get("method") == "provider_count_required_v1":
        encoded = canonical(payload)
        return dict(
            payload_tokens=None,
            framing_margin_tokens=margin,
            input_token_bound=len(encoded.encode()) + margin,
            payload_bytes=len(encoded.encode()),
            encoding=None,
            request_sha256=digest(payload),
            method=options["method"],
            exact_provider_count=False,
            provider_count_required=True,
            planning_basis="UTF-8 payload byte upper bound, not a measured model token count",
        )
    if options.get("method") != "tiktoken_payload_plus_margin_v1":
        raise ValueError("Configure a verified input-token counting method")
    encoded = canonical(payload)
    encoding = _encoding(settings["model_id"])
    count = len(encoding.encode(encoded, disallowed_special=()))
    return dict(
        payload_tokens=count,
        framing_margin_tokens=margin,
        input_token_bound=count + margin,
        payload_bytes=len(encoded.encode()),
        encoding=encoding.name,
        tokenizer_version=importlib.metadata.version("tiktoken"),
        tokenizer_sha256=TABLE_SHA256,
        request_sha256=digest(payload),
        method=options["method"],
        exact_provider_count=False,
    )


def provider_count_payload(payload, settings, transport):
    if settings.get("api_format") != "responses":
        raise ValueError(
            "Exact provider counting requires the same Responses payload used for execution"
        )
    result = count_payload(payload, settings)
    counted = transport(
        {
            k: v
            for k, v in payload.items()
            if k
            in {
                "model",
                "input",
                "instructions",
                "text",
                "reasoning",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
            }
        }
    )
    n = counted.get("input_tokens")
    if type(n) is not int or n < 0:
        raise ValueError("Invalid provider input-token count")
    result.update(
        payload_tokens=n,
        input_token_bound=n + settings["token_counting"]["framing_margin_tokens"],
        exact_provider_count=True,
        provider_count_required=False,
        count_endpoint="/v1/responses/input_tokens",
        planning_basis="Provider model-specific input count plus framing reserve",
    )
    if result["input_token_bound"] > settings["max_input_tokens"]:
        raise ValueError(
            "Provider input-token count exceeds frozen ceiling before reservation"
        )
    return result


def validate_payload(payload, settings):
    count = count_payload(payload, settings)
    if count["input_token_bound"] > settings["max_input_tokens"]:
        raise ValueError(
            "Request exceeds frozen input-token ceiling before reservation/transport"
        )
    return count


def audit_prompts(config):
    from .engine import selected_items
    from .prompts import generation_prompt
    from .judge import judge_prompt, rubric_for
    from .clients import build_payload, generation_schema, judgment_schema

    items = selected_items(config, "all")
    generation, judgment = [], []
    for item in items:
        for family in ["P0", "P1", "P2"]:
            p = generation_prompt(item, family, protocols=config.get("protocols"))
            s = config["generator"]
            data = validate_payload(
                build_payload(
                    s,
                    system=p["system"],
                    user=p["user"],
                    schema=generation_schema(item["item_id"], p["cap"]),
                ),
                s,
            )
            generation.append(
                dict(item_id=item["item_id"], prompt_family=family, **data)
            )
        feature = dict(
            feature_id="f_" + "0" * 20, text="requires a specific semantic property"
        )
        s = config["judge"]
        version = rubric_for(s)
        p = judge_prompt(item, feature, rubric_version=version)
        data = validate_payload(
            build_payload(
                s,
                system=p["system"],
                user=p["user"],
                schema=judgment_schema(
                    item["item_id"], feature["feature_id"], rubric_version=version
                ),
            ),
            s,
        )
        # Parser permits 300 Unicode characters; JSON control escaping can use
        # six bytes per character, larger than the four-byte UTF-8 maximum.
        # BPE tokens cannot outnumber bytes. Reserve the full replacement cost.
        upper = data["input_token_bound"] + 1800
        if upper > s["max_input_tokens"]:
            raise ValueError(
                "Possible full-length feature exceeds judgment input-token ceiling"
            )
        judgment.append(
            dict(item_id=item["item_id"], max_feature_input_token_bound=upper, **data)
        )

    def stats(rows):
        ordered = sorted(r["input_token_bound"] for r in rows)
        return dict(
            n_templates=len(rows),
            max_input_token_bound=max(ordered),
            median_input_token_bound=ordered[len(ordered) // 2],
            p95_input_token_bound=ordered[math.ceil(0.95 * len(ordered)) - 1],
            max_payload_bytes=max(r["payload_bytes"] for r in rows),
        )

    return dict(
        n_items=len(items),
        generation=stats(generation),
        judgment=dict(
            stats(judgment),
            max_input_token_bound_with_feature_reserve=max(
                r["max_feature_input_token_bound"] for r in judgment
            ),
        ),
        largest_generation=max(generation, key=lambda r: r["input_token_bound"]),
        generation_requests=generation,
        judgment_requests=judgment,
        declared_output_tokens=dict(
            generation=config["generator"]["max_output_tokens"],
            judgment=config["judge"]["max_output_tokens"],
        ),
        byte_lengths_are_token_counts=False,
        provider_usage_is_billing_authority=True,
    )


def main():
    import argparse, json
    from .engine import load_config
    from .io import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/feature_space_pilot.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        result = audit_prompts(load_config(args.config))
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Token audit failed: {error}\n")


if __name__ == "__main__":
    main()

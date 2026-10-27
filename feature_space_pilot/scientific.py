"""Read-only scientific/source preflight, separated from language equivalence claims."""

from __future__ import annotations
from collections import Counter
import csv

from .io import file_hash, materialized
from .registry import seal_item
from .references import load_reference, validate_item_alignment


def load_translation_reviews(path, *, expected_sha256):
    path = materialized(path)
    if file_hash(path) != expected_sha256:
        raise ValueError("Translation review hash mismatch")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "item_id",
            "german_original",
            "published_loose_english",
            "reviewed_english",
            "reviewer",
            "decision",
        }
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing translation review columns")
        rows = list(reader)
    result = {}
    for row in rows:
        iid = row["item_id"]
        if not iid or iid in result:
            raise ValueError("Duplicate or missing translation review ID")
        decision = row["decision"].strip().lower()
        if decision not in {"pending", "approved", "rejected"}:
            raise ValueError("Unknown translation review decision")
        if decision in {"approved", "rejected"} and not row["reviewer"].strip():
            raise ValueError("Translation decision requires a named reviewer")
        if decision == "approved" and not row["reviewed_english"].strip():
            raise ValueError("Approved translation requires reviewed English")
        result[iid] = dict(row, decision=decision, approved=decision == "approved")
    return result


def apply_language_condition(items, config):
    from .engine import resolve

    condition = config.get("human_language_condition")
    if condition is None:
        return items
    if condition not in {"published_english_exploratory", "german_primary"}:
        raise ValueError("Unknown human language condition")
    reviews = load_translation_reviews(
        resolve(config, config["translation_review_file"]),
        expected_sha256=config["translation_review_sha256"],
    )
    abstract_ids = {
        i["item_id"] for i in items if i["source_dataset"] == "harpaintner_2018"
    }
    if abstract_ids != reviews.keys():
        raise ValueError("Translation review IDs must match all 24 abstract items")
    output = []
    for original in items:
        item = dict(original)
        if item["source_dataset"] == "harpaintner_2018":
            review = reviews[item["item_id"]]
            if (
                review["german_original"] != item["source_text"]
                or review["published_loose_english"] != item["text"]
            ):
                raise ValueError("Translation review source-label mismatch")
            item.update(
                english_text=item["text"],
                original_german=item["source_text"],
                stimulus_condition=condition,
                translation_review=review,
            )
            if condition == "german_primary":
                item.update(
                    text=item["source_text"],
                    language="de",
                    translation_status="original",
                )
            elif review["approved"]:
                item.update(
                    text=review["reviewed_english"], translation_status="reviewed"
                )
            elif review["decision"] == "rejected":
                raise ValueError(
                    f"Rejected English stimulus must be resolved before generation: {item['item_id']}"
                )
        elif config.get("vinson_syntax_markers"):
            prefix = "to" if item["selection_stratum"] == "actionV" else "the"
            item.update(
                text=prefix + " " + item["source_text"].lower(),
                stimulus_condition="vinson_published_syntax",
            )
        output.append(seal_item(item))
    return output


def scientific_audit(config):
    from .engine import selected_items, _source_specs
    from .source_adapters import load_vinson_zip
    from .prompts import generation_prompt

    items = selected_items(config, "human")
    blockers, warnings, sources = [], [], []
    by_source = {}
    for spec in _source_specs(config):
        if not spec.get("primary", True):
            continue
        try:
            ref = load_reference(spec, live=True)
            if spec["dataset_id"] == "vinson_2008":
                raw = load_vinson_zip(spec["raw_path"])
                with open(spec["path"], encoding="utf-8-sig", newline="") as handle:
                    derived = list(csv.DictReader(handle))
                normalize = lambda records: Counter(
                    tuple(sorted((k, str(v)) for k, v in r.items())) for r in records
                )
                if normalize(raw["rows"]) != normalize(derived):
                    raise ValueError("Vinson raw/derived crosswalk mismatch")
            by_source[spec["dataset_id"]] = ref
            sources.append(
                dict(
                    dataset_id=spec["dataset_id"],
                    available=True,
                    raw_sha256=spec.get("raw_sha256"),
                    sha256=spec["sha256"],
                    data_level=spec["data_level"],
                    verification_basis=spec.get("verification_basis"),
                )
            )
            if spec.get("publisher_download_verified") is False:
                warnings.append(
                    "Harpaintner source supplied by user; workbook/schema/selected rows verified, publisher-download chain not independently established"
                )
        except (ValueError, OSError, KeyError) as error:
            blockers.append(f"{spec['dataset_id']}: {error}")
    if Counter(i["target_type"] for i in items) != dict(
        concept=24, event=24, object=12
    ):
        blockers.append("Human inventory must have exact 24/24/12 quotas")
    availability = []
    for item in items:
        ref = by_source.get(item["source_dataset"])
        reason = "Verified reference unavailable"
        if ref:
            try:
                validate_item_alignment(item, ref)
            except ValueError as error:
                reason = str(error)
                if (
                    item.get("stimulus_condition") != "published_english_exploratory"
                    or item.get("translation_review", {}).get("decision") != "pending"
                ):
                    blockers.append(reason)
            else:
                reason = None
        availability.append(
            dict(item_id=item["item_id"], available=reason is None, reason=reason)
        )
        try:
            generation_prompt(item, "P0", live=True, protocols=config.get("protocols"))
        except ValueError as error:
            blockers.append(str(error))
    pending = sum(
        i["source_dataset"] == "harpaintner_2018"
        and i.get("translation_review", {}).get("decision") == "pending"
        for i in items
    )
    if (
        pending
        and config.get("human_language_condition") == "published_english_exploratory"
    ):
        warnings.append(
            f"{pending} published loose English translations are exploratory stimuli; cross-language human category comparisons remain unavailable until named review"
        )
    abstract = {
        i["item_id"] for i in items if i["source_dataset"] == "harpaintner_2018"
    }
    return dict(
        generation_ready=not blockers,
        primary_language_condition=config.get("human_language_condition"),
        selected_items=len(items),
        sources=sources,
        pending_translation_reviews=pending,
        abstract_comparison_ready=all(
            r["available"] for r in availability if r["item_id"] in abstract
        ),
        item_comparison_availability=availability,
        blockers=sorted(set(blockers)),
        warnings=warnings,
    )


def main():
    import argparse
    import json
    from .engine import load_config
    from .io import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/feature_space_pilot.json")
    parser.add_argument("--output")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Verify pinned source bytes then make references read-only for this checkout",
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.prepare:
            from .engine import _source_specs

            pending = []
            for spec in _source_specs(config):
                for path_key, hash_key in [
                    ("path", "sha256"),
                    ("raw_path", "raw_sha256"),
                    ("crosswalk_path", "crosswalk_sha256"),
                ]:
                    if spec.get(path_key):
                        path = materialized(spec[path_key])
                        if file_hash(path) != spec.get(hash_key):
                            raise ValueError(
                                "Source preparation refuses changed/unpinned bytes"
                            )
                        pending.append(path)
            for path in pending:
                path.chmod(path.stat().st_mode & ~0o222)
        result = scientific_audit(config)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Scientific audit failed: {error}\n")


if __name__ == "__main__":
    main()

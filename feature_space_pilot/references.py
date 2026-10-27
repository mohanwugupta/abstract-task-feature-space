"""Evaluation-only reference ingestion. Never imported by model prompt builders."""

from __future__ import annotations
import csv
import random
from pathlib import Path
from .io import file_hash, materialized


def load_reference(spec, *, live=False, model_language=None):
    if live and spec.get("kind") != "human":
        raise ValueError("A synthetic reference cannot pass a live human gate")
    if live and not spec.get("verified"):
        raise ValueError(
            "Human reference must be verified from the source and codebook"
        )
    required = [
        "dataset_id",
        "kind",
        "path",
        "sha256",
        "language",
        "data_level",
        "columns",
        "participants",
        "instructions",
        "license",
        "url",
        "codebook",
    ]
    if any(not spec.get(k) for k in required):
        raise ValueError(
            "Missing reference provenance, instructions, license, or columns"
        )
    path = materialized(spec["path"])
    if file_hash(path) != spec["sha256"]:
        raise ValueError("Reference hash mismatch")
    if path.stat().st_mode & 0o222:
        raise ValueError("Reference must be read-only (chmod 444)")
    if spec.get("raw_path"):
        raw = materialized(spec["raw_path"])
        if not spec.get("raw_sha256") or file_hash(raw) != spec["raw_sha256"]:
            raise ValueError("Raw reference hash mismatch")
        if raw.stat().st_mode & 0o222:
            raise ValueError("Raw reference must be read-only (chmod 444)")
        if spec.get("adapter") == "human_pilot_category_shares_v1":
            from .source_adapters import verify_harpaintner_workbook

            verified = verify_harpaintner_workbook(
                raw, path, expected_sha256=spec["raw_sha256"]
            )
            crosswalk = materialized(spec["crosswalk_path"])
            if file_hash(crosswalk) != spec.get("crosswalk_sha256"):
                raise ValueError("Workbook crosswalk artifact hash mismatch")
            import json

            if json.loads(crosswalk.read_text()) != verified:
                raise ValueError(
                    "Workbook crosswalk artifact does not match raw source"
                )
    if spec["data_level"] not in {
        "item_features",
        "participant_features",
        "category_counts",
    }:
        raise ValueError("Unsupported reference data_level")
    if spec.get("adapter") == "human_pilot_category_shares_v1":
        from .human_inputs import CATEGORY_COLUMNS, load_abstract_category_rows

        expected_columns = dict(
            item="original_german", language="source_language", **CATEGORY_COLUMNS
        )
        if (
            spec["data_level"] != "category_counts"
            or spec["columns"] != expected_columns
        ):
            raise ValueError("Unsupported category-proportion reference columns")
        rows = load_abstract_category_rows(path)
    else:
        rows = _read_reference_csv(path, spec)
    if not rows or any(not r.get("item", "").strip() for r in rows):
        raise ValueError("Empty reference or missing item strings")
    if any(r.get("language", spec["language"]) != spec["language"] for r in rows):
        raise ValueError("Reference language does not match declared language")
    language = model_language or spec["language"]
    if language != spec["language"]:
        mapping = spec.get("translation_mapping", {})
        if (
            not spec.get("translation_reviewed")
            or not mapping
            or any(r["item"] not in mapping for r in rows)
        ):
            raise ValueError(
                "A reviewed translation mapping is required for aligned language"
            )
        if spec["data_level"] != "category_counts" and not spec.get(
            "feature_translation_reviewed"
        ):
            raise ValueError(
                "Feature translation must also be reviewed for lexical recovery"
            )
    return dict(spec, rows=rows, model_language=language)


def _read_reference_csv(path, spec):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = spec["columns"]
        needed = (
            {"item", "category", "count"}
            if spec["data_level"] == "category_counts"
            else {"item", "feature"}
        )
        if not needed <= columns.keys() or not set(columns.values()) <= set(
            reader.fieldnames or []
        ):
            raise ValueError("Unsupported reference columns")
        return [{k: r[v] for k, v in columns.items()} for r in reader]


def validate_item_alignment(item, ref):
    """Check the actual source label and reviewed cross-language mapping."""
    source_text = item.get("source_text", item["text"])
    if source_text not in {r["item"] for r in ref["rows"]}:
        raise ValueError(
            f"Human stimulus absent from verified reference: {item['item_id']}"
        )
    if item["language"] == ref["language"]:
        if item.get("stimulus_condition") == "vinson_published_syntax":
            prefix = "to" if item["selection_stratum"] == "actionV" else "the"
            if item["text"] == prefix + " " + source_text.lower():
                return
        if source_text != item["text"]:
            raise ValueError("Human item text does not match its same-language source")
        return
    if item.get("translation_status") != "reviewed":
        raise ValueError("Human stimulus language requires reviewed translation")
    if (
        not ref.get("translation_reviewed")
        or ref.get("translation_mapping", {}).get(source_text) != item["text"]
    ):
        raise ValueError(
            "Human stimulus requires an exact reviewed translation mapping"
        )


def select_stimuli(rows, quotas, *, seed):
    """Hash-blind stratified sampling of labels; excludes ambiguous translations."""
    rng = random.Random(seed)
    selected = []
    for stratum, count in sorted(quotas.items()):
        candidates = sorted(
            {
                r["item"]
                for r in rows
                if r.get("stratum") == stratum and not r.get("ambiguous", False)
            }
        )
        if len(candidates) < count:
            raise ValueError(
                f"Insufficient coverage for {stratum}: {len(candidates)} < {count}"
            )
        chosen = rng.sample(candidates, count)
        held = set(rng.sample(chosen, max(1, count // 4))) if count >= 3 else set()
        selected.extend(
            dict(
                item=text,
                stratum=stratum,
                split="held_out" if text in held else "development",
            )
            for text in sorted(chosen)
        )
    return selected

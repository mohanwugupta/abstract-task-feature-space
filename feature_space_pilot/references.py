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
    if spec["data_level"] not in {
        "item_features",
        "participant_features",
        "category_counts",
    }:
        raise ValueError("Unsupported reference data_level")
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
        rows = [{k: r[v] for k, v in columns.items()} for r in reader]
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

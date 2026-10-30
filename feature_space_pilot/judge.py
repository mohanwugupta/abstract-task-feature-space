"""Strict binary applicability rubric for the feature-space pilot."""

from __future__ import annotations
import json
from .io import canonical, digest
from .prompts import DIRECTORY

LEGACY_RUBRIC = "binary_v1"
RUBRIC = "binary_v2"


def rubric_for(settings=None):
    """Missing flags in frozen historical settings mean v1, never retroactive v2."""
    version = (
        RUBRIC if settings is None else settings.get("rubric_version", LEGACY_RUBRIC)
    )
    if version not in {LEGACY_RUBRIC, RUBRIC}:
        raise ValueError("Unknown applicability rubric version")
    return version


def judge_prompt(item, feature, *, rubric_version=RUBRIC):
    rubric_for(dict(rubric_version=rubric_version))
    system = (DIRECTORY / (rubric_version + ".txt")).read_text().strip()
    user = canonical(
        dict(
            item_id=item["item_id"],
            text=item["text"],
            language=item["language"],
            feature_id=feature["feature_id"],
            proposition=feature["text"],
        )
    )
    return dict(
        system=system,
        user=user,
        prompt_hash=digest([system, user]),
        rubric_version=rubric_version,
    )


def parse_judgment(raw, item_id, feature_id, *, rubric_version=RUBRIC):
    rubric_for(dict(rubric_version=rubric_version))
    try:
        r = json.loads(raw)
    except (ValueError, TypeError) as e:
        raise ValueError("Malformed judgment JSON") from e
    if not isinstance(r, dict) or set(r) != {
        "item_id",
        "feature_id",
        "applies",
        "status",
        "evidence",
    }:
        raise ValueError("Invalid binary judgment schema")
    if r["item_id"] != item_id or r["feature_id"] != feature_id:
        raise ValueError("Judgment target mismatch")
    if (
        not isinstance(r["evidence"], str)
        or not r["evidence"].strip()
        or len(r["evidence"]) > 1200
    ):
        raise ValueError("Brief evidence note required")
    if (
        rubric_version == LEGACY_RUBRIC
        and r["status"] == "ambiguous"
        and r["applies"] is None
    ):
        return r
    if (
        r["status"] != "ok"
        or type(r["applies"]) is not int
        or r["applies"] not in (0, 1)
    ):
        raise ValueError("Judgment must be exactly integer 0 or 1 with status=ok")
    return r


def build_matrix(item_ids, feature_ids, judgments):
    scope = {(i, f) for i in item_ids for f in feature_ids}
    cells = {}
    for r in judgments:
        key = (r["item_id"], r["feature_id"])
        if key not in scope:
            raise ValueError("Judgment outside declared scope")
        if key in cells:
            raise ValueError("Duplicate judgment cell")
        if (
            r.get("status") != "ok"
            or type(r.get("applies")) is not int
            or r["applies"] not in (0, 1)
        ):
            raise ValueError(
                "Unresolved judgment; incomplete matrix requires adjudication"
            )
        cells[key] = r["applies"]
    if set(cells) != scope:
        raise ValueError(f"incomplete matrix: {len(scope-cells.keys())} missing cells")
    return [[cells[i, f] for f in feature_ids] for i in item_ids]

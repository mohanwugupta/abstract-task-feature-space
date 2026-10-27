"""Versioned item registry. Situations are an independent axis."""

from __future__ import annotations
from .io import digest, read_jsonl

TYPES = {
    "concept",
    "event",
    "object",
    "domain",
    "task_type",
    "task_instance",
    "situation",
    "situated_task",
}
REQUIRED = {
    "item_id",
    "text",
    "target_type",
    "level",
    "parent_id",
    "task_id",
    "situation_ids",
    "source_dataset",
    "language",
    "selection_stratum",
    "translation_status",
    "content_hash",
}


def seal_item(item):
    item = dict(item)
    item.pop("content_hash", None)
    return dict(item, content_hash=digest(item))


def validate_items(items):
    ids = [x.get("item_id") for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate item IDs")
    by_id = {x["item_id"]: x for x in items}
    allowed_parents = {
        "task_type": "domain",
        "task_instance": "task_type",
        "situated_task": "task_instance",
    }
    for x in items:
        if REQUIRED - x.keys():
            raise ValueError(f"Missing item fields: {sorted(REQUIRED - x.keys())}")
        if x["target_type"] not in TYPES:
            raise ValueError("Invalid target_type")
        for key in ["item_id", "text", "language", "source_dataset", "level"]:
            if not isinstance(x[key], str) or not x[key].strip():
                raise ValueError(f"Empty or invalid {key}")
        if seal_item(x)["content_hash"] != x["content_hash"]:
            raise ValueError(f'Stale item hash: {x["item_id"]}')
        if not isinstance(x["situation_ids"], list) or len(
            set(x["situation_ids"])
        ) != len(x["situation_ids"]):
            raise ValueError("Invalid situation_ids")
        parent = x["parent_id"]
        if parent:
            if parent not in by_id or parent == x["item_id"]:
                raise ValueError("Missing or cyclic parent")
            if by_id[parent]["target_type"] != allowed_parents.get(x["target_type"]):
                raise ValueError("Wrong parent type")
            if by_id[parent]["task_id"] != x["task_id"]:
                raise ValueError("Inconsistent parent task_id")
        if x["target_type"] == "situated_task" and (
            not parent or not x["situation_ids"]
        ):
            raise ValueError("Situated task requires parent and situation")
        if x["situation_ids"] and x["target_type"] != "situated_task":
            raise ValueError("Only situated tasks reference a situation")
        for sid in x["situation_ids"]:
            if sid not in by_id or by_id[sid]["target_type"] != "situation":
                raise ValueError(f"Unknown situation: {sid}")
        if x["target_type"] == "situation" and (parent or x["task_id"]):
            raise ValueError("Situation must be independent of task hierarchy")
    return items


def load_items(path):
    return validate_items(read_jsonl(path))

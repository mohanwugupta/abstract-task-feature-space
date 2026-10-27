"""Canonical serialization and materialized-input guards; no model dependencies."""

from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialized(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing materialized file: {path}")
    if (
        path.read_bytes()[:200]
        .lstrip()
        .startswith(b"version https://git-lfs.github.com/spec/v1")
    ):
        raise ValueError(
            f"Git LFS pointer: {path}; run git lfs pull or supply the actual source file."
        )
    return path


def read_jsonl(path):
    return [
        json.loads(line)
        for line in materialized(path).read_text().splitlines()
        if line.strip()
    ]


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(canonical(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )
    tmp.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: canonical(v) if isinstance(v, (dict, list, tuple)) else v
                    for k, v in row.items()
                }
            )

"""One validated read-only adapter shared by every visualization."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from ..io import digest, file_hash, materialized, read_jsonl
from ..registry import validate_items
from ..report import verify_manifest


class CellState(IntEnum):
    OUT_OF_SCOPE = -2
    UNRESOLVED = -1
    INAPPLICABLE = 0
    APPLIES = 1


SYMBOLS = {
    CellState.APPLIES: "1",
    CellState.INAPPLICABLE: "0",
    CellState.UNRESOLVED: "?",
    CellState.OUT_OF_SCOPE: "N/A",
}
SYNTHETIC_BANNER = "SYNTHETIC DEMONSTRATION — NOT EMPIRICAL RESULTS"


def _json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _csv(path, required=False):
    if not path.exists():
        if required:
            raise ValueError(f"Missing required visualization input: {path.name}")
        return []
    with materialized(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _bool(value):
    if value in (True, "True", "true", "1"):
        return True
    if value in (False, "False", "false", "0"):
        return False
    raise ValueError("Invalid candidate selection flag")


@dataclass(frozen=True)
class VizRun:
    path: Path
    manifest: dict
    manifest_hash: str
    fingerprint: str
    items: dict
    features: dict
    scope: dict
    excluded: frozenset
    judgments: dict
    candidates: tuple
    generations: tuple
    jobs: dict
    attempts: tuple
    composition_audit: tuple
    manual_audits: dict
    interaction_reviews: tuple
    human_summary: dict
    embedding_summary: dict
    config: dict
    evaluation: dict
    sources: tuple

    @property
    def demo(self):
        return bool(self.manifest.get("demo"))

    @property
    def n_in_scope(self):
        return len(self.scope["item_ids"]) * len(self.scope["feature_ids"]) - sum(
            i in self.scope["item_ids"] and f in self.scope["feature_ids"]
            for i, f in self.excluded
        )

    @property
    def task_item_ids(self):
        return tuple(
            i
            for i, x in self.items.items()
            if x["target_type"] not in {"concept", "event", "object"}
        )

    def cell(self, item_id, feature_id):
        if item_id not in self.items:
            raise ValueError(f"Unknown item ID: {item_id}")
        if feature_id not in self.features:
            raise ValueError(f"Unknown feature ID: {feature_id}")
        if (
            item_id not in self.scope["item_ids"]
            or feature_id not in self.scope["feature_ids"]
            or (item_id, feature_id) in self.excluded
        ):
            return CellState.OUT_OF_SCOPE
        record = self.judgments.get((item_id, feature_id))
        return CellState(record["applies"]) if record else CellState.UNRESOLVED

    def inspect(self, feature_id):
        if feature_id not in self.features:
            raise ValueError("Unknown feature ID")
        return dict(
            **self.features[feature_id],
            origins=[
                dict(
                    item_id=r["origin_item"],
                    prompt_family=r["prompt_family"],
                    replicate=r["replicate"],
                    job_id=r["job_id"],
                    raw_phrase=r["phrase"],
                    normalization=r.get("normalization"),
                    selected=r["selected"],
                )
                for r in self.candidates
                if r["feature_id"] == feature_id
            ],
            judgments=[
                dict(
                    item_id=i,
                    text=self.items[i]["text"],
                    state=SYMBOLS[self.cell(i, feature_id)],
                    evidence=self.judgments.get((i, feature_id), {}).get(
                        "evidence", "No resolved in-scope judgment"
                    ),
                    judge_model=self.judgments.get((i, feature_id), {}).get(
                        "judge_model"
                    ),
                )
                for i in self.items
            ],
            primary_attempts=[
                r for r in self.attempts if r.get("feature_id") == feature_id
            ],
            manual_reviews=[
                dict(item_id=i, **a[feature_id])
                for i, a in self.manual_audits.items()
                if feature_id in a
            ],
        )


def load_run(run_dir):
    path = Path(run_dir).expanduser().resolve()
    try:
        manifest = verify_manifest(path)
    except (OSError, ValueError, KeyError) as e:
        raise ValueError(f"Run manifest/integrity error: {e}") from e
    if not isinstance(manifest.get("run_id"), str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", manifest["run_id"]
    ):
        raise ValueError("Invalid run ID: expected a path-safe run identity")
    scope = _json(path / "matrix_scope.json")
    if not isinstance(scope, dict) or not {"item_ids", "feature_ids"} <= scope.keys():
        raise ValueError(
            "Missing matrix scope metadata; generation-only partial runs have no declared judgment scope"
        )
    item_rows = validate_items(read_jsonl(path / "stimuli.jsonl"))
    items = {r["item_id"]: r for r in item_rows}
    feature_rows = _csv(path / "feature_bank.csv", True)
    if len({r["feature_id"] for r in feature_rows}) != len(feature_rows):
        raise ValueError("Duplicate feature ID")
    features = {
        r["feature_id"]: dict(
            r, merge_decisions=json.loads(r.get("merge_decisions") or "[]")
        )
        for r in feature_rows
    }
    if not features:
        raise ValueError("No frozen feature bank")
    for key, known in [("item_ids", items), ("feature_ids", features)]:
        if len(scope[key]) != len(set(scope[key])):
            raise ValueError("Duplicate scope IDs")
        if set(scope[key]) - known.keys():
            raise ValueError("Unknown scope IDs")
    hashes = {r.get("inventory_hash") for r in feature_rows}
    if len(hashes) != 1 or (
        manifest.get("candidate_inventory_hash")
        and hashes != {manifest["candidate_inventory_hash"]}
    ):
        raise ValueError("Feature inventory hash mismatch")
    excluded = set()
    for value in scope.get("excluded_combinations", []):
        pair = (
            (value["item_id"], value["feature_id"])
            if isinstance(value, dict)
            else tuple(value)
        )
        if len(pair) != 2 or pair[0] not in items or pair[1] not in features:
            raise ValueError("Unknown excluded scope IDs")
        excluded.add(pair)
    judgments = {}
    for row in _csv(path / "binary_judgments.csv", True):
        key = (row["item_id"], row["feature_id"])
        if key[0] not in items or key[1] not in features:
            raise ValueError("Unknown binary judgment IDs")
        if key in judgments:
            raise ValueError("Duplicate binary judgment cell")
        if (
            key in excluded
            or key[0] not in scope["item_ids"]
            or key[1] not in scope["feature_ids"]
        ):
            raise ValueError("Judgment outside declared scope")
        if row["applies"] not in {"0", "1"}:
            raise ValueError("Judgments must be strict 0/1")
        judgments[key] = dict(row, applies=int(row["applies"]))
    jobs_rows = (
        read_jsonl(path / "generation_jobs.jsonl")
        if (path / "generation_jobs.jsonl").exists()
        else []
    )
    jobs = {r["job_id"]: r for r in jobs_rows}
    if len(jobs) != len(jobs_rows):
        raise ValueError("Duplicate generation job IDs")
    generation_rows = (
        read_jsonl(path / "generations.jsonl")
        if (path / "generations.jsonl").exists()
        else []
    )
    generations = [r for r in generation_rows if r.get("parse_status") == "ok"]
    if len({r["job_id"] for r in generations}) != len(generations):
        raise ValueError("Duplicate successful generation records")
    generated = {r["job_id"]: r for r in generations}
    for r in generations:
        if r["item_id"] not in items or r["job_id"] not in jobs:
            raise ValueError("Unknown generation join ID")
        if any(
            r[k] != jobs[r["job_id"]][k]
            for k in ["item_id", "prompt_family", "replicate"]
        ):
            raise ValueError("Inconsistent generation join")
    candidates = []
    seen = set()
    for row in _csv(path / "feature_candidates.csv"):
        r = dict(
            row,
            replicate=int(row["replicate"]),
            raw_feature_index=int(row["raw_feature_index"]),
            selected=_bool(row["selected"]),
        )
        if r["origin_item"] not in items or r["job_id"] not in generated:
            raise ValueError("Unknown candidate provenance IDs")
        if r["selected"] and r["feature_id"] not in features:
            raise ValueError("Unknown selected candidate feature ID")
        origin = generated[r["job_id"]]
        if (
            r["origin_item"] != origin["item_id"]
            or r["replicate"] != origin["replicate"]
            or r["prompt_family"] != origin["prompt_family"]
        ):
            raise ValueError("Inconsistent candidate provenance join")
        key = (r["job_id"], r["raw_feature_index"])
        if key in seen:
            raise ValueError("Duplicate candidate provenance row")
        seen.add(key)
        if (
            not 0 <= r["raw_feature_index"] < len(origin["features"])
            or origin["features"][r["raw_feature_index"]] != r["phrase"]
        ):
            raise ValueError("Candidate phrase does not match raw generation")
        candidates.append(r)
    if (path / "binary_matrix.csv").exists():
        with (path / "binary_matrix.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            matrix = list(reader)
            if reader.fieldnames != ["item_id"] + scope["feature_ids"]:
                raise ValueError("Matrix feature order disagrees with scope")
        if [r["item_id"] for r in matrix] != scope["item_ids"]:
            raise ValueError("Matrix item order disagrees with scope")
        for r in matrix:
            for fid in scope["feature_ids"]:
                if (
                    r[fid] not in {"0", "1"}
                    or (r["item_id"], fid) not in judgments
                    or int(r[fid]) != judgments[r["item_id"], fid]["applies"]
                ):
                    raise ValueError(
                        "Matrix contradicts resolved judgments or fills missing cells"
                    )
    config = _json(path / "frozen_config.json", {})
    evaluation = _json(path / "evaluation_inputs.json", {})
    sources = _json(path / "frozen_source_registry.json", {}).get("sources", [])
    manifest_hash = file_hash(path / "manifest.json")
    return VizRun(
        path,
        manifest,
        manifest_hash,
        digest([str(path), manifest_hash, manifest.get("evaluation_inputs_hash")]),
        items,
        features,
        scope,
        frozenset(excluded),
        judgments,
        tuple(candidates),
        tuple(generations),
        jobs,
        (
            tuple(read_jsonl(path / "judgments.jsonl"))
            if (path / "judgments.jsonl").exists()
            else ()
        ),
        tuple(_json(path / "compositionality_audit.json", [])),
        _json(path / "manual_relational_audits.json", {}),
        tuple(_csv(path / "interaction_review.csv")),
        _json(path / "human_norm_comparison.json", {}),
        _json(path / "embedding_rdm_comparison.json", {}),
        config,
        evaluation,
        tuple(sources),
    )

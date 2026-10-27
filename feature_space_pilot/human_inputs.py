"""Hash-pinned human CSV inventory and evaluation-only category-share adapter.

Uploaded bytes are never rewritten. Stimulus text is kept separate from human
labels; prompt builders continue to whitelist only ID, text and language.
"""

from __future__ import annotations
import csv
import math
import random
import re
from collections import Counter

from .io import file_hash, materialized
from .registry import seal_item, validate_items
from .prompts import normalize_phrase

CATEGORY_COLUMNS = {
    "SM": "human_sm",
    "IS/E": "human_is_e",
    "SC": "human_sc",
    "VA": "human_va",
}
CLASSES = {
    "actionN": "event",
    "actionV": "event",
    "object": "object",
    "abstract_concept": "concept",
}
SOURCE_ALIASES = {
    "vinson_2008": "vinson_2008",
    "harpaintner_2018": "harpaintner_2018",
    "harpaintner_2018_original_supplement": "harpaintner_2018",
}
REQUIRED = {
    "item_id",
    "item_text",
    "original_german",
    "source_dataset",
    "source_language",
    "stimulus_language",
    "item_class",
    "selection_stratum",
    "translation_status",
    "source_excel_row",
    "source_file_sha256",
    "source_url",
    "human_modal_category",
    "abstractness_rating",
    "familiarity_rating",
    *CATEGORY_COLUMNS.values(),
}
ABSTRACT_STRATA = {
    "sensorimotor",
    "internal_state_emotion",
    "social_constellation",
    "verbal_association",
}


def _read(path, expected_sha256=None):
    path = materialized(path)
    fingerprint = file_hash(path)
    if expected_sha256 is not None and fingerprint != expected_sha256:
        raise ValueError(f"Human CSV input hash mismatch: {path.name}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not REQUIRED <= set(fields):
            raise ValueError("Missing or duplicate human CSV columns")
        records = list(reader)
    if not records or any(
        None in r or any(v is None for v in r.values()) for r in records
    ):
        raise ValueError("Empty or malformed human CSV rows")
    return records, fingerprint


def _validate(records, *, abstract_only=False):
    ids = [r["item_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate human CSV item IDs")
    for row in records:
        for key in [
            "item_id",
            "item_text",
            "source_dataset",
            "source_language",
            "stimulus_language",
            "selection_stratum",
            "translation_status",
            "source_url",
        ]:
            if not row[key].strip():
                raise ValueError(f"Missing human CSV {key}")
        cls = row["item_class"]
        if cls not in CLASSES or (abstract_only and cls != "abstract_concept"):
            raise ValueError("Unsupported human CSV item class")
        expected_source = (
            "harpaintner_2018" if cls == "abstract_concept" else "vinson_2008"
        )
        if SOURCE_ALIASES.get(row["source_dataset"]) != expected_source:
            raise ValueError("Human CSV class/source mismatch")
        if row.get("split", "unassigned") not in {
            "",
            "unassigned",
            "development",
            "held_out",
        }:
            raise ValueError("Invalid human CSV split")
        if cls != "abstract_concept":
            if row["source_language"] != "en" or row["stimulus_language"] != "en":
                raise ValueError("Vinson CSV language mismatch")
            if row["selection_stratum"] != cls:
                raise ValueError("Vinson CSV class/stratum mismatch")
            if row.get("split") not in {"development", "held_out"}:
                raise ValueError("Vinson CSV must preserve its frozen split")
            continue
        if (
            row["source_language"] != "de"
            or row["stimulus_language"] not in {"en", "de"}
            or not row["original_german"].strip()
        ):
            raise ValueError(
                "Abstract CSV requires original German and declared language"
            )
        if (
            row["stimulus_language"] == "de"
            and row["item_text"] != row["original_german"]
        ):
            raise ValueError("German stimulus differs from original source text")
        if row["selection_stratum"] not in ABSTRACT_STRATA:
            raise ValueError("Unknown abstract CSV stratum")
        if not re.fullmatch(r"[0-9a-f]{64}", row["source_file_sha256"]):
            raise ValueError("Missing abstract publication source hash")
        if not row["source_excel_row"].isdigit() or int(row["source_excel_row"]) < 1:
            raise ValueError("Invalid abstract source Excel row")
        try:
            values = [float(row[k]) for k in CATEGORY_COLUMNS.values()]
        except ValueError:
            raise ValueError("Invalid human category proportion") from None
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Invalid human category proportion")
        if abs(sum(values) - 1) > 0.0005:
            raise ValueError(
                "Human category proportions must sum to one within rounding tolerance"
            )
        if row["human_modal_category"] not in CATEGORY_COLUMNS:
            raise ValueError("Unknown human modal category")
        for key in ["abstractness_rating", "familiarity_rating"]:
            try:
                value = float(row[key])
            except ValueError:
                raise ValueError(f"Invalid human {key}") from None
            if not math.isfinite(value) or not 1 <= value <= 7:
                raise ValueError(f"Invalid human {key}")


def _abstract_inventory(records):
    abstract = [r for r in records if r["item_class"] == "abstract_concept"]
    for column in ["original_german", "item_text"]:
        labels = [normalize_phrase(r[column]) for r in abstract]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate abstract source/translated labels: {column}")
    if len(abstract) != 24 or Counter(r["selection_stratum"] for r in abstract) != {
        s: 6 for s in ABSTRACT_STRATA
    }:
        raise ValueError("Abstract CSV requires 24 items, six per declared stratum")
    return abstract


def load_human_csv(
    path, *, expected_sha256=None, abstract_path=None, abstract_sha256=None, seed=42
):
    records, fingerprint = _read(path, expected_sha256)
    _validate(records)
    abstract = _abstract_inventory(records)
    if Counter(r["item_class"] for r in records) != dict(
        actionN=12, actionV=12, object=12, abstract_concept=24
    ):
        raise ValueError("Human CSV requires the frozen 12/12/12/24 class quotas")
    companion_hash = None
    if abstract_path is not None:
        companion, companion_hash = _read(abstract_path, abstract_sha256)
        _validate(companion, abstract_only=True)
        _abstract_inventory(companion)
        by_id = {r["item_id"]: r for r in abstract}
        if set(by_id) != {r["item_id"] for r in companion}:
            raise ValueError("Human inventory and abstract companion IDs disagree")
        for row in companion:
            if any(row[k] != by_id[row["item_id"]].get(k) for k in row if k != "split"):
                raise ValueError(
                    f"Human inventory and abstract companion values disagree: {row['item_id']}"
                )
    # Freeze previously unassigned splits by ID within source strata, not human outcomes.
    splits = {r["item_id"]: r.get("split", "unassigned") for r in records}
    rng = random.Random(seed)
    for stratum in sorted(ABSTRACT_STRATA):
        group = sorted(
            (r for r in abstract if r["selection_stratum"] == stratum),
            key=lambda r: r["item_id"],
        )
        pending = [
            r["item_id"] for r in group if splits[r["item_id"]] in {"", "unassigned"}
        ]
        if not pending:
            continue
        needed = len(group) // 4 - sum(
            splits[r["item_id"]] == "held_out" for r in group
        )
        if not 0 <= needed <= len(pending):
            raise ValueError(
                "Assigned abstract splits conflict with quarter-held-out policy"
            )
        held_out = set(rng.sample(pending, needed))
        splits.update(
            {iid: "held_out" if iid in held_out else "development" for iid in pending}
        )
    items = []
    for row in sorted(records, key=lambda r: r["item_id"]):
        abstract_item = row["item_class"] == "abstract_concept"
        target_type = CLASSES[row["item_class"]]
        item = dict(
            item_id=row["item_id"],
            text=row["item_text"],
            language=row["stimulus_language"],
            target_type=target_type,
            level=target_type,
            parent_id=None,
            task_id=None,
            situation_ids=[],
            source_dataset=SOURCE_ALIASES[row["source_dataset"]],
            source_dataset_original=row["source_dataset"],
            source_language=row["source_language"],
            source_text=row["original_german"] if abstract_item else row["item_text"],
            translation_status=row["translation_status"],
            selection_stratum=row["selection_stratum"],
            split=splits[row["item_id"]],
            source_split=row.get("split", "unassigned"),
            stimulus_csv_sha256=fingerprint,
            source_url=row["source_url"],
            split_policy="preserve_assigned_else_seeded_within_stratum_quarter_v1",
        )
        if abstract_item:
            item.update(
                source_excel_row=int(row["source_excel_row"]),
                source_file_sha256=row["source_file_sha256"],
                abstract_csv_sha256=companion_hash,
            )
        items.append(seal_item(item))
    return validate_items(items)


def load_abstract_category_rows(path):
    """Unpivot supplied rounded proportions; never fabricate feature productions."""
    records, _ = _read(path)
    _validate(records, abstract_only=True)
    _abstract_inventory(records)
    return [
        dict(
            item=row["original_german"],
            item_id=row["item_id"],
            category=category,
            count=float(row[column]),
            language=row["source_language"],
            source_excel_row=int(row["source_excel_row"]),
        )
        for row in sorted(records, key=lambda r: r["item_id"])
        for category, column in CATEGORY_COLUMNS.items()
    ]


def main():
    """Audit/normalize the configured input files offline, without study execution."""
    import argparse
    import json
    from pathlib import Path
    from .engine import load_config, resolve, selected_items
    from .io import canonical, digest, write_csv, write_json

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", default="configs/feature_space_pilot.json")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if config.get("demo") or not config.get("human_abstract_summary"):
            raise ValueError("Input audit requires the configured real human CSV pair")
        items = selected_items(config, "human")
        inputs = {
            key: resolve(config, config[key]).resolve()
            for key in ["human_stimuli", "human_abstract_summary"]
        }
        fingerprints = {key: file_hash(path) for key, path in inputs.items()}
        categories = load_abstract_category_rows(inputs["human_abstract_summary"])
        summary = dict(
            adapter_version="human_csv_v1",
            input_hashes=fingerprints,
            n_items=len(items),
            n_abstract=sum(i["target_type"] == "concept" for i in items),
            n_category_rows=len(categories),
            n_unreviewed_translations=sum(
                i["language"] != i["source_language"]
                and i["translation_status"] != "reviewed"
                for i in items
            ),
            classes=dict(sorted(Counter(i["target_type"] for i in items).items())),
            splits=dict(sorted(Counter(i["split"] for i in items).items())),
            held_out_item_ids=sorted(
                i["item_id"] for i in items if i["split"] == "held_out"
            ),
            seed=config["seed"],
            scientific_source_verified=False,
            verification_note="Structural input audit only; publication, protocol, translation and source provenance gates remain separate",
            category_measurement="rounded_category_proportions_not_integer_counts",
        )
        output = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else Path(config["root"])
            / "artifacts/human_input_audit"
            / digest(summary)[:16]
        )
        if any(path.is_relative_to(output) for path in inputs.values()):
            raise ValueError(
                "Audit output must be separate from the source input directory"
            )
        output.mkdir(parents=True, exist_ok=True)
        summary["output_dir"] = str(output)
        (output / "stimuli.jsonl").write_text(
            "".join(canonical(i) + "\n" for i in items), encoding="utf-8"
        )
        write_csv(
            output / "abstract_category_shares.csv",
            categories,
            ["item", "item_id", "category", "count", "language", "source_excel_row"],
        )
        summary["artifacts"] = {
            name: file_hash(output / name)
            for name in ["stimuli.jsonl", "abstract_category_shares.csv"]
        }
        write_json(output / "input_manifest.json", summary)
        if fingerprints != {key: file_hash(path) for key, path in inputs.items()}:
            raise ValueError("Human source CSV changed during audit")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Human input audit failed: {error}\n")


if __name__ == "__main__":
    main()

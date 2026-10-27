"""Verified raw-format adapters; derived files never replace verbatim references."""

from __future__ import annotations
import csv
import io
import zipfile
import math
from .io import materialized, file_hash


def verify_harpaintner_workbook(path, summary_path, *, expected_sha256):
    """Verify the supplied summary against exact workbook cells, not translations."""
    from openpyxl import load_workbook

    path = materialized(path)
    if file_hash(path) != expected_sha256:
        raise ValueError("Harpaintner raw workbook hash mismatch")
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook["Wortliste"]
        table = list(sheet.iter_rows(values_only=True))
    finally:
        workbook.close()
    columns = {name: n for n, name in enumerate(table[3])}
    mapping = {
        "original_german": "original german abstract concepts",
        "item_text": "loose english translations",
        "human_sm": "%SM",
        "human_is_e": "%IS/E",
        "human_sc": "%SC",
        "human_va": "%VA",
        "abstractness_rating": "abstractness rating",
        "familiarity_rating": "familiarity rating",
    }
    if not set(mapping.values()) <= columns.keys():
        raise ValueError("Unsupported Harpaintner workbook columns")
    if sum(bool(r[0]) for r in table[4:]) != 296:
        raise ValueError("Harpaintner workbook must contain 296 concepts")
    with materialized(summary_path).open(encoding="utf-8-sig", newline="") as handle:
        selected = list(csv.DictReader(handle))
    if len(selected) != 24 or len({r["item_id"] for r in selected}) != 24:
        raise ValueError("Workbook crosswalk requires 24 unique selected concepts")
    matches = []
    for row in sorted(selected, key=lambda r: r["item_id"]):
        number = int(row["source_excel_row"])
        if row["source_file_sha256"] != expected_sha256 or not 5 <= number <= len(
            table
        ):
            raise ValueError("Workbook crosswalk source row/hash mismatch")
        source = table[number - 1]
        for field, column in mapping.items():
            actual = source[columns[column]]
            same = (
                row[field] == actual
                if field in {"original_german", "item_text"}
                else math.isclose(
                    float(row[field]), float(actual), abs_tol=0.00005, rel_tol=0
                )
            )
            if not same:
                raise ValueError(
                    f"Workbook crosswalk mismatch: {row['item_id']} / {field}"
                )
        matches.append(
            dict(
                item_id=row["item_id"],
                source_excel_row=number,
                original_german=row["original_german"],
                published_loose_english=row["item_text"],
                fields_checked=sorted(mapping),
            )
        )
    return dict(
        adapter_version="harpaintner_workbook_crosswalk_v1",
        raw_sha256=expected_sha256,
        summary_sha256=file_hash(summary_path),
        workbook_items=296,
        matched_items=len(matches),
        matches=matches,
        discrepancies=[],
        publisher_download_verified=False,
        provenance="User-supplied workbook; exact internal derivation verified against primary publication schema",
    )


def load_vinson_zip(path, *, expected_items=456):
    path = materialized(path)
    if path.stat().st_mode & 0o222:
        raise ValueError("Raw reference must be read-only")
    with zipfile.ZipFile(path) as z:

        def table(name):
            data = z.read("Vinson-BRM-2008/" + name).decode("utf-8-sig")
            return [
                row
                for row in csv.reader(io.StringIO(data), delimiter="\t")
                if any(c.strip() for c in row)
            ]

        words = table("word_categories.txt")
        features = table("feature_list_and_types.txt")
        if len(words) - 1 != expected_items:
            raise ValueError(f"Expected {expected_items} items, found {len(words)-1}")
        if len(words[0]) != 4 or len(features[0]) != 6:
            raise ValueError("Unsupported Vinson columns")
        items = []
        for index, row in enumerate(words[1:], 1):
            if int(row[0]) != index or row[2] not in {"object", "actionV", "actionN"}:
                raise ValueError("Vinson item ordering/type mismatch")
            items.append(
                dict(index=index, item=row[1], stratum=row[2], original_category=row[3])
            )
        feature_text = []
        categories = {}
        for index, row in enumerate(features[1:], 1):
            if int(row[0]) != index:
                raise ValueError("Vinson feature ordering mismatch")
            feature_text.append(row[1])
            categories[row[1]] = [
                name for name, value in zip(features[0][2:], row[2:]) if int(value) == 1
            ]
        left = table("feature_weight_matrix_1_256.txt")
        right = table("feature_weight_matrix_257_456.txt")
        if len(left) != len(feature_text) or (
            expected_items > 256 and len(right) != len(feature_text)
        ):
            raise ValueError("Vinson matrix feature axis mismatch")
        joined = []
        for idx, row in enumerate(left):
            values = row + (right[idx] if right else [])
            if len(values) != expected_items:
                raise ValueError("Vinson matrix item axis mismatch")
            weights = [int(x) for x in values]
            if any(x < 0 for x in weights):
                raise ValueError("Negative feature weight")
            joined.append(weights)
        weights = [list(row) for row in zip(*joined)]
        rows = []
        for item, vector in zip(items, weights):
            for phrase, weight in zip(feature_text, vector):
                if weight > 0:
                    for category in categories[phrase] or [""]:
                        rows.append(
                            dict(
                                item=item["item"],
                                feature=phrase,
                                frequency=weight,
                                category=category,
                                language="en",
                                stratum=item["stratum"],
                                original_category=item["original_category"],
                            )
                        )
        return dict(
            items=items,
            item_count=len(items),
            features=feature_text,
            feature_categories=categories,
            weights=weights,
            rows=rows,
            raw_sha256=file_hash(path),
            axis_policy="Raw feature rows and ordered word columns transposed; positive weights indicate listed productions; no participant reconstruction",
        )

"""Verified raw-format adapters; derived files never replace verbatim references."""

from __future__ import annotations
import csv
import io
import zipfile
from .io import materialized, file_hash


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

"""One global binary scope, with explicit unresolved states outside final matrices."""

from collections import defaultdict
import numpy as np


def matrix_state(items, features, records):
    ii = {x["item_id"]: n for n, x in enumerate(items)}
    ff = {x["feature_id"]: n for n, x in enumerate(features)}
    if len(ii) != len(items) or len(ff) != len(features):
        raise ValueError("Duplicate global matrix identity")
    matrix = np.full((len(items), len(features)), -1, dtype=np.int8)
    evidence = defaultdict(list)
    for r in records:
        key = r["item_id"], r["feature_id"]
        if key[0] not in ii or key[1] not in ff:
            raise ValueError("Judgment outside global scope")
        evidence[key].append(r)
    unresolved = []
    for item in items:
        for f in features:
            key = item["item_id"], f["feature_id"]
            rows = evidence.get(key, [])
            resolved = [
                r
                for r in rows
                if r.get("parse_status") == "ok" and r.get("status") == "ok"
            ]
            if any(
                type(r.get("applies")) is not int or r["applies"] not in (0, 1)
                for r in resolved
            ):
                raise ValueError("Invalid non-binary resolved global judgment")
            values = {r["applies"] for r in resolved}
            if len(values) == 1:
                matrix[ii[key[0]], ff[key[1]]] = next(iter(values))
                continue
            reason = (
                "conflicting_resolved_evidence"
                if len(values) > 1
                else (
                    "ambiguous"
                    if any(r.get("status") == "ambiguous" for r in rows)
                    else "invalid" if rows else "missing"
                )
            )
            unresolved.append(dict(item_id=key[0], feature_id=key[1], reason=reason))
    return dict(
        matrix=matrix,
        planned_cells=len(items) * len(features),
        resolved_cells=int((matrix >= 0).sum()),
        unresolved=unresolved,
        complete=not unresolved,
        item_ids=list(ii),
        feature_ids=list(ff),
    )

"""Descriptive task views that preserve unresolved cells and full vocabulary."""

from collections import defaultdict
import numpy as np


def task_audit(items, features, matrix):
    a = np.asarray(matrix)
    if a.shape != (len(items), len(features)) or not np.isin(a, [-1, 0, 1]).all():
        raise ValueError("Task scope or ternary matrix is invalid")
    rows = []
    for n, item in enumerate(items):
        yes = int((a[n] == 1).sum())
        no = int((a[n] == 0).sum())
        unknown = int((a[n] < 0).sum())
        rows.append(
            dict(
                item_id=item["item_id"],
                task_id=item.get("task_id"),
                level=item.get("level"),
                applicable_observed=yes,
                inapplicable=no,
                unresolved=unknown,
                applicable_total=yes if not unknown else None,
            )
        )
    return dict(
        n_items=len(items),
        n_features=len(features),
        planned_cells=int(a.size),
        resolved_cells=int((a >= 0).sum()),
        applicable_observed=int((a == 1).sum()),
        inapplicable=int((a == 0).sum()),
        unresolved_cells=int((a < 0).sum()),
        items=rows,
    )


def pattern_groups(matrix, features):
    a = np.asarray(matrix)
    if a.ndim != 2 or a.shape[1] != len(features) or not np.isin(a, [-1, 0, 1]).all():
        raise ValueError("Invalid task pattern matrix")
    complete = (a >= 0).all(axis=0)
    positive = (a == 1).any(axis=0)
    groups = defaultdict(list)
    for n, f in enumerate(features):
        if complete[n] and positive[n]:
            groups[tuple(map(int, a[:, n]))].append(f)
    rows = []
    for pattern, members in groups.items():
        members = sorted(members, key=lambda f: f["feature_id"])
        rows.append(
            dict(
                pattern=list(pattern),
                n_features=len(members),
                feature_ids=[f["feature_id"] for f in members],
                texts=[f["text"] for f in members],
            )
        )
    rows.sort(key=lambda r: (-r["n_features"], r["pattern"]))
    return dict(
        complete_columns=int(complete.sum()),
        positive_complete_columns=int((complete & positive).sum()),
        all_zero_columns=int((complete & ~positive).sum()),
        unknown_columns=int((~complete).sum()),
        groups=rows,
        semantic_merge=False,
        interpretation="Identical binary profiles on these 15 items; these are co-applicability groups, not semantic equivalences or an inferred abstraction hierarchy. Complete-case selection is potentially biased.",
    )


def select_examples(items, features, matrix, origins, *, per_item=2):
    """Display selection only: known-positive examples discovered for each item."""
    lookup = {f["feature_id"]: k for k, f in enumerate(features)}
    origin = defaultdict(set)
    for o in origins:
        origin[o["item_id"]].add(o["canonical_id"])
    selected = []
    seen = set()
    for n, item in enumerate(items):
        candidates = [
            lookup[f]
            for f in origin[item["item_id"]]
            if f in lookup and matrix[n, lookup[f]] == 1 and f not in seen
        ]
        candidates.sort(
            key=lambda k: (
                -int((matrix[:, k] >= 0).sum()),
                int((matrix[:, k] == 1).sum()),
                features[k]["text"],
            )
        )
        for k in candidates[:per_item]:
            selected.append(
                dict(
                    feature_id=features[k]["feature_id"],
                    proposition=features[k]["text"],
                    display_origin=item["item_id"],
                )
            )
            seen.add(features[k]["feature_id"])
    return selected


def inheritance_flags(items, features, matrix, task_feature_ids):
    """Screen parent-positive/child-negative cells for interpretation, not repair."""
    index = {i["item_id"]: n for n, i in enumerate(items)}
    rows = []
    for n, item in enumerate(items):
        if (
            item.get("target_type") != "situated_task"
            or item.get("parent_id") not in index
        ):
            continue
        parent = index[item["parent_id"]]
        if not item["text"].startswith(items[parent]["text"]):
            continue
        for k, f in enumerate(features):
            if (
                f["feature_id"] in task_feature_ids
                and matrix[parent, k] == 1
                and matrix[n, k] == 0
            ):
                rows.append(
                    dict(
                        parent_id=item["parent_id"],
                        situated_id=item["item_id"],
                        feature_id=f["feature_id"],
                        proposition=f["text"],
                        parent_applies=1,
                        situated_applies=0,
                        review_required=True,
                        confirmed_error=False,
                        interpretation="Inherited-text disagreement; inspect unchanged explicit requirements first. Some context-sensitive properties can legitimately change. No label has been corrected.",
                    )
                )
    return rows

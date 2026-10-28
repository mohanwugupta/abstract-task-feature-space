"""Independent task analyses and explicitly provisional interpretation metadata."""

from collections import Counter
from difflib import SequenceMatcher
import re
import numpy as np
from ..full.analysis import (
    composition_report,
    task_contrast,
    discovery_report,
)
from ..task_focus.analysis import pattern_groups, inheritance_flags

FAMILY_RULES = {
    "Goals and outputs": r"\b(goal|purpose|output|deliver|produc|creat|compose|report|poem|reconcil|resolve|result|artifact|objective|completion)\w*\b",
    "Operations and information requirements": r"\b(calculat|compar|match|check|inspect|read|record|input|information|data|arithmet|sum|identif|infer|select|sort|allocat|track|verif|payment|invoice|interpret|revision|edit|process|prepar)\w*\b",
    "Constraints and resources": r"\b(constraint|requir|must|limit|deadline|time|minute|resource|format|length|line|rhyme|currency|cent|exactly|fixed|restrict|available|material|rule|within|cannot|prohibit)\w*\b",
    "Quality and evaluation criteria": r"\b(quality|evaluat|accurac|accurate|correct|error|review|judge|assess|score|rubric|criterion|criteria|coheren|originality|creativity|aesthetic|success|compliance|precision|audit|ambigu|valid|consisten|reliab|complete)\w*\b",
    "Social context and beneficiaries": r"\b(user|client|customer|audience|reader|reviewer|evaluator|external|beneficiar|social|communicat|human|collaborat|recipient|stakeholder|requester)\w*\b",
    "Relationships and trade-offs": r"\b(if|when|unless|because|caus|depend|trade.?off|balance|conflict|interact|tension|compromis|priorit|under|despite|whereas|while|affect|increase|decrease|reduce|pressure|both|between)\w*\b",
}
SPECIFIC = r"\b\d+\b|\b(five|eight|nineteen|abab|cdcd|wet pavement|streetlight|invoice.?id|payment.?id|integer cents)\b"
LINGUISTIC = r"\b(the word|word itself|syllable|spelling|letters? long|grammatical|lexical|target.{0,20}label|labeled|labelled)\b"
OPERATION = r"\b(requir|involv|produc|compar|check|match|allocat|creat|generat|evaluat|communicat|verif|depend|can|may|goal|constraint)\w*\b"


def annotate_features(bank):
    rows = []
    for f in bank["features"]:
        text = f["text"].lower()
        hits = {
            label: sorted(set(re.findall(pattern, text)))
            for label, pattern in FAMILY_RULES.items()
        }
        families = [label for label, matches in hits.items() if matches]
        linguistic = bool(re.search(LINGUISTIC, text))
        specific = bool(re.search(SPECIFIC, text))
        functional = (
            bool(re.search(OPERATION, text)) and not linguistic and not specific
        )
        specificity = (
            "linguistic_or_label_metadata"
            if linguistic
            else (
                "instance_detail_candidate"
                if specific
                else (
                    "functional_or_general_candidate"
                    if functional
                    else "scope_requires_review"
                )
            )
        )
        evaluative = (
            bool(
                set(families)
                & {
                    "Quality and evaluation criteria",
                    "Constraints and resources",
                    "Relationships and trade-offs",
                }
            )
            and not linguistic
        )
        rows.append(
            dict(
                feature_id=f["feature_id"],
                proposition=f["text"],
                semantic_families=families,
                specificity=specificity,
                generalizable_functional_candidate=functional,
                evaluatively_relevant_candidate=evaluative,
                potential_irrelevance=linguistic,
                annotation_method="offline_provisional_rule_coding",
                rule_evidence={k: v for k, v in hits.items() if v},
                human_reviewed=False,
                semantic_equivalence=False,
                preference_value=None,
                behavioral_validation_required=True,
            )
        )
    return rows


def _transitions(items, features, matrix):
    im = {i["item_id"]: n for n, i in enumerate(items)}
    rows = []
    for n, child in enumerate(items):
        pid = child.get("parent_id")
        if pid not in im:
            continue
        parent = items[im[pid]]
        for k, f in enumerate(features):
            a, b = int(matrix[im[pid], k]), int(matrix[n, k])
            change = (
                "unresolved"
                if min(a, b) < 0
                else (
                    "retained"
                    if a == b == 1
                    else (
                        "gained"
                        if a == 0 and b == 1
                        else (
                            "lost_or_inconsistent"
                            if a == 1 and b == 0
                            else "absent_both"
                        )
                    )
                )
            )
            rows.append(
                dict(
                    parent_id=pid,
                    child_id=child["item_id"],
                    task_id=child.get("task_id"),
                    parent_level=parent.get("level"),
                    child_level=child.get("level"),
                    feature_id=f["feature_id"],
                    proposition=f["text"],
                    parent_applies=a if a >= 0 else "?",
                    child_applies=b if b >= 0 else "?",
                    change=change,
                    unchanged_parent_text=child["text"].startswith(parent["text"]),
                )
            )
    return rows


def _family_features(items, features, matrix, annotations):
    groups = {
        task: [n for n, i in enumerate(items) if i.get("task_id") == task]
        for task in ("poem", "finance")
    }
    rows = []
    for k, f in enumerate(features):
        states = {}
        for task, ns in groups.items():
            vals = matrix[ns, k]
            states[task] = (
                "present"
                if (vals == 1).any()
                else "absent" if len(ns) and (vals == 0).all() else "unresolved"
            )
        a, b = states["poem"], states["finance"]
        category = (
            "shared"
            if a == b == "present"
            else (
                "poem_distinctive"
                if a == "present" and b == "absent"
                else (
                    "finance_distinctive"
                    if b == "present" and a == "absent"
                    else "neither" if a == b == "absent" else "unresolved"
                )
            )
        )
        rows.append(
            dict(
                feature_id=f["feature_id"],
                proposition=f["text"],
                poem_family=a,
                finance_family=b,
                comparison=category,
                generalizable_candidate=category == "shared"
                and annotations[k]["generalizable_functional_candidate"],
                definition="Family presence: at least one positive item. Family absence: all six task variants independently zero. Unknowns prevent a distinctiveness claim.",
            )
        )
    return rows


def analyze_tasks(items, bank, generations, jobs, state, *, replicates=3):
    features = bank["features"]
    a = state["matrix"]
    if a.shape != (len(items), len(features)) or not np.isin(a, [-1, 0, 1]).all():
        raise ValueError("Task-only matrix dimensions or values invalid")
    annotations = annotate_features(bank)
    compositions = composition_report(
        items, bank, generations, state, replicates=replicates
    )
    im = {i["item_id"]: n for n, i in enumerate(items)}
    ff = {f["feature_id"]: n for n, f in enumerate(features)}
    texts = {f["feature_id"]: f["text"] for f in features}
    composition_features = []
    for r in compositions:
        r["counts"] = {
            k: len(r[k])
            for k in (
                "recovered_both",
                "component_only_discovered",
                "direct_only_discovered",
                "constituent_discovery_gaps",
                "interaction_candidates",
                "unresolved_features",
            )
        }
        for fid in sorted(set(r["component_bank"]) | set(r["direct_bank"])):
            values = {
                role: int(a[im[iid], ff[fid]])
                for role, iid in (
                    ("T", r["task_item_id"]),
                    ("S", r["situation_id"]),
                    ("TS", r["item_id"]),
                )
            }
            if fid in r["recovered_both"]:
                category = "shared_discovery"
            elif fid in r["component_only_discovered"]:
                category = "component_only_discovery"
            elif fid in r["constituent_discovery_gaps"]:
                category = "constituent_omission"
            elif fid in r["interaction_candidates"]:
                category = "interaction_pattern_candidate"
            elif values["TS"] == 0:
                category = "direct_only_inapplicable"
            else:
                category = "direct_only_unresolved"
            composition_features.append(
                dict(
                    item_id=r["item_id"],
                    task_id=r["task_id"],
                    prompt_family=r["prompt_family"],
                    sampling_mode=r["sampling_mode"],
                    feature_id=fid,
                    proposition=texts[fid],
                    category=category,
                    **{role: (v if v >= 0 else "?") for role, v in values.items()},
                    relational_text_candidate=bool(
                        re.search(
                            FAMILY_RULES["Relationships and trade-offs"], texts[fid]
                        )
                    ),
                    generation_coverage_complete=r["generation_coverage_complete"],
                    confirmed_interaction=False
                )
            )
    transitions = _transitions(items, features, a)
    transition_counts = []
    for parent, child in dict.fromkeys(
        (r["parent_id"], r["child_id"]) for r in transitions
    ):
        rows = [
            r
            for r in transitions
            if r["parent_id"] == parent and r["child_id"] == child
        ]
        transition_counts.append(
            dict(
                parent_id=parent,
                child_id=child,
                counts=dict(Counter(r["change"] for r in rows)),
            )
        )
    family_features = _family_features(items, features, a, annotations)
    return dict(
        annotations=annotations,
        compositions=compositions,
        composition_features=composition_features,
        transitions=transitions,
        transition_counts=transition_counts,
        family_features=family_features,
        family_counts=dict(Counter(r["comparison"] for r in family_features)),
        instance_contrast=task_contrast(items, features, state),
        pattern_groups=pattern_groups(a, features),
        inheritance_flags=inheritance_flags(items, features, a, set(ff)),
        discovery_curves=discovery_report(items, bank, generations, jobs, state),
    )


def diagnostic_audit(items, bank, state, analysis, *, per_category=4):
    """A small balanced screening sample; zero automatic corrections or approvals."""
    annotations = {r["feature_id"]: r for r in analysis["annotations"]}
    features = bank["features"]
    im = {i["item_id"]: n for n, i in enumerate(items)}
    fm = {f["feature_id"]: k for k, f in enumerate(features)}
    flags = []

    def add(category, fid, iid, reason, **extra):
        flags.append(
            dict(
                category=category,
                feature_id=fid,
                proposition=features[fm[fid]]["text"],
                item_id=iid,
                source_text=items[im[iid]]["text"] if iid in im else "",
                reason=reason,
                review_status="suspected_or_descriptive_requires_review",
                verified_correction=False,
                original_labels_preserved=True,
                **extra
            )
        )

    for r in analysis["inheritance_flags"]:
        add(
            "inherited_requirement_disagreement",
            r["feature_id"],
            r["situated_id"],
            r["interpretation"],
            parent_id=r["parent_id"],
        )
    candidate_seen = set()
    for r in analysis["composition_features"]:
        key = r["item_id"], r["feature_id"]
        if r["category"] != "interaction_pattern_candidate" or key in candidate_seen:
            continue
        candidate_seen.add(key)
        child = items[im[r["item_id"]]]
        parent = items[im[child["parent_id"]]]
        situation = items[im[child["situation_ids"][0]]]
        add(
            "interaction_candidate_grounding",
            r["feature_id"],
            r["item_id"],
            "A 0/0/1 pattern may reflect instruction binding, a component false negative or a true conjunction. Inspect all three descriptions before interpreting novelty.",
            parent_text=parent["text"],
            situation_text=situation["text"],
            T=r["T"],
            S=r["S"],
            TS=r["TS"],
        )
    for f in features:
        fid = f["feature_id"]
        text = f["text"].lower()
        annotation = annotations[fid]
        for n, i in enumerate(items):
            if state["matrix"][n, fm[fid]] != 1:
                continue
            if (
                i.get("task_id") == "poem"
                and re.search(
                    r"\b(invoice|payment|reconciliation|currency|financial|bank statements?|cash flow)\b",
                    text,
                )
                or i.get("task_id") == "finance"
                and re.search(
                    r"\b(rhyme|poem|poetry|poetic|stanza|imagery|metaphor|meter|creative.production)\b",
                    text,
                )
            ):
                add(
                    "cross_domain_positive",
                    fid,
                    i["item_id"],
                    "Lexical domain mismatch with a positive judgment; inspect actual grounding rather than automatically rejecting",
                )
        if annotation["potential_irrelevance"]:
            add(
                "linguistic_or_label_property",
                fid,
                "",
                "May describe a word or dataset label rather than a task demand",
            )
        if annotation["specificity"] == "instance_detail_candidate":
            origins = [o for o in bank["origins"] if o["canonical_id"] == fid]
            add(
                "explicit_instruction_or_instance_detail",
                fid,
                origins[0]["item_id"] if origins else "",
                "Concrete number, format or instance cue; can be valid but generalization and preference relevance are unverified",
            )
        if not annotation["evaluatively_relevant_candidate"]:
            add(
                "preference_relevance_unclear",
                fid,
                "",
                "Descriptive feature without a declared evaluation cue; this does not establish irrelevance",
            )
    # Suggest close text pairs without semantic merging or changing any binary cell.
    tokens = {f["feature_id"]: set(re.findall(r"\w+", f["text"])) for f in features}
    for k, f in enumerate(features):
        for g in features[k + 1 :]:
            x, y = tokens[f["feature_id"]], tokens[g["feature_id"]]
            if (
                len(x & y) / max(1, len(x | y)) >= 0.7
                and SequenceMatcher(None, f["text"], g["text"]).ratio() >= 0.76
            ):
                add(
                    "near_duplicate_review",
                    f["feature_id"],
                    "",
                    "Text similarity only; preserve negation, modality, conditions and argument roles",
                    other_feature_id=g["feature_id"],
                    other_proposition=g["text"],
                )
    sample = []
    for category in sorted({r["category"] for r in flags}):
        candidates = sorted(
            (r for r in flags if r["category"] == category),
            key=lambda r: (
                not bool(re.search(r"lines 1 and 3 must rhyme", r["proposition"])),
                r["proposition"],
                r["item_id"],
            ),
        )
        sample.extend(candidates[:per_category])
    return dict(
        all_flags=flags,
        diagnostic_set=sample,
        counts=dict(Counter(r["category"] for r in flags)),
        verified_corrections=[],
        manual_reviews_completed=0,
        interpretation="Screening and descriptive coding only. Counts are flags, not error rates. No source or matrix value is altered.",
    )


def behavioral_candidates(rows, annotations, flags):
    """Prioritize interpretable unflagged candidates; never infer preference values."""
    am = {r["feature_id"]: r for r in annotations}
    risky = {
        r["feature_id"]
        for r in flags
        if r["category"]
        in {
            "cross_domain_positive",
            "inherited_requirement_disagreement",
            "linguistic_or_label_property",
        }
    }
    candidates = []
    for r in rows:
        annotation = am[r["feature_id"]]
        if (
            r["feature_id"] in risky
            or not annotation["evaluatively_relevant_candidate"]
            or not annotation["generalizable_functional_candidate"]
            or r["comparison"]
            not in {"shared", "poem_distinctive", "finance_distinctive"}
        ):
            continue
        if not re.search(
            r"\b(quality|evaluat|review|accurac|accurate|correct|deadline|time|resource|constraint|trade.?off|error|check|match|information|audience)\w*\b",
            r["proposition"],
        ):
            continue
        candidates.append(dict(r))
    return sorted(
        candidates,
        key=lambda r: (
            r["comparison"] != "shared",
            len(r["proposition"]),
            r["proposition"],
        ),
    )

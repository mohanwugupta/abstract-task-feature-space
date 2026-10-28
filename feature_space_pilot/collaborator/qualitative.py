"""Transparent descriptive cue coding, with full raw repetitions for interpretation."""

from collections import defaultdict
from difflib import SequenceMatcher
import re
import numpy as np
from ..full.bank import protected_compatible
from ..prompts import normalize_phrase

LINGUISTIC = r"\b(word|noun|verb|phrase|determiner|syllable|spelling|grammatical|singular|plural|intransitiv\w*|transitiv\w*|infinitive|lexical|definite article|target.{0,25}label\w*|labelled|labeled)\b"
CUES = {
    "operational_or_relational": r"\b(match|calculat|check|compar|review|revis|edit|sum|sort|reconcil|identif|record|require|depend|caus|between|if|when|unless|process|produce|develop|create)\w*\b",
    "resource_demands": r"\b(time|deadline|minute|money|cost|budget|resource|effort|tool|material|within|capacity|workload)\w*\b",
    "incentives": r"\b(incentive|reward|penalt|bonus|profit|compensation|payment for|reputation|punish)\w*\b",
    "uncertainty": r"\b(uncertain|ambigu|unpredictab|unknown|risk|probabil|unclear)\w*\b",
    "conflict_or_tradeoff": r"\b(trade.?off|conflict|compromis|competing|tension|versus|sacrifice|balance between|force a trade)\w*\b",
    "consequences": r"\b(consequence|lead to|result in|affect|cause|prevent|increase|decrease|reduce|impact)\w*\b",
    "evaluation_feedback": r"\b(evaluat|assess|review|feedback|quality|criteria|criterion|rubric|correct|accurac|accurate|traceab|score)\w*\b",
    "social_context": r"\b(audience|client|user|reviewer|recipient|stakeholder|collaborat|beneficiar|communicat)\w*\b",
}
STOP = {
    "a",
    "an",
    "the",
    "it",
    "is",
    "are",
    "be",
    "has",
    "have",
    "of",
    "to",
    "in",
    "and",
    "or",
    "for",
    "as",
    "with",
    "that",
    "this",
    "its",
    "can",
    "may",
    "must",
    "will",
    "by",
    "on",
    "from",
    "at",
    "than",
    "each",
}


def linguistic_feature_mask(features):
    return np.array(
        [
            any(
                re.search(LINGUISTIC, t.lower())
                for t in [f["text"]] + f.get("source_phrases", [])
            )
            for f in features
        ],
        dtype=bool,
    )


def code_proposition(proposition, target_text):
    text = proposition.lower()
    target = target_text.lower()
    codes = {key: bool(re.search(pattern, text)) for key, pattern in CUES.items()}
    linguistic = bool(re.search(LINGUISTIC, text))
    specific = bool(
        re.search(
            r"\b\d+\b|\b(inv-\d+|pay-\d+|abab|cdcd|five|eight|twenty|nineteen)\b", text
        )
    )
    words = [w for w in re.findall(r"\w+", text) if w not in STOP]
    tw = set(re.findall(r"\w+", target))
    overlap = sum(w in tw for w in words) / len(words) if words else 0
    restatement = bool(
        (len(text) > 12 and text.rstrip(".") in target)
        or overlap >= 0.6
        and len(words) >= 3
        or overlap >= 0.5
        and re.search(
            r"\b(asks|requests|requires|specifies|must|lists|names|requested|the text)\b",
            text,
        )
    )
    codes.update(
        linguistic_or_label_metadata=linguistic,
        instance_detail=specific,
        instruction_restatement_candidate=restatement,
        generalizable_functional_candidate=any(codes.values())
        and not linguistic
        and not specific,
        unsupported_verified=False,
        annotation_method="offline_lexical_screening_not_semantic_gold",
    )
    return codes


def qualitative_comparison(items, local, generations, jobs):
    im = {i["item_id"]: i for i in items}
    accepted = {g["job_id"]: g for g in generations if g.get("parse_status") == "ok"}
    origin = {
        iid: {
            (o["generation_job_id"], o["raw_feature_index"]): o
            for o in t["bank"]["origins"]
        }
        for iid, t in local["targets"].items()
    }
    verdict = {
        iid: {j["feature_id"]: j for j in t["judgments"]}
        for iid, t in local["targets"].items()
    }
    raw = []
    calls = []
    near = []
    metrics = list(CUES) + [
        "linguistic_or_label_metadata",
        "instance_detail",
        "instruction_restatement_candidate",
        "generalizable_functional_candidate",
    ]
    for job in jobs:
        iid = job["item_id"]
        if iid not in im:
            continue
        g = accepted.get(job["job_id"])
        call = dict(
            item_id=iid,
            prompt_family=job["prompt_family"],
            replicate=job["replicate"],
            level=im[iid].get("level"),
            task_id=im[iid].get("task_id"),
            generation_job_id=job["job_id"],
            accepted=g is not None,
            raw_features=None,
            canonical_features=None,
            **{m: None for m in metrics}
        )
        if g is None:
            calls.append(call)
            continue
        codes = []
        ids = set()
        for k, text in enumerate(g["features"]):
            o = origin[iid][g["job_id"], k]
            j = verdict[iid][o["canonical_id"]]
            c = code_proposition(text, im[iid]["text"])
            codes.append(c)
            ids.add(o["canonical_id"])
            raw.append(
                dict(
                    item_id=iid,
                    target=im[iid]["text"],
                    level=im[iid].get("level"),
                    task_id=im[iid].get("task_id"),
                    prompt_family=g["prompt_family"],
                    replicate=g["replicate"],
                    raw_feature_index=k,
                    proposition=text,
                    feature_id=o["canonical_id"],
                    local_applies=j["applies"],
                    local_status=j["status"],
                    source_run_id=o["source_run_id"],
                    source_record_hash=o["source_record_hash"],
                    generation_job_id=g["job_id"],
                    generation_model=g.get("generation_model"),
                    generation_prompt_hash=g.get("prompt_hash"),
                    **c
                )
            )
        call.update(
            raw_features=len(codes),
            canonical_features=len(ids),
            within_call_canonical_redundancy=len(codes) - len(ids),
            **{
                m: sum(c[m] for c in codes) / len(codes) if codes else None
                for m in metrics
            }
        )
        calls.append(call)
    target_summaries = []
    patterns = []
    for iid in im:
        rs = [r for r in raw if r["item_id"] == iid]
        byfeature = defaultdict(lambda: defaultdict(set))
        for r in rs:
            byfeature[r["feature_id"]][r["prompt_family"]].add(r["replicate"])
        for fid, families in byfeature.items():
            example = next(r["proposition"] for r in rs if r["feature_id"] == fid)
            patterns.append(
                dict(
                    item_id=iid,
                    feature_id=fid,
                    proposition=example,
                    **{
                        f + "_repetitions": len(families.get(f, set()))
                        for f in ("P0", "P1", "P2")
                    },
                    observed_prompt_count=len(families),
                    all_three_observed=len(families) == 3,
                    interpretation="Discovery presence, not proof of exclusive elicitation or applicability elsewhere"
                )
            )
        for f in ("P0", "P1", "P2"):
            cs = [c for c in calls if c["item_id"] == iid and c["prompt_family"] == f]
            observed = [c for c in cs if c["accepted"]]
            fs = {r["feature_id"] for r in rs if r["prompt_family"] == f}
            values = [verdict[iid][fid]["applies"] for fid in fs]
            target_summaries.append(
                dict(
                    item_id=iid,
                    prompt_family=f,
                    level=im[iid].get("level"),
                    planned_calls=len(cs),
                    accepted_calls=len(observed),
                    coverage_complete=len(observed) == len(cs) == 3,
                    observed_canonical_features=len(fs),
                    raw_features=sum(c["raw_features"] for c in observed),
                    local_yes=sum(v == 1 for v in values),
                    local_no=sum(v == 0 for v in values),
                    local_unknown=sum(v is None for v in values),
                    **{
                        m: (
                            float(np.mean([c[m] for c in observed if c[m] is not None]))
                            if any(c[m] is not None for c in observed)
                            else None
                        )
                        for m in metrics
                    }
                )
            )
        distinct = {}
        for r in rs:
            distinct.setdefault(
                (r["feature_id"], normalize_phrase(r["proposition"])), r
            )
        ds = list(distinct.values())
        for n, a in enumerate(ds):
            x = set(re.findall(r"\w+", a["proposition"].lower()))
            for b in ds[n + 1 :]:
                if a["feature_id"] == b["feature_id"]:
                    continue
                y = set(re.findall(r"\w+", b["proposition"].lower()))
                score = len(x & y) / max(1, len(x | y))
                if (
                    score >= 0.7
                    and SequenceMatcher(
                        None, a["proposition"].lower(), b["proposition"].lower()
                    ).ratio()
                    >= 0.76
                ):
                    near.append(
                        dict(
                            item_id=iid,
                            left=a["proposition"],
                            right=b["proposition"],
                            left_prompt=a["prompt_family"],
                            right_prompt=b["prompt_family"],
                            left_job=a["generation_job_id"],
                            right_job=b["generation_job_id"],
                            lexical_jaccard=score,
                            protected_meaning_difference=not protected_compatible(
                                a["proposition"], b["proposition"]
                            ),
                            semantic_equivalence_confirmed=False,
                        )
                    )
    cohort = sorted(
        iid
        for iid in im
        if all(
            next(
                r
                for r in target_summaries
                if r["item_id"] == iid and r["prompt_family"] == f
            )["coverage_complete"]
            for f in ("P0", "P1", "P2")
        )
    )
    summaries = []
    for f in ("P0", "P1", "P2"):
        rs = [
            r
            for r in target_summaries
            if r["prompt_family"] == f and r["item_id"] in cohort
        ]
        summaries.append(
            dict(
                prompt_family=f,
                n_complete_paired_targets=len(rs),
                mean_canonical_features=(
                    float(np.mean([r["observed_canonical_features"] for r in rs]))
                    if rs
                    else None
                ),
                **{
                    m: (
                        float(np.mean([r[m] for r in rs if r[m] is not None]))
                        if any(r[m] is not None for r in rs)
                        else None
                    )
                    for m in metrics
                }
            )
        )
    reviews = []
    seen = set()
    categories = {
        "judge_rejected": lambda r: r["local_applies"] == 0,
        "unresolved": lambda r: r["local_applies"] is None,
        "label_metadata": lambda r: r["linguistic_or_label_metadata"],
        "instance_detail": lambda r: r["instance_detail"],
        "instruction_restatement": lambda r: r["instruction_restatement_candidate"],
        "resource_demands": lambda r: r["resource_demands"],
        "uncertainty_or_conflict": lambda r: r["uncertainty"]
        or r["conflict_or_tradeoff"],
        "consequences_or_incentives": lambda r: r["consequences"] or r["incentives"],
    }
    for label, predicate in categories.items():
        selected = []
        for f in ("P0", "P1", "P2"):
            candidates = [r for r in raw if r["prompt_family"] == f and predicate(r)]
            candidates.sort(
                key=lambda r: (r["item_id"], r["replicate"], len(r["proposition"]))
            )
            for r in candidates:
                key = r["item_id"], r["generation_job_id"], r["raw_feature_index"]
                if key in seen:
                    continue
                reviews.append(
                    dict(
                        r,
                        selection_reason=label,
                        manual_review_status="pending",
                        reviewer="",
                        verified_correction=False,
                    )
                )
                seen.add(key)
                break
    return dict(
        raw_features=raw,
        call_metrics=calls,
        target_prompt_metrics=target_summaries,
        prompt_summary=summaries,
        fixed_complete_target_ids=cohort,
        prompt_feature_patterns=patterns,
        near_duplicate_pairs=near,
        manual_review=reviews,
        methods="Cue proportions are descriptive lexical screening, averaged within calls, then within target and across the same fully sampled targets. They are not semantic gold or independent feature observations. Local judge negatives/unknowns are review evidence, not verified errors.",
    )

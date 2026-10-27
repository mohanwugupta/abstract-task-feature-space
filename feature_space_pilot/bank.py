"""Exact proposition equivalence; uncertain semantic merges require explicit human review."""

from __future__ import annotations
import difflib
import re
from .io import digest
from .prompts import normalize_phrase


def protected_signature(text):
    """Conservative guard; a reviewed decision never overrides a known semantic distinction."""
    words = normalize_phrase(text).split()
    protected = {
        "not",
        "never",
        "no",
        "may",
        "can",
        "must",
        "should",
        "if",
        "unless",
        "when",
        "user",
        "agent",
        "assistant",
        "reviewer",
        "customer",
        "participant",
        "model",
    }
    return tuple(w for w in words if w in protected)


def consolidate(records, *, cap=200, merges=()):
    if type(cap) is not int or cap < 1:
        raise ValueError("Candidate cap must be positive")
    candidates = []
    phrases = {}
    for record in records:
        for index, phrase in enumerate(record["features"]):
            normalized = normalize_phrase(phrase)
            phrases.setdefault(normalized, normalized)
            candidates.append(
                dict(
                    phrase=phrase,
                    normalized=normalized,
                    job_id=record["job_id"],
                    origin_item=record["item_id"],
                    prompt_family=record["prompt_family"],
                    replicate=record["replicate"],
                    raw_feature_index=index,
                )
            )
    mapping = {p: p for p in phrases}
    for merge in merges:
        a, b = normalize_phrase(merge["left"]), normalize_phrase(merge["right"])
        if a not in phrases or b not in phrases:
            raise ValueError("Unknown reviewed merge phrase")
        if merge.get("decision") != "merge":
            continue
        if not merge.get("reviewer"):
            raise ValueError("Semantic merges require a named reviewer")
        if protected_signature(a) != protected_signature(b):
            raise ValueError("Merge changes role, negation, modality or condition")
        canonical = min(mapping[a], mapping[b])
        old = {mapping[a], mapping[b]}
        mapping = {k: canonical if v in old else v for k, v in mapping.items()}
    texts = sorted(set(mapping.values()), key=lambda p: (digest(p), p))
    selected = texts[
        :cap
    ]  # preregistered hash ordering; no human match or judge labels
    inventory_hash = digest(selected)
    ids = {p: "f_" + digest(p)[:20] for p in texts}
    bank = [
        dict(
            feature_id=ids[p],
            text=p,
            cluster_id=ids[p],
            inventory_hash=inventory_hash,
            merge_decisions=[
                m for m in merges if mapping.get(normalize_phrase(m["left"])) == p
            ],
        )
        for p in selected
    ]
    for c in candidates:
        c.update(
            feature_id=ids[mapping[c["normalized"]]],
            selected=mapping[c["normalized"]] in selected,
            normalization="NFC_casefold_space_terminal_period_v1",
        )
    review = []
    # Bound quadratic ambiguity review to the frozen inventory; every excluded
    # phrase remains in the candidate provenance table for coverage accounting.
    ordered = sorted(p for p in phrases if mapping[p] in selected)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if (
                difflib.SequenceMatcher(None, a, b).ratio() >= 0.72
                and mapping[a] != mapping[b]
            ):
                review.append(
                    dict(
                        left=a,
                        right=b,
                        decision="pending",
                        reviewer="",
                        reason="similar wording; no automatic semantic merge",
                    )
                )
    return bank, candidates, review

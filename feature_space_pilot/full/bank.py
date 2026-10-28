"""Complete vocabulary, narrow explicit equivalences and conservative reviewed clusters."""

from collections import Counter, defaultdict
import heapq
import json
import re
from ..io import digest
from ..prompts import normalize_phrase

VERSION = "complete_bank_v1"
RULES = {"needs ": "requires ", "is made up of ": "is composed of "}
CONTRACTIONS = {
    "isn't": "is not",
    "aren't": "are not",
    "doesn't": "does not",
    "don't": "do not",
    "can't": "cannot",
    "mustn't": "must not",
    "shouldn't": "should not",
    "won't": "will not",
}


def proof_key(text):
    text = normalize_phrase(text).replace("’", "'")
    for a, b in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(a) + r"\b", b, text)
    for a, b in RULES.items():
        if text.startswith(a):
            return b + text[len(a) :]
    return text


def protected_signature(text):
    text = proof_key(text)
    words = re.findall(r"[^\W_]+", text)
    negation = tuple(
        w for w in words if w in {"no", "not", "never", "cannot", "without", "neither"}
    )
    modality = tuple(
        w
        for w in words
        if w
        in {
            "can",
            "cannot",
            "may",
            "must",
            "should",
            "could",
            "might",
            "would",
            "will",
            "always",
            "typically",
            "often",
            "sometimes",
        }
    )
    condition = tuple(
        re.findall(r"\b(if|unless|when|because|provided that|due to|therefore)\b", text)
    )
    relation = re.search(
        r"\b(requires|rewards|rewarded|causes|caused|cause|prevents|prevented|conflicts|depends|implies|enables)\b",
        text,
    )
    families = {
        "rewarded": "rewards",
        "caused": "causes",
        "cause": "causes",
        "prevented": "prevents",
    }
    edge = families.get(relation.group(), relation.group()) if relation else None
    role_matches = list(
        re.finditer(
            r"\b(user|assistant|agent|reviewer|customer|participant|model|client|employer|employee)\b",
            text,
        )
    )
    roles = tuple(
        (m.group(), m.start() < relation.start() if relation else m.start() == 0)
        for m in role_matches
    )
    numbers = tuple(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    connectives = tuple(w for w in words if w in {"and", "or"})
    return negation, modality, condition, edge, roles, numbers, connectives


def protected_compatible(left, right):
    if protected_signature(left) != protected_signature(right):
        return False

    def arguments(text):
        text = proof_key(text)
        match = re.search(
            r"\b(requires|rewards|causes|prevents|depends on|implies|enables)\b", text
        )
        if not match:
            return None
        return text[: match.start()].strip(), text[match.end() :].strip()

    a, b = arguments(left), arguments(right)
    # A named semantic review may support paraphrased arguments, but cannot
    # silently reverse an observed subject/object pair.
    if a and b and a[0] and a[1] and a[0] != a[1] and a == (b[1], b[0]):
        return False
    return True


def build_bank(generations, *, reviews=()):
    origins = []
    excluded = []
    phrases = {}
    keys = {}
    for record in generations:
        if record.get("parse_status") != "ok":
            try:
                rejected = json.loads(record.get("raw_response", ""))
            except (TypeError, ValueError):
                rejected = {}
            candidates = (
                rejected.get("features", []) if isinstance(rejected, dict) else []
            )
            for n, phrase in enumerate(
                candidates if isinstance(candidates, list) else []
            ):
                excluded.append(
                    dict(
                        source_run_id=record.get("source_run_id"),
                        generation_job_id=record["job_id"],
                        item_id=record["item_id"],
                        attempt=record.get("attempt"),
                        raw_feature_index=n,
                        phrase=phrase,
                        exclusion_reason="Original response failed the frozen generation protocol: "
                        + str(record.get("error")),
                        canonical_id=None,
                    )
                )
            continue
        for n, phrase in enumerate(record["features"]):
            if not isinstance(phrase, str) or not phrase.strip():
                raise ValueError("Invalid accepted generation feature")
            normalized = normalize_phrase(phrase)
            key = proof_key(normalized)
            phrases[normalized] = normalized
            keys[normalized] = key
            origins.append(
                dict(
                    source_run_id=record.get("source_run_id", "unknown"),
                    source_record_hash=record.get("record_hash"),
                    generation_job_id=record["job_id"],
                    generation_model=record.get(
                        "generation_model", record.get("model_id")
                    ),
                    item_id=record["item_id"],
                    prompt_family=record["prompt_family"],
                    replicate=record["replicate"],
                    attempt=record.get("attempt", 0),
                    raw_feature_index=n,
                    phrase=phrase,
                    normalized=normalized,
                    legacy_feature_id="f_" + digest(normalized)[:20],
                    equivalence_key=key,
                    excluded=False,
                )
            )
    aliases = {}
    for phrase, key in sorted(keys.items()):
        aliases.setdefault(key, phrase)
    normalized_reviews = []
    for review in reviews:
        left, right = normalize_phrase(review["left"]), normalize_phrase(
            review["right"]
        )
        a = left if left in phrases else aliases.get(left)
        b = right if right in phrases else aliases.get(right)
        if a is None or b is None:
            raise ValueError("Unknown reviewed feature phrase")
        normalized_reviews.append(
            dict(review, left=a, right=b, submitted_left=left, submitted_right=right)
        )
    reviews = normalized_reviews
    # An explicit uncertainty/distinction takes precedence over an automatic
    # rewrite. Split the whole challenged proof class to avoid hidden chaining.
    challenged = set()
    for review in reviews:
        a, b = normalize_phrase(review["left"]), normalize_phrase(review["right"])
        if a not in keys or b not in keys:
            raise ValueError("Unknown reviewed feature phrase")
        if a == b and review["decision"] in {"distinct", "related", "uncertain"}:
            raise ValueError(
                "Identical normalized text needs explicit sense-qualified features"
            )
        if review["decision"] != "merge" and keys[a] == keys[b]:
            challenged.add(keys[a])
    keys = {p: (p if k in challenged else k) for p, k in keys.items()}
    for origin in origins:
        origin["equivalence_key"] = keys[origin["normalized"]]
    decisions = []
    edges = {}
    for review in reviews:
        a, b = normalize_phrase(review["left"]), normalize_phrase(review["right"])
        if a not in phrases or b not in phrases:
            raise ValueError("Unknown reviewed feature phrase")
        decision = review["decision"]
        if decision not in {"merge", "distinct", "related", "uncertain", "pending"}:
            raise ValueError("Invalid equivalence decision")
        if review.get("review_kind") == "automated_bidirectional_entailment":
            from .equivalence import validate_automated_review

            validate_automated_review(review)
        elif decision != "pending" and (
            not review.get("reviewer") or not review.get("reason")
        ):
            raise ValueError("Named equivalence reviewer and reason required")
        if decision == "merge" and not protected_compatible(a, b):
            raise ValueError(
                "Review would change protected negation/modality/condition/role/relation meaning"
            )
        pair = tuple(sorted((keys[a], keys[b])))
        if pair in edges and edges[pair] != decision:
            raise ValueError("Conflicting equivalence reviews")
        edges[pair] = decision
        decisions.append(dict(review, left=a, right=b))
    groups = {k: {k} for k in set(keys.values())}
    root = {k: k for k in groups}
    blocked = []
    for (a, b), decision in sorted(edges.items()):
        if decision != "merge" or root[a] == root[b]:
            continue
        left, right = groups[root[a]], groups[root[b]]
        if not all(
            edges.get(tuple(sorted((x, y)))) == "merge" for x in left for y in right
        ):
            blocked.append(
                dict(
                    left=a,
                    right=b,
                    reason="Complete-link evidence missing; transitive chain is not sufficient",
                )
            )
            continue
        combined = left | right
        representative = min(combined, key=lambda x: (digest(x), x))
        del groups[root[a]]
        del groups[root[b]]
        groups[representative] = combined
        for key in combined:
            root[key] = representative
    source_phrases = defaultdict(list)
    for p, k in keys.items():
        source_phrases[root[k]].append(p)
    automated_groups = {
        root[keys[r["left"]]]
        for r in decisions
        if r.get("review_kind") == "automated_bidirectional_entailment"
        and r["decision"] == "merge"
        and root[keys[r["left"]]] == root[keys[r["right"]]]
    }
    features = []
    byid = {}
    for representative, members in sorted(
        groups.items(), key=lambda x: (digest(x[0]), x[0])
    ):
        text = min(members, key=lambda x: (digest(x), x))
        fid = "f_" + digest(text)[:20]
        feature = dict(
            feature_id=fid,
            text=text,
            equivalence_keys=sorted(members),
            source_phrases=sorted(source_phrases[representative]),
            n_origins=0,
            human_reviewed=False,
        )
        features.append(feature)
        byid[fid] = feature
    for o in origins:
        rep = root[o["equivalence_key"]]
        fid = "f_" + digest(rep)[:20]
        o["canonical_id"] = fid
        automated = rep in automated_groups
        o["consolidation_method"] = (
            (
                "automated_semantic_equivalence"
                if automated
                else "reviewed_semantic_equivalence"
            )
            if len(groups[rep]) > 1
            else (
                "conservative_semantic_rule"
                if o["normalized"] != o["equivalence_key"]
                else "exact_normalization"
            )
        )
        byid[fid]["n_origins"] += 1
    inventory_hash = digest(
        [
            VERSION,
            [
                (f["feature_id"], f["text"], f["equivalence_keys"], f["source_phrases"])
                for f in features
            ],
        ]
    )
    for f in features:
        f["inventory_hash"] = inventory_hash
    return dict(
        version=VERSION,
        feature_cap=None,
        source_feature_count=len(origins),
        normalized_unique=len(phrases),
        features=features,
        origins=origins,
        exclusions=excluded,
        inventory_hash=inventory_hash,
        review_decisions=sorted(decisions, key=lambda r: (r["left"], r["right"])),
        blocked_chains=blocked,
        human_reviewed=False,
        consolidation_policy="Narrow declared rewrite proofs plus complete-link named equivalence reviews; other proposals stay distinct",
    )


def propose_equivalences(bank, *, neighbors=8):
    """Retrieval proposes reviews, never removes features or proves equivalence."""
    features = bank["features"]
    n = len(features)
    postings = defaultdict(set)
    tokens = {}
    stop = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "be",
        "has",
        "have",
        "to",
        "of",
        "and",
        "or",
        "in",
        "on",
        "for",
        "by",
        "with",
        "it",
        "its",
        "can",
        "may",
        "requires",
    }
    for f in features:
        words = set(re.findall(r"[^\W_]+", f["text"])) - stop
        tokens[f["feature_id"]] = words
        for word in words:
            postings[word].add(f["feature_id"])
    lookup = {f["feature_id"]: f for f in features}
    pairs = {}
    for f in features:
        candidates = set()
        for word in tokens[f["feature_id"]]:
            if len(postings[word]) <= max(50, int(n * 0.03)):
                candidates.update(postings[word])
        scored = []
        for other in candidates - {f["feature_id"]}:
            a, b = tokens[f["feature_id"]], tokens[other]
            score = len(a & b) / len(a | b) if a | b else 0
            if score > 0:
                scored.append((score, other))
        for score, other in heapq.nlargest(neighbors, scored):
            key = tuple(sorted((f["feature_id"], other)))
            a, b = map(lookup.get, key)
            pairs[key] = dict(
                left_id=key[0],
                right_id=key[1],
                left=a["text"],
                right=b["text"],
                retrieval_score=score,
                protected_compatible=protected_compatible(a["text"], b["text"]),
                decision="pending",
                reviewer="",
                reason="",
                automatic_merge=False,
                note="Similarity is a review proposal, not semantic equivalence; unproposed pairs are not established distinct",
            )
    return [pairs[k] for k in sorted(pairs)]

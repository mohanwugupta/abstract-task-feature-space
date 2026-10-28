"""Complete vocabulary construction is independent of human matching or bank caps."""

import copy
import pytest


def generation(features, job="g0", item="i0"):
    return dict(
        job_id=job,
        item_id=item,
        prompt_family="P0",
        replicate=0,
        attempt=0,
        parse_status="ok",
        features=features,
        source_run_id="fixture",
        generation_model="fixture-generator",
    )


def review(a, b, decision="merge"):
    return dict(
        left=a,
        right=b,
        decision=decision,
        reviewer="fixture-reviewer",
        reason="Synthetic equivalence review",
    )


def test_full_bank_never_truncates_p0_or_rare_features():
    from feature_space_pilot.full.bank import build_bank

    phrases = ["rare distinct property " + str(n) for n in range(231)]
    result = build_bank([generation(phrases)])
    assert len(result["features"]) == 231 and len(result["origins"]) == 231
    assert all(o["canonical_id"] and not o.get("excluded") for o in result["origins"])
    assert result["feature_cap"] is None and result["source_feature_count"] == 231


def test_semantic_rules_merge_equivalents_and_preserve_all_origins():
    from feature_space_pilot.full.bank import build_bank

    rows = [
        generation(["Requires accuracy.", "requires accuracy", "accuracy is rewarded"]),
        generation(["needs accuracy"], "g1", "i1"),
    ]
    result = build_bank(rows)
    assert len(result["features"]) == 2 and len(result["origins"]) == 4
    ids = {
        o["canonical_id"] for o in result["origins"] if "rewarded" not in o["phrase"]
    }
    assert len(ids) == 1
    f = next(f for f in result["features"] if f["feature_id"] in ids)
    assert f["text"] == "requires accuracy" and f["n_origins"] == 3
    assert any(
        o["consolidation_method"] == "conservative_semantic_rule"
        for o in result["origins"]
    )
    assert result["human_reviewed"] is False


@pytest.mark.parametrize(
    "left,right",
    [
        ("requires accuracy", "accuracy is rewarded"),
        ("can cause harm", "must cause harm"),
        ("causes harm", "does not cause harm"),
        ("requires checking if the user asks", "requires checking"),
        ("user rewards assistant", "assistant rewards user"),
        ("accuracy causes trust", "trust causes accuracy"),
    ],
)
def test_review_cannot_erase_protected_meaning(left, right):
    from feature_space_pilot.full.bank import build_bank

    result = build_bank([generation([left, right])])
    assert len(result["features"]) == 2
    with pytest.raises(ValueError, match="protected"):
        build_bank([generation([left, right])], reviews=[review(left, right)])


def test_related_and_uncertain_reviews_never_merge():
    from feature_space_pilot.full.bank import build_bank

    result = build_bank(
        [generation(["supports inspection", "permits analysis"])],
        reviews=[review("supports inspection", "permits analysis", "related")],
    )
    assert len(result["features"]) == 2
    assert result["review_decisions"][0]["decision"] == "related"


def test_equivalence_does_not_chain_across_unreviewed_pairs():
    from feature_space_pilot.full.bank import build_bank

    rows = [generation(["alpha", "beta", "gamma"])]
    reviews = [review("alpha", "beta"), review("beta", "gamma")]
    assert len(build_bank(rows, reviews=reviews)["features"]) == 2
    assert (
        len(build_bank(rows, reviews=reviews + [review("alpha", "gamma")])["features"])
        == 1
    )


def test_full_bank_ids_and_hash_ignore_input_order_but_keep_provenance():
    from feature_space_pilot.full.bank import build_bank

    rows = [generation(["needs accuracy", "rare property"]), "unused"][:1]
    a = build_bank(rows)
    b = build_bank([generation(list(reversed(rows[0]["features"])))])
    assert a["inventory_hash"] == b["inventory_hash"]
    assert sorted((f["feature_id"], f["text"]) for f in a["features"]) == sorted(
        (f["feature_id"], f["text"]) for f in b["features"]
    )
    assert {o["phrase"] for o in a["origins"]} == set(rows[0]["features"])


def test_similarity_proposals_never_change_or_drop_feature_inventory():
    from feature_space_pilot.full.bank import build_bank, propose_equivalences

    result = build_bank(
        [
            generation(
                [
                    "requires accurate arithmetic",
                    "accuracy is rewarded",
                    "supports rapid arithmetic",
                ]
            )
        ]
    )
    original = copy.deepcopy(result)
    proposals = propose_equivalences(result)
    assert result == original and len(result["features"]) == 3
    assert all(
        p["decision"] == "pending" and p["automatic_merge"] is False for p in proposals
    )


def test_explicit_uncertain_review_overrides_a_rewrite_rule():
    from feature_space_pilot.full.bank import build_bank

    rows = [generation(["needs accuracy", "requires accuracy"])]
    result = build_bank(
        rows, reviews=[review("needs accuracy", "requires accuracy", "uncertain")]
    )
    assert len(result["features"]) == 2 and len(result["origins"]) == 2


def test_named_review_can_merge_a_real_paraphrase_without_erasing_argument_roles():
    from feature_space_pilot.full.bank import build_bank

    a, b = "requires accurate calculations", "requires exact arithmetic"
    result = build_bank([generation([a, b])], reviews=[review(a, b)])
    assert len(result["features"]) == 1 and len(result["origins"]) == 2


def test_review_cannot_change_numeric_conditions_or_conjunction_logic():
    from feature_space_pilot.full.bank import build_bank

    for a, b in [
        ("has a 5 minute deadline", "has a 10 minute deadline"),
        ("requires checking and accuracy", "requires checking or accuracy"),
    ]:
        with pytest.raises(ValueError, match="protected"):
            build_bank([generation([a, b])], reviews=[review(a, b)])


def test_reviews_can_use_canonical_display_text_created_by_a_safe_rule():
    from feature_space_pilot.full.bank import build_bank

    rows = [generation(["needs exact arithmetic", "requires accurate calculations"])]
    result = build_bank(
        rows,
        reviews=[review("requires exact arithmetic", "requires accurate calculations")],
    )
    assert len(result["features"]) == 1 and len(result["origins"]) == 2

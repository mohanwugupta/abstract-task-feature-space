"""Empirical prompt descriptions use outputs and local labels, not prompt intent."""

import copy
import pytest


def test_cue_coding_keeps_decision_dimensions_separate_and_is_not_a_validity_label():
    from feature_space_pilot.collaborator.qualitative import code_proposition

    r = code_proposition(
        "A deadline may force a trade-off between accuracy and delivery time.",
        "Deliver this accurately in five minutes.",
    )
    assert r["resource_demands"] and r["conflict_or_tradeoff"]
    assert not r["incentives"] and not r["unsupported_verified"]
    r = code_proposition(
        "The target is labeled Creative production.", "Creative production"
    )
    assert (
        r["linguistic_or_label_metadata"]
        and not r["generalizable_functional_candidate"]
    )


def test_reliability_review_never_turns_coder_agreement_or_unanimity_into_ceiling():
    from feature_space_pilot.collaborator.literature import reliability_summary

    evidence = dict(
        records=[
            dict(
                paper="Harpaintner2018",
                unit="researcher_coding",
                metric="joint_probability_agreement",
                value=0.7679,
                participant_similarity_ceiling=False,
                locator="p5",
                citation_url="https://doi.org/10.3389/fpsyg.2018.01748",
            )
        ]
    )
    out = reliability_summary(evidence)
    assert out["participant_similarity_ceiling"] is None and not out["loocv_performed"]
    evidence["records"][0]["participant_similarity_ceiling"] = True
    with pytest.raises(ValueError, match="coding"):
        reliability_summary(evidence)


def test_content_only_sensitivity_removes_grammar_columns_without_changing_rows():
    from feature_space_pilot.collaborator.qualitative import linguistic_feature_mask

    features = [
        dict(text="is a mammal", source_phrases=["is a mammal"]),
        dict(text="is a noun phrase", source_phrases=["is a noun phrase"]),
        dict(
            text="can be used transitively", source_phrases=["can be used transitively"]
        ),
    ]
    assert linguistic_feature_mask(features).tolist() == [False, True, True]

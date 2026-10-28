"""Semantic alignment is supervised, typed, mass-limited, and fully auditable."""

import numpy as np
import pytest


def fixture():
    items = [
        dict(
            item_id="h",
            text="the pull",
            source_text="PULL",
            selection_stratum="actionN",
        )
    ]
    human = dict(
        item_ids=["h"],
        feature_labels=["move", "force", "sing"],
        matrix=np.array([[4, 2, 0]], dtype=float),
    )
    occurrences = [
        dict(
            item_id="h",
            prompt_family="P0",
            replicate=r,
            feature_id=f,
            proposition=text,
            generation_job_id="g" + str(r),
            raw_feature_index=k,
        )
        for r, f, text, k in [
            (0, "f1", "moves with force", 0),
            (0, "f2", "moves", 1),
            (1, "f1", "moves with force", 0),
        ]
    ]
    inv = dict(
        items=items,
        norm_features=[
            dict(item_id="h", human_id="h0", label="move"),
            dict(item_id="h", human_id="h1", label="force"),
            dict(item_id="h", human_id="h2", label="sing"),
        ],
        propositions=[
            dict(item_id="h", proposition_id="p1", text="moves with force"),
            dict(item_id="h", proposition_id="p2", text="moves"),
        ],
        origins=[
            dict(o, proposition_id="p1" if o["feature_id"] == "f1" else "p2")
            for o in occurrences
        ],
    )
    records = [
        dict(
            item_id="h",
            parse_status="ok",
            record_hash="fixture",
            results=[
                dict(
                    proposition_id="p1",
                    matches=["h0", "h1", "h2"],
                    related=[],
                    uncertain=[],
                    evidence="Fixture multi-label match",
                ),
                dict(
                    proposition_id="p2",
                    matches=["h0"],
                    related=[],
                    uncertain=[],
                    evidence="Fixture singleton",
                ),
            ],
        )
    ]
    return (
        items,
        human,
        dict(occurrences=occurrences, replicates=3),
        dict(inventory=inv, records=records),
    )


def test_one_proposition_has_at_most_one_unit_of_mass_and_labels_are_capped_per_rep():
    from feature_space_pilot.collaborator.alignment import align_features

    items, human, discovery, semantic = fixture()
    out = align_features(items, human, discovery, semantic)
    assert out["frequency"]["P0"].tolist() == pytest.approx(np.array([[0.5, 1 / 3, 0]]))
    assert all(
        sum(r["label_weights"].values()) <= 1 + 1e-12 for r in out["mapping_audit"]
    )
    assert out["single_label_only"]["P0"].tolist() == [[1 / 3, 0, 0]]
    assert out["inventory_audit"][0]["unexpected_target_labels"] == ["sing"]
    assert out["supervised_alignment"] and not out["human_verified"]


def test_related_and_uncertain_are_not_clear_matches_and_unscored_is_not_absence():
    from feature_space_pilot.collaborator.alignment import align_features

    items, human, discovery, semantic = fixture()
    row = semantic["records"][0]["results"][0]
    row.update(matches=[], related=["h0"], uncertain=["h1"])
    out = align_features(items, human, discovery, semantic)
    p0 = next(r for r in out["concept_scores"] if r["prompt_family"] == "P0")
    assert p0["unmatched_units"] == 2 and p0["unscored_units"] == 0
    semantic["records"] = []
    p0 = next(
        r
        for r in align_features(items, human, discovery, semantic)["concept_scores"]
        if r["prompt_family"] == "P0"
    )
    assert p0["cosine"] is None and p0["unscored_units"] == 3


def test_matching_provenance_cannot_import_a_task_proposition_or_wrong_text():
    from feature_space_pilot.collaborator.alignment import align_features

    items, human, discovery, semantic = fixture()
    semantic["inventory"]["propositions"][0]["text"] = "unrelated task feature"
    with pytest.raises(ValueError, match="provenance"):
        align_features(items, human, discovery, semantic)


def test_duplicate_published_feature_labels_keep_coordinates_and_are_not_guessed():
    from feature_space_pilot.collaborator.alignment import align_features

    items, human, discovery, semantic = fixture()
    human.update(
        feature_labels=["move", "move", "force"],
        matrix=np.array([[4, 2, 1]], dtype=float),
    )
    out = align_features(items, human, discovery, semantic)
    assert out["frequency"]["P0"].shape == (1, 3)
    assert out["frequency"]["P0"][0, :2].tolist() == [0, 0]
    assert out["inventory_audit"][0]["ambiguous_codebook_labels"] == ["move"]

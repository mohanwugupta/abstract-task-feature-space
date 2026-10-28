import copy
import pytest


def test_composition_separates_discovery_gap_inapplicability_and_interaction_candidates():
    from feature_space_pilot.full.analysis import classify_composition

    t = {"rhyme"}
    s = {"standalone"}
    direct = {"rhyme", "interaction", "gap"}
    values = {
        "T": dict(rhyme=1, standalone=0, interaction=0, gap=1),
        "S": dict(rhyme=0, standalone=1, interaction=0, gap=0),
        "TS": dict(rhyme=1, standalone=0, interaction=1, gap=1),
    }
    r = classify_composition(t, s, direct, values)
    assert r["recovered_both"] == ["rhyme"]
    assert r["component_generated_but_inapplicable"] == ["standalone"]
    assert r["direct_only_applicable"] == ["gap", "interaction"]
    assert r["constituent_discovery_gaps"] == ["gap"]
    assert (
        r["interaction_candidates"] == ["interaction"]
        and not r["confirmed_interactions"]
    )


def test_missing_constituent_judgment_never_becomes_evidence_of_emergence():
    from feature_space_pilot.full.analysis import classify_composition

    values = {"T": {"r": None}, "S": {"r": 0}, "TS": {"r": 1}}
    r = classify_composition(set(), set(), {"r"}, values)
    assert not r["interaction_candidates"] and r["unresolved_features"] == ["r"]
    with pytest.raises(ValueError, match="reviewer"):
        classify_composition(
            set(),
            set(),
            {"r"},
            {"T": {"r": 0}, "S": {"r": 0}, "TS": {"r": 1}},
            reviews={"r": dict(confirmed=True)},
        )


def test_complete_matrix_statistics_are_gated_and_legacy_comparison_uses_same_judge():
    from feature_space_pilot.full.analysis import structure_summary
    from feature_space_pilot.full.matrix import matrix_state

    items = [dict(item_id="a"), dict(item_id="b")]
    features = [dict(feature_id="x"), dict(feature_id="y")]
    partial = matrix_state(
        items,
        features,
        [dict(item_id="a", feature_id="x", parse_status="ok", status="ok", applies=1)],
    )
    r = structure_summary(
        items, features, partial, legacy_subsets={"original150": ["x"]}
    )
    assert r["geometry"]["available"] is False
    assert (
        r["items"][0]["applicable_features"] is None
        and r["items"][0]["observed_applicable_lower_bound"] == 1
    )
    rows = [
        dict(
            item_id=i,
            feature_id=f,
            parse_status="ok",
            status="ok",
            applies=int(i == "a" or f == "y"),
        )
        for i in ["a", "b"]
        for f in ["x", "y"]
    ]
    complete = matrix_state(items, features, rows)
    r = structure_summary(
        items, features, complete, legacy_subsets={"original150": ["x"]}
    )
    assert r["geometry"]["available"] and r["items"][0]["applicable_features"] == 2
    assert (
        r["legacy_comparisons"]["original150"]["comparison_policy"]
        == "same_global_judge_and_item_scope"
    )


def test_full_composition_keeps_original_three_call_budget_per_family():
    from feature_space_pilot.full.analysis import composition_report
    from feature_space_pilot.full.bank import build_bank
    from feature_space_pilot.full.matrix import matrix_state
    from test_full_bank import generation

    items = [
        dict(item_id="T", target_type="task_instance"),
        dict(item_id="S", target_type="situation"),
        dict(
            item_id="TS",
            target_type="situated_task",
            parent_id="T",
            situation_ids=["S"],
            task_id="poem",
        ),
    ]
    records = []
    for iid, phrase in [
        ("T", "requires rhyme"),
        ("S", "has a deadline"),
        ("TS", "checking conflicts with the deadline"),
    ]:
        for family in ["P0", "P1", "P2"]:
            for n in range(3):
                r = generation([phrase], iid + family + str(n), iid)
                r.update(
                    prompt_family=family,
                    replicate=n,
                    usage=dict(input_tokens=10, output_tokens=10),
                )
                records.append(r)
    bank = build_bank(records)
    rows = [
        dict(
            item_id=i["item_id"],
            feature_id=f["feature_id"],
            status="ok",
            parse_status="ok",
            applies=1,
        )
        for i in items
        for f in bank["features"]
    ]
    state = matrix_state(items, bank["features"], rows)
    report = composition_report(items, bank, records, state)
    primary = [r for r in report if r["sampling_mode"] == "equal_calls"]
    assert len(primary) == 3 and all(
        r["component_calls"] == r["direct_calls"] == 3 for r in primary
    )
    assert primary[0]["sampling_allocation"] == dict(T=2, S=1, TS=3)
    assert primary[1]["sampling_allocation"] == dict(T=1, S=2, TS=3)


def test_actual_task_contrast_requires_known_binary_cells_on_both_sides():
    from feature_space_pilot.full.analysis import task_contrast
    from feature_space_pilot.full.matrix import matrix_state

    items = [
        dict(item_id="poem_T", task_id="poem", target_type="task_instance"),
        dict(item_id="finance_T", task_id="finance", target_type="task_instance"),
    ]
    features = [
        dict(feature_id="rhyme", text="requires rhyme"),
        dict(feature_id="shared", text="has quality criteria"),
        dict(feature_id="unknown", text="a pending property"),
    ]
    rows = [
        dict(item_id=i, feature_id=f, parse_status="ok", status="ok", applies=v)
        for i, f, v in [
            ("poem_T", "rhyme", 1),
            ("finance_T", "rhyme", 0),
            ("poem_T", "shared", 1),
            ("finance_T", "shared", 1),
            ("poem_T", "unknown", 1),
        ]
    ]
    result = task_contrast(items, features, matrix_state(items, features, rows))
    assert {r["feature_id"]: r["comparison"] for r in result} == dict(
        rhyme="poem_only", shared="shared", unknown="unresolved"
    )
    assert (
        next(r for r in result if r["feature_id"] == "rhyme")["proposition"]
        == "requires rhyme"
    )


def test_usage_keeps_generation_and_atomic_models_separate_and_missing_usage_explicit():
    from feature_space_pilot.full.analysis import observed_usage

    source = dict(
        snapshots=[
            dict(
                manifest=dict(run_id="old"),
                config=dict(
                    generator=dict(model_id="mini"), judge=dict(model_id="mini")
                ),
                generations=[
                    dict(
                        job_id="g",
                        attempt=0,
                        usage=dict(input_tokens=100, output_tokens=80),
                    )
                ],
                judgments=[
                    dict(
                        job_id="j",
                        attempt=0,
                        usage=dict(input_tokens=300, output_tokens=40),
                    ),
                    dict(job_id="k", attempt=0, usage={}),
                ],
            )
        ]
    )
    result = observed_usage(source, [], dict(model_id="luna"))
    assert (
        next(r for r in result["groups"] if r["stage"] == "generation")["model_id"]
        == "mini"
    )
    atomic = next(r for r in result["groups"] if r["stage"] == "judgment")
    assert (
        atomic["model_id"] == "mini"
        and atomic["known_usage_requests"] == 1
        and atomic["missing_usage_requests"] == 1
    )
    assert atomic["mean_output_tokens_known"] == 40
    assert result["configured_global_judge"] == "luna"
    assert result["global_judge_atomic_usage_available"] is False


def test_saturation_summary_uses_same_complete_items_at_every_point_and_family():
    from feature_space_pilot.full.analysis import saturation_summary

    curves = [
        dict(
            item_id=i,
            stratum="task",
            prompt_family=f,
            sampling_mode="protocol_comparison_cap",
            replicates_included=n,
            generation_complete=not (i == "partial" and f == "P2" and n == 3),
            canonical_discoveries=n * (2 if i == "complete" else 100),
        )
        for i in ["complete", "partial"]
        for f in ["P0", "P1", "P2"]
        for n in [1, 2, 3]
    ]
    result = saturation_summary(curves, replicates=3)
    assert len(result) == 9 and all(
        r["item_ids"] == ["complete"] and r["n_items"] == 1 for r in result
    )
    assert [
        r["mean_canonical_discoveries"] for r in result if r["prompt_family"] == "P0"
    ] == [2.0, 4.0, 6.0]

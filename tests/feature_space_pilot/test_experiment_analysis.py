"""Interpretation overlays cannot change membership, truth values or emergence claims."""

import copy
import numpy as np
from feature_space_pilot.full.bank import build_bank
from test_experiment_banks import item, generation


def test_overlapping_semantic_tags_and_specificity_are_supplementary():
    from feature_space_pilot.experiments.analysis import annotate_features

    bank = build_bank(
        [
            generation(
                "t",
                [
                    "requires accurate calculations under a deadline",
                    "contains exactly 19 payment records",
                    "the word has two syllables",
                ],
            )
        ]
    )
    before = copy.deepcopy(bank)
    rows = annotate_features(bank)
    r = next(r for r in rows if "accurate" in r["proposition"])
    assert {
        "Operations and information requirements",
        "Constraints and resources",
        "Quality and evaluation criteria",
    } <= set(r["semantic_families"])
    assert (
        not r["human_reviewed"]
        and r["annotation_method"] == "offline_provisional_rule_coding"
    )
    assert (
        next(r for r in rows if "19" in r["proposition"])["specificity"]
        == "instance_detail_candidate"
    )
    assert (
        next(r for r in rows if "syllables" in r["proposition"])["specificity"]
        == "linguistic_or_label_metadata"
    )
    assert bank == before and len(rows) == 3


def test_coding_does_not_confuse_creative_domain_or_financial_budget_with_evaluation_or_resources():
    from feature_space_pilot.experiments.analysis import annotate_features

    bank = build_bank(
        [
            generation(
                "t",
                [
                    "creative production can include producing music",
                    "it involves preparing budgets",
                ],
            )
        ]
    )
    rows = annotate_features(bank)
    assert (
        "Quality and evaluation criteria"
        not in next(r for r in rows if "music" in r["proposition"])["semantic_families"]
    )
    assert (
        "Constraints and resources"
        not in next(r for r in rows if "budgets" in r["proposition"])[
            "semantic_families"
        ]
    )


def test_preference_candidate_shortlist_excludes_flagged_properties_without_changing_labels():
    from feature_space_pilot.experiments.analysis import behavioral_candidates

    rows = [
        dict(
            feature_id="deadline",
            proposition="a project may have a deadline",
            comparison="shared",
        ),
        dict(feature_id="rhyme", proposition="it can use rhyme", comparison="shared"),
    ]
    annotations = [
        dict(
            feature_id=r["feature_id"],
            evaluatively_relevant_candidate=True,
            generalizable_functional_candidate=True,
            semantic_families=["Constraints and resources"],
        )
        for r in rows
    ]
    flags = [dict(feature_id="rhyme", category="cross_domain_positive")]
    before = copy.deepcopy((rows, annotations, flags))
    selected = behavioral_candidates(rows, annotations, flags)
    assert [r["feature_id"] for r in selected] == ["deadline"]
    assert (rows, annotations, flags) == before


def test_diagnostics_include_interaction_grounding_without_automatic_confirmation():
    from feature_space_pilot.experiments.analysis import (
        diagnostic_audit,
        annotate_features,
    )

    items = [
        item("T", "task_instance", task_id="finance"),
        item("S", "situation"),
        item(
            "TS", "situated_task", task_id="finance", parent_id="T", situation_ids=["S"]
        ),
    ]
    bank = build_bank(
        [generation("TS", ["the report must be delivered within five minutes"])]
    )
    fid = bank["features"][0]["feature_id"]
    analysis = dict(
        annotations=annotate_features(bank),
        inheritance_flags=[],
        composition_features=[
            dict(
                feature_id=fid,
                item_id="TS",
                category="interaction_pattern_candidate",
                T=0,
                S=0,
                TS=1,
            )
        ],
    )
    state = dict(matrix=np.array([[0], [0], [1]], dtype=np.int8))
    out = diagnostic_audit(items, bank, state, analysis)
    assert out["counts"]["interaction_candidate_grounding"] == 1
    row = next(
        r
        for r in out["diagnostic_set"]
        if r["category"] == "interaction_candidate_grounding"
    )
    assert row["parent_text"] == "T" and row["situation_text"] == "S"
    assert not row["verified_correction"] and out["verified_corrections"] == []


def test_typed_creative_production_property_on_finance_is_screened_not_corrected():
    from feature_space_pilot.experiments.analysis import (
        diagnostic_audit,
        annotate_features,
    )

    items = [item("finance_T", "task_instance", task_id="finance")]
    bank = build_bank(
        [generation("finance_T", ["a creative-production task may have a deadline"])]
    )
    state = dict(matrix=np.array([[1]], dtype=np.int8))
    analysis = dict(
        annotations=annotate_features(bank),
        inheritance_flags=[],
        composition_features=[],
    )
    out = diagnostic_audit(items, bank, state, analysis)
    assert out["counts"].get("cross_domain_positive") == 1
    assert state["matrix"].tolist() == [[1]] and not out["verified_corrections"]


def test_task_report_uses_task_only_scope_and_unknowns_in_composition():
    from feature_space_pilot.experiments.analysis import analyze_tasks

    items = [
        item("poem_T", "task_instance", task_id="poem", level="task_instance"),
        item("S_deadline", "situation", level="situation"),
        item(
            "poem_S_deadline",
            "situated_task",
            parent_id="poem_T",
            situation_ids=["S_deadline"],
            task_id="poem",
            level="situated_task",
        ),
    ]
    gs = [
        generation("poem_T", ["requires accuracy"], f, n)
        for f in ("P0", "P1", "P2")
        for n in range(3)
    ]
    gs += [
        generation("S_deadline", ["has limited time"], f, n)
        for f in ("P0", "P1", "P2")
        for n in range(3)
    ]
    gs += [
        generation(
            "poem_S_deadline",
            ["requires accuracy", "requires checking under time pressure"],
            f,
            n,
        )
        for f in ("P0", "P1", "P2")
        for n in range(3)
    ]
    bank = build_bank(gs)
    ff = {f["text"]: n for n, f in enumerate(bank["features"])}
    a = np.full((3, len(ff)), -1, dtype=np.int8)
    a[:, ff["requires accuracy"]] = [1, 0, 1]
    a[:, ff["has limited time"]] = [0, 1, 1]
    a[:, ff["requires checking under time pressure"]] = [0, 0, 1]
    jobs = [
        {k: g[k] for k in ("job_id", "item_id", "prompt_family", "replicate")}
        for g in gs
    ]
    out = analyze_tasks(
        items, bank, gs, jobs, dict(matrix=a, complete=True), replicates=3
    )
    rows = out["compositions"]
    assert len(rows) == 6
    assert all(r["interaction_candidates"] for r in rows)
    assert all(not r["confirmed_interactions"] for r in rows)
    assert all(r["generation_coverage_complete"] for r in rows)
    assert all(r["proposition"] for r in out["composition_features"])
    a[0, ff["requires checking under time pressure"]] = -1
    out = analyze_tasks(
        items, bank, gs, jobs, dict(matrix=a, complete=False), replicates=3
    )
    assert all(not r["interaction_candidates"] for r in out["compositions"])
    assert all(not r["available"] for r in out["compositions"])


def test_missing_generations_cannot_be_reported_as_equal_complete_sampling():
    from feature_space_pilot.experiments.analysis import analyze_tasks

    items = [
        item("T", "task_instance", task_id="poem"),
        item("S", "situation"),
        item("TS", "situated_task", task_id="poem", parent_id="T", situation_ids=["S"]),
    ]
    gs = [
        generation(i, ["requires accuracy"], f, n)
        for i in ("T", "S", "TS")
        for f in ("P0", "P1", "P2")
        for n in range(3)
    ]
    bank = build_bank(gs)
    state = dict(matrix=np.ones((3, 1), dtype=np.int8), complete=True)
    out = analyze_tasks(items, bank, gs[:-1], [], state, replicates=3)
    p2 = [r for r in out["compositions"] if r["prompt_family"] == "P2"]
    assert all(not r["available"] and not r["generation_coverage_complete"] for r in p2)


def test_human_validation_scores_only_generated_benchmark_origins_and_reports_limits(
    tmp_path,
):
    from feature_space_pilot.experiments.human import analyze_human

    items = [
        item("h_event", "event", split="development"),
        item("h_object", "object", split="held_out"),
        item("h_abstract", "concept", split="held_out"),
    ]
    gs = [
        generation(i["item_id"], ["has an observable effect"], f)
        for i in items
        for f in ("P0", "P1", "P2")
    ]
    bank = build_bank(gs)
    inv = dict(
        comparison_cap=10,
        items=items[:2],
        propositions=[],
        norm_features=[],
        origins=[],
    )
    records = []
    for i in items[:2]:
        pid = "p_" + i["item_id"]
        inv["propositions"].append(
            dict(
                item_id=i["item_id"],
                proposition_id=pid,
                text="has an observable effect",
            )
        )
        inv["norm_features"].append(
            dict(
                item_id=i["item_id"],
                human_id="h0",
                label="effect",
                categories=["synthetic"],
            )
        )
        for g in (g for g in gs if g["item_id"] == i["item_id"]):
            inv["origins"].append(
                dict(
                    item_id=i["item_id"],
                    proposition_id=pid,
                    generation_job_id=g["job_id"],
                    raw_feature_index=0,
                    prompt_family=g["prompt_family"],
                    replicate=0,
                )
            )
        records.append(
            dict(
                parse_status="ok",
                item_id=i["item_id"],
                results=[
                    dict(
                        proposition_id=pid,
                        matches=["h0"],
                        related=[],
                        uncertain=[],
                        evidence="Synthetic norm match",
                    )
                ],
            )
        )
    semantic = dict(inventory=inv, records=records, manifest={})
    state = dict(
        matrix=np.ones((3, len(bank["features"])), dtype=np.int8), complete=True
    )
    out = analyze_human(items, bank, gs, [], state, semantic, replicates=1)
    assert all(r["mean_recall"] == 1 for r in out["family_summary"].values())
    assert out["reference_scope"]["individual_feature_norm_targets"] == 2
    assert out["reference_scope"]["abstract_category_only_targets"] == 1
    assert out["precision_against_truth"] is None
    from feature_space_pilot.experiments.reports import write_human_report

    partial = analyze_human(
        items, bank, gs, [], state, dict(semantic, records=[]), replicates=1
    )
    write_human_report(
        tmp_path,
        items,
        dict(bank, bank_id="synthetic-human-bank"),
        dict(
            audit_summary=dict(
                planned_cells=3, reused_binary_cells=3, unresolved_cells=0
            )
        ),
        partial,
        dict(planned=9, completed=9),
    )
    assert "unavailable" in (tmp_path / "scientific_report.md").read_text()
    inv["origins"][0]["generation_job_id"] = "task-origin"
    import pytest

    with pytest.raises(ValueError, match="human generation provenance"):
        analyze_human(items, bank, gs, [], state, semantic, replicates=1)

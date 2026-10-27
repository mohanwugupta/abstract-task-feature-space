from dataclasses import replace
import pytest
from feature_space_pilot.viz.load import load_run, CellState


def test_composition_sets_match_audit_and_delta_identity(viz_run_path):
    from feature_space_pilot.viz.derive import composition_rows

    run = load_run(viz_run_path)
    rows = composition_rows(run)
    assert len(rows) == 6
    for row in rows:
        assert set(row["incremental_applicable"]) == set(
            row["direct_applicable"]
        ) - set(row["component_applicable"])
        assert row["component_calls"] == row["direct_calls"]
        assert all(v == "unreviewed" for v in row["interaction_status"].values())
        assert not row["confirmed_interactions"]
    first = rows[0]
    bad = dict(run.composition_audit[0], incremental_applicable=[])
    with pytest.raises(ValueError, match="audit|diverge"):
        composition_rows(
            replace(run, composition_audit=(bad,) + run.composition_audit[1:])
        )
    unmatched = dict(run.composition_audit[0], direct_calls=999)
    with pytest.raises(ValueError, match="budget|calls"):
        composition_rows(
            replace(run, composition_audit=(unmatched,) + run.composition_audit[1:])
        )


def test_manual_interactions_require_named_review_and_rejections_stay_rejected(
    viz_run_path,
):
    from feature_space_pilot.viz.derive import composition_rows

    run = load_run(viz_run_path)
    audit = run.composition_audit[0]
    fid = audit["interaction_candidates"][0]
    iid = audit["item_id"]
    manual = {
        iid: {
            fid: dict(
                confirmed=False,
                reviewer="independent-person",
                note="Not a relational property",
            )
        }
    }
    row = composition_rows(replace(run, manual_audits=manual))[0]
    assert (
        row["interaction_status"][fid] == "rejected"
        and not row["confirmed_interactions"]
    )
    manual[iid][fid]["confirmed"] = True
    confirmed = dict(audit, confirmed_interactions=[fid])
    row = composition_rows(
        replace(
            run,
            manual_audits=manual,
            composition_audit=(confirmed,) + run.composition_audit[1:],
        )
    )[0]
    assert row["interaction_status"][fid] == "confirmed"
    manual[iid][fid].pop("reviewer")
    with pytest.raises(ValueError, match="review"):
        composition_rows(
            replace(
                run,
                manual_audits=manual,
                composition_audit=(confirmed,) + run.composition_audit[1:],
            )
        )


def test_atlas_uses_judgments_and_matrix_keeps_unknown_states(viz_run_path):
    from feature_space_pilot.viz.derive import atlas, matrix_data, select_features

    run = load_run(viz_run_path)
    result = atlas(run, "poem_T", "finance_T")
    assert result["shared"] and result["left_only"] and result["right_only"]
    assert all(
        run.cell("poem_T", r["feature_id"]) == 1
        and run.cell("finance_T", r["feature_id"]) == 1
        for r in result["shared"]
    )
    changed = replace(
        run, candidates=tuple(r for r in run.candidates if r["origin_item"] != "poem_T")
    )
    assert atlas(changed, "poem_T", "finance_T")["shared"] == result["shared"]
    fid = result["shared"][0]["feature_id"]
    cells = dict(run.judgments)
    cells.pop(("poem_T", fid))
    incomplete = replace(run, judgments=cells, excluded=frozenset({("finance_T", fid)}))
    result = atlas(incomplete, "poem_T", "finance_T")
    assert fid in {r["feature_id"] for r in result["unresolved"]}
    matrix = matrix_data(incomplete, ["poem_T", "finance_T"], [fid], cluster=True)
    assert matrix["states"] == [["?"], ["N/A"]]
    assert matrix["clustering_available"] is False
    assert set(select_features(run, prompts=["P0"])) <= set(run.features)


def test_discovery_pools_exact_union_and_preserves_exclusions(viz_run_path):
    from feature_space_pilot.viz.derive import discovery_rows, contribution_rows

    run = load_run(viz_run_path)
    fid = next(iter(run.features))
    records = []
    candidates = []
    for j, item in enumerate(["poem_T", "finance_T"]):
        job = f"fixture-{j}"
        phrases = ["same exact phrase"] + (["excluded proposal"] if j else [])
        records.append(
            dict(
                job_id=job,
                item_id=item,
                prompt_family="P0",
                replicate=0,
                features=phrases,
                usage=dict(input_tokens=10, output_tokens=5),
            )
        )
        for ix, p in enumerate(phrases):
            candidates.append(
                dict(
                    job_id=job,
                    origin_item=item,
                    prompt_family="P0",
                    replicate=0,
                    phrase=p,
                    normalized=p,
                    feature_id=fid if ix == 0 else "unselected",
                    selected=ix == 0,
                    raw_feature_index=ix,
                )
            )
    toy = replace(run, generations=tuple(records), candidates=tuple(candidates))
    raw = discovery_rows(toy, representation="raw")
    assert raw[-1]["unique"] == 2 and raw[-1]["call_count"] == 2
    canonical = discovery_rows(toy, representation="canonical")
    assert canonical[-1]["unique"] == 1 and canonical[-1]["excluded_proposals"] == 1
    contributions = contribution_rows(toy, by="prompt_family")
    assert (
        contributions[0]["selected_unique"] == 1
        and contributions[0]["excluded_unique"] == 1
    )


def test_discovery_reports_uncapped_source_protocol_alongside_numeric_caps(
    viz_run_path,
):
    from dataclasses import replace
    from feature_space_pilot.viz.load import load_run
    from feature_space_pilot.viz.derive import discovery_rows

    run = load_run(viz_run_path)
    jobs = {k: dict(v) for k, v in run.jobs.items()}
    first = next(k for k, j in jobs.items() if j["prompt_family"] == "P0")
    jobs[first]["cap"] = None
    toy = replace(run, jobs=jobs)
    rows = discovery_rows(toy)
    p0 = [r for r in rows if r["prompt_family"] == "P0"]
    assert p0 and any(None in r["output_caps"] and 10 in r["output_caps"] for r in p0)

import csv
import json


def test_report_exports_all_cells_and_gates_incomplete_scientific_claims(tmp_path):
    from feature_space_pilot.full.report import write_full_report
    from feature_space_pilot.full.sources import collect_sources
    from feature_space_pilot.full.bank import build_bank, propose_equivalences
    from feature_space_pilot.full.matrix import matrix_state
    from feature_space_pilot.full.budget import cost_plan, profile_inputs, IndexedLedger
    from feature_space_pilot.engine import load_config
    from test_full_sources import source_fixture

    p = source_fixture(tmp_path, "a")
    q = source_fixture(tmp_path, "b")
    source = collect_sources([p, q], allow_synthetic=True, expected_items=2)
    for item in source["items"]:
        item["source_dataset"] = "synthetic_demo"
    bank = build_bank(source["generations"])
    source["generations"].append(
        dict(
            job_id="failed-fixture",
            item_id="a",
            prompt_family="P0",
            replicate=0,
            attempt=1,
            parse_status="error",
            generation_model="fixture",
            usage={},
        )
    )
    c = load_config("configs/feature_space_pilot.json")
    c.update(root=str(tmp_path), demo=True)
    profile = profile_inputs(
        source["items"], c["judge"], lambda p: dict(input_tokens=250)
    )
    reuse = dict(
        reused=[],
        unresolved_prior=[],
        external_pending={},
        rejected_counts={},
        policy="fixture",
    )
    with IndexedLedger(
        tmp_path / "ledger.jsonl", cap=100, stage_caps=c["stage_caps"]
    ) as ledger:
        cost = cost_plan(
            source["items"],
            bank["features"],
            c,
            ledger,
            reuse,
            profile=profile,
            generations_complete=True,
        )
    partial = matrix_state(source["items"], bank["features"], [])
    out = tmp_path / "report"
    report = write_full_report(
        out,
        source,
        bank,
        partial,
        c,
        cost,
        reuse,
        profile=profile,
        proposals=[
            dict(
                left_id="fixture_a",
                right_id="fixture_b",
                left="synthetic first",
                right="synthetic second",
                retrieval_score=0.5,
                protected_compatible=True,
                decision="pending",
                reviewer="",
                automatic_merge=False,
                reason="synthetic export fixture",
                note="Unresolved review proposal",
            )
        ],
        render=False,
    )
    assert report["matrix_complete"] is False and report["synthetic"] is True
    assert not (out / "binary_matrix.csv").exists()
    rows = list(csv.DictReader((out / "applicability_matrix_partial.csv").open()))
    assert len(rows) == 2 and all(
        row[bank["features"][0]["feature_id"]] == "?" for row in rows
    )
    assert len(list(csv.DictReader((out / "blocked_cells.csv").open()))) == 2
    assert "SYNTHETIC" in (out / "scientific_report.md").read_text()
    assert (out / "task_shared_distinct_features.csv").exists()
    assert (out / "observed_usage.json").exists()
    assert (out / "semantic_review_queue.csv").exists()
    assert (out / "blocked_equivalence_chains.json").exists()
    before = {r.name: r.read_bytes() for r in p.iterdir()}
    values = [
        dict(
            item_id=i["item_id"],
            feature_id=f["feature_id"],
            parse_status="ok",
            status="ok",
            applies=1,
        )
        for i in source["items"]
        for f in bank["features"]
    ]
    complete = matrix_state(source["items"], bank["features"], values)
    final = tmp_path / "final"
    write_full_report(
        final,
        source,
        bank,
        complete,
        c,
        cost,
        reuse,
        profile=profile,
        proposals=[],
        render=False,
    )
    assert (final / "binary_matrix.csv").exists()
    assert {r.name: r.read_bytes() for r in p.iterdir()} == before

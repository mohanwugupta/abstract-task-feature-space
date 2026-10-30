import copy
import pytest
from feature_space_pilot.io import canonical, digest


def settings():
    from feature_space_pilot.engine import load_config

    return dict(
        load_config("configs/feature_space_pilot.json")["judge"],
        rubric_version="binary_v1",
    )


def fixture():
    from feature_space_pilot.full.reuse import cell_job

    item = dict(
        item_id="pencil",
        text="the pencil",
        language="en",
        content_hash=digest("the pencil"),
    )
    feature = dict(
        feature_id="f_" + digest("requires accuracy")[:20], text="requires accuracy"
    )
    s = settings()
    job = cell_job(item, feature, s)
    raw = canonical(
        dict(
            item_id=item["item_id"],
            feature_id=feature["feature_id"],
            applies=1,
            status="ok",
            evidence="Synthetic fixture",
        )
    )
    r = dict(
        job_id=job["job_id"],
        item_id=item["item_id"],
        feature_id=feature["feature_id"],
        applies=1,
        status="ok",
        parse_status="ok",
        raw_response=raw,
        response_hash=digest(raw),
        prompt_hash=job["prompt_hash"],
        rubric_version="binary_v1",
        judge_model=s["model_id"],
        attempt=0,
    )
    r["record_hash"] = digest(r)
    source = dict(
        manifest=dict(run_id="fixture", stage="human"),
        manifest_sha256="fixture-sha",
        config=dict(judge=s),
        items=[item],
        baseline_features=[feature],
        judgment_jobs=[job],
        judgments=[r],
        pending=[],
    )
    return item, feature, source


def test_reuse_requires_literal_prompt_model_item_and_feature_compatibility():
    from feature_space_pilot.full.reuse import reuse_evidence

    item, feature, source = fixture()
    out = reuse_evidence([source], [item], [feature], settings())
    assert (
        len(out["reused"]) == 1
        and out["reused"][0]["source_journal_record_hash"]
        == source["judgments"][0]["record_hash"]
    )
    for field in ["model", "item", "feature", "prompt"]:
        s = copy.deepcopy(source)
        if field == "model":
            s["config"]["judge"]["model_id"] = "different-model"
        elif field == "item":
            s["items"][0]["text"] = "different target"
        elif field == "feature":
            s["baseline_features"][0]["text"] = "accuracy is rewarded"
        else:
            s["judgment_jobs"][0]["system"] = "different rubric wording"
        assert not reuse_evidence([s], [item], [feature], settings())["reused"]


def test_pending_source_requests_are_held_and_ambiguous_cells_not_reused_as_zero():
    from feature_space_pilot.full.reuse import cell_job, reuse_evidence

    item, feature, source = fixture()
    j = cell_job(item, feature, settings())
    source["judgments"] = []
    source["pending"] = [
        dict(
            job_id=j["job_id"],
            stage="judgment",
            body=j["payload"],
            item_id=item["item_id"],
            feature_id=feature["feature_id"],
            batch_id="existing",
            ticket="held",
            status="in_progress",
        )
    ]
    out = reuse_evidence([source], [item], [feature], settings())
    assert j["job_id"] in out["external_pending"] and not out["reused"]
    source["pending"] = []
    source["judgments"] = fixture()[2]["judgments"]
    r = source["judgments"][0]
    r.update(status="ambiguous", applies=None)
    r["raw_response"] = canonical(
        {k: r[k] for k in ["item_id", "feature_id", "applies", "status"]}
        | dict(evidence="Uncertain fixture")
    )
    out = reuse_evidence([source], [item], [feature], settings())
    assert (
        not out["reused"] and out["unresolved_prior"][0]["reason"] == "prior_ambiguous"
    )


def test_matrix_is_global_and_never_fills_missing_or_ambiguous_with_zero():
    from feature_space_pilot.full.matrix import matrix_state

    items = [dict(item_id="a"), dict(item_id="b")]
    features = [dict(feature_id="x"), dict(feature_id="y")]
    rows = [
        dict(item_id="a", feature_id="x", parse_status="ok", status="ok", applies=1),
        dict(item_id="b", feature_id="x", parse_status="ok", status="ok", applies=0),
        dict(
            item_id="a",
            feature_id="y",
            parse_status="ok",
            status="ambiguous",
            applies=None,
        ),
    ]
    state = matrix_state(items, features, rows)
    assert state["matrix"].tolist() == [[1, -1], [0, -1]]
    assert state["complete"] is False and state["planned_cells"] == 4
    assert {x["reason"] for x in state["unresolved"]} == {"ambiguous", "missing"}
    complete = matrix_state(
        items,
        features,
        rows[:2]
        + [
            dict(item_id=i, feature_id="y", parse_status="ok", status="ok", applies=0)
            for i in ["a", "b"]
        ],
    )
    assert complete["complete"] and set(complete["matrix"].ravel()) == {0, 1}


def test_duplicate_compatible_evidence_is_idempotent_but_conflicts_stay_unresolved():
    from feature_space_pilot.full.matrix import matrix_state

    row = dict(item_id="a", feature_id="x", parse_status="ok", status="ok", applies=1)
    s = matrix_state([dict(item_id="a")], [dict(feature_id="x")], [row, row])
    assert s["complete"]
    s = matrix_state(
        [dict(item_id="a")], [dict(feature_id="x")], [row, dict(row, applies=0)]
    )
    assert (
        not s["complete"]
        and s["unresolved"][0]["reason"] == "conflicting_resolved_evidence"
    )


def test_compatible_invalid_receipts_count_attempts_without_becoming_binary_evidence():
    from feature_space_pilot.full.reuse import reuse_evidence

    item, feature, source = fixture()
    record = source["judgments"][0]
    record.update(parse_status="error", raw_response="truncated")
    out = reuse_evidence([source], [item], [feature], settings())
    assert not out["reused"] and len(out["compatible_attempts"]) == 1
    assert out["compatible_attempts"][0]["job_id"] == record["job_id"]
    source["config"]["judge"]["model_id"] = "another-model"
    assert not reuse_evidence([source], [item], [feature], settings())[
        "compatible_attempts"
    ]

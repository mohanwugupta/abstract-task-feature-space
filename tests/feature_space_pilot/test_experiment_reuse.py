"""Offline reuse keeps the original request identity, and all unknown states."""

import copy
import pytest
from feature_space_pilot.io import canonical, digest
from feature_space_pilot.full.bank import build_bank
from feature_space_pilot.full.reuse import cell_job
from test_experiment_banks import item, generation


SETTINGS = dict(
    provider="fixture",
    base_url="https://fixture.invalid",
    model_id="fixture-judge",
    revision="fixture-judge-v1",
    max_output_tokens=400,
    reasoning_effort="low",
    supports_structured_outputs=True,
)


def fixture(value=1, status="ok"):
    from feature_space_pilot.experiments.banks import separate_banks

    items = [item("t", "task_instance"), item("h", "concept")]
    gs = [
        generation("t", ["requires accuracy", "permits inspection"]),
        generation("h", ["has a surface"]),
    ]
    pooled = build_bank(gs)
    banks = separate_banks(items, gs, pooled)
    f = next(f for f in pooled["features"] if f["text"] == "requires accuracy")
    j = cell_job(items[0], f, SETTINGS)
    parsed = dict(
        item_id="t",
        feature_id=f["feature_id"],
        applies=value,
        status=status,
        evidence="Synthetic applicability evidence",
    )
    r = dict(
        **parsed,
        raw_response=canonical(parsed),
        parse_status="ok",
        attempt=0,
        judge_model=SETTINGS["model_id"],
        job_id=j["job_id"],
        prompt_hash=j["prompt_hash"],
        payload_sha256=j["payload_sha256"],
        rubric_version="binary_v1"
    )
    r["response_hash"] = digest(r["raw_response"])
    r["record_hash"] = digest(r)
    source = dict(
        source_id="fixture",
        items=items,
        features=pooled["features"],
        settings=SETTINGS,
        literal_system=j["system"],
        records=[r],
    )
    return items, pooled, banks, source


def resign(record):
    record["response_hash"] = digest(record["raw_response"])
    record["record_hash"] = digest(
        {k: v for k, v in record.items() if k != "record_hash"}
    )


def test_id_only_reorganization_reuses_judgment_and_has_correct_dimensions():
    from feature_space_pilot.experiments.evidence import reconstruct_matrix

    items, pooled, banks, source = fixture()
    before = copy.deepcopy(source)
    out = reconstruct_matrix([items[0]], banks["task"], [source], SETTINGS)
    assert out["state"]["matrix"].shape == (1, 2)
    assert sorted(out["state"]["matrix"].ravel()) == [-1, 1]
    assert out["audit_summary"]["reused_binary_cells"] == 1
    cell = next(r for r in out["cell_audit"] if r["status"] == "reused_binary")
    assert cell["source_record_hashes"] == [source["records"][0]["record_hash"]]
    assert cell["feature_id"].startswith("tf_")
    assert source == before


@pytest.mark.parametrize(
    "change", ["text", "language", "model", "rubric", "prompt", "payload"]
)
def test_incompatible_requests_are_never_reused(change):
    from feature_space_pilot.experiments.evidence import reconstruct_matrix

    items, pooled, banks, s = fixture()
    if change == "text":
        s["items"][0] = dict(s["items"][0], text="different target")
    if change == "language":
        s["items"][0] = dict(s["items"][0], language="de")
    if change == "model":
        s["settings"] = dict(SETTINGS, revision="other-revision")
    if change == "rubric":
        s["records"][0]["rubric_version"] = "different-rubric"
    if change == "prompt":
        s["records"][0]["prompt_hash"] = "changed"
    if change == "payload":
        s["records"][0]["payload_sha256"] = "changed"
    resign(s["records"][0])
    out = reconstruct_matrix([items[0]], banks["task"], [s], SETTINGS)
    assert out["audit_summary"]["reused_binary_cells"] == 0
    assert out["audit_summary"]["rejected_receipts"] == 1
    assert (out["state"]["matrix"] == -1).all()


def test_ambiguous_and_conflicting_evidence_stays_unknown():
    from feature_space_pilot.experiments.evidence import reconstruct_matrix

    items, _, banks, s = fixture(None, "ambiguous")
    out = reconstruct_matrix([items[0]], banks["task"], [s], SETTINGS)
    assert (out["state"]["matrix"] == -1).all()
    assert "ambiguous" in {r["status"] for r in out["cell_audit"]}
    items, _, banks, a = fixture(1)
    b = fixture(0)[3]
    out = reconstruct_matrix([items[0]], banks["task"], [a, b], SETTINGS)
    assert (out["state"]["matrix"] == -1).all()
    assert "conflicting_resolved_evidence" in {r["status"] for r in out["cell_audit"]}


def test_tampered_receipts_fail_closed():
    from feature_space_pilot.experiments.evidence import reconstruct_matrix

    items, _, banks, source = fixture()
    source["records"][0]["applies"] = 0
    with pytest.raises(ValueError, match="hash"):
        reconstruct_matrix([items[0]], banks["task"], [source], SETTINGS)


def test_changed_canonical_wording_reuses_only_a_direct_safe_equivalence():
    from feature_space_pilot.experiments.banks import separate_banks
    from feature_space_pilot.experiments.evidence import reconstruct_matrix
    from test_experiment_banks import review

    left, right = sorted(
        ["requires accurate calculations", "requires exact arithmetic"], key=digest
    )
    items = [item("h", "concept"), item("t", "task_instance")]
    gs = [generation("h", [left]), generation("t", [right])]
    pooled = build_bank(gs, reviews=[review(left, right)])
    old = pooled["features"][0]
    assert old["text"] == left
    bank = separate_banks(items, gs, pooled)["task"]
    assert bank["features"][0]["text"] == right
    j = cell_job(items[1], old, SETTINGS)
    parsed = dict(
        item_id="t",
        feature_id=old["feature_id"],
        applies=1,
        status="ok",
        evidence="Synthetic equivalent proposition",
    )
    r = dict(
        parsed,
        raw_response=canonical(parsed),
        parse_status="ok",
        judge_model=SETTINGS["model_id"],
        job_id=j["job_id"],
        prompt_hash=j["prompt_hash"],
        payload_sha256=j["payload_sha256"],
        rubric_version="binary_v1",
    )
    resign(r)
    source = dict(
        source_id="fixture",
        items=items,
        features=pooled["features"],
        settings=SETTINGS,
        literal_system=j["system"],
        records=[r],
    )
    out = reconstruct_matrix([items[1]], bank, [source], SETTINGS)
    assert out["state"]["matrix"].tolist() == [[1]]
    assert (
        out["reused_receipts"][0]["meaning_proof"]["method"]
        == "direct_reviewed_equivalence"
    )
    unsafe = copy.deepcopy(bank)
    unsafe["old_to_new"][0].update(reusable_meaning=False, proof=None)
    assert reconstruct_matrix([items[1]], unsafe, [source], SETTINGS)["state"][
        "matrix"
    ].tolist() == [[-1]]

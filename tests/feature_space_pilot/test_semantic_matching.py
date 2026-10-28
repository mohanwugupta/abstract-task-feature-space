"""Automated matching is tested offline; synthetic grades never become human review."""

import copy
import json
from pathlib import Path
import pytest


def inventory():
    from feature_space_pilot.matching.protocol import build_inventory

    items = {
        "pencil": dict(
            item_id="pencil",
            text="the pencil",
            source_text="PENCIL",
            source_dataset="vinson_2008",
            split="development",
            language="en",
            target_type="object",
        ),
        "abstract": dict(
            item_id="abstract",
            text="creativity",
            source_text="Kreativität",
            source_dataset="harpaintner_2018",
            split="held_out",
            language="en",
            target_type="concept",
        ),
    }
    generations = [
        dict(
            job_id="g0",
            item_id="pencil",
            prompt_family="P0",
            replicate=0,
            parse_status="ok",
            features=["Has a graphite core.", "Is held in the hand."]
            + ["filler " + str(i) for i in range(12)],
        ),
        dict(
            job_id="g1",
            item_id="pencil",
            prompt_family="P1",
            replicate=0,
            parse_status="ok",
            features=["has a graphite core"],
        ),
        dict(
            job_id="g2",
            item_id="abstract",
            prompt_family="P1",
            replicate=0,
            parse_status="ok",
            features=["can produce new ideas"],
        ),
    ]
    human = [
        dict(item="PENCIL", feature="graphite", category="Perceptual"),
        dict(item="PENCIL", feature="handheld", category="Motoric"),
        dict(item="PENCIL", feature="graphite", category="Visual"),
    ]
    return build_inventory(items, generations, human)


def job():
    from feature_space_pilot.matching.protocol import build_jobs

    return next(j for j in build_jobs(inventory()) if not j.get("is_control"))


def response(job):
    return dict(
        item_id=job["item_id"],
        reviewed_all_labels=True,
        results=[
            dict(
                proposition_id=p["proposition_id"],
                matches=[],
                related=[],
                uncertain=[],
                evidence="Fixture",
            )
            for p in job["propositions"]
        ],
    )


def test_inventory_uses_all_human_labels_and_caps_each_original_response_before_dedup():
    inv = inventory()
    assert [i["item_id"] for i in inv["items"]] == ["pencil"]
    assert len(inv["norm_features"]) == 2 and len(inv["propositions"]) == 10
    assert {p["text"] for p in inv["propositions"]} == {
        "has a graphite core",
        "is held in the hand",
    } | {"filler " + str(i) for i in range(8)}
    graphite = next(
        p for p in inv["propositions"] if p["text"] == "has a graphite core"
    )
    assert (
        len(
            [
                o
                for o in inv["origins"]
                if o["proposition_id"] == graphite["proposition_id"]
            ]
        )
        == 2
    )


def test_matching_prompt_hides_origins_model_frequencies_and_split():
    from feature_space_pilot.matching.protocol import payload_for_job

    j = job()
    data = payload_for_job(j)
    assert len(data["norm_features"]) == 2
    assert not {"prompt_family", "replicate", "split", "frequency", "model_id"} & set(
        data
    )
    serialized = json.dumps(data)
    assert "P0" not in serialized and "development" not in serialized


@pytest.mark.parametrize(
    "mutation",
    ["unknown_label", "missing_prop", "duplicate_prop", "overlap", "not_reviewed"],
)
def test_parser_rejects_malformed_or_incomplete_matching(mutation):
    from feature_space_pilot.matching.protocol import parse_response

    j = job()
    r = response(j)
    h = j["norm_features"][0]["human_id"]
    if mutation == "unknown_label":
        r["results"][0]["matches"] = ["not-a-label"]
    if mutation == "missing_prop":
        r["results"].pop()
    if mutation == "duplicate_prop":
        r["results"].append(copy.deepcopy(r["results"][0]))
    if mutation == "overlap":
        r["results"][0].update(matches=[h], uncertain=[h])
    if mutation == "not_reviewed":
        r["reviewed_all_labels"] = False
    with pytest.raises(ValueError):
        parse_response(json.dumps(r), j)


def test_aggregation_counts_a_human_feature_once_and_keeps_unscored_work_unknown():
    from feature_space_pilot.matching.analysis import aggregate

    inv = inventory()
    j = job()
    r = response(j)
    r["results"][0]["matches"] = [j["norm_features"][0]["human_id"]]
    records = [
        dict(
            job_id=j["job_id"],
            item_id=j["item_id"],
            parse_status="ok",
            results=r["results"],
            is_control=False,
        )
    ]
    result = aggregate(inv, records)
    row = next(
        row
        for row in result["recovery"]
        if row["prompt_family"] == "P0" and row["replicates_included"] == 1
    )
    assert row["n_scored_propositions"] < row["n_expected_propositions"]
    assert row["recall"] is None and row["observed_recall_lower_bound"] == 0.5
    assert row["n_matched_human_features"] == 1
    assert result["human_reviewed"] is False


class FakeMatchingAPI:
    def __init__(self):
        from test_batch import FakeBatchAPI

        self.base = FakeBatchAPI()

    def upload(self, path):
        return self.base.upload(path)

    def create(self, file_id, metadata):
        return self.base.create(file_id, metadata)

    def list(self):
        return self.base.list()

    def count(self, payload):
        return self.base.count(payload)

    def content(self, fid):
        return self.base.content(fid)

    def retrieve(self, bid):
        from feature_space_pilot.io import canonical

        b = self.base.batches[bid]
        rows = []
        for row in reversed(
            [
                json.loads(line)
                for line in self.base.files[b["input_file_id"]].splitlines()
            ]
        ):
            body = row["body"]
            j = json.loads(body["input"][1]["content"])
            r = response(j)
            for score, proposition in zip(r["results"], j["propositions"]):
                text = proposition["text"]
                score["evidence"] = (
                    "Synthetic fixture, not an empirical matching result"
                )
                if "graphite" in text and "no graphite" not in text:
                    score["matches"] = ["h000"]
                elif "held" in text and "hand" in text:
                    score["matches"] = ["h001"]
            payload = dict(
                status="completed",
                model=body["model"],
                output=[
                    dict(
                        type="message",
                        content=[dict(type="output_text", text=canonical(r))],
                    )
                ],
                usage=dict(input_tokens=100, output_tokens=200),
            )
            rows.append(
                canonical(
                    dict(
                        custom_id=row["custom_id"],
                        response=dict(status_code=200, body=payload),
                        error=None,
                    )
                )
            )
        fid = "output-" + bid
        self.base.files[fid] = "\n".join(rows) + "\n"
        return dict(
            b,
            status="completed",
            output_file_id=fid,
            request_counts=dict(total=len(rows), completed=len(rows), failed=0),
        )


def fixture_plan(tmp_path):
    from feature_space_pilot.matching.run import load_settings, make_design

    root = Path(__file__).resolve().parents[2]
    c = load_settings(root / "configs/feature_space_semantic_matching.json")
    c.update(root=str(tmp_path), artifact_root="shared_ledger", demo=True)
    c["semantic_options"]["analysis_root"] = "matching"
    return make_design(
        inventory(),
        dict(
            source_run_id="synthetic_matching_fixture",
            source_manifest_sha256="fixture-v1",
            source_verified=False,
        ),
        c,
    )


def test_semantic_batch_pipeline_reserves_collects_once_and_never_claims_human_review(
    tmp_path,
):
    from feature_space_pilot.matching.run import advance, preflight

    p = fixture_plan(tmp_path)
    api = FakeMatchingAPI()
    first = advance(p, api=api, preflight_hash=preflight(p)["preflight_hash"])
    assert first["status"] == "batch_pending" and first["completed_requests"] == 0
    second = advance(p, api=api, preflight_hash=preflight(p)["preflight_hash"])
    assert second["status"] == "complete" and second["scored_propositions"] == 10
    assert second["human_reviewed"] is False and second["control_checks"]["passed"]
    assert second["demo"] is True and second["billed_execution"] is False
    assert "SYNTHETIC" in (Path(p["output_dir"]) / "analysis_summary.md").read_text()
    paid = second["analysis_accounted_usd"]
    before = (Path(p["output_dir"]) / "judgments.jsonl").read_bytes()
    third = advance(p, api=api, preflight_hash=preflight(p)["preflight_hash"])
    assert third["analysis_accounted_usd"] == paid and api.base.creates == 1
    assert (Path(p["output_dir"]) / "judgments.jsonl").read_bytes() == before


def test_semantic_frozen_input_tampering_blocks_before_more_submissions(tmp_path):
    from feature_space_pilot.matching.run import advance, preflight

    p = fixture_plan(tmp_path)
    api = FakeMatchingAPI()
    advance(p, api=api, preflight_hash=preflight(p)["preflight_hash"])
    path = Path(p["output_dir"]) / "inventory.json"
    data = json.loads(path.read_text())
    data["propositions"][0]["text"] = "changed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Frozen semantic"):
        advance(p, api=api, preflight_hash=preflight(p)["preflight_hash"])
    assert api.base.creates == 1


def test_live_matching_rejects_injected_or_unverified_sources_before_spending(tmp_path):
    from feature_space_pilot.matching.run import advance, preflight

    p = fixture_plan(tmp_path)
    p["config"]["demo"] = False
    api = FakeMatchingAPI()
    with pytest.raises(ValueError, match="verified sources"):
        advance(
            p, api=api, execute_live=True, preflight_hash=preflight(p)["preflight_hash"]
        )
    assert api.base.creates == 0 and not Path(p["output_dir"]).exists()


def test_prompt_comparisons_use_identical_completely_scored_targets():
    from feature_space_pilot.matching.analysis import paired_summary

    rows = []
    for item in ["a", "b", "c"]:
        for index, family in enumerate(["P0", "P1", "P2"]):
            complete = not (
                (item == "a" and family == "P1") or (item == "b" and family == "P0")
            )
            rows.append(
                dict(
                    item_id=item,
                    prompt_family=family,
                    scoring_complete=complete,
                    recall=(index + 1) / 10 if item == "c" else 0.9,
                )
            )
    summary, ids = paired_summary(rows)
    assert ids == ["c"]
    assert [summary[f]["mean_recall"] for f in ["P0", "P1", "P2"]] == [0.1, 0.2, 0.3]

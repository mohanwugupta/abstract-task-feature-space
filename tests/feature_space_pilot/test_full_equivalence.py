import copy
import json
import pytest
from feature_space_pilot.io import canonical, digest


def job():
    from feature_space_pilot.full.equivalence import build_jobs
    from feature_space_pilot.full.bank import build_bank, propose_equivalences
    from test_full_bank import generation

    bank = build_bank(
        [
            generation(
                [
                    "requires accurate arithmetic",
                    "requires exact arithmetic",
                    "arithmetic is rewarded",
                ]
            )
        ]
    )
    return build_jobs(bank, propose_equivalences(bank))[0]


def answer(j):
    return dict(
        results=[
            dict(
                pair_id=p["pair_id"],
                decision="equivalent",
                left_entails_right=True,
                right_entails_left=True,
                preserves={
                    k: True
                    for k in [
                        "negation",
                        "modality",
                        "roles",
                        "conditions",
                        "causal_direction",
                        "conjunction",
                        "quantification",
                    ]
                },
                evidence="Synthetic equivalence fixture",
            )
            for p in j["pairs"]
        ]
    )


def test_automatic_merge_requires_both_entailments_and_all_protection_checks():
    from feature_space_pilot.full.equivalence import parse_response

    j = job()
    a = answer(j)
    assert parse_response(canonical(a), j)["results"][0]["merge_eligible"]
    for change in ["one_way", "role", "related"]:
        b = copy.deepcopy(a)
        if change == "one_way":
            b["results"][0]["right_entails_left"] = False
        elif change == "role":
            b["results"][0]["preserves"]["roles"] = False
        else:
            b["results"][0]["decision"] = "related"
        assert parse_response(canonical(b), j)["results"][0]["merge_eligible"] is False


def test_equivalence_parser_rejects_missing_or_invented_pairs():
    from feature_space_pilot.full.equivalence import parse_response

    j = job()
    a = answer(j)
    with pytest.raises(ValueError, match="pair"):
        parse_response(canonical(dict(results=[])), j)
    a["results"][0]["pair_id"] = "invented"
    with pytest.raises(ValueError, match="pair"):
        parse_response(canonical(a), j)


def test_automated_receipts_merge_without_claiming_a_named_human_review():
    from feature_space_pilot.full.equivalence import (
        parse_response,
        reviews_from_records,
    )
    from feature_space_pilot.full.bank import build_bank
    from test_full_bank import generation

    j = job()
    raw = canonical(answer(j))
    parsed = parse_response(raw, j)
    r = dict(
        job_id=j["job_id"],
        parse_status="ok",
        raw_response=raw,
        response_hash=digest(raw),
        pair_job=j,
        grader_model="synthetic-fixture-model",
        **parsed
    )
    r["record_hash"] = digest(r)
    reviews = reviews_from_records([r])
    result = build_bank(
        [
            generation(
                [
                    "requires accurate arithmetic",
                    "requires exact arithmetic",
                    "arithmetic is rewarded",
                ]
            )
        ],
        reviews=reviews,
    )
    assert len(result["features"]) == 2
    assert (
        reviews[0]["human_reviewed"] is False
        and reviews[0]["review_kind"] == "automated_bidirectional_entailment"
    )
    assert not result["human_reviewed"]


def test_proposals_for_incompatible_predicates_are_never_sent_for_automatic_merge():
    from feature_space_pilot.full.equivalence import build_jobs
    from feature_space_pilot.full.bank import build_bank, propose_equivalences
    from test_full_bank import generation

    b = build_bank([generation(["requires accuracy", "accuracy is rewarded"])])
    assert not build_jobs(b, propose_equivalences(b))


def test_equivalence_batch_uses_separate_journal_and_model_review_provenance(tmp_path):
    from feature_space_pilot.full.equivalence import (
        EquivalenceSession,
        reviews_from_records,
    )
    from feature_space_pilot.full.budget import IndexedLedger
    from feature_space_pilot.engine import load_config, _read
    from test_batch import FakeBatchAPI

    c = load_config("configs/feature_space_pilot.json")
    c["demo"] = True
    c["equivalence"] = dict(c["judge"], max_output_tokens=2000)

    class API(FakeBatchAPI):
        def retrieve(self, bid):
            b = self.batches[bid]
            rows = []
            for line in self.files[b["input_file_id"]].splitlines():
                x = json.loads(line)
                data = json.loads(x["body"]["input"][1]["content"])
                j = dict(pairs=data["pairs"])
                body = dict(
                    status="completed",
                    model=x["body"]["model"],
                    output=[
                        dict(
                            type="message",
                            content=[
                                dict(type="output_text", text=canonical(answer(j)))
                            ],
                        )
                    ],
                    usage=dict(input_tokens=100, output_tokens=100),
                )
                rows.append(
                    dict(
                        custom_id=x["custom_id"],
                        response=dict(status_code=200, body=body),
                        error=None,
                    )
                )
            fid = "out-" + bid
            self.files[fid] = "\n".join(canonical(r) for r in rows) + "\n"
            return dict(
                b,
                status="completed",
                output_file_id=fid,
                request_counts=dict(total=len(rows), completed=len(rows), failed=0),
            )

    api = API()
    out = tmp_path / "equivalence"
    out.mkdir()
    with IndexedLedger(
        tmp_path / "ledger.jsonl", cap=100, stage_caps=c["stage_caps"]
    ) as ledger:
        s = EquivalenceSession(c, out, ledger, api=api)
        s.enqueue_pair_job(job())
        s.flush("adjudication")
        s.refresh()
        assert not s.active and not (out / "judgments.jsonl").exists()
        receipts = _read(out / "equivalences.jsonl")
        reviews = reviews_from_records(receipts)
        assert reviews and all(r["human_reviewed"] is False for r in reviews)
        paid = ledger.total
        s.refresh()
        assert ledger.total == paid and api.creates == 1

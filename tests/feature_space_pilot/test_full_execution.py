import json
from pathlib import Path
import pytest


def execution_fixture(tmp):
    from feature_space_pilot.engine import load_config
    from feature_space_pilot.full.budget import profile_inputs
    from feature_space_pilot.io import digest

    c = load_config("configs/feature_space_pilot.json")
    c.update(root=str(tmp), demo=True, batch_max_requests=1, batch_max_in_flight=3)
    items = [
        dict(
            item_id="a",
            text="the pencil",
            language="en",
            content_hash=digest("the pencil"),
        ),
        dict(
            item_id="b",
            text="the ruler",
            language="en",
            content_hash=digest("the ruler"),
        ),
    ]
    f = dict(
        feature_id="f_" + digest("requires accuracy")[:20], text="requires accuracy"
    )
    profile = profile_inputs(items, c["judge"], lambda p: dict(input_tokens=250))
    return c, items, f, profile


def test_full_batch_round_compacts_state_and_resumes_without_replaying_cells(tmp_path):
    from feature_space_pilot.full.execution import FullSession
    from feature_space_pilot.full.budget import IndexedLedger
    from test_batch import FakeBatchAPI

    c, items, f, profile = execution_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    api = FakeBatchAPI()
    with IndexedLedger(
        tmp_path / "ledger.jsonl", cap=100, stage_caps=c["stage_caps"]
    ) as ledger:
        s = FullSession(c, out, ledger, api=api)
        for i in items:
            s.enqueue_cell(i, f, profile)
        s.flush("judgment")
        assert api.creates == 2
        s.refresh()
        assert (
            not s.active and len(s.state["tasks"]) == 0 and len(s.state["batches"]) == 0
        )
        settled = ledger.total
        before = (out / "judgments.jsonl").read_bytes()
        resumed = FullSession(c, out, ledger, api=api)
        for i in items:
            resumed.enqueue_cell(i, f, profile)
        resumed.flush("judgment")
        resumed.refresh()
        assert api.creates == 2 and ledger.total == settled
        assert (out / "judgments.jsonl").read_bytes() == before
        archive = [
            json.loads(l)
            for l in (out / "completed_batches.jsonl").read_text().splitlines()
        ]
        assert len(archive) == 2 and all(a["input_sha256"] for a in archive)


def test_full_batch_budget_checks_before_any_submission(tmp_path):
    from feature_space_pilot.full.execution import FullSession
    from feature_space_pilot.full.budget import IndexedLedger
    from test_batch import FakeBatchAPI

    c, items, f, profile = execution_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    api = FakeBatchAPI()
    with IndexedLedger(
        tmp_path / "ledger.jsonl", cap=0.000001, stage_caps={"judgment": 0.000001}
    ) as ledger:
        s = FullSession(c, out, ledger, api=api)
        s.enqueue_cell(items[0], f, profile)
        with pytest.raises(ValueError, match="budget"):
            s.flush("judgment")
        assert api.creates == 0 and ledger.total == 0


def test_preexisting_unlocated_job_is_held_instead_of_submitted_again(tmp_path):
    from feature_space_pilot.full.execution import FullSession
    from feature_space_pilot.full.reuse import cell_job
    from feature_space_pilot.full.budget import IndexedLedger
    from test_batch import FakeBatchAPI

    c, items, f, profile = execution_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    api = FakeBatchAPI()
    with IndexedLedger(
        tmp_path / "ledger.jsonl", cap=100, stage_caps=c["stage_caps"]
    ) as ledger:
        jid = cell_job(items[0], f, c["judge"])["job_id"]
        ledger.reserve(jid, "judgment", 0.001)
        s = FullSession(c, out, ledger, api=api)
        s.enqueue_cell(items[0], f, profile)
        s.flush("judgment")
        assert api.creates == 0 and jid in s.unlocated_prior_jobs


def test_owned_invalid_receipt_can_retry_after_collection_compacts_its_batch(tmp_path):
    from feature_space_pilot.full.execution import FullSession
    from feature_space_pilot.full.budget import IndexedLedger
    from test_batch import FakeBatchAPI

    c, items, f, profile = execution_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    class API(FakeBatchAPI):
        def retrieve(self, bid):
            b = super().retrieve(bid)
            if bid == "batch-1":
                rows = [
                    json.loads(l) for l in self.files[b["output_file_id"]].splitlines()
                ]
                for r in rows:
                    r["response"]["body"]["status"] = "incomplete"
                self.files[b["output_file_id"]] = (
                    "\n".join(json.dumps(r) for r in rows) + "\n"
                )
            return b

    api = API()
    with IndexedLedger(
        tmp_path / "ledger.jsonl", cap=100, stage_caps=c["stage_caps"]
    ) as ledger:
        s = FullSession(c, out, ledger, api=api)
        s.enqueue_cell(items[0], f, profile)
        s.flush("judgment")
        s.refresh()
        s.enqueue_cell(items[0], f, profile)
        s.flush("judgment")
        assert api.creates == 2 and not s.unlocated_prior_jobs

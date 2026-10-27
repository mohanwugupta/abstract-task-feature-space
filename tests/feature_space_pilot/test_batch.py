"""No live requests: exercise actual batch preparation, durable state and ingestion."""

import json
from pathlib import Path
import pytest
from feature_space_pilot.io import canonical
from feature_space_pilot.engine import load_config
from feature_space_pilot.budget import Ledger
from feature_space_pilot.clients import FakeClient, generation_schema

ROOT = Path(__file__).resolve().parents[2]


class FakeBatchAPI:
    def __init__(self):
        self.files = {}
        self.batches = {}
        self.creates = 0
        self.on_create = None
        self.timeout_after_create = False
        self.corrupt = False

    def upload(self, path):
        fid = "file-" + str(len(self.files))
        self.files[fid] = Path(path).read_text()
        return dict(id=fid)

    def create(self, file_id, metadata):
        self.creates += 1
        if self.on_create:
            self.on_create()
        bid = "batch-" + str(self.creates)
        self.batches[bid] = dict(
            id=bid, status="validating", input_file_id=file_id, metadata=metadata
        )
        if self.timeout_after_create:
            raise TimeoutError("simulated lost acknowledgement")
        return self.batches[bid]

    def list(self):
        return list(self.batches.values())

    def count(self, payload):
        return dict(input_tokens=100)

    def retrieve(self, bid):
        b = self.batches[bid]
        lines = []
        for row in reversed(
            [json.loads(l) for l in self.files[b["input_file_id"]].splitlines()]
        ):
            body = row["body"]
            fake = FakeClient()
            messages = body.get("messages", body.get("input"))
            schema = (
                body["text"]["format"]["schema"]
                if "input" in body
                else body["response_format"]["json_schema"]["schema"]
            )
            raw, usage = fake.complete(
                system=messages[0]["content"],
                user=messages[1]["content"],
                schema=schema,
            )
            response = dict(
                model=body["model"],
                id="response-" + row["custom_id"],
                choices=[dict(message=dict(content=raw), finish_reason="stop")],
                usage=dict(
                    prompt_tokens=usage["input_tokens"],
                    completion_tokens=usage["output_tokens"],
                ),
            )
            if "input" in body:
                response = dict(
                    model=body["model"],
                    id="response-" + row["custom_id"],
                    status="completed",
                    output=[
                        dict(
                            type="message", content=[dict(type="output_text", text=raw)]
                        )
                    ],
                    usage=dict(
                        input_tokens=usage["input_tokens"],
                        output_tokens=usage["output_tokens"],
                    ),
                )
            lines.append(
                canonical(
                    dict(
                        custom_id="foreign" if self.corrupt else row["custom_id"],
                        response=dict(status_code=200, body=response),
                        error=None,
                    )
                )
            )
        fid = "output-" + bid
        self.files[fid] = "\n".join(lines) + "\n"
        return dict(
            b,
            status="completed",
            output_file_id=fid,
            request_counts=dict(completed=len(lines), failed=0, total=len(lines)),
        )

    def content(self, fid):
        return self.files[fid]


def setup_session(tmp_path, api, ledger):
    from feature_space_pilot.batch import BatchSession

    c = load_config(ROOT / "configs/feature_space_pilot.json")
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    return BatchSession(c, run, ledger, api=api)


def enqueue(session, count=2):
    c = load_config(ROOT / "configs/feature_space_pilot.json")
    for n in range(count):
        item = "item" + str(n)
        session.enqueue(
            prompt=dict(
                system="Fixture",
                user=canonical(
                    dict(item_id=item, text="writing a poem", language="en")
                ),
            ),
            schema=generation_schema(item, 4),
            settings=c["generator"],
            job_id="job" + str(n),
            stage="generation",
            journal_name="generations.jsonl",
            record_base=dict(item_id=item, prompt_family="P1", replicate=0),
        )


def test_batch_pre_reserves_then_ingests_unordered_results_once(tmp_path):
    from feature_space_pilot.engine import _read

    api = FakeBatchAPI()
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(generation=1)
    ) as ledger:
        session = setup_session(tmp_path, api, ledger)
        enqueue(session)

        def before_create():
            assert len(ledger.events) == 2

        api.on_create = before_create
        session.flush("generation")
        assert len(ledger.events) == 2 and api.creates == 1
        session.refresh()
        records = _read(tmp_path / "run/generations.jsonl")
        assert {r["job_id"] for r in records} == {"job0", "job1"}
        assert all(
            r["parse_status"] == "ok" and r["execution_mode"] == "batch"
            for r in records
        )
        assert ledger.report()["unsettled_reservations"] == 0
        expected = 2 * (80 * 0.05 + 40 * 0.25) / 1_000_000
        assert ledger.total == pytest.approx(expected)
        before = (tmp_path / "run/generations.jsonl").read_bytes()
        balance = ledger.total
        session.refresh()
        assert ledger.total == balance
        assert (tmp_path / "run/generations.jsonl").read_bytes() == before


def test_uncertain_creation_recovers_by_metadata_without_resubmission(tmp_path):
    from feature_space_pilot.batch import BatchSession

    api = FakeBatchAPI()
    api.timeout_after_create = True
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(generation=1)
    ) as ledger:
        session = setup_session(tmp_path, api, ledger)
        enqueue(session, 1)
        with pytest.raises(ValueError, match="uncertain|acknowledgement"):
            session.flush("generation")
        resumed = setup_session(tmp_path, api, ledger)
        resumed.refresh()
        assert api.creates == 1 and (tmp_path / "run/generations.jsonl").exists()
        assert ledger.report()["unsettled_reservations"] == 0


def test_unmatched_create_does_not_block_other_collections_or_resubmit(tmp_path):
    api = FakeBatchAPI()
    create = api.create
    api.create = lambda *args: (_ for _ in ()).throw(
        TimeoutError("lost before acknowledgement")
    )
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(generation=1)
    ) as ledger:
        session = setup_session(tmp_path, api, ledger)
        session.max_in_flight = 2
        enqueue(session, 1)
        with pytest.raises(ValueError, match="uncertain"):
            session.flush("generation")
        api.create = create
        # Resume as after a crash; the unmatched job stays reserved/in flight.
        resumed = setup_session(tmp_path, api, ledger)
        resumed.max_in_flight = 2
        enqueue(resumed, 2)
        resumed.flush("generation")
        resumed.refresh()
        from feature_space_pilot.engine import _read

        assert [r["job_id"] for r in _read(tmp_path / "run/generations.jsonl")] == [
            "job1"
        ]
        assert ledger.report()["unsettled_reservations"] == 1
        assert api.creates == 1 and resumed.active
        resumed.refresh()
        assert api.creates == 1


def test_one_transient_batch_read_does_not_block_other_results(tmp_path):
    from feature_space_pilot.batch import BatchSession, BatchTransientError
    from feature_space_pilot.engine import _read

    api = FakeBatchAPI()
    c = load_config(ROOT / "configs/feature_space_pilot.json")
    c.update(batch_max_requests=1, batch_max_in_flight=2)
    out = tmp_path / "run"
    out.mkdir()
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(generation=1)
    ) as ledger:
        session = BatchSession(c, out, ledger, api=api)
        enqueue(session, 2)
        session.flush("generation")
        retrieve = api.retrieve

        def interrupted(bid):
            if bid == "batch-1":
                raise BatchTransientError("temporary network failure")
            return retrieve(bid)

        api.retrieve = interrupted
        session.refresh()
        assert len(_read(out / "generations.jsonl")) == 1
        assert ledger.report()["unsettled_reservations"] == 1
        api.retrieve = retrieve
        session.refresh()
        assert len(_read(out / "generations.jsonl")) == 2 and api.creates == 2


def test_transport_classifies_connection_failure_without_logging_details(monkeypatch):
    import urllib.error
    import urllib.request
    from feature_space_pilot.batch import OpenAIBatchAPI, BatchTransientError

    c = load_config(ROOT / "configs/feature_space_pilot.json")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-a-real-key")

    def unavailable(*args, **kwargs):
        raise urllib.error.URLError("provider-private-detail")

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    with pytest.raises(BatchTransientError, match="connection failure") as error:
        OpenAIBatchAPI(c["judge"]).list()
    assert "provider-private-detail" not in str(error.value)


def test_unknown_output_id_never_settles_or_creates_journal(tmp_path):
    api = FakeBatchAPI()
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(generation=1)
    ) as ledger:
        session = setup_session(tmp_path, api, ledger)
        enqueue(session)
        session.flush("generation")
        api.corrupt = True
        with pytest.raises(ValueError, match="custom_id|request ID"):
            session.refresh()
        assert ledger.report()["unsettled_reservations"] == 2
        assert not (tmp_path / "run/generations.jsonl").exists()


def test_batch_submission_fails_before_any_call_when_reservation_exceeds_cap(tmp_path):
    api = FakeBatchAPI()
    with Ledger(
        tmp_path / "ledger.jsonl", cap=0.000001, stage_caps=dict(generation=0.000001)
    ) as ledger:
        session = setup_session(tmp_path, api, ledger)
        enqueue(session)
        with pytest.raises(ValueError, match="budget"):
            session.flush("generation")
        assert not ledger.events and api.creates == 0


def test_default_batch_discount_and_online_override_do_not_change_standard_prices():
    from feature_space_pilot.execution import execution_config

    c = load_config(ROOT / "configs/feature_space_pilot.json")
    assert c["execution_mode"] == "batch"
    assert c["generator"]["prices"]["input_per_million"] == 0.05
    assert c["judge"]["prices"]["output_per_million"] == 0.25
    online = execution_config(c, "online")
    assert online["generator"]["prices"]["input_per_million"] == 0.1
    assert c["generator"]["prices"]["input_per_million"] == 0.05


def test_real_pipeline_advances_two_batch_phases_and_never_replays_completed_jobs(
    tmp_path, monkeypatch
):
    import feature_space_pilot.batch as module
    import feature_space_pilot.engine as engine
    from feature_space_pilot.report import verify_manifest

    api = FakeBatchAPI()
    monkeypatch.setattr(module, "OpenAIBatchAPI", lambda settings: api)
    monkeypatch.setattr(engine, "load_api_credentials", lambda config: None)
    c = load_config(ROOT / "configs/feature_space_pilot.json")
    c["artifact_root"] = str(tmp_path / "artifacts")

    def advance():
        return engine.run_pipeline(
            c,
            "tasks",
            execute_live=True,
            preflight_hash=engine.preflight(c, "tasks")["preflight_hash"],
        )

    first = advance()
    assert (
        first["status"] == "batch_pending" and first["completed_generation_jobs"] == 0
    )
    second = advance()
    assert (
        second["status"] == "batch_pending"
        and second["completed_generation_jobs"] == 135
    )
    third = advance()
    assert (
        third["status"] == "complete"
        and third["resolved_judgments"] == third["planned_judgments"]
    )
    assert api.creates == 2
    verify_manifest(third["output_dir"])
    fourth = advance()
    assert fourth["status"] == "complete" and api.creates == 2


def test_expired_batch_preserves_successes_and_releases_only_explicitly_unexecuted(
    tmp_path,
):
    from feature_space_pilot.engine import _read

    api = FakeBatchAPI()
    original = api.retrieve

    def expired(bid):
        result = original(bid)
        fid = result["output_file_id"]
        rows = [json.loads(l) for l in api.files[fid].splitlines()]
        api.files[fid] = canonical(rows[0]) + "\n"
        error_id = "errors"
        api.files[error_id] = (
            canonical(
                dict(
                    custom_id=rows[1]["custom_id"],
                    response=None,
                    error=dict(code="batch_expired"),
                )
            )
            + "\n"
        )
        return dict(result, status="expired", error_file_id=error_id)

    api.retrieve = expired
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(generation=1)
    ) as ledger:
        session = setup_session(tmp_path, api, ledger)
        enqueue(session)
        session.flush("generation")
        session.refresh()
        records = _read(tmp_path / "run/generations.jsonl")
        assert sum(r["parse_status"] == "ok" for r in records) == 1
        assert ledger.report()["unsettled_reservations"] == 0
        assert ledger.total == pytest.approx((80 * 0.05 + 40 * 0.25) / 1_000_000)


def test_pre_creation_upload_interruption_recovers_existing_reservations(tmp_path):
    api = FakeBatchAPI()
    upload = api.upload
    api.upload = lambda path: (_ for _ in ()).throw(TimeoutError("upload interrupted"))
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(generation=1)
    ) as ledger:
        session = setup_session(tmp_path, api, ledger)
        enqueue(session)
        with pytest.raises(TimeoutError):
            session.flush("generation")
        assert len(ledger.events) == 2
        api.upload = upload
        resumed = setup_session(tmp_path, api, ledger)
        resumed.refresh()
        assert api.creates == 1 and ledger.report()["unsettled_reservations"] == 0
        assert sum(e["event"] == "reserve" for e in ledger.events) == 2


def test_luna_responses_payload_requires_provider_count_before_any_reservation(
    tmp_path,
):
    c = load_config(ROOT / "configs/feature_space_pilot.json")
    assert (
        c["generator"]["model_id"] == "gpt-6-luna"
        and c["generator"]["api_format"] == "responses"
    )
    api = FakeBatchAPI()
    api.count = lambda payload: dict(input_tokens=20000)
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(generation=1)
    ) as ledger:
        session = setup_session(tmp_path, api, ledger)
        with pytest.raises(ValueError, match="Provider input-token count"):
            enqueue(session, 1)
        assert ledger.events == [] and api.creates == 0


def test_batch_job_jsonl_uses_same_responses_schema_and_frozen_body(tmp_path):
    api = FakeBatchAPI()
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(generation=1)
    ) as ledger:
        session = setup_session(tmp_path, api, ledger)
        enqueue(session, 1)
        session.flush("generation")
        batch = next(iter(api.batches.values()))
        row = json.loads(api.files[batch["input_file_id"]])
        assert row["url"] == "/v1/responses"
        assert (
            row["body"]["model"] == "gpt-6-luna"
            and row["body"]["reasoning"]["effort"] == "low"
        )
        assert row["body"]["text"]["format"]["strict"]
        assert row["body"]["input"][1]["content"] == canonical(
            dict(item_id="item0", text="writing a poem", language="en")
        )


def test_batch_pricing_cannot_be_used_by_synchronous_client():
    from feature_space_pilot.clients import HostedClient

    c = load_config(ROOT / "configs/feature_space_pilot.json")
    with pytest.raises(ValueError, match="online|synchronous"):
        HostedClient(c["generator"])


@pytest.mark.parametrize("after_append", [False, True])
def test_collection_crash_after_settlement_recovers_without_double_charge(
    tmp_path, monkeypatch, after_append
):
    import feature_space_pilot.batch as module
    from feature_space_pilot.engine import _read

    api = FakeBatchAPI()
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(generation=1)
    ) as ledger:
        session = setup_session(tmp_path, api, ledger)
        enqueue(session, 1)
        session.flush("generation")
        original = module.append_jsonl

        def interrupted(path, record):
            if after_append:
                original(path, record)
            raise OSError("simulated collection interruption")

        monkeypatch.setattr(module, "append_jsonl", interrupted)
        with pytest.raises(OSError):
            session.refresh()
        assert sum(e["event"] == "settle" for e in ledger.events) == 1
        paid = ledger.total
        monkeypatch.setattr(module, "append_jsonl", original)
        resumed = setup_session(tmp_path, api, ledger)
        resumed.refresh()
        assert api.creates == 1 and ledger.total == paid
        assert len(_read(tmp_path / "run/generations.jsonl")) == 1
        assert sum(e["event"] == "settle" for e in ledger.events) == 1


def test_batch_chunk_limit_advances_only_unfinished_jobs(tmp_path):
    from feature_space_pilot.batch import BatchSession
    from feature_space_pilot.engine import _read

    api = FakeBatchAPI()
    c = load_config(ROOT / "configs/feature_space_pilot.json")
    c["batch_max_requests"] = 1
    c["batch_max_in_flight"] = 1
    out = tmp_path / "run"
    out.mkdir()
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(generation=1)
    ) as ledger:
        for n in range(2):
            session = BatchSession(c, out, ledger, api=api)
            session.refresh()
            done = {r["job_id"] for r in _read(out / "generations.jsonl")}
            # The engine skips successful IDs, so only remaining jobs enter enqueue.
            for item in range(2):
                if "job" + str(item) not in done:
                    session.enqueue(
                        prompt=dict(
                            system="Fixture",
                            user=canonical(
                                dict(
                                    item_id="item" + str(item),
                                    text="writing a poem",
                                    language="en",
                                )
                            ),
                        ),
                        schema=generation_schema("item" + str(item), 4),
                        settings=c["generator"],
                        job_id="job" + str(item),
                        stage="generation",
                        journal_name="generations.jsonl",
                        record_base=dict(
                            item_id="item" + str(item), prompt_family="P1", replicate=0
                        ),
                    )
            session.flush("generation")
            assert api.creates == n + 1
        session.refresh()
        assert len(_read(out / "generations.jsonl")) == 2
        assert ledger.report()["unsettled_reservations"] == 0


def test_multiple_chunks_submit_before_waiting_and_never_duplicate_active_jobs(
    tmp_path,
):
    from feature_space_pilot.batch import BatchSession

    api = FakeBatchAPI()
    c = load_config(ROOT / "configs/feature_space_pilot.json")
    c.update(
        batch_max_requests=1, batch_max_in_flight=2, batch_max_queued_input_tokens=2000
    )
    out = tmp_path / "run"
    out.mkdir()
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(generation=1)
    ) as ledger:
        session = BatchSession(c, out, ledger, api=api)
        enqueue(session, 3)
        session.flush("generation")
        assert api.creates == 2
        assert len(session.state["tasks"]) == 2
        resumed = BatchSession(c, out, ledger, api=api)
        enqueue(resumed, 3)
        resumed.flush("generation")
        assert api.creates == 2 and len(ledger.events) == 2
        resumed.refresh()
        assert ledger.report()["unsettled_reservations"] == 0


def test_concurrent_chunks_obey_combined_input_token_window(tmp_path):
    from feature_space_pilot.batch import BatchSession

    api = FakeBatchAPI()
    c = load_config(ROOT / "configs/feature_space_pilot.json")
    c.update(
        batch_max_requests=1, batch_max_in_flight=4, batch_max_queued_input_tokens=700
    )
    out = tmp_path / "run"
    out.mkdir()
    with Ledger(
        tmp_path / "ledger.jsonl", cap=1, stage_caps=dict(generation=1)
    ) as ledger:
        session = BatchSession(c, out, ledger, api=api)
        enqueue(session, 3)
        session.flush("generation")
        assert api.creates == 1  # Each exact count + reserve is 612 tokens.
        session.refresh()
        assert ledger.report()["unsettled_reservations"] == 0

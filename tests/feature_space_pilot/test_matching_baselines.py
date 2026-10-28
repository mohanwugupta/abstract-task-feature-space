"""Synthetic baseline checks run offline and never establish semantic accuracy."""

import copy
import json
from pathlib import Path
import numpy as np
import pytest


def inventory():
    return dict(
        items=[
            dict(item_id=i, text=i, split=s)
            for i, s in [("dev", "development"), ("test", "held_out")]
        ],
        norm_features=[
            dict(item_id=i, human_id=h, label=t, categories=[])
            for i in ["dev", "test"]
            for h, t in [("h0", "graphite"), ("h1", "handheld")]
        ],
        propositions=[
            dict(item_id=i, proposition_id=p, text=t)
            for i in ["dev", "test"]
            for p, t in [("p0", "has a graphite core"), ("p1", "contains no graphite")]
        ],
        origins=[],
        comparison_cap=10,
    )


def decisions(inv=None):
    inv = inv or inventory()
    return {
        (i["item_id"], p): dict(
            matches=["h0"] if p == "p0" else [],
            related=[],
            uncertain=["h1"] if p == "p1" else [],
        )
        for i in inv["items"]
        for p in ["p0", "p1"]
    }


def test_lexical_scores_preserve_negation_and_short_label_asymmetry():
    from feature_space_pilot.baselines.core import lexical_scores

    s = lexical_scores("Has a graphite core.", "graphite")
    assert (
        s["exact"] == 0
        and s["token_jaccard"] == 0.25
        and s["label_token_coverage"] == 1
    )
    # A deliberately naive baseline can misclassify a contradiction; no hidden LLM repair.
    assert (
        lexical_scores("contains no graphite", "graphite")["label_token_coverage"] == 1
    )
    assert lexical_scores(" A  B. ", "a b")["exact"] == 1
    assert lexical_scores("", "")["token_jaccard"] == 0


def test_pair_universe_is_within_target_and_unknown_judge_labels_stay_unknown():
    from feature_space_pilot.baselines.core import build_pairs

    inv = inventory()
    d = decisions()
    del d["test", "p0"]
    rows = build_pairs(inv, d)
    assert len(rows) == 8
    assert sum(r["judge_relation"] == "unscored" for r in rows) == 2
    assert sum(r["judge_relation"] == "uncertain" for r in rows) == 2
    assert sum(r["judge_label"] is None for r in rows) == 4
    assert (
        next(
            r
            for r in rows
            if r["item_id"] == "dev"
            and r["proposition_id"] == "p1"
            and r["human_id"] == "h1"
        )["judge_relation"]
        == "uncertain"
    )
    assert all(r["item_id"] in {"dev", "test"} for r in rows)


def test_thresholds_never_use_heldout_or_incomplete_development_targets():
    from feature_space_pilot.baselines.core import build_pairs, calibrate, evaluate

    rows = build_pairs(inventory(), decisions())
    fit = calibrate(rows, inventory())
    threshold = fit["methods"]["token_jaccard"]["threshold"]
    altered = copy.deepcopy(rows)
    for r in altered:
        if r["split"] == "held_out" and r["judge_label"] is not None:
            r["judge_label"] = 1 - r["judge_label"]
    assert calibrate(altered, inventory()) == fit
    broken = copy.deepcopy(rows)
    broken[0]["judge_relation"] = "unscored"
    broken[0]["judge_label"] = None
    assert calibrate(broken, inventory())["available"] is False
    result = evaluate(rows, fit, split="held_out")
    assert result["methods"]["token_jaccard"]["threshold"] == threshold
    assert result["n_pairs"] == 3 and result["uncertain_pairs_excluded"] == 1
    assert result["reference_kind"] == "automated_luna_decisions_not_human_gold"


def test_cosine_requires_complete_nonzero_finite_vectors_and_uses_ids():
    from feature_space_pilot.baselines.core import cosine_scores, embedding_texts

    inv = inventory()
    texts = embedding_texts(inv)
    assert len(texts) == 8 and all(
        "Target:" in x["text"] and "Feature:" in x["text"] for x in texts
    )
    vectors = {x["text_id"]: np.array([1.0, 0.0]) for x in texts}
    scores = cosine_scores(inv, vectors)
    assert len(scores) == 8 and all(np.isclose(v, 1) for v in scores.values())
    vectors.pop(next(iter(vectors)))
    with pytest.raises(ValueError, match="Missing"):
        cosine_scores(inv, vectors)


def embedding_config(tmp):
    from feature_space_pilot.baselines.embedding import settings
    from feature_space_pilot.engine import load_config

    c = load_config("configs/feature_space_pilot.json")
    c.update(root=str(tmp), artifact_root="shared", demo=True)
    c["baseline_embedding"] = settings()
    return c


class FakeEmbeddingAPI:
    def __init__(self):
        from test_batch import FakeBatchAPI

        self.base = FakeBatchAPI()
        self.base.mode = "normal"

    def upload(self, p):
        return self.base.upload(p)

    def create(self, f, metadata):
        return self.base.create(f, metadata)

    def list(self):
        return self.base.list()

    def content(self, f):
        return self.base.content(f)

    def retrieve(self, bid):
        b = self.base.batches[bid]
        rows = []
        for row in [
            json.loads(l) for l in self.base.files[b["input_file_id"]].splitlines()
        ]:
            assert row["url"] == "/v1/embeddings"
            body = row["body"]
            assert set(body) == {"model", "input", "encoding_format", "dimensions"}
            n = body["dimensions"]
            data = [
                dict(index=i, embedding=[float(i + 1)] + [0.0] * (n - 1))
                for i in reversed(range(len(body["input"])))
            ]
            rows.append(
                dict(
                    custom_id=row["custom_id"],
                    error=None,
                    response=dict(
                        status_code=200,
                        body=dict(
                            model=body["model"],
                            data=data,
                            usage=dict(prompt_tokens=40, total_tokens=40),
                        ),
                    ),
                )
            )
        fid = "emb-output-" + bid
        self.base.files[fid] = "\n".join(json.dumps(r) for r in rows) + "\n"
        return dict(
            b,
            status="completed",
            output_file_id=fid,
            request_counts=dict(total=len(rows), completed=len(rows), failed=0),
        )


def test_embedding_batch_uses_embedding_endpoint_and_settles_once(tmp_path):
    from feature_space_pilot.baselines.embedding import (
        EmbeddingSession,
        make_jobs,
        load_vectors,
    )
    from feature_space_pilot.budget import Ledger
    from feature_space_pilot.baselines.core import embedding_texts

    c = embedding_config(tmp_path)
    out = tmp_path / "embedding"
    out.mkdir()
    api = FakeEmbeddingAPI()
    jobs = make_jobs(embedding_texts(inventory()), c["baseline_embedding"])
    with Ledger(
        tmp_path / "ledger.jsonl", cap=100, stage_caps=c["stage_caps"]
    ) as ledger:
        s = EmbeddingSession(c, out, ledger, api=api)
        for j in jobs:
            s.enqueue_job(j)
        s.flush("embeddings")
        reserved = ledger.total
        assert s.active and api.base.creates == 1
        s.refresh()
        assert not s.active and ledger.total < reserved
        before = ledger.total
        raw = (out / "embeddings.jsonl").read_bytes()
        s.refresh()
        assert ledger.total == before and (out / "embeddings.jsonl").read_bytes() == raw
    vectors = load_vectors(out, jobs)
    assert len(vectors) == 8 and all(len(v) == 1536 for v in vectors.values())
    for job in jobs:
        assert [float(vectors[tid][0]) for tid in job["text_ids"]] == list(
            range(1, len(job["text_ids"]) + 1)
        )


def test_embedding_cost_guard_precedes_upload(tmp_path):
    from feature_space_pilot.baselines.embedding import EmbeddingSession, make_jobs
    from feature_space_pilot.baselines.core import embedding_texts
    from feature_space_pilot.budget import Ledger

    c = embedding_config(tmp_path)
    out = tmp_path / "embedding"
    out.mkdir()
    api = FakeEmbeddingAPI()
    with Ledger(
        tmp_path / "ledger.jsonl", cap=0.000001, stage_caps={"embeddings": 0.000001}
    ) as ledger:
        s = EmbeddingSession(c, out, ledger, api=api)
        for j in make_jobs(embedding_texts(inventory()), c["baseline_embedding"]):
            s.enqueue_job(j)
        with pytest.raises(ValueError, match="budget"):
            s.flush("embeddings")
        assert api.base.creates == 0 and ledger.total == 0


def test_embedding_planning_is_offline_and_labels_byte_bounds(monkeypatch):
    import socket
    from feature_space_pilot.baselines.embedding import make_jobs, settings
    from feature_space_pilot.baselines.core import embedding_texts

    def blocked(*a, **k):
        raise AssertionError("Offline baseline planning attempted network")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    jobs = make_jobs(embedding_texts(inventory()), settings())
    assert jobs[0]["token_budget"]["payload_tokens"] is None
    assert jobs[0]["token_budget"]["method"] == "embedding_utf8_byte_upper_bound_v1"


def matching_fixture(tmp_path):
    from feature_space_pilot.matching.protocol import build_jobs
    from feature_space_pilot.io import (
        canonical,
        digest,
        write_json,
        file_hash,
        append_jsonl,
    )

    source = tmp_path / "source"
    source.mkdir()
    inv = inventory()
    jobs = build_jobs(inv, include_control=False)
    grades = decisions()
    write_json(source / "inventory.json", inv)
    write_json(source / "jobs.json", jobs)
    for job in jobs:
        results = [
            dict(
                proposition_id=p["proposition_id"],
                evidence="Synthetic matching fixture",
                **grades[job["item_id"], p["proposition_id"]]
            )
            for p in job["propositions"]
        ]
        raw = canonical(
            dict(item_id=job["item_id"], reviewed_all_labels=True, results=results)
        )
        r = dict(
            job_id=job["job_id"],
            item_id=job["item_id"],
            parse_status="ok",
            is_control=False,
            raw_response=raw,
            response_hash=digest(raw),
            results=results,
        )
        r["record_hash"] = digest(r)
        append_jsonl(source / "judgments.jsonl", r)
    write_json(
        source / "analysis_manifest.json",
        dict(
            analysis_id="synthetic_semantic_fixture",
            demo=True,
            billed_execution=False,
            source=dict(source_verified=False),
            grader=dict(model_id="synthetic_fixture"),
            control_checks=dict(passed=True),
            artifacts={p.name: file_hash(p) for p in source.iterdir()},
        ),
    )
    return source


def test_baseline_pipeline_is_separate_resumable_and_keeps_synthetic_labels(tmp_path):
    from feature_space_pilot.baselines.run import make_plan, preflight, advance

    source = matching_fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    plan = make_plan(source, embedding_config(tmp_path))
    api = FakeEmbeddingAPI()
    a = advance(
        plan, api=api, preflight_hash=preflight(plan)["preflight_hash"], render=False
    )
    assert a["status"] == "batch_pending" and a["embedding_vectors"] == 0
    b = advance(
        plan, api=api, preflight_hash=preflight(plan)["preflight_hash"], render=False
    )
    assert b["status"] == "complete" and b["embedding_vectors"] == 8
    assert b["demo"] is True and b["human_reviewed"] is False
    report = Path(b["report_dir"])
    summary = json.loads((report / "summary.json").read_text())
    assert (
        summary["held_out"]["reference_kind"]
        == "automated_luna_decisions_not_human_gold"
    )
    assert "embedding_cosine" in summary["held_out"]["methods"]
    advance(
        plan, api=api, preflight_hash=preflight(plan)["preflight_hash"], render=False
    )
    assert api.base.creates == 1
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_tampered_matching_snapshot_is_rejected_before_baseline_spending(tmp_path):
    from feature_space_pilot.baselines.run import make_plan

    source = matching_fixture(tmp_path)
    (source / "inventory.json").write_text("{}")
    with pytest.raises(ValueError, match="snapshot"):
        make_plan(source, embedding_config(tmp_path))


def test_snapshot_records_exact_manifest_file_digest(tmp_path):
    from feature_space_pilot.baselines.run import load_snapshot
    from feature_space_pilot.io import file_hash

    source = matching_fixture(tmp_path)
    assert load_snapshot(source)["manifest_hash"] == file_hash(
        source / "analysis_manifest.json"
    )


def test_report_snapshot_is_immutable_on_repeated_render(tmp_path):
    from feature_space_pilot.baselines.report import write_report
    from feature_space_pilot.baselines.core import build_pairs
    from feature_space_pilot.baselines.run import load_snapshot

    source = matching_fixture(tmp_path)
    snap = load_snapshot(source)
    out = tmp_path / "baseline" / "reports" / "checkpoint"
    out.parent.parent.mkdir(parents=True)
    pairs = build_pairs(inventory(), decisions())
    write_report(out, pairs, inventory(), snap, render=True)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
    write_report(out, pairs, inventory(), snap, render=True)
    assert {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()
    } == before


def test_embedding_receipt_crash_after_settlement_never_double_charges(
    tmp_path, monkeypatch
):
    from feature_space_pilot.baselines import embedding
    from feature_space_pilot.baselines.core import embedding_texts
    from feature_space_pilot.budget import Ledger

    c = embedding_config(tmp_path)
    out = tmp_path / "embedding"
    out.mkdir()
    api = FakeEmbeddingAPI()
    jobs = embedding.make_jobs(embedding_texts(inventory()), c["baseline_embedding"])
    append = embedding.append_jsonl

    def crash(*a, **k):
        raise RuntimeError("Simulated crash before journal append")

    with Ledger(
        tmp_path / "ledger.jsonl", cap=100, stage_caps=c["stage_caps"]
    ) as ledger:
        s = embedding.EmbeddingSession(c, out, ledger, api=api)
        for j in jobs:
            s.enqueue_job(j)
        s.flush("embeddings")
        monkeypatch.setattr(embedding, "append_jsonl", crash)
        with pytest.raises(RuntimeError, match="Simulated"):
            s.refresh()
        paid = ledger.total
        monkeypatch.setattr(embedding, "append_jsonl", append)
        resumed = embedding.EmbeddingSession(c, out, ledger, api=api)
        resumed.refresh()
        assert ledger.total == paid and not resumed.active and api.base.creates == 1
        assert len(embedding.read_receipts(out / "embeddings.jsonl")) == len(jobs)


def test_terminal_incomplete_judge_stops_with_review_status(tmp_path):
    from feature_space_pilot.baselines.run import make_plan, preflight, advance
    from feature_space_pilot.io import file_hash, write_json

    source = matching_fixture(tmp_path)
    path = source / "judgments.jsonl"
    path.write_text(path.read_text().splitlines()[1] + "\n")
    m = json.loads((source / "analysis_manifest.json").read_text())
    m["artifacts"]["judgments.jsonl"] = file_hash(path)
    m["status"] = "needs_review"
    write_json(source / "analysis_manifest.json", m)
    plan = make_plan(source, embedding_config(tmp_path))
    api = FakeEmbeddingAPI()
    advance(
        plan, api=api, preflight_hash=preflight(plan)["preflight_hash"], render=False
    )
    done = advance(
        plan, api=api, preflight_hash=preflight(plan)["preflight_hash"], render=False
    )
    assert done["status"] == "needs_review"


def test_provider_poll_metadata_does_not_duplicate_baseline_report(tmp_path):
    from feature_space_pilot.baselines.run import make_plan, preflight, advance
    from feature_space_pilot.io import write_json

    source = matching_fixture(tmp_path)
    plan = make_plan(source, embedding_config(tmp_path))
    api = FakeEmbeddingAPI()
    advance(
        plan, api=api, preflight_hash=preflight(plan)["preflight_hash"], render=False
    )
    b = advance(
        plan, api=api, preflight_hash=preflight(plan)["preflight_hash"], render=False
    )
    m = json.loads((source / "analysis_manifest.json").read_text())
    m["provider_poll_counter"] = 1
    write_json(source / "analysis_manifest.json", m)
    c = advance(
        plan, api=api, preflight_hash=preflight(plan)["preflight_hash"], render=False
    )
    assert c["report_dir"] == b["report_dir"]
    assert c["source_manifest_sha256"] == b["source_manifest_sha256"]

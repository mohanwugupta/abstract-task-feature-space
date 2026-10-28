"""A complete scientific follow-up runs without network or empirical mutations."""

import copy
import json
import socket
import numpy as np
import pytest
from feature_space_pilot.io import file_hash, digest
from feature_space_pilot.full.bank import build_bank
from feature_space_pilot.experiments.banks import separate_banks
from test_experiment_banks import item, generation


def bundle_fixture(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    original = root / "data.txt"
    original.write_text("immutable evidence\n")
    human_items = [
        item(
            "h" + word,
            kind,
            source_dataset="vinson_2008",
            source_text=word,
            selection_stratum=st,
        )
        for word, kind, st in [
            ("A", "event", "actionN"),
            ("B", "object", "object"),
            ("C", "object", "object"),
            ("D", "event", "actionV"),
        ]
    ]
    raw = dict(
        features=["move", "force", "sound"],
        items=[
            dict(index=n + 1, item=i["source_text"], stratum=i["selection_stratum"])
            for n, i in enumerate(human_items)
        ],
        weights=[[5, 2, 0], [4, 1, 1], [0, 2, 5], [0, 1, 4]],
    )
    human_gens = [
        generation(
            i["item_id"],
            [t for t, w in zip(raw["features"], raw["weights"][n]) if w],
            f,
            r,
        )
        for n, i in enumerate(human_items)
        for f in ("P0", "P1", "P2")
        for r in range(3)
    ]
    tasks = [
        item("poem_T", "task_instance", task_id="poem", level="task_instance"),
        item("S_deadline", "situation", level="situation"),
        item(
            "poem_S_deadline",
            "situated_task",
            task_id="poem",
            parent_id="poem_T",
            situation_ids=["S_deadline"],
            level="situated_task",
        ),
    ]
    task_gens = [
        generation(
            i["item_id"],
            (
                ["base property"]
                if n == 0
                else (
                    ["has limited time"]
                    if n == 1
                    else [
                        "base property",
                        "has limited time",
                        "requires checking under time pressure",
                    ]
                )
            ),
            f,
            r,
        )
        for n, i in enumerate(tasks)
        for f in ("P0", "P1", "P2")
        for r in range(3)
    ]
    allgens = human_gens + task_gens
    banks = separate_banks(human_items + tasks, allgens, build_bank(allgens))
    inv = dict(
        items=human_items,
        norm_features=[],
        origins=[],
        propositions=[],
        comparison_cap=10,
    )
    records = []
    for n, i in enumerate(human_items):
        hid = {text: "n" + str(k) for k, text in enumerate(raw["features"])}
        inv["norm_features"].extend(
            dict(item_id=i["item_id"], human_id=hid[t], label=t)
            for t, w in zip(raw["features"], raw["weights"][n])
            if w
        )
        results = []
        for k, t in enumerate(
            t for t, w in zip(raw["features"], raw["weights"][n]) if w
        ):
            pid = i["item_id"] + t
            inv["propositions"].append(
                dict(item_id=i["item_id"], proposition_id=pid, text=t)
            )
            results.append(
                dict(
                    proposition_id=pid,
                    matches=[hid[t]],
                    related=[],
                    uncertain=[],
                    evidence="Synthetic match",
                )
            )
            for g in (g for g in human_gens if g["item_id"] == i["item_id"]):
                inv["origins"].append(
                    dict(
                        item_id=i["item_id"],
                        generation_job_id=g["job_id"],
                        raw_feature_index=k,
                        prompt_family=g["prompt_family"],
                        replicate=g["replicate"],
                        proposition_id=pid,
                    )
                )
        records.append(
            dict(
                item_id=i["item_id"],
                parse_status="ok",
                record_hash=digest(i),
                results=results,
            )
        )
    cells = [
        dict(
            item_id=i["item_id"],
            feature_id=f["feature_id"],
            applies="1",
            status="reused_binary",
            source_record_hashes=[digest([i, f])],
        )
        for i in tasks
        for f in banks["task"]["features"]
    ]
    jobs = lambda gs: [
        dict(
            **{k: g[k] for k in ("item_id", "job_id", "prompt_family", "replicate")},
            system="Synthetic prompt",
            cap=10
        )
        for g in gs
    ]
    return dict(
        human_items=human_items,
        human_bank=banks["human"],
        human_generations=human_gens,
        human_jobs=jobs(human_gens),
        task_items=tasks,
        task_bank=banks["task"],
        task_generations=task_gens,
        task_jobs=jobs(task_gens),
        historical_task_cells=cells,
        semantic=dict(inventory=inv, records=records),
        raw_vinson=raw,
        abstract_reference=[],
        demo=True,
        source_export=str(root),
        provenance=dict(
            immutable_files={str(original): file_hash(original)}, journal_prefixes=[]
        ),
    )


def test_complete_offline_outputs_have_rsa_alignment_and_irregular_local_scope(
    tmp_path, monkeypatch
):
    from feature_space_pilot.collaborator.run import analyze, verify

    bundle = bundle_fixture(tmp_path)
    before = (tmp_path / "original/data.txt").read_bytes()

    def forbidden(*a, **k):
        raise AssertionError("Scientific follow-up attempted network/paid inference")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    out = analyze(bundle, tmp_path / "results", n_permutations=19, render=True)
    s = json.loads((out / "summary.json").read_text())
    assert (
        s["model_api_calls"] == 0
        and s["task"]["primary_cells"] < s["task"]["historical_cells"]
    )
    assert len(s["human"]["primary_rsa"]) == 3
    assert (out / "human/feature_aligned_cosine.csv").exists()
    assert (out / "task/prompt_comparison.html").exists()
    assert (out / "task/target_inventories/poem_T/judgments.csv").exists()
    assert not (out / "task/applicability_matrix.csv").exists()
    assert "SYNTHETIC" in (out / "scientific_report.md").read_text()
    assert (tmp_path / "original/data.txt").read_bytes() == before
    assert verify(out)["verified"]
    assert analyze(bundle, tmp_path / "results", n_permutations=19, render=True) == out
    (out / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact"):
        verify(out)


def test_output_cannot_be_written_into_any_original_empirical_run(tmp_path):
    from feature_space_pilot.collaborator.run import analyze

    bundle = bundle_fixture(tmp_path)
    source = tmp_path / "separated_export"
    source.mkdir()
    bundle["source_export"] = str(source)
    bundle["provenance"]["original_sources"] = [dict(path=str(tmp_path / "original"))]
    with pytest.raises(ValueError, match="separate"):
        analyze(
            bundle,
            tmp_path / "original" / "new_reports",
            n_permutations=9,
            render=False,
        )

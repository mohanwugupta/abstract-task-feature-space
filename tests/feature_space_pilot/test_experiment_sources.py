"""Active journals can grow, but offline reconstruction never changes their prefix."""

import hashlib
import json
import pytest
from feature_space_pilot.io import canonical, digest


def write_frozen(path, values, **manifest):
    path.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        (path / name).write_text(
            json.dumps(value)
            if not name.endswith(".jsonl")
            else "".join(canonical(r) + "\n" for r in value)
        )
    hashes = {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in values
    }
    (path / "full_manifest.json").write_text(
        json.dumps(dict(frozen_hashes=hashes, **manifest))
    )
    return hashes


def source_fixture(tmp_path):
    from test_experiment_banks import item, generation
    from test_experiment_reuse import SETTINGS
    from feature_space_pilot.full.bank import build_bank

    items = [item("h", "concept"), item("t", "task_instance")]
    gens = [generation("h", ["is warm"]), generation("t", ["requires accuracy"])]
    roots, source_hashes = [], {}
    for i, g in zip(items, gens):
        g["raw_response"] = canonical(
            dict(item_id=g["item_id"], features=g["features"])
        )
        g["response_hash"] = digest(g["raw_response"])
        g["record_hash"] = digest(
            {
                k: v
                for k, v in g.items()
                if k not in {"record_hash", "source_run_id", "generation_model"}
            }
        )
        p = tmp_path / g["source_run_id"]
        values = {
            "stimuli.jsonl": [i],
            "generation_jobs.jsonl": [
                {k: g[k] for k in ("job_id", "item_id", "prompt_family", "replicate")}
            ],
            "frozen_config.json": dict(generator=dict(model_id="fixture-generator")),
        }
        source_hashes[g["source_run_id"]] = write_frozen(p, values)
        (p / "manifest.json").write_text(
            json.dumps(
                dict(
                    run_id=g["source_run_id"],
                    stage="human" if i["item_id"] == "h" else "tasks",
                )
            )
        )
        roots.append(str(p))
    checkpoint = dict(
        items=items,
        generations=gens,
        frozen_source_inputs=source_hashes,
        coverage={
            i["item_id"]: dict(planned=1, completed=1, failed_attempts=0) for i in items
        },
        missing_generation_jobs=[],
    )
    checkpoint["checkpoint_hash"] = digest(checkpoint)
    p = tmp_path / "pooled"
    c = dict(
        root=str(tmp_path),
        judge=SETTINGS,
        full_options=dict(source_runs=roots),
        demo=True,
        replicates=1,
    )
    from feature_space_pilot.judge import judge_prompt

    literal = judge_prompt(items[0], dict(feature_id="f", text="x"))["system"]
    write_frozen(
        p,
        {
            "stimuli.json": items,
            "frozen_config.json": c,
            "frozen_global_bank.json": build_bank(gens),
            "generation_checkpoint.json": checkpoint,
            "literal_binary_prompt.json": dict(system=literal),
        },
        analysis_id="fixture",
        demo=True,
    )
    return p


def test_verified_loader_uses_frozen_generation_provenance_and_never_writes(tmp_path):
    from feature_space_pilot.experiments.sources import load_inputs

    root = source_fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    inputs = load_inputs(root)
    assert len(inputs["items"]) == 2 and len(inputs["generations"]) == 2
    assert len(inputs["generation_jobs"]) == 2
    assert inputs["semantic_snapshot"] is None
    assert {
        str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    } == before
    (root / "stimuli.json").write_text("[]")
    with pytest.raises(ValueError, match="frozen"):
        load_inputs(root)


def test_prefix_snapshot_ignores_inflight_partial_line_and_preserves_original(tmp_path):
    from feature_space_pilot.experiments.sources import journal_prefix, verify_preserved

    p = tmp_path / "journal.jsonl"
    good = canonical(dict(example="complete")) + "\n"
    p.write_text(good + '{"inflight":')
    before = p.read_bytes()
    rows, checkpoint = journal_prefix(p)
    assert rows == [dict(example="complete")]
    assert checkpoint["bytes"] == len(good.encode())
    assert p.read_bytes() == before
    with p.open("a") as f:
        f.write("true}\n")
    assert verify_preserved(dict(immutable_files={}, journal_prefixes=[checkpoint]))[
        "verified"
    ]
    p.write_text(good.replace("complete", "modified") + "\n")
    with pytest.raises(ValueError, match="prefix"):
        verify_preserved(dict(immutable_files={}, journal_prefixes=[checkpoint]))


def test_immutable_inputs_cannot_be_resealed_or_rewritten(tmp_path):
    from feature_space_pilot.experiments.sources import verify_preserved

    p = tmp_path / "frozen.json"
    p.write_text('{"empirical":true}\n')
    evidence = dict(
        immutable_files={str(p): hashlib.sha256(p.read_bytes()).hexdigest()},
        journal_prefixes=[],
    )
    assert verify_preserved(evidence)["verified"]
    p.write_text("{}\n")
    with pytest.raises(ValueError, match="immutable"):
        verify_preserved(evidence)


def test_category_reference_is_preserved_as_aggregate_data_not_feature_strings(
    tmp_path,
):
    from feature_space_pilot.experiments.sources import load_inputs

    root = source_fixture(tmp_path)
    ref = tmp_path / "abstract.csv"
    ref.write_text(
        "\ufeffitem_id,human_sm,human_is_e,human_sc,human_va\nh,0.5,0.2,0.2,0.1\n"
    )
    config = json.loads((root / "frozen_config.json").read_text())
    config.update(
        human_abstract_summary=str(ref),
        human_abstract_summary_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
    )
    values = {
        name: json.loads((root / name).read_text())
        for name in json.loads((root / "full_manifest.json").read_text())[
            "frozen_hashes"
        ]
    }
    values["frozen_config.json"] = config
    write_frozen(root, values, analysis_id="fixture", demo=True)
    inputs = load_inputs(root)
    assert inputs["category_reference"][0]["human_sm"] == "0.5"
    assert "feature" not in inputs["category_reference"][0]
    assert str(ref) in inputs["provenance"]["immutable_files"]

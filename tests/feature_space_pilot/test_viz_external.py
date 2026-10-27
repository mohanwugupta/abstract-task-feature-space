import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from feature_space_pilot.io import digest, file_hash
from feature_space_pilot.viz.load import load_run


def test_synthetic_human_metrics_are_gated_and_generated_geometry_is_exact(
    viz_run_path,
):
    from feature_space_pilot.viz.derive import (
        human_sources,
        geometry_data,
        compare_rdms,
    )

    run = load_run(viz_run_path)
    assert all(not r["available"] for r in human_sources(run))
    g = geometry_data(run, ["poem_T", "finance_T"])
    assert g["metric"] == "jaccard" and g["item_ids"] == ["poem_T", "finance_T"]
    assert g["rdm"][0][0] == 0 and 0 < g["rdm"][0][1] <= 1
    a = np.array(
        [[0, 0.1, 0.4, 0.6], [0.1, 0, 0.7, 0.8], [0.4, 0.7, 0, 0.2], [0.6, 0.8, 0.2, 0]]
    )
    result = compare_rdms(a, a, item_ids=["a", "b", "c", "d"], seed=42, bootstrap=25)
    assert result["spearman"] == pytest.approx(1)
    assert result["bootstrap_unit"] == "items_excluding_same_original_item"


def test_embeddings_require_raw_verified_vectors_and_exact_alignment(
    viz_run_path, tmp_path
):
    from feature_space_pilot.viz.derive import embedding_data

    run = load_run(viz_run_path)
    assert embedding_data(run)["available"] is False
    rows = [
        dict(
            item_id=i,
            text_hash=digest(x["text"]),
            vector=[1.0, 0.0] if n % 2 else [-1.0, 0.0],
        )
        for n, (i, x) in enumerate(run.items.items())
    ]
    artifact = dict(
        model_id="fixture-dense",
        revision="fixture-1",
        source_url="https://example.test/documentation",
        items=rows,
    )
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps(artifact))
    evaluation = dict(embedding_file=str(path), embedding_sha256=file_hash(path))
    g = embedding_data(replace(run, evaluation=evaluation))
    assert g["available"] and g["metric"] == "cosine" and g["rdm"][0][1] == 2
    artifact["items"][0]["text_hash"] = "wrong"
    path.write_text(json.dumps(artifact))
    evaluation["embedding_sha256"] = file_hash(path)
    with pytest.raises(ValueError, match="text|alignment"):
        embedding_data(replace(run, evaluation=evaluation))


def test_known_binary_jaccard_and_rdm_order_are_explicit(viz_run_path):
    from feature_space_pilot.viz.derive import geometry_data, compare_rdms

    run = load_run(viz_run_path)
    items = list(run.items)[:3]
    features = list(run.features)[:2]
    cells = {
        (i, f): dict(applies=v, evidence="Fixture")
        for i, row in zip(items, [[1, 0], [1, 1], [0, 0]])
        for f, v in zip(features, row)
    }
    toy = replace(run, judgments=cells)
    matrix = geometry_data(toy, items, features)
    assert matrix["rdm"] == [[0, 0.5, 1], [0.5, 0, 1], [1, 1, 0]]
    with pytest.raises(ValueError, match="order"):
        compare_rdms(
            matrix["rdm"],
            matrix["rdm"],
            item_ids=items,
            other_item_ids=list(reversed(items)),
        )


def test_verified_human_language_and_category_reviewer_are_checked(
    viz_run_path, tmp_path
):
    import csv
    from feature_space_pilot.viz.derive import human_sources

    run = load_run(viz_run_path)
    path = tmp_path / "norms.csv"
    path.write_text(
        "item,feature,language,category\nword,listed property,de,internal\n"
    )
    path.chmod(0o444)
    spec = dict(
        dataset_id="fixture-human",
        kind="human",
        verified=True,
        path=str(path),
        sha256=file_hash(path),
        language="de",
        data_level="item_features",
        participants=4,
        instructions="Offline test fixture only",
        license="Synthetic test fixture",
        url="https://example.test/source",
        codebook="https://example.test/codebook",
        columns={k: k for k in ["item", "feature", "language", "category"]},
    )
    item = dict(
        next(iter(run.items.values())),
        item_id="human",
        source_dataset="fixture-human",
        text="word",
        language="en",
        target_type="concept",
    )
    toy = replace(
        run,
        manifest=dict(run.manifest, demo=False),
        items={"human": item},
        sources=(spec,),
    )
    assert "language" in human_sources(toy)[0]["reason"]
    item["language"] = "de"
    codes = tmp_path / "coding.csv"
    codes.write_text("item_id,feature_id,category,reviewer\nhuman,f,internal,\n")
    toy = replace(
        toy,
        evaluation=dict(
            category_coding_file=str(codes), category_coding_sha256=file_hash(codes)
        ),
    )
    assert "reviewer" in human_sources(toy)[0]["reason"]


def test_static_external_geometry_keeps_verified_sources_separate(
    viz_run_path, tmp_path
):
    from feature_space_pilot.viz.figures import static_geometry
    import matplotlib.pyplot as plt

    run = load_run(viz_run_path)
    features = list(run.features)[:2]
    items = {}
    sources = []
    cells = {}
    for dataset in ["source-a", "source-b"]:
        path = tmp_path / (dataset + ".csv")
        path.write_text(
            "item,feature,language\nword0,f0,en\nword1,f0,en\nword1,f1,en\nword2,f1,en\nword3,f2,en\n"
        )
        path.chmod(0o444)
        sources.append(
            dict(
                dataset_id=dataset,
                kind="human",
                verified=True,
                path=str(path),
                sha256=file_hash(path),
                language="en",
                data_level="item_features",
                participants=4,
                instructions="Test fixture",
                license="Test fixture",
                url="https://example.test/source",
                codebook="https://example.test/codebook",
                columns={k: k for k in ["item", "feature", "language"]},
            )
        )
        for n in range(4):
            iid = dataset + str(n)
            items[iid] = dict(
                next(iter(run.items.values())),
                item_id=iid,
                text="word" + str(n),
                language="en",
                source_dataset=dataset,
                target_type="concept",
            )
            for j, fid in enumerate(features):
                cells[iid, fid] = dict(
                    applies=int((n + j) % 2 == 0), evidence="Fixture"
                )
    toy = replace(
        run,
        manifest=dict(run.manifest, demo=False),
        items=items,
        sources=tuple(sources),
        judgments=cells,
        scope=dict(item_ids=list(items), feature_ids=features),
    )
    fig, data, _ = static_geometry(toy)
    assert len(data["sources"]) == 2
    assert {s["source_dataset"] for s in data["sources"]} == {"source-a", "source-b"}
    assert all(
        all(
            i.startswith(source["source_dataset"])
            for i in source["panels"][0]["item_ids"]
        )
        for source in data["sources"]
    )
    plt.close(fig)


def test_reviewed_category_only_human_source_has_a_static_profile(
    viz_run_path, tmp_path
):
    import matplotlib.pyplot as plt
    from feature_space_pilot.viz.figures import static_human

    run = load_run(viz_run_path)
    path = tmp_path / "category_norms.csv"
    path.write_text(
        "item,category,count,language\nword,internal,3,en\nword,social,1,en\n"
    )
    path.chmod(0o444)
    spec = dict(
        dataset_id="category-fixture",
        kind="human",
        verified=True,
        path=str(path),
        sha256=file_hash(path),
        language="en",
        data_level="category_counts",
        participants=4,
        instructions="Test fixture",
        license="Test fixture",
        url="https://example.test/source",
        codebook="https://example.test/codebook",
        columns={k: k for k in ["item", "category", "count", "language"]},
    )
    item = dict(
        next(iter(run.items.values())),
        item_id="human",
        text="word",
        language="en",
        source_dataset="category-fixture",
        target_type="concept",
    )
    codes = tmp_path / "codes.csv"
    codes.write_text(
        "item_id,feature_id,category,reviewer\nhuman,f,internal,reviewer-1\n"
    )
    profile = dict(
        item_id="human",
        human_profile={"internal": 0.75, "social": 0.25},
        generated_profiles={"P0": {"internal": 1.0}},
        coding_coverage={
            "P0": {"coded_applicable_features": 1, "applicable_features": 1}
        },
    )
    toy = replace(
        run,
        manifest=dict(run.manifest, demo=False),
        items={"human": item},
        sources=(spec,),
        human_summary={"recovery": [], "category_profiles": [profile]},
        evaluation=dict(
            category_coding_file=str(codes), category_coding_sha256=file_hash(codes)
        ),
    )
    fig, data, caption = static_human(toy)
    assert fig is not None
    assert (
        "coded" in caption
        and data["sources"][0]["source_dataset"] == "category-fixture"
    )
    plt.close(fig)

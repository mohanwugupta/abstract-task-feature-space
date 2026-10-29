import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path
import pytest
from feature_space_pilot.source_adapters import load_vinson_zip

ROOT = Path(__file__).resolve().parents[2]


def test_vinson_adapter_exact_axes_and_category_vocabulary(tmp_path):
    p = tmp_path / "tiny.zip"
    with zipfile.ZipFile(p, "w") as z:
        prefix = "Vinson-BRM-2008/"
        z.writestr(
            prefix + "feature_list_and_types.txt",
            "feature#\tfeature\tVisual\tPerceptual\tFunctional\tMotoric\n1\tred\t1\t0\t0\t0\n2\tmoves\t0\t0\t0\t1\n",
        )
        z.writestr(
            prefix + "word_categories.txt",
            "word#\tword\ttype\tcategory\n1\tBALL\tobject\ttoy\n2\tRUN\tactionV\tmotion\n",
        )
        z.writestr(prefix + "feature_weight_matrix_1_256.txt", "2\t0\n0\t3\n")
        z.writestr(prefix + "feature_weight_matrix_257_456.txt", "\n")
    p.chmod(0o444)
    source = load_vinson_zip(p, expected_items=2)
    assert source["item_count"] == 2
    assert source["features"] == ["red", "moves"]
    assert source["weights"] == [[2, 0], [0, 3]]
    assert source["items"][1]["stratum"] == "actionV"
    assert source["feature_categories"]["moves"] == ["Motoric"]
    with pytest.raises(ValueError, match="456|items"):
        load_vinson_zip(p)


def test_cli_dry_runs_and_demo_entrypoints(tmp_path):
    # Source validation must not depend on the user's accumulated paid spending.
    config = json.loads((ROOT / "configs/feature_space_pilot.json").read_text())
    config.update(root=str(ROOT), artifact_root=str(tmp_path / "campaign"))
    config_path = tmp_path / "isolated_config.json"
    config_path.write_text(json.dumps(config))
    common = [
        sys.executable,
        "-m",
        "feature_space_pilot.run",
        "--config",
        str(config_path),
    ]
    for stage, count in [("human", 540), ("tasks", 135)]:
        p = subprocess.run(
            common + ["--stage", stage, "--dry-run"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stderr
        result = json.loads(p.stdout)
        assert result["generation_jobs"] == count
        assert result["live_ready"] is True and not result["blockers"]
    p = subprocess.run(
        common + ["--stage", "all", "--demo", "--output-root", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    m = json.loads(p.stdout)
    assert m["status"] == "complete" and m["completed_generation_jobs"] == 675
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_space_pilot.report",
            "--run-dir",
            m["output_dir"],
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0 and "SYNTHETIC DEMONSTRATION" in p.stdout


def test_missing_human_data_does_not_underestimate_declared_matrix_scope():
    from feature_space_pilot.engine import load_config, preflight

    config = load_config(ROOT / "configs/feature_space_pilot.json")
    plan = preflight(config, "human")
    assert plan["judgment_cells_upper_bound"] == 60 * config["candidate_cap"] == 9000
    assert plan["scientific_preflight"]["generation_ready"]
    assert not plan["scientific_preflight"]["abstract_comparison_ready"]

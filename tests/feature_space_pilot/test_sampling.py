import json
from pathlib import Path

import pytest

from feature_space_pilot.clients import FakeClient
from feature_space_pilot.engine import (
    demo_config,
    load_config,
    preflight,
    run_pipeline,
    selected_items,
)
from feature_space_pilot.planner import plan_generation


ROOT = Path(__file__).resolve().parents[2]


def test_default_three_repetitions_plan_675_calls_and_task_stage_135(tmp_path):
    c = demo_config(tmp_path)
    assert c["replicates"] == 3
    jobs = plan_generation(selected_items(c, "all"), c["generator"])
    assert len(jobs) == 675
    assert preflight(c, "tasks")["generation_jobs"] == 135
    live = load_config(ROOT / "configs/feature_space_pilot.json")
    assert live["replicates"] == 3
    assert preflight(live, "human")["generation_jobs"] == 540
    assert preflight(live, "tasks")["generation_jobs"] == 135


def test_odd_three_repetition_run_keeps_all_direct_samples_and_equal_call_budgets(
    tmp_path,
):
    c = demo_config(tmp_path)
    c["replicates"] = 3
    result = run_pipeline(c, "tasks", client=FakeClient())
    assert result["status"] == "complete"
    assert result["completed_generation_jobs"] == 135
    out = Path(result["output_dir"])
    compositions = json.loads((out / "compositionality_audit.json").read_text())
    assert len(compositions) == 6
    task_calls = situation_calls = 0
    for record in compositions:
        assert record["component_calls"] == record["direct_calls"] == 9
        assert record["balance_rule"] == "equal_calls_alternating_component_extra_v1"
        allocation = record["sampling_allocation"]
        assert set(allocation) == {"P0", "P1", "P2"}
        for plan in allocation.values():
            assert plan["task"] + plan["situation"] == plan["direct"] == 3
            assert {plan["task"], plan["situation"]} == {1, 2}
            task_calls += plan["task"]
            situation_calls += plan["situation"]
    assert task_calls == situation_calls == 27
    report = (out / "pilot_report.md").read_text()
    assert "alternates" in report and "sampling_allocation" in report


def test_component_allocation_preserves_even_design_and_rejects_invalid_counts():
    from feature_space_pilot.composition import matched_sampling_allocation

    a = matched_sampling_allocation(3, composition_index=0)
    b = matched_sampling_allocation(3, composition_index=1)
    for family in ["P0", "P1", "P2"]:
        assert a[family]["task"] == b[family]["situation"]
        assert a[family]["situation"] == b[family]["task"]
        assert a[family]["direct"] == b[family]["direct"] == 3
    for n in [2, 4, 10]:
        assert all(
            plan == dict(task=n // 2, situation=n // 2, direct=n)
            for plan in matched_sampling_allocation(n, composition_index=0).values()
        )
    for invalid in [0, 1, True, 2.5]:
        with pytest.raises(ValueError):
            matched_sampling_allocation(invalid, composition_index=0)

from pathlib import Path
import pytest
from feature_space_pilot.clients import FakeClient
from feature_space_pilot.engine import demo_config, run_pipeline


@pytest.fixture
def viz_run_path(tmp_path):
    c = demo_config(tmp_path)
    c["replicates"] = 2
    return Path(run_pipeline(c, "tasks", client=FakeClient())["output_dir"])

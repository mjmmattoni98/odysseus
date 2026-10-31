"""Run the conversation controls' JavaScript regressions in the pytest suite."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_assistant_controls_js():
    result = subprocess.run(
        ["node", "--test", "tests/assistant_controls.test.mjs"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

"""Runs the shared calendar time-range formatter tests under pytest.

Behavior lives in tests/calendar_time_range.test.mjs (node:test, no DOM).
This wrapper only exists so the JS suite runs in the normal pytest job.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_HAS_NODE = shutil.which("node") is not None


@pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")
def test_calendar_time_range_behavior():
    result = subprocess.run(
        ["node", "--test", "tests/calendar_time_range.test.mjs"],
        cwd=_REPO,
        capture_output=True,
        timeout=30,
        text=True,
        env={**os.environ, "TZ": "America/New_York"},
    )
    if result.returncode != 0:
        raise AssertionError(
            f"node --test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

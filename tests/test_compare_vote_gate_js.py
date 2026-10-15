"""Runs the compare vote-gate browser suite under pytest.

Behavior lives in tests/compareVoteGate.test.mjs: Vote and Tie open only once
two panes finished with an answer, and only those panes are recorded as the
round's contestants. It drives Playwright's Chromium, so it skips on a
checkout without ``npm ci``.
"""

import subprocess
from pathlib import Path

import pytest

from tests.helpers.cli_loader import load_script

_REPO = Path(__file__).resolve().parent.parent
_HAS_BROWSER = load_script("css_snapshot.py").playwright_available()


@pytest.mark.skipif(not _HAS_BROWSER, reason="node with the playwright package is required (npm ci)")
def test_compare_vote_gate_behavior():
    result = subprocess.run(
        ["node", "--test", "tests/compareVoteGate.test.mjs"],
        cwd=_REPO,
        capture_output=True,
        timeout=120,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"node --test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

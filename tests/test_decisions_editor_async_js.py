"""Runs the decision editor's async-ordering browser suite under pytest.

Behavior lives in tests/decisionsEditorAsync.test.mjs: calendar sync, record
and archive results land on the decision they were started for when the user
opens another one mid-request, and prototype-named ids score in the editor.
It drives Playwright's Chromium, so it skips on a checkout without ``npm ci``.
"""

import subprocess
from pathlib import Path

import pytest

from tests.helpers.cli_loader import load_script

_REPO = Path(__file__).resolve().parent.parent
_HAS_BROWSER = load_script("css_snapshot.py").playwright_available()


@pytest.mark.skipif(not _HAS_BROWSER, reason="node with the playwright package is required (npm ci)")
def test_decisions_editor_async_behavior():
    result = subprocess.run(
        ["node", "--test", "tests/decisionsEditorAsync.test.mjs"],
        cwd=_REPO,
        capture_output=True,
        timeout=120,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"node --test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

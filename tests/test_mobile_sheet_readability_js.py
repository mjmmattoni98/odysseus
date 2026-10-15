"""Runs the shipped-cascade readability checks under pytest.

Behavior lives in tests/mobileSheetClose.test.mjs (visible, labelled Close on
375px tool sheets) and tests/mutedTextContrast.test.mjs (muted tokens keep
WCAG AA in every built-in theme; default density sidebar text size). Both drive
Playwright's Chromium, so they skip on a checkout without ``npm ci``.
"""

import subprocess
from pathlib import Path

import pytest

from tests.helpers.cli_loader import load_script

_REPO = Path(__file__).resolve().parent.parent
_HAS_BROWSER = load_script("css_snapshot.py").playwright_available()


@pytest.mark.skipif(not _HAS_BROWSER, reason="node with the playwright package is required (npm ci)")
@pytest.mark.parametrize("spec", ["mobileSheetClose.test.mjs", "mutedTextContrast.test.mjs"])
def test_shipped_cascade_readability(spec):
    result = subprocess.run(
        ["node", "--test", f"tests/{spec}"],
        cwd=_REPO,
        capture_output=True,
        timeout=120,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"node --test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

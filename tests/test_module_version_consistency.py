"""Every importer must request a versioned ES module with the same ``?v=``.

Browsers key ES modules by full URL, so ``chatRenderer.js?v=a`` and
``chatRenderer.js?v=b`` load as two separate instances with duplicated state
and listeners. A cache-bust bump must update index.html and every import.
"""
import collections
import re
from pathlib import Path

_STATIC = Path(__file__).resolve().parents[1] / "static"
_VERSIONED = re.compile(r"([A-Za-z0-9_\-./]+\.js)\?v=([A-Za-z0-9_]+)")


def _module_key(ref: str) -> str:
    ref = ref.replace("/static/js/", "").replace("/static/", "")
    while ref.startswith(("./", "../")):
        ref = ref.split("/", 1)[1]
    return ref[3:] if ref.startswith("js/") else ref


def test_versioned_module_urls_agree_across_importers():
    sources = list(_STATIC.glob("*.js")) + list(_STATIC.glob("js/**/*.js")) + [_STATIC / "index.html"]
    versions = collections.defaultdict(set)
    for path in sources:
        for ref, version in _VERSIONED.findall(path.read_text(encoding="utf-8", errors="ignore")):
            versions[_module_key(ref)].add(version)
    mismatched = {name: sorted(v) for name, v in versions.items() if len(v) > 1}
    assert not mismatched, f"modules imported under different ?v= URLs: {mismatched}"

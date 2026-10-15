"""Primary/official source counts agree between saved coverage and the report.

Regression for the 2026-10-12 QA audit: a run with three official NASA
sources and one secondary source saved ``primary_or_official == 3`` while the
rendered report said "0 Primary sources", because the report counted only
``source_kind == "primary"``.
"""
import json
import re
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from src.deep_research import DeepResearcher
from src.research_navigator import assess_source
from src.visual_report import _source_evidence_summary, generate_visual_report

FIXTURE = Path(__file__).parent / "fixtures" / "research_nasa_official_run.json"
# The audit check script reads the rendered count with this pattern.
AUDIT_PRIMARY_RE = re.compile(
    r'class="evidence-value">(\d+)</span>\s*<span class="evidence-label">Primary sources'
)


def _payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _researcher_from(payload):
    researcher = DeepResearcher.__new__(DeepResearcher)
    researcher.findings = [dict(f) for f in payload["findings"]]
    researcher.analyzed_urls = [dict(a) for a in payload["analyzed_urls"]]
    return researcher


def test_report_counts_official_sources_as_primary_evidence():
    payload = _payload()
    html = generate_visual_report(
        payload["query"], payload["raw_report"], sources=payload["sources"],
        stats=payload["stats"], session_id="rp-f4202c23768b",
    )

    assert AUDIT_PRIMARY_RE.findall(html) == ["3"]
    soup = BeautifulSoup(html, "html.parser")
    label = next(node for node in soup.select(".evidence-label")
                 if node.get_text(strip=True).startswith("Primary sources"))
    assert label.select_one(".evidence-caption").get_text(strip=True) == "incl. 3 official"
    official_badges = [b for b in soup.select(".source-badge.primary") if b.get_text(strip=True) == "official"]
    assert len(official_badges) == 3


def test_rendered_count_matches_saved_source_coverage():
    payload = _payload()
    coverage = _researcher_from(payload)._source_coverage()
    html = generate_visual_report(payload["query"], payload["raw_report"], sources=payload["sources"])

    rendered = int(AUDIT_PRIMARY_RE.findall(html)[0])
    assert coverage["primary_or_official"] == 3
    assert rendered == coverage["primary_or_official"]
    # source_mix still describes every analyzed page, including the failure.
    assert coverage["source_mix"] == {"official": 3, "secondary": 2}


def test_unread_official_page_does_not_count_as_first_party_evidence():
    payload = _payload()
    researcher = _researcher_from(payload)
    researcher.analyzed_urls.append({
        "url": "https://www.jpl.nasa.gov/missions/", "title": "JPL", "source_kind": "official",
    })

    coverage = researcher._source_coverage()

    # The page was analyzed but never read into a source, so it is not
    # evidence the report can rely on (and the report would not count it).
    assert coverage["primary_or_official"] == 3
    assert coverage["source_mix"]["official"] == 4


def test_evidence_summary_splits_official_from_project_sources():
    summary = _source_evidence_summary([
        {"url": "https://nasa.gov/a", "source_kind": "official"},
        {"url": "https://docs.example.com/b", "source_kind": "primary"},
        {"url": "https://blog.example.net/c", "source_kind": "secondary"},
    ])
    assert summary["primary"] == 2
    assert summary["official"] == 1


# ── Labels and counts use hostname boundaries (review finding 8) ──

@pytest.mark.parametrize("url, kind", [
    ("https://github.com.evil.com/acme/tool", "secondary"),
    ("https://nasa.gov.evil.com/mars", "secondary"),
    ("https://mydocs.evil.com/x", "secondary"),
    ("https://en.wikipedia.org/wiki/Official_language", "secondary"),
    ("https://github.com/acme/tool", "primary"),
    ("https://docs.python.org/3/", "primary"),
    ("https://www.nasa.gov/mars/", "official"),
    ("https://www.gov.uk/x", "official"),
])
def test_source_labels_match_hostnames_on_label_boundaries(url, kind):
    assert assess_source(url, summary="Useful evidence.").kind == kind


def test_spoofed_and_secondary_pages_are_not_counted_as_first_party():
    researcher = DeepResearcher.__new__(DeepResearcher)
    researcher.findings = [
        {"url": "https://github.com.evil.com/acme/tool", "summary": "Useful evidence."},
        {"url": "https://nasa.gov.evil.com/mars", "summary": "Useful evidence."},
        {"url": "https://en.wikipedia.org/wiki/Official_language", "summary": "Useful evidence."},
    ]
    researcher.analyzed_urls = []

    coverage = researcher._source_coverage()

    assert coverage["primary_or_official"] == 0
    assert "primary/official evidence missing" in coverage["gaps"]

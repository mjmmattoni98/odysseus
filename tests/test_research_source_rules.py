"""Per-run source rules and citation checks for Deep Research.

The 2026-10-12 QA audit asked for "only official NASA sources", yet the run
read Wikipedia, kept it as a source, and the report cited a NASA page that was
never opened. Rules now come from the launch form/API or unambiguous wording
in the question, are enforced at search selection, fetch time and in the
final source list, and every cited link that was not read is flagged.
"""
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup
from fastapi import HTTPException

from src.deep_research import DeepResearcher
from src.research_navigator import ResearchAction, ResearchNavigator, ResearchPage
from src.research_sources import (
    SourceRules, canonical_url, cited_urls, effective_source_rules, infer_source_rules,
    normalize_domain, normalize_domain_list, verify_citations,
)
from src.visual_report import generate_visual_report

FIXTURE = Path(__file__).parent / "fixtures" / "research_nasa_official_run.json"
AUDIT_QUESTION = json.loads(FIXTURE.read_text(encoding="utf-8"))["query"]


# ── Normalization and inference ──

@pytest.mark.parametrize("raw, expected", [
    ("nasa.gov", "nasa.gov"),
    ("https://www.NASA.gov/mars/", "nasa.gov"),
    ("site:science.nasa.gov", "science.nasa.gov"),
    ("*.nasa.gov", "nasa.gov"),
    ("wikipedia", "wikipedia"),
    ("bücher.de", "xn--bcher-kva.de"),
])
def test_normalize_domain(raw, expected):
    assert normalize_domain(raw) == expected


@pytest.mark.parametrize("raw", ["", "bad domain", "a..b", "-x.com", "x" * 64 + ".com"])
def test_normalize_domain_rejects_invalid(raw):
    with pytest.raises(ValueError):
        normalize_domain(raw)


def test_normalize_domain_list_dedupes_and_limits():
    assert normalize_domain_list("nasa.gov, www.nasa.gov esa.int") == ("nasa.gov", "esa.int")
    with pytest.raises(ValueError):
        normalize_domain_list([f"d{i}.com" for i in range(21)])


def test_audit_question_infers_nasa_official_rules():
    rules = infer_source_rules(AUDIT_QUESTION)
    assert rules.allowed_domains == ("nasa",)
    assert rules.official_only is True
    assert rules.inferred  # the report can say where the rules came from


@pytest.mark.parametrize("question, allowed, excluded, official", [
    ("site:nasa.gov mars rovers -site:reddit.com", ("nasa.gov",), ("reddit.com",), False),
    ("Compare Rust and Go, no Wikipedia please", (), ("wikipedia",), False),
    ("only use docs.python.org for asyncio", ("docs.python.org",), (), False),
    ("Summarize the policy, cite only primary sources", (), (), True),
    # Ordinary wording must not become a rule.
    ("What is the only official language of Switzerland?", (), (), False),
    ("only US sources on inflation", (), (), False),
    ("how to avoid python pitfalls", (), (), False),
])
def test_infer_source_rules(question, allowed, excluded, official):
    rules = infer_source_rules(question)
    assert (rules.allowed_domains, rules.excluded_domains, rules.official_only) == (allowed, excluded, official)


def test_explicit_allowed_domains_replace_inferred_ones_and_exclusions_accumulate():
    rules = effective_source_rules(
        {"allowed_domains": ["esa.int"], "excluded_domains": ["reddit.com"]},
        "only NASA sources, no Wikipedia",
    )
    assert rules.allowed_domains == ("esa.int",)
    assert rules.excluded_domains == ("reddit.com", "wikipedia")


def test_rule_violations():
    rules = effective_source_rules(None, AUDIT_QUESTION)
    assert rules.violation("https://science.nasa.gov/planetary-science/programs/mars-exploration/") == ""
    assert "outside allowed domains" in rules.violation("https://en.wikipedia.org/wiki/Mars_Exploration_Program")
    assert "outside allowed domains" in rules.violation("https://staging.mars.jpllab.net/")
    # Host named after NASA but not an official source.
    assert rules.violation("https://nasa.com/") == "not an official source"
    explicit = SourceRules(allowed_domains=("python.org",), official_only=True)
    assert explicit.allows("https://www.python.org/about/")  # explicitly allowed counts as official
    assert SourceRules(excluded_domains=("wikipedia",)).violation("https://de.wikipedia.org/x") == "excluded domain wikipedia"


# ── Engine enforcement ──

class _Navigator(ResearchNavigator):
    def __init__(self, results=None):
        super().__init__()
        self.results = results or []
        self.searches = []
        self.fetches = []

    async def search(self, query, *, count=10):
        self.searches.append(query)
        return list(self.results) if ("site:" not in query or self.results) else []

    async def fetch(self, url, *, timeout=10, max_bytes=None):
        self.fetches.append(url)
        return ResearchPage(url=url, title="Page", content=f"Evidence from {url}", success=True)

    async def browser_read(self, url, *, timeout=45):
        self.fetches.append(url)
        return ResearchPage(url=url, title="Page", content=f"Evidence from {url}", success=True, retrieval="browser")


def _researcher(nav, rules):
    researcher = DeepResearcher.__new__(DeepResearcher)
    researcher.navigator = nav
    researcher.source_rules = rules
    researcher.search_provider_override = None
    researcher.queries_used = set()
    researcher.urls_fetched = set()
    researcher.analyzed_urls = []
    researcher.providers_used = []
    researcher.action_trace = []
    researcher.navigation_trace = []
    researcher.findings = []
    researcher.session_id = "rules-test"
    researcher.max_urls_per_round = 3
    researcher.extraction_concurrency = 2
    researcher.max_content_chars = 15000
    researcher.extraction_max_tokens = 1024
    researcher.extraction_timeout = 30
    researcher.mechanical_think = None
    researcher.context_window = None
    researcher._cancelled = False
    researcher._start_time = time.time()
    researcher.max_time = 999
    researcher._progress = None
    researcher.round_count = 1
    researcher.llm_model = "m"
    researcher.category = None

    async def extract(messages, **kwargs):
        return json.dumps({"rational": "r", "summary": "Useful evidence.", "evidence": "e"})

    researcher._llm = extract
    return researcher


NASA_RULES = effective_source_rules(None, AUDIT_QUESTION)
SEARCH_HITS = [
    {"url": "https://en.wikipedia.org/wiki/Mars_Exploration_Program", "title": "Mars Exploration Program",
     "content": "Mars Exploration Program official overview"},
    {"url": "https://science.nasa.gov/planetary-science/programs/mars-exploration/",
     "title": "Mars Exploration Program", "content": "Mars Exploration Program purpose"},
    {"url": "https://staging.mars.jpllab.net/", "title": "NASA Mars Exploration",
     "content": "Mars Exploration Program"},
]


def test_search_and_fetch_never_read_pages_outside_the_rules():
    nav = _Navigator(SEARCH_HITS)
    researcher = _researcher(nav, NASA_RULES)

    findings = asyncio.run(researcher._execute_research_actions(
        [ResearchAction("web_search", {"query": "Mars Exploration Program purpose"}),
         ResearchAction("web_fetch", {"url": "https://en.wikipedia.org/wiki/Mars"})],
        AUDIT_QUESTION,
    ))

    assert nav.fetches == ["https://science.nasa.gov/planetary-science/programs/mars-exploration/"]
    assert [f["url"] for f in findings] == nav.fetches
    assert researcher.rule_filtered_urls == 3
    assert researcher.get_stats()["Source rules"] == "only nasa; official sources only"


def test_fetch_time_gate_blocks_any_path():
    nav = _Navigator()
    researcher = _researcher(nav, NASA_RULES)

    assert asyncio.run(researcher._fetch_and_extract(
        "https://en.wikipedia.org/wiki/Mars", AUDIT_QUESTION, "Mars", prefer_browser=True,
    )) is None
    assert nav.fetches == []
    assert researcher.navigation_trace[-1]["status"] == "blocked"


def test_planner_fetch_actions_outside_rules_are_rejected_with_reason():
    researcher = _researcher(_Navigator(), NASA_RULES)
    rejected = []
    kept = researcher._normalize_research_actions(
        [ResearchAction("web_fetch", {"url": "https://en.wikipedia.org/wiki/Mars"}),
         ResearchAction("web_fetch", {"url": "https://www.nasa.gov/mars/"})],
        max_actions=4, rejected=rejected,
    )
    assert [a.args["url"] for a in kept] == ["https://www.nasa.gov/mars/"]
    assert rejected[0]["reason"].startswith("source rule:")


def test_single_allowed_domain_scopes_search_and_falls_back_when_empty():
    nav = _Navigator()  # returns nothing for site: queries, nothing at all here
    researcher = _researcher(nav, SourceRules(allowed_domains=("nasa.gov",)))
    asyncio.run(researcher._search("mars rovers"))
    assert nav.searches == ["mars rovers site:nasa.gov", "mars rovers"]


def test_final_prompt_lists_only_read_pages_and_rules():
    researcher = _researcher(_Navigator(), NASA_RULES)
    researcher.findings = [
        {"url": "https://www.nasa.gov/mars/", "summary": "Useful.", "source_kind": "official"},
        {"url": "https://en.wikipedia.org/wiki/Mars", "summary": "Useful.", "source_kind": "secondary"},
    ]
    block = researcher._citable_sources_block()
    assert "https://www.nasa.gov/mars/" in block
    assert "wikipedia" not in block
    assert "Only use pages on: nasa" in researcher._rules_prompt_suffix()


# ── Saved sources and citation check ──

def test_handler_source_list_excludes_rule_violations():
    from src.research_handler import ResearchHandler
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    sources = ResearchHandler._extract_sources(payload["findings"], NASA_RULES)
    assert [s["url"] for s in sources] == [s["url"] for s in payload["sources"][:3]]


def test_citation_check_flags_unread_and_excluded_links():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    warnings = verify_citations(
        payload["raw_report"],
        read_urls=[s["url"] for s in payload["sources"][:3]],
        attempted_urls=[a["url"] for a in payload["analyzed_urls"]],
        rules=NASA_RULES,
    )
    assert [(w["url"], w["status"]) for w in warnings] == [
        ("https://www.nasa.gov/missions/mars-science-laboratory/", "not_opened"),
        ("https://en.wikipedia.org/wiki/Mars_Exploration_Program", "excluded"),
    ]


def test_citation_check_marks_failed_reads_and_ignores_images_and_code():
    report = (
        "See [staging](https://staging.mars.jpllab.net/) and ![chart](https://img.example.com/c.png).\n\n"
        "```\ncurl https://example.org/api\n```\n"
        "Read: https://www.nasa.gov/mars/?utm_source=x"
    )
    assert cited_urls(report) == ["https://staging.mars.jpllab.net/", "https://www.nasa.gov/mars/?utm_source=x"]
    warnings = verify_citations(report, read_urls=["https://nasa.gov/mars"],
                                attempted_urls=["https://staging.mars.jpllab.net"])
    assert [(w["url"], w["status"]) for w in warnings] == [("https://staging.mars.jpllab.net/", "unread")]


def test_save_result_persists_rules_and_citation_warnings(tmp_path, monkeypatch):
    import src.research_handler as rh
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    monkeypatch.setattr(rh, "RESEARCH_DATA_DIR", tmp_path)
    monkeypatch.setattr("src.event_bus.fire_event", lambda *a, **k: None)
    researcher = SimpleNamespace(
        findings=payload["findings"], analyzed_urls=payload["analyzed_urls"], navigation_trace=[],
        action_trace=[], _source_state_summary=lambda: "", _source_coverage=lambda: {},
    )
    entry = {"query": payload["query"], "status": "done", "result": "r", "raw_report": payload["raw_report"],
             "researcher": researcher, "started_at": 1.0, "owner": "alice",
             "source_rules": NASA_RULES.to_dict()}
    handler = rh.ResearchHandler.__new__(rh.ResearchHandler)

    handler._save_result("rp-rules0000001", entry)

    saved = json.loads((tmp_path / "rp-rules0000001.json").read_text())
    assert [s["url"] for s in saved["sources"]] == [s["url"] for s in payload["sources"][:3]]
    assert saved["source_rules"]["official_only"] is True
    assert {w["status"] for w in saved["citation_warnings"]} == {"not_opened", "excluded"}


def test_visual_report_shows_rules_and_flags_unread_citations():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    warnings = verify_citations(payload["raw_report"], read_urls=[s["url"] for s in payload["sources"][:3]],
                                rules=NASA_RULES)
    soup = BeautifulSoup(generate_visual_report(
        payload["query"], payload["raw_report"], sources=payload["sources"][:3],
        citation_warnings=warnings, source_rules=NASA_RULES.to_dict(), session_id="rp-x",
    ), "html.parser")

    check = soup.select_one(".citation-check")
    assert "2 cited links were not read" in check.get_text(" ", strip=True)
    flagged = {a["href"] for a in soup.select(".content a.citation-unverified")}
    assert flagged == {"https://www.nasa.gov/missions/mars-science-laboratory/",
                       "https://en.wikipedia.org/wiki/Mars_Exploration_Program"}
    assert "only nasa; official sources only" in soup.select_one(".source-rules-note").get_text()


# ── API validation ──

def test_start_route_rejects_invalid_domains(monkeypatch):
    import routes.research.research_routes as rr

    monkeypatch.setattr(rr, "get_current_user", lambda request: "alice")
    monkeypatch.setattr("src.auth_helpers.require_privilege", lambda request, privilege: "alice")
    started = []
    handler = SimpleNamespace(_active_tasks={}, start_research=lambda **kw: started.append(kw))
    router = rr.setup_research_routes(handler)
    start = next(r.endpoint for r in router.routes if r.path == "/api/research/start" and "POST" in r.methods)
    body = SimpleNamespace(query="q", max_rounds=0, search_provider=None, endpoint_id=None, model=None,
                           max_time=300, extraction_timeout=None, extraction_concurrency=None, category=None,
                           origin_chat_id=None, allowed_domains=["nasa..gov"], excluded_domains=None,
                           official_only=False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(start(body=body, request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))))
    assert exc.value.status_code == 400
    assert "Invalid domain" in exc.value.detail
    assert started == []


# ── Malformed citations never block saving (review finding 5) ──

@pytest.mark.parametrize("url", ["https://example.com:bad/x", "https://example.com:99999/x"])
def test_canonical_url_tolerates_malformed_ports(url):
    # Not canonicalizable, so it is compared verbatim instead of raising.
    assert canonical_url(url) == url


def test_citation_check_flags_malformed_citation_instead_of_raising():
    warnings = verify_citations("See [x](https://example.com:bad/x).", read_urls=["https://nasa.gov/mars"])
    assert [(w["url"], w["status"]) for w in warnings] == [("https://example.com:bad/x", "not_opened")]


def _save_entry(payload, raw_report):
    researcher = SimpleNamespace(
        findings=payload["findings"], analyzed_urls=payload["analyzed_urls"], navigation_trace=[],
        action_trace=[], _source_state_summary=lambda: "", _source_coverage=lambda: {},
    )
    return {"query": payload["query"], "status": "done", "result": "r", "raw_report": raw_report,
            "researcher": researcher, "started_at": 1.0, "owner": "alice",
            "source_rules": NASA_RULES.to_dict()}


def test_save_result_persists_report_with_malformed_citation(tmp_path, monkeypatch):
    import src.research_handler as rh
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    monkeypatch.setattr(rh, "RESEARCH_DATA_DIR", tmp_path)
    monkeypatch.setattr("src.event_bus.fire_event", lambda *a, **k: None)
    handler = rh.ResearchHandler.__new__(rh.ResearchHandler)

    handler._save_result("rp-badport00001", _save_entry(payload, "Answer [x](https://example.com:bad/x)."))

    saved = json.loads((tmp_path / "rp-badport00001.json").read_text())
    assert saved["raw_report"] == "Answer [x](https://example.com:bad/x)."
    assert [w["url"] for w in saved["citation_warnings"]] == ["https://example.com:bad/x"]


def test_save_result_persists_report_when_citation_diagnostics_fail(tmp_path, monkeypatch):
    import src.research_handler as rh
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    monkeypatch.setattr(rh, "RESEARCH_DATA_DIR", tmp_path)
    monkeypatch.setattr("src.event_bus.fire_event", lambda *a, **k: None)

    def broken_check(*args, **kwargs):
        raise RuntimeError("diagnostics bug")

    monkeypatch.setattr(rh, "verify_citations", broken_check)
    handler = rh.ResearchHandler.__new__(rh.ResearchHandler)

    handler._save_result("rp-diagfail0001", _save_entry(payload, payload["raw_report"]))

    saved = json.loads((tmp_path / "rp-diagfail0001.json").read_text())
    assert saved["raw_report"] == payload["raw_report"]
    assert saved["citation_warnings"] == []
    assert len(saved["sources"]) == 3


# ── Conflicting rules never broaden access (review finding 7) ──

def test_excluding_every_allowed_domain_does_not_lift_the_restriction():
    rules = effective_source_rules({"allowed_domains": ["nasa.gov"], "excluded_domains": ["nasa.gov"]}, "mars")
    assert rules.allowed_domains == ("nasa.gov",)
    assert not rules.allows("https://www.esa.int/")
    assert not rules.allows("https://www.nasa.gov/mars/")
    assert rules.permits_nothing


def test_query_exclusion_merged_with_panel_allow_list_stays_restricted():
    rules = effective_source_rules({"allowed_domains": ["nasa.gov"]}, "Mars rovers, exclude nasa.gov")
    assert not rules.allows("https://en.wikipedia.org/wiki/Mars")
    assert rules.permits_nothing


def test_conflict_inside_the_question_stays_restricted():
    rules = infer_source_rules("mars rovers site:science.nasa.gov -site:nasa.gov")
    assert not rules.allows("https://en.wikipedia.org/wiki/Mars")
    assert rules.permits_nothing


def test_partial_exclusion_keeps_the_remaining_allowed_domains():
    rules = effective_source_rules(
        {"allowed_domains": ["nasa.gov", "esa.int"], "excluded_domains": ["nasa.gov"]}, "q")
    assert rules.allowed_domains == ("esa.int",)
    assert rules.allows("https://www.esa.int/x")
    assert not rules.allows("https://nasa.gov/x")
    assert not rules.allows("https://example.com/")
    assert not rules.permits_nothing


@pytest.mark.parametrize("query, allowed, excluded", [
    ("q", ["nasa.gov"], ["nasa.gov"]),
    ("q", ["science.nasa.gov"], ["nasa.gov"]),
    ("Mars rovers, exclude nasa.gov", ["nasa.gov"], None),
])
def test_start_route_rejects_rules_that_permit_nothing(monkeypatch, query, allowed, excluded):
    import routes.research.research_routes as rr

    monkeypatch.setattr(rr, "get_current_user", lambda request: "alice")
    monkeypatch.setattr("src.auth_helpers.require_privilege", lambda request, privilege: "alice")
    started = []
    handler = SimpleNamespace(_active_tasks={}, start_research=lambda **kw: started.append(kw))
    router = rr.setup_research_routes(handler)
    start = next(r.endpoint for r in router.routes if r.path == "/api/research/start" and "POST" in r.methods)
    body = SimpleNamespace(query=query, max_rounds=0, search_provider=None, endpoint_id=None, model=None,
                           max_time=300, extraction_timeout=None, extraction_concurrency=None, category=None,
                           origin_chat_id=None, allowed_domains=allowed, excluded_domains=excluded,
                           official_only=False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(start(body=body, request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))))
    assert exc.value.status_code == 400
    assert "excluded" in exc.value.detail.lower()
    assert started == []


# ── Official-only matches official hostnames on label boundaries (review finding 8) ──

OFFICIAL_ONLY = SourceRules(official_only=True)


@pytest.mark.parametrize("url", [
    "https://en.wikipedia.org/wiki/Official_language",       # path word
    "https://evil.com/official/documentation/release-notes",  # path words
    "https://github.com.evil.com/acme/tool",                  # hostname substring
    "https://nasa.gov.evil.com/mars",                          # ".gov." substring
    "https://ec.europa.eu.evil.com/x",
    "https://notgov.com/",
    "https://docs.example.com/guide",  # vendor docs are first-party, not official
])
def test_official_only_rejects_secondary_and_spoofed_pages(url):
    assert OFFICIAL_ONLY.violation(url) == "not an official source"


@pytest.mark.parametrize("url", [
    "https://www.nasa.gov/mars/", "https://science.nasa.gov/x", "https://www.gov.uk/guidance",
    "https://www.nsw.gov.au/x", "https://www.gob.mx/x", "https://www.service-public.gouv.fr/x",
    "https://ec.europa.eu/x", "https://www.who.int/x", "https://www.army.mil/x",
    "https://web.mit.edu/x", "https://www.ox.ac.uk/x", "https://www.iso.org/standard/1.html",
])
def test_official_only_accepts_official_domains(url):
    assert OFFICIAL_ONLY.allows(url)


def test_audit_rules_reject_a_spoofed_nasa_host():
    # The "nasa" label rule admits the host; official-only must not.
    assert NASA_RULES.violation("https://nasa.gov.evil.com/mars") == "not an official source"


# ── Rules apply to the final URL after redirects (review finding 9) ──

class _RedirectResponse:
    """httpx-like response whose ``url`` is where the redirects ended."""

    def __init__(self, text, final_url):
        self.text = text
        self.content = text.encode("utf-8")
        self.headers = {"Content-Type": "text/html"}
        self.status_code = 200
        self.url = final_url

    def raise_for_status(self):
        return None


def test_fetcher_reports_the_final_url_after_redirects(monkeypatch, tmp_path):
    from services.search import content as content_mod

    monkeypatch.setattr(content_mod, "CONTENT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(content_mod, "_cache_result", lambda *a, **k: None)
    html = "<html><body><main><p>" + "Mars evidence. " * 60 + "</p></main></body></html>"
    monkeypatch.setattr(content_mod, "_get_public_url",
                        lambda url, **kw: _RedirectResponse(html, "https://publisher.example.com/a"))

    page = content_mod.fetch_webpage_content("https://www.nasa.gov/redirect")

    assert page["url"] == "https://www.nasa.gov/redirect"
    assert page["final_url"] == "https://publisher.example.com/a"
    normalized = ResearchNavigator._normalize_fetch_page("https://www.nasa.gov/redirect", page)
    assert normalized.final_url == "https://publisher.example.com/a"


class _RedirectNavigator(_Navigator):
    def __init__(self, final_url):
        super().__init__()
        self.final_url = final_url

    async def fetch(self, url, *, timeout=10, max_bytes=None):
        self.fetches.append(url)
        return ResearchPage(url=url, title="Page", content="Mars evidence.", success=True,
                            final_url=self.final_url)


def test_redirect_to_a_page_outside_the_rules_is_never_extracted():
    researcher = _researcher(_RedirectNavigator("https://en.wikipedia.org/wiki/Mars"), NASA_RULES)
    extractions = []

    async def extract(messages, **kwargs):
        extractions.append(messages)
        return json.dumps({"rational": "r", "summary": "Useful evidence.", "evidence": "e"})

    researcher._llm = extract

    out = asyncio.run(researcher._fetch_and_extract("https://www.nasa.gov/mars/", AUDIT_QUESTION, "Mars"))

    assert out is None
    assert extractions == []
    blocked = researcher.navigation_trace[-1]
    assert blocked["status"] == "blocked"
    assert "en.wikipedia.org" in blocked["error"] and "source rule" in blocked["error"]


def test_allowed_redirect_records_the_final_url_on_the_source():
    from src.research_handler import ResearchHandler
    researcher = _researcher(_RedirectNavigator("https://science.nasa.gov/mars/"), NASA_RULES)

    finding = asyncio.run(researcher._fetch_and_extract("https://www.nasa.gov/mars/", AUDIT_QUESTION, "Mars"))

    assert finding["url"] == "https://www.nasa.gov/mars/"
    assert finding["final_url"] == "https://science.nasa.gov/mars/"
    source = ResearchHandler._extract_sources([finding], NASA_RULES)[0]
    assert source["final_url"] == "https://science.nasa.gov/mars/"
    # Citing the page where the content was actually read is not "never opened".
    assert verify_citations("[Mars](https://science.nasa.gov/mars/)",
                            read_urls=ResearchHandler._read_urls([source])) == []


def test_saved_finding_whose_final_url_breaks_the_rules_is_not_a_source():
    from src.research_sources import usable_findings
    finding = {"url": "https://www.nasa.gov/mars/", "final_url": "https://en.wikipedia.org/wiki/Mars",
               "summary": "Useful evidence."}
    assert usable_findings([finding], NASA_RULES) == []

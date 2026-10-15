"""Refreshing saved research creates a new linked version with a change summary.

The previous version's file is never modified; the new one records
``lineage_id``/``version``/``previous_version_id`` and a summary of which
evidence changed (by URL and page-content fingerprint) plus a short written
comparison of the conclusions.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup
from fastapi import HTTPException

from src.deep_research import DeepResearcher
from src.research_navigator import ResearchAction
from src.research_sources import SourceRules
from src.research_versions import (
    build_change_summary, compare_sources, list_versions, refresh_seed_urls,
)
from src.visual_report import generate_visual_report


def _src(url, content_hash="", title=""):
    source = {"url": url, "title": title or url}
    if content_hash:
        source["content_hash"] = content_hash
    return source


def test_compare_sources_by_canonical_url_and_fingerprint():
    previous = [
        _src("https://www.nasa.gov/mars/", "aaa"),
        _src("https://science.nasa.gov/mep/", "bbb"),
        _src("https://en.wikipedia.org/wiki/Mars", "ccc"),
        _src("https://legacy.example.gov/page"),  # saved before fingerprints existed
    ]
    current = [
        _src("https://nasa.gov/mars", "aaa"),          # same page, different spelling
        _src("https://science.nasa.gov/mep/", "zzz"),  # content changed
        _src("https://legacy.example.gov/page", "ddd"),
        _src("https://mars.nasa.gov/news/", "eee"),
    ]
    diff = compare_sources(previous, current)

    def urls(key):
        return [item["url"] for item in diff[key]]

    assert urls("unchanged") == ["https://nasa.gov/mars"]
    assert urls("changed") == ["https://science.nasa.gov/mep/"]
    assert urls("unverified") == ["https://legacy.example.gov/page"]
    assert urls("added") == ["https://mars.nasa.gov/news/"]
    assert urls("removed") == ["https://en.wikipedia.org/wiki/Mars"]


def test_change_summary_records_counts_and_note():
    summary = build_change_summary(
        {"id": "rp-old", "version": 1, "sources": [_src("https://a.gov/x", "1")]},
        [_src("https://a.gov/x", "2"), _src("https://b.gov/y", "3")],
        conclusion_summary="- Conclusion strengthened.",
    )
    assert summary["previous_version_id"] == "rp-old"
    assert summary["counts"]["changed"] == 1 and summary["counts"]["added"] == 1
    assert summary["conclusions"] == "- Conclusion strengthened."


def test_refresh_seed_urls_respect_rules_and_limit():
    previous = {"sources": [_src(f"https://www.nasa.gov/{i}") for i in range(8)]
                + [_src("https://en.wikipedia.org/wiki/Mars")]}
    seeds = refresh_seed_urls(previous, SourceRules(allowed_domains=("nasa.gov",)))
    assert len(seeds) == 6
    assert all("nasa.gov" in url for url in seeds)


def test_research_rereads_seed_urls_in_round_one():
    researcher = DeepResearcher(llm_endpoint="http://localhost:11434", llm_model="m",
                                max_rounds=1, min_rounds=1)
    researcher._emit = lambda **k: None
    seen = []

    async def plan(question):
        return "plan"

    async def no_category(question, research_plan=""):
        return None

    async def planned(question, report, round_num):
        return [ResearchAction("web_search", {"query": "fresh query"})]

    async def execute(actions, question):
        seen.extend(actions)
        return []

    researcher._create_plan = plan
    researcher._classify_category = no_category
    researcher._plan_research_actions = planned
    researcher._execute_research_actions = execute

    asyncio.run(researcher.research("q", seed_urls=["https://www.nasa.gov/mars/"]))

    reads = [(a.tool, a.args.get("url"), a.args.get("requested_by")) for a in seen if a.tool == "web_fetch"]
    assert reads == [("web_fetch", "https://www.nasa.gov/mars/", "refresh")]
    assert any(a.tool == "web_search" for a in seen)


# ── Handler persistence ──

def _handler(monkeypatch, tmp_path):
    import src.research_handler as rh
    monkeypatch.setattr(rh, "RESEARCH_DATA_DIR", tmp_path)
    monkeypatch.setattr("src.event_bus.fire_event", lambda *a, **k: None)
    return rh.ResearchHandler.__new__(rh.ResearchHandler)


def _refresh_entry(previous):
    findings = [
        {"url": "https://www.nasa.gov/mars/", "title": "Mars", "summary": "Robotic exploration of Mars.",
         "content_hash": "new-hash", "source_kind": "official"},
    ]
    researcher = SimpleNamespace(findings=findings, analyzed_urls=[], navigation_trace=[], action_trace=[],
                                 _source_state_summary=lambda: "", _source_coverage=lambda: {})
    return {"query": "q", "status": "done", "result": "r", "raw_report": "New answer.",
            "researcher": researcher, "started_at": 1.0, "owner": "alice",
            "previous_version": previous, "run_settings": {"report_length": "brief"}}


PREVIOUS = {"id": "rp-oldversion01", "lineage_id": "rp-oldversion01", "version": 1, "next_version": 2,
            "sources": [_src("https://www.nasa.gov/mars/", "old-hash")], "raw_report": "Old answer."}


def test_save_result_links_new_version_and_keeps_deterministic_diff(monkeypatch, tmp_path):
    handler = _handler(monkeypatch, tmp_path)
    handler._save_result("rp-newversion01", _refresh_entry(dict(PREVIOUS)))

    saved = json.loads((tmp_path / "rp-newversion01.json").read_text())
    assert (saved["lineage_id"], saved["version"], saved["previous_version_id"]) == (
        "rp-oldversion01", 2, "rp-oldversion01")
    assert saved["change_summary"]["counts"]["changed"] == 1
    assert saved["settings"] == {"report_length": "brief"}
    assert saved["sources"][0]["content_hash"] == "new-hash"


def test_refresh_summary_adds_written_comparison(monkeypatch, tmp_path):
    import src.research_versions as rv
    handler = _handler(monkeypatch, tmp_path)
    seen = {}

    async def fake_summary(question, previous_report, new_report, diff, **kwargs):
        seen.update(previous=previous_report, new=new_report, changed=len(diff["changed"]))
        return "- The purpose statement is unchanged."

    monkeypatch.setattr(rv, "summarize_conclusion_changes", fake_summary)
    entry = _refresh_entry(dict(PREVIOUS))
    asyncio.run(handler._summarize_refresh(entry, "http://localhost:11434", "m"))

    assert seen == {"previous": "Old answer.", "new": "New answer.", "changed": 1}
    assert entry["change_summary"]["conclusions"] == "- The purpose statement is unchanged."


# ── Routes ──

@pytest.fixture
def research_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "deep_research"
    data_dir.mkdir()
    monkeypatch.setattr("routes.research_routes.DEEP_RESEARCH_DIR", str(data_dir))
    return data_dir


def _write(data_dir, session_id, **data):
    path = data_dir / f"{session_id}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _router(monkeypatch, handler):
    import routes.research.research_routes as rr
    monkeypatch.setattr(rr, "get_current_user", lambda request: request.state.current_user)
    monkeypatch.setattr("src.auth_helpers.require_privilege",
                        lambda request, privilege: request.state.current_user)
    monkeypatch.setattr(rr, "resolve_endpoint",
                        lambda purpose, owner=None, **k: ("http://localhost:11434", "m", {})
                        if purpose == "research" else ("", "", {}))
    return rr.setup_research_routes(handler)


def _route(router, path, method):
    return next(r.endpoint for r in router.routes if r.path == path and method in r.methods)


def _request(user):
    return SimpleNamespace(state=SimpleNamespace(current_user=user),
                           app=SimpleNamespace(state=SimpleNamespace()))


class _CapturingHandler:
    def __init__(self):
        self._active_tasks = {}
        self.started = {}

    def start_research(self, **kwargs):
        self.started = kwargs
        return {"source_rules": kwargs.get("source_rules"), "report_length": {"name": kwargs.get("report_length")}}


SAVED = dict(
    owner="alice", query="Summarize NASA's Mars program briefly using only official NASA sources.",
    status="done", category="factcheck", completed_at=100.0,
    raw_report="Old answer.", result="Old answer.",
    sources=[_src("https://www.nasa.gov/mars/", "old-hash")],
    source_rules={"allowed_domains": ["nasa"], "excluded_domains": [], "official_only": True},
    settings={"max_rounds": 1, "search_provider": "searxng", "report_length": "brief", "max_time": 240},
)


def test_refresh_reruns_with_saved_settings_and_preserves_previous_version(monkeypatch, research_dir):
    path = _write(research_dir, "rp-oldversion01", **SAVED)
    before = path.read_bytes()
    handler = _CapturingHandler()
    refresh = _route(_router(monkeypatch, handler), "/api/research/{session_id}/refresh", "POST")

    out = asyncio.run(refresh(session_id="rp-oldversion01", request=_request("alice")))

    started = handler.started
    assert started["query"] == SAVED["query"]
    assert (started["max_rounds"], started["search_provider"], started["max_time"]) == (1, "searxng", 240)
    assert started["category"] == "factcheck"
    assert started["report_length"] == "brief"
    assert started["source_rules"] == SAVED["source_rules"]
    assert started["previous_version"]["id"] == "rp-oldversion01"
    assert started["previous_version"]["next_version"] == 2
    assert started["previous_version"]["raw_report"] == "Old answer."
    assert out["refresh_of"] == "rp-oldversion01" and out["version"] == 2
    assert out["session_id"] == started["session_id"] != "rp-oldversion01"
    assert path.read_bytes() == before  # the previous answer is never rewritten


def test_refresh_is_owner_scoped(monkeypatch, research_dir):
    _write(research_dir, "rp-oldversion01", **SAVED)
    handler = _CapturingHandler()
    refresh = _route(_router(monkeypatch, handler), "/api/research/{session_id}/refresh", "POST")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(refresh(session_id="rp-oldversion01", request=_request("mallory")))
    assert exc.value.status_code == 404
    assert handler.started == {}


def test_library_lists_newest_version_and_versions_route_lists_all(monkeypatch, research_dir):
    _write(research_dir, "rp-oldversion01", **SAVED)
    _write(research_dir, "rp-newversion01", **{**SAVED, "completed_at": 200.0, "lineage_id": "rp-oldversion01",
                                              "version": 2, "previous_version_id": "rp-oldversion01",
                                              "citation_warnings": [{"url": "https://x.gov"}]})
    _write(research_dir, "rp-unrelated001", **{**SAVED, "query": "Other question"})
    router = _router(monkeypatch, _CapturingHandler())
    library = asyncio.run(_route(router, "/api/research/library", "GET")(
        request=_request("alice"), search=None, sort="recent", limit=50, archived=False))

    by_id = {item["id"]: item for item in library["research"]}
    assert set(by_id) == {"rp-newversion01", "rp-unrelated001"}
    assert by_id["rp-newversion01"]["version_count"] == 2
    assert by_id["rp-newversion01"]["citation_warning_count"] == 1

    versions = asyncio.run(_route(router, "/api/research/{session_id}/versions", "GET")(
        session_id="rp-newversion01", request=_request("alice")))
    assert [v["id"] for v in versions["versions"]] == ["rp-oldversion01", "rp-newversion01"]
    assert list_versions(research_dir, "rp-oldversion01", "mallory") == []


def test_visual_report_shows_version_history_and_change_summary():
    soup = BeautifulSoup(generate_visual_report(
        "q", "## Answer\n\nBody.", sources=[],
        session_id="rp-newversion01",
        change_summary={"previous_version_id": "rp-oldversion01", "previous_version": 1,
                        "counts": {"added": 1, "changed": 1},
                        "added": [{"url": "https://mars.nasa.gov/news/", "title": "News"}],
                        "conclusions": "- The program's stated purpose is unchanged."},
        versions=[{"id": "rp-oldversion01", "version": 1, "completed_at": 100.0},
                  {"id": "rp-newversion01", "version": 2, "completed_at": 200.0}],
    ), "html.parser")

    strip = soup.select_one(".version-strip")
    assert "Version 2 of 2" in strip.get_text(" ", strip=True)
    assert strip.select_one("a")["href"] == "/api/research/report/rp-oldversion01"
    change = soup.select_one(".change-summary")
    assert "What changed since version 1" in change.get_text(" ", strip=True)
    assert "stated purpose is unchanged" in change.get_text(" ", strip=True)
    assert soup.select_one("#btn-refresh-research") is not None


# ── A refresh re-reads pages instead of reusing the content cache (review finding 6) ──

class _TextResponse:
    def __init__(self, text, url):
        self.text = text
        self.content = text.encode("utf-8")
        self.headers = {"Content-Type": "text/plain"}
        self.status_code = 200
        self.url = url

    def raise_for_status(self):
        return None


def test_fetcher_bypass_cache_refetches_and_replaces_the_cached_copy(monkeypatch, tmp_path):
    from services.search import content as content_mod
    monkeypatch.setattr(content_mod, "CONTENT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(content_mod, "content_cache_index", {})
    live = {"text": "Old page text."}
    fetched = []

    def get(url, **kwargs):
        fetched.append(url)
        return _TextResponse(live["text"], url)

    monkeypatch.setattr(content_mod, "_get_public_url", get)
    url = "https://www.nasa.gov/mars.txt"

    assert content_mod.fetch_webpage_content(url)["content"] == "Old page text."
    live["text"] = "New page text."
    # Ordinary reads keep the content cache.
    assert content_mod.fetch_webpage_content(url)["content"] == "Old page text."
    assert content_mod.fetch_webpage_content(url, bypass_cache=True)["content"] == "New page text."
    # The fresh copy replaces the stale one for later ordinary reads.
    assert content_mod.fetch_webpage_content(url)["content"] == "New page text."
    assert len(fetched) == 2


def test_navigator_fresh_fetch_bypasses_the_content_cache(monkeypatch):
    from src.research_navigator import ResearchNavigator
    calls = []

    def fake_fetch(url, **kwargs):
        calls.append(kwargs)
        return {"success": True, "content": "Mars evidence.", "title": "Mars"}

    monkeypatch.setattr("src.search.content.fetch_webpage_content", fake_fetch)
    asyncio.run(ResearchNavigator(fresh_fetch=True).fetch("https://www.nasa.gov/mars/"))
    asyncio.run(ResearchNavigator().fetch("https://www.nasa.gov/mars/"))

    assert calls[0].get("bypass_cache") is True
    assert "bypass_cache" not in calls[1]


def test_fresh_fetch_researcher_reads_pages_past_the_cache(monkeypatch):
    calls = []

    def fake_fetch(url, **kwargs):
        calls.append(kwargs)
        return {"success": True, "content": "Mars evidence.", "title": "Mars"}

    monkeypatch.setattr("src.search.content.fetch_webpage_content", fake_fetch)
    researcher = DeepResearcher(llm_endpoint="http://localhost:11434", llm_model="m", fresh_fetch=True)
    researcher._emit = lambda **k: None

    async def extract(messages, **kwargs):
        return json.dumps({"rational": "r", "summary": "Useful evidence.", "evidence": "e"})

    researcher._llm = extract
    finding = asyncio.run(researcher._fetch_and_extract("https://www.nasa.gov/mars/", "q", "Mars"))

    assert finding["url"] == "https://www.nasa.gov/mars/"
    assert calls == [{"timeout": 10, "bypass_cache": True}]


class _KwargsResearcher:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.findings = []
        self.evolving_report = ""
        self.category = None
        _KwargsResearcher.last = self

    async def research(self, query, **kwargs):
        return "report"

    def get_stats(self):
        return {"Rounds": 1}


def _researcher_kwargs(monkeypatch, **call_kwargs):
    import src.deep_research as dr
    import src.research_handler as rh
    import src.research_presets as rp

    monkeypatch.setattr(dr, "DeepResearcher", _KwargsResearcher)

    async def ok_probe(*a, **k):
        return None

    monkeypatch.setattr(rh.ResearchHandler, "_probe_endpoint", staticmethod(ok_probe))
    profile = rp.ResearchProfile("auto", rp.PRESETS["medium"], 32768, None, "test")
    monkeypatch.setattr(rp, "resolve_research_profile", lambda url, model, **kw: profile)
    handler = rh.ResearchHandler.__new__(rh.ResearchHandler)
    handler._legacy_engine = None
    handler._active_tasks = {}
    asyncio.run(handler.call_research_service("q", "http://localhost:11434", "m", **call_kwargs))
    return _KwargsResearcher.last.kwargs


def test_only_a_refresh_builds_a_fresh_fetch_researcher(monkeypatch):
    assert _researcher_kwargs(monkeypatch, fresh_fetch=True)["fresh_fetch"] is True
    assert _researcher_kwargs(monkeypatch)["fresh_fetch"] is False


@pytest.mark.parametrize("kwargs, expected", [
    ({"previous_version": PREVIOUS, "fresh_fetch": True}, True),
    ({"previous_version": PREVIOUS}, True),  # any refresh compares fresh page text
    ({}, False),
])
def test_start_research_passes_fresh_fetch_to_the_research_run(monkeypatch, tmp_path, kwargs, expected):
    import src.research_handler as rh
    monkeypatch.setattr(rh, "RESEARCH_DATA_DIR", tmp_path)
    handler = rh.ResearchHandler.__new__(rh.ResearchHandler)
    handler._legacy_engine = None
    handler._active_tasks = {}
    seen = {}

    async def fake_service(*args, **service_kwargs):
        seen.update(service_kwargs)
        return "report"

    async def no_summary(*args, **kw):
        return None

    handler.call_research_service = fake_service
    handler._summarize_refresh = no_summary
    handler._save_result = lambda *a, **k: None

    async def run():
        handler.start_research(session_id="rp-refresh00001", query="q", llm_endpoint="http://x", llm_model="m",
                               hard_timeout=60, **{k: (dict(v) if isinstance(v, dict) else v)
                                                   for k, v in kwargs.items()})
        await handler._active_tasks["rp-refresh00001"]["task"]

    asyncio.run(run())
    assert seen.get("fresh_fetch", False) is expected


def test_refresh_route_asks_for_fresh_pages_and_start_route_does_not(monkeypatch, research_dir):
    _write(research_dir, "rp-oldversion01", **SAVED)
    handler = _CapturingHandler()
    router = _router(monkeypatch, handler)

    asyncio.run(_route(router, "/api/research/{session_id}/refresh", "POST")(
        session_id="rp-oldversion01", request=_request("alice")))
    assert handler.started["fresh_fetch"] is True

    body = SimpleNamespace(query="q", max_rounds=0, search_provider=None, endpoint_id=None, model=None,
                           max_time=300, extraction_timeout=None, extraction_concurrency=None,
                           category=None, origin_chat_id=None)
    asyncio.run(_route(router, "/api/research/start", "POST")(body=body, request=_request("alice")))
    assert not handler.started.get("fresh_fetch")

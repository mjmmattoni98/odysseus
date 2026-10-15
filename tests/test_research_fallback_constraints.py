"""A failed Deep Research run never falls back to unconstrained results.

When the engine raised (for example on a malformed URL such as
``https://[bad`` in the question), the handler fell back to the legacy engine
or to a basic web search that ignored the run's source rules, re-read pages
from the content cache on a refresh, and returned text far past the
requested word limit. The fallback now fails closed when it cannot apply the
source rules or the refresh, and is cut to the word limit otherwise.
"""
import asyncio
import json

import pytest

import src.deep_research as dr
import src.research_handler as rh
import src.research_presets as rp
from src.deep_research import DeepResearcher
from src.research_navigator import ResearchAction, ResearchNavigator, ResearchPage
from src.research_presets import count_report_words

RULED_QUESTION = ("Summarize https://[bad using only official NASA sources, no Wikipedia, "
                  "in 120 words")
LIMITED_QUESTION = "Summarize the Mars Exploration Program in 120 words"
WIKIPEDIA_DUMP = "Mars article from https://en.wikipedia.org/wiki/Mars says the planet is red. " * 50


# ── The trigger: malformed URLs no longer crash the engine ──

def test_malformed_urls_are_skipped_not_raised():
    urls = DeepResearcher._extract_explicit_urls("See https://[bad and https://www.nasa.gov/mars/.")
    assert urls == ["https://www.nasa.gov/mars/"]
    assert DeepResearcher._result_host("https://[bad/x") == ""


class _Nav(ResearchNavigator):
    async def search(self, query, *, count=10):
        return [{"url": "https://[bad/x", "title": "bad", "content": "mars"},
                {"url": "https://www.nasa.gov/mars/", "title": "Mars", "content": "Mars exploration"}]

    async def fetch(self, url, *, timeout=10, max_bytes=None):
        return ResearchPage(url=url, title="Mars", content="Mars evidence.", success=True)


def test_research_with_a_malformed_url_in_the_question_completes():
    researcher = DeepResearcher(llm_endpoint="http://localhost:11434", llm_model="m",
                                max_rounds=1, min_rounds=1, report_word_limit=150)
    researcher.navigator = _Nav()
    researcher._emit = lambda **k: None

    async def llm(messages, **kwargs):
        return json.dumps({"rational": "r", "summary": "Useful evidence.", "evidence": "e"})

    async def plan(question):
        return "plan"

    async def no_category(question, research_plan=""):
        return None

    async def actions(question, report, round_num):
        return [ResearchAction("web_search", {"query": "mars"})]

    researcher._llm = llm
    researcher._create_plan = plan
    researcher._classify_category = no_category
    researcher._plan_research_actions = actions

    report = asyncio.run(researcher.research(RULED_QUESTION))

    assert report
    assert [f["url"] for f in researcher.findings] == ["https://www.nasa.gov/mars/"]


# ── The fallback honors the run's constraints ──

class _FailingResearcher:
    def __init__(self, **kwargs):
        self.findings = []
        self.evolving_report = ""
        self.category = None

    async def research(self, query, **kwargs):
        raise ValueError("Invalid IPv6 URL")

    def get_stats(self):
        return {}


@pytest.fixture
def failing_engine(monkeypatch):
    """Engine that always raises; records every fallback search."""
    monkeypatch.setattr(dr, "DeepResearcher", _FailingResearcher)

    async def ok_probe(*a, **k):
        return None

    monkeypatch.setattr(rh.ResearchHandler, "_probe_endpoint", staticmethod(ok_probe))
    profile = rp.ResearchProfile("auto", rp.PRESETS["medium"], 32768, None, "test")
    monkeypatch.setattr(rp, "resolve_research_profile", lambda url, model, **kw: profile)
    searches = []

    def basic_search(query, **kwargs):
        searches.append(query)
        return WIKIPEDIA_DUMP

    monkeypatch.setattr("src.search.comprehensive_web_search", basic_search)
    handler = rh.ResearchHandler.__new__(rh.ResearchHandler)
    handler._legacy_engine = None
    handler._active_tasks = {}
    return handler, searches


def _run(handler, question, **kwargs):
    return asyncio.run(handler.call_research_service(question, "http://localhost:11434", "m", **kwargs))


def test_failed_run_with_source_rules_fails_closed(failing_engine):
    handler, searches = failing_engine

    out = _run(handler, RULED_QUESTION)

    assert searches == []
    assert "Mars article from" not in out and "en.wikipedia.org" not in out
    assert "source rules" in out and "Invalid IPv6 URL" in out
    assert count_report_words(out) <= 150


def test_failed_refresh_fails_closed(failing_engine):
    handler, searches = failing_engine

    out = _run(handler, "Summarize the Mars Exploration Program", fresh_fetch=True)

    assert searches == []
    assert "refresh" in out


def test_failed_run_falls_back_to_basic_search_within_the_word_limit(failing_engine):
    handler, searches = failing_engine
    limit = rp.resolve_report_length(None, LIMITED_QUESTION).word_limit
    assert count_report_words(WIKIPEDIA_DUMP) > limit

    out = _run(handler, LIMITED_QUESTION)

    assert searches == [LIMITED_QUESTION]
    assert "Basic Search Fallback" in out
    assert count_report_words(out) <= limit


class _LegacyEngine:
    def __init__(self):
        self.calls = []
        self.findings = []
        self.source_reports = []
        self.progress_tracker = None

    def start_research(self, query, max_time):
        self.calls.append(query)
        return WIKIPEDIA_DUMP


def test_legacy_engine_fallback_respects_rules_and_word_limit(failing_engine):
    handler, searches = failing_engine
    legacy = _LegacyEngine()
    handler._legacy_engine = legacy

    ruled = _run(handler, RULED_QUESTION)
    assert legacy.calls == [] and "Mars article from" not in ruled

    limited = _run(handler, LIMITED_QUESTION)
    assert legacy.calls == [LIMITED_QUESTION]
    body = limited.split("## Research Summary")[-1]
    assert count_report_words(body.split("---", 1)[-1]) <= rp.resolve_report_length(None, LIMITED_QUESTION).word_limit
    assert searches == []


# ── Failed runs end as errors, and a failed refresh never becomes a version ──

PREVIOUS_VERSION = {"id": "rp-goodversion1", "lineage_id": "rp-goodversion1", "version": 1, "next_version": 2,
                    "sources": [{"url": "https://www.nasa.gov/mars/", "content_hash": "h1"}],
                    "raw_report": "Old answer."}


class _NoInformationResearcher(_FailingResearcher):
    async def research(self, query, **kwargs):
        self.evolving_report = ""
        return "No information could be gathered for this question."


def _start(handler, monkeypatch, tmp_path, question, **kwargs):
    monkeypatch.setattr(rh, "RESEARCH_DATA_DIR", tmp_path)
    summaries = []

    async def summarize(entry, *args, **kw):
        summaries.append(entry)

    handler._summarize_refresh = summarize

    async def run():
        handler.start_research(session_id="rp-attempt00001", query=question, llm_endpoint="http://x",
                               llm_model="m", hard_timeout=60, owner="alice", **kwargs)
        await handler._active_tasks["rp-attempt00001"]["task"]

    asyncio.run(run())
    return handler._active_tasks["rp-attempt00001"], summaries


def test_fail_closed_run_ends_as_an_error_not_a_saved_report(failing_engine, monkeypatch, tmp_path):
    handler, searches = failing_engine

    entry, _ = _start(handler, monkeypatch, tmp_path, RULED_QUESTION)

    assert entry["status"] == "error"
    assert handler.get_status("rp-attempt00001")["status"] == "error"
    assert "Invalid IPv6 URL" in entry["result"] and "source rules" in entry["result"]
    assert not entry["result"].startswith("#")  # shown as a plain error message
    assert list(tmp_path.glob("*.json")) == []
    assert searches == []


@pytest.mark.parametrize("engine", [_FailingResearcher, _NoInformationResearcher])
def test_failed_refresh_is_not_saved_as_a_version(failing_engine, monkeypatch, tmp_path, engine):
    from src.research_versions import list_versions
    handler, _searches = failing_engine
    monkeypatch.setattr(dr, "DeepResearcher", engine)

    entry, summaries = _start(handler, monkeypatch, tmp_path, "Summarize the Mars Exploration Program",
                              previous_version=dict(PREVIOUS_VERSION), fresh_fetch=True)

    assert entry["status"] == "error"
    assert "no new version was saved" in entry["result"] and "version 1" in entry["result"]
    assert summaries == [] and "change_summary" not in entry  # never a "removed: all" diff
    assert list(tmp_path.glob("*.json")) == []
    assert list_versions(tmp_path, "rp-goodversion1", "alice") == []


def test_refresh_partial_result_without_sources_is_not_saved(monkeypatch, tmp_path):
    handler = rh.ResearchHandler.__new__(rh.ResearchHandler)
    handler._legacy_engine = None
    handler._active_tasks = {}
    monkeypatch.setattr(rh, "RESEARCH_DATA_DIR", tmp_path)
    stalled_researcher = _FailingResearcher()
    stalled_researcher.evolving_report = "Partial notes without any readable source."

    async def stalled(*args, _task_entry=None, **kwargs):
        _task_entry["researcher"] = stalled_researcher
        await asyncio.sleep(5)

    handler.call_research_service = stalled

    async def run():
        handler.start_research(session_id="rp-attempt00002", query="q", llm_endpoint="http://x", llm_model="m",
                               hard_timeout=0.05, owner="alice", previous_version=dict(PREVIOUS_VERSION))
        await handler._active_tasks["rp-attempt00002"]["task"]

    asyncio.run(run())
    entry = handler._active_tasks["rp-attempt00002"]
    assert entry["status"] == "error"
    assert "no new version was saved" in entry["result"]
    assert list(tmp_path.glob("*.json")) == []

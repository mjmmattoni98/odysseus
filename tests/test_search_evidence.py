import pytest
from services.search import core
from src.assistant_preferences import assistant_turn, AssistantPreferences, search_reports


@pytest.fixture
def search(monkeypatch):
    monkeypatch.setattr(core, "_get_search_settings", lambda: {"search_provider": "searxng"})
    monkeypatch.setattr(core, "_get_result_count", lambda: 3)
    monkeypatch.setattr(core, "_build_provider_chain", lambda provider: ["searxng", "duckduckgo"])
    monkeypatch.setattr(core, "rank_search_results", lambda query, rows: rows)
    return core


def test_records_fallback_and_distinguishes_page_text_from_snippets(search, monkeypatch):
    def provider(name, *args):
        if name == "searxng":
            raise RuntimeError("untrusted upstream error must not enter UI status")
        return [{"url": f"https://example.com/{n}", "title": str(n), "snippet": "snippet"} for n in range(4)]
    def fetch(url, *args, **kwargs):
        return {"url": url, "title": "Example page", "success": url.endswith("0"), "content": "page text" if url.endswith("0") else ""}
    monkeypatch.setattr(search, "_call_provider", provider)
    monkeypatch.setattr(search, "fetch_webpage_content", fetch)
    with assistant_turn(AssistantPreferences()):
        text, sources = search.comprehensive_web_search("test", max_pages=2, return_sources=True)
        report = search_reports()[0]
    # Page 1 failed, so the next-ranked result was tried in its place.
    assert [s["read_status"] for s in sources] == ["read", "failed", "failed", "snippet"]
    assert report["fallback"] is True
    assert report["provider"] == "duckduckgo"
    assert report["pages_read"] == 1
    assert report["pages_failed"] == 2
    assert "untrusted" not in str(report)
    assert "page text" in text


def test_failed_search_has_a_report_even_without_sources(search, monkeypatch):
    monkeypatch.setattr(search, "_call_provider", lambda *args: [])
    report = {}
    text, sources = search.comprehensive_web_search("test", return_sources=True, status=report)
    assert sources == []
    assert report["state"] == "empty"
    assert report["pages_read"] == 0


def test_multiple_searches_keep_citations_stable_and_report_partial_text(search, monkeypatch):
    rounds = [["a", "b"], ["b", "c"]]
    def provider(*args):
        return [{"url": f"https://example.com/{name}", "title": name, "snippet": name} for name in rounds.pop(0)]
    monkeypatch.setattr(search, "_call_provider", provider)
    monkeypatch.setattr(search, "fetch_webpage_content", lambda url, *a, **k: {"url": url, "title": url, "success": True, "content": "x" * 4000})
    with assistant_turn(AssistantPreferences()):
        _, first = search.comprehensive_web_search("first", return_sources=True)
        output, second = search.comprehensive_web_search("second", return_sources=True)
    assert [s["citation"] for s in first] == [1, 2]
    assert [s["citation"] for s in second] == [2, 3]
    assert "[2] b" in output and "[3] c" in output
    assert "[CONTENT 2]" in output and "[CONTENT 3]" in output
    assert all(s["partial"] for s in first + second)


def test_unreadable_top_pages_are_replaced_by_the_next_results(search, monkeypatch):
    names = ["blocked1", "blocked2", "c", "d", "e", "f"]
    monkeypatch.setattr(search, "_get_result_count", lambda: len(names))
    monkeypatch.setattr(search, "_call_provider", lambda *args: [
        {"url": f"https://example.com/{name}", "title": name, "snippet": name} for name in names
    ])
    fetched = []
    def fetch(url, *args, **kwargs):
        fetched.append(url)
        ok = "blocked" not in url
        return {"url": url, "title": url, "success": ok, "content": "page text" if ok else ""}
    monkeypatch.setattr(search, "fetch_webpage_content", fetch)
    report = {}
    _, sources = search.comprehensive_web_search("test", max_pages=3, return_sources=True, status=report)
    assert [s["read_status"] for s in sources] == ["failed", "failed", "read", "read", "read", "snippet"]
    assert report["pages_read"] == 3
    assert len(fetched) == 5


def test_readable_top_pages_do_not_trigger_extra_fetches(search, monkeypatch):
    monkeypatch.setattr(search, "_get_result_count", lambda: 5)
    monkeypatch.setattr(search, "_call_provider", lambda *args: [
        {"url": f"https://example.com/{n}", "title": str(n), "snippet": str(n)} for n in range(5)
    ])
    fetched = []
    def fetch(url, *args, **kwargs):
        fetched.append(url)
        return {"url": url, "title": url, "success": True, "content": "page text"}
    monkeypatch.setattr(search, "fetch_webpage_content", fetch)
    search.comprehensive_web_search("test", max_pages=3, return_sources=True)
    assert len(fetched) == 3

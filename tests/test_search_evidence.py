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
        return [{"url": f"https://example.com/{n}", "title": str(n), "snippet": "snippet"} for n in range(3)]
    def fetch(url, *args, **kwargs):
        return {"url": url, "title": "Example page", "success": url.endswith("0"), "content": "page text" if url.endswith("0") else ""}
    monkeypatch.setattr(search, "_call_provider", provider)
    monkeypatch.setattr(search, "fetch_webpage_content", fetch)
    with assistant_turn(AssistantPreferences()):
        text, sources = search.comprehensive_web_search("test", max_pages=2, return_sources=True)
        report = search_reports()[0]
    assert [s["read_status"] for s in sources] == ["read", "failed", "snippet"]
    assert report["fallback"] is True
    assert report["provider"] == "duckduckgo"
    assert report["pages_read"] == 1
    assert report["pages_failed"] == 1
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

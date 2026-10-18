"""Brave must receive the caller's query, not a synthesized boolean expression.

The Brave provider used to rewrite every query through ``build_enhanced_query``,
which split natural language on "and"/"or", AND-joined the fragments, quoted
capitalized words, and OR-appended question-type boosts. A chat turn such as
"Tell me where to download X and where to get Y" became
``(Tell me ... "Tell") AND ((where to get Y) OR (location))`` and Brave
returned one or two unrelated hits. The other providers already pass the query
through unchanged; recency is a request parameter, not query text.
"""

import pytest

from services.search import providers


class _Response:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"web": {"results": [
            {"title": "t", "url": "https://example.com/a", "description": "d"},
        ]}}


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, "_get_search_settings", lambda: {}, raising=False)
    monkeypatch.setattr(providers, "_safesearch_for", lambda *_a, **_k: None, raising=False)
    monkeypatch.setenv("DATA_BRAVE_API_KEY", "k")

    def fake_get(url, *, params=None, **_kwargs):
        calls.append(params)
        return _Response()

    monkeypatch.setattr(providers.httpx, "get", fake_get)
    return calls


def test_natural_language_query_reaches_brave_unchanged(sent):
    query = "Tell me where to download the shadps4 emulator and where to report bugs"

    results = providers.brave_search(query, 5)

    assert results and results[0]["url"] == "https://example.com/a"
    assert sent[0]["q"] == query


def test_site_operator_is_preserved(sent):
    providers.brave_search("shadps4 releases site:github.com", 5)

    assert sent[0]["q"] == "shadps4 releases site:github.com"


def test_time_filter_uses_freshness_param_not_query_text(sent):
    providers.brave_search("rust 2.0 release", 5, time_filter="week")

    assert sent[0]["q"] == "rust 2.0 release"
    assert sent[0]["freshness"] == "week"

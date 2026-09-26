"""Tests for the cached Ollama capability oracle (src/ollama_capabilities.py)."""
import pytest

import src.ollama_capabilities as oc


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.is_success = 200 <= status < 300

    def json(self):
        return self._payload


def _patch_post(monkeypatch, payload, calls):
    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        if isinstance(payload, Exception):
            raise payload
        return payload

    monkeypatch.setattr(oc.httpx, "post", fake_post)


@pytest.fixture(autouse=True)
def _clear_cache():
    oc.reset_cache()
    yield
    oc.reset_cache()


def test_api_root_only_for_ollama_ports():
    assert oc.ollama_api_root("http://localhost:11434/v1") == "http://localhost:11434"
    assert (
        oc.ollama_api_root("http://host.docker.internal:11434/v1/chat/completions")
        == "http://host.docker.internal:11434"
    )
    assert oc.ollama_api_root("https://ollama.com/api/chat") == "https://ollama.com"
    assert oc.ollama_api_root("http://localhost:1234/v1") == ""
    assert oc.ollama_api_root("") == ""


def test_capability_tokens_parsed_and_cached(monkeypatch):
    calls = []
    _patch_post(monkeypatch, _Resp({"capabilities": ["completion", "tools", "thinking"]}), calls)

    tokens = oc.capability_tokens("http://localhost:11434/v1", "lfm2.5:8b")
    assert tokens == frozenset({"completion", "tools", "thinking"})

    oc.capability_tokens("http://localhost:11434/v1", "lfm2.5:8b")
    assert len(calls) == 1, "second lookup must be served from cache"
    assert calls[0][1] == {"model": "lfm2.5:8b"}


def test_supports_helpers(monkeypatch):
    calls = []
    _patch_post(monkeypatch, _Resp({"capabilities": ["tools"]}), calls)
    assert oc.supports_tool_calls("http://localhost:11434/v1", "a") is True
    assert oc.supports_thinking("http://localhost:11434/v1", "a") is False


def test_failure_is_cached_and_returns_none(monkeypatch):
    calls = []
    _patch_post(monkeypatch, _Resp({}, status=404), calls)
    assert oc.capability_tokens("http://localhost:11434/v1", "a") is None
    oc.capability_tokens("http://localhost:11434/v1", "a")
    assert len(calls) == 1, "failures must be cached briefly instead of re-probing"


def test_transport_error_returns_none(monkeypatch):
    calls = []
    _patch_post(monkeypatch, RuntimeError("connection refused"), calls)
    assert oc.capability_tokens("http://localhost:11434/v1", "a") is None


def test_missing_capabilities_list_is_unknown(monkeypatch):
    calls = []
    _patch_post(monkeypatch, _Resp({}), calls)
    assert oc.capability_tokens("http://localhost:11434/v1", "a") is None


def test_non_ollama_port_never_probes(monkeypatch):
    calls = []
    _patch_post(monkeypatch, _Resp({"capabilities": ["tools"]}), calls)
    assert oc.capability_tokens("http://localhost:1234/v1", "a") is None
    assert calls == []


class TestRouteSupportsThinking:
    def test_capability_report_wins(self, monkeypatch):
        from src import llm_core

        monkeypatch.setattr("src.ollama_capabilities.supports_thinking", lambda url, model: True)
        assert llm_core._route_supports_thinking("http://localhost:11434/v1", "ornith-1.5:9b") is True

        monkeypatch.setattr("src.ollama_capabilities.supports_thinking", lambda url, model: False)
        assert llm_core._route_supports_thinking("http://localhost:11434/v1", "qwen3.8:27b") is False

    def test_name_heuristics_fall_back_when_unknown(self, monkeypatch):
        from src import llm_core

        monkeypatch.setattr("src.ollama_capabilities.supports_thinking", lambda url, model: None)
        assert llm_core._route_supports_thinking("http://localhost:11434/v1", "qwen3.8:27b") is True
        assert llm_core._route_supports_thinking("http://localhost:11434/v1", "lfm2.5:8b") is False
        # Non-Ollama routes never probe and use the name list directly.
        assert llm_core._route_supports_thinking("https://api.example.com/v1", "qwen3.8:27b") is True

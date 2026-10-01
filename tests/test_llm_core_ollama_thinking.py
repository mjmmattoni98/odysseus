"""Tests for Ollama /v1 thinking-suppression helpers.

Covers:
- _is_ollama_openai_compat_url: URL classification (local host + /v1 path)
- think: false is injected into the payload for Ollama /v1 thinking models
- think: false is NOT injected for non-thinking models or non-Ollama /v1 endpoints
"""
import asyncio
import json

import pytest

from src import llm_core


@pytest.fixture(autouse=True)
def _no_real_capability_probes(monkeypatch):
    """Keep payload tests hermetic: the Ollama capability probe falls back to
    the name heuristics instead of hitting a real server on port 11434, and the
    thinking effort defaults to "auto" regardless of the local settings file."""
    monkeypatch.setattr("src.ollama_capabilities.supports_thinking", lambda url, model: None)
    monkeypatch.setattr(llm_core, "_ollama_thinking_effort", lambda: "auto")
    yield
    import src.ollama_capabilities as oc
    oc.reset_cache()


def _fingerprint(monkeypatch, ollama_roots):
    """Answer /api/version probes: Ollama only for ``ollama_roots``."""
    import src.ollama_capabilities as oc
    oc.reset_cache()
    monkeypatch.setattr(oc, "_probe_version", lambda root, timeout: "0.34.4" if root in ollama_roots else None)
    monkeypatch.setattr(oc, "_registered_kind", lambda url: None)


# ---------------------------------------------------------------------------
# Fake HTTP client — captures the outgoing payload without network I/O
# ---------------------------------------------------------------------------

class _FakeResp:
    status_code = 200

    async def aiter_lines(self):
        # Yield a minimal done event so stream_llm exits cleanly
        yield json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})
        yield "data: [DONE]"

    async def aread(self):
        return b""


class _FakeStreamCtx:
    def __init__(self, captured):
        self._captured = captured

    async def __aenter__(self):
        return _FakeResp()

    async def __aexit__(self, *a):
        return False


class _FakeClient:
    """Minimal stand-in for httpx.AsyncClient that captures request payload."""

    def __init__(self):
        self.captured_payload = {}

    def stream(self, method, url, **kw):
        self.captured_payload = kw.get("json") or {}
        return _FakeStreamCtx(self.captured_payload)


def _capture_payload(monkeypatch, url, model):
    """Run stream_llm, intercept the HTTP payload, and return it."""
    client = _FakeClient()
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: client)
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda u: False)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "get_context_length", lambda u, m: 32768)

    async def run():
        return [c async for c in llm_core.stream_llm(
            url, model, [{"role": "user", "content": "hi"}],
        )]

    asyncio.run(run())
    return client.captured_payload


# ---------------------------------------------------------------------------
# _is_ollama_openai_compat_url — pure function, no I/O
# ---------------------------------------------------------------------------

class TestIsOllamaOpenAICompatUrl:
    """Unit tests for the URL classifier that gates think-suppression."""

    # Positive cases — should be True
    def test_default_port_v1_root(self):
        assert llm_core._is_ollama_openai_compat_url("http://127.0.0.1:11434/v1")

    def test_default_port_chat_completions(self):
        assert llm_core._is_ollama_openai_compat_url("http://127.0.0.1:11434/v1/chat/completions")

    def test_localhost_default_port(self):
        assert llm_core._is_ollama_openai_compat_url("http://localhost:11434/v1")

    def test_localhost_default_port_with_path(self):
        assert llm_core._is_ollama_openai_compat_url("http://localhost:11434/v1/chat/completions")

    def test_loopback_ipv6(self):
        # IPv6 addresses in URLs require square brackets per RFC 3986
        assert llm_core._is_ollama_openai_compat_url("http://[::1]:11434/v1")

    def test_any_local_non_default_port(self, monkeypatch):
        """Localhost on a non-default port (custom OLLAMA_HOST) matches once
        the server answered /api/version like Ollama."""
        _fingerprint(monkeypatch, {"http://127.0.0.1:11435"})
        assert llm_core._is_ollama_openai_compat_url("http://127.0.0.1:11435/v1")

    def test_localhost_non_default_port_that_is_not_ollama(self, monkeypatch):
        """llama.cpp/LM Studio/vLLM /v1 servers on localhost are not Ollama
        and must not receive Ollama-only thinking fields."""
        _fingerprint(monkeypatch, set())
        assert not llm_core._is_ollama_openai_compat_url("http://localhost:8080/v1/chat/completions")

    def test_zero_dot_zero_host(self):
        assert llm_core._is_ollama_openai_compat_url("http://0.0.0.0:11434/v1")

    # Negative cases — should be False
    def test_openai_api_v1(self):
        """Real OpenAI endpoint must never match, even though path is /v1."""
        assert not llm_core._is_ollama_openai_compat_url("https://api.openai.com/v1")

    def test_openai_chat_completions(self):
        assert not llm_core._is_ollama_openai_compat_url("https://api.openai.com/v1/chat/completions")

    def test_ollama_native_api_path(self):
        """The native /api path is a different surface and must not match /v1."""
        assert not llm_core._is_ollama_openai_compat_url("http://localhost:11434/api")

    def test_ollama_native_api_chat(self):
        assert not llm_core._is_ollama_openai_compat_url("http://localhost:11434/api/chat")

    def test_remote_openrouter(self):
        assert not llm_core._is_ollama_openai_compat_url("https://openrouter.ai/api/v1")

    def test_empty_string(self):
        assert not llm_core._is_ollama_openai_compat_url("")

    def test_none_like_empty(self):
        assert not llm_core._is_ollama_openai_compat_url(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Payload injection — think: false only when both conditions hold
# ---------------------------------------------------------------------------

class TestThinkSuppression:
    """Assert think:false is present/absent in the outgoing HTTP payload."""

    def test_think_false_for_ollama_v1_thinking_model(self, monkeypatch):
        """think:false must be set for qwen3 on Ollama /v1."""
        payload = _capture_payload(
            monkeypatch, "http://127.0.0.1:11434/v1/chat/completions", "qwen3:14b"
        )
        assert payload.get("think") is False

    def test_no_think_for_ollama_v1_non_thinking_model(self, monkeypatch):
        """think must NOT be set for a plain (non-thinking) model on Ollama /v1."""
        payload = _capture_payload(
            monkeypatch, "http://127.0.0.1:11434/v1/chat/completions", "llama3.2:3b"
        )
        assert "think" not in payload

    def test_no_think_for_openai_endpoint_with_thinking_model_name(self, monkeypatch):
        """think must NOT leak to a real OpenAI endpoint even if the model name
        matches a thinking pattern — the URL guard is what matters."""
        payload = _capture_payload(
            monkeypatch, "https://api.openai.com/v1/chat/completions", "qwen3:14b"
        )
        assert "think" not in payload

    def test_think_false_for_non_default_port_thinking_model(self, monkeypatch):
        """Custom-port localhost Ollama (e.g. OLLAMA_HOST=0.0.0.0:11435) must
        also receive think:false once fingerprinted as Ollama."""
        _fingerprint(monkeypatch, {"http://127.0.0.1:11435"})
        import src.ollama_capabilities as oc
        assert oc.is_ollama_url("http://127.0.0.1:11435/v1")  # warm the cache outside the loop
        payload = _capture_payload(
            monkeypatch, "http://127.0.0.1:11435/v1/chat/completions", "qwen3:14b"
        )
        assert payload.get("think") is False


class TestOllamaKeepAlive:
    """ODYSSEUS_OLLAMA_KEEP_ALIVE is a native-Ollama setting only.

    Ollama's /v1 adapter discards the field, so sending it on the compat
    surface would be a silent no-op; cloud is excluded too (the residency
    concern is local, and cloud rejects/normalizes request options).
    """

    def test_compat_surface_does_not_send_keep_alive(self, monkeypatch):
        monkeypatch.setenv("ODYSSEUS_OLLAMA_KEEP_ALIVE", "30m")
        payload = _capture_payload(
            monkeypatch, "http://127.0.0.1:11434/v1/chat/completions", "qwen3:14b"
        )
        assert "keep_alive" not in payload

    def test_native_absent_by_default(self, monkeypatch):
        monkeypatch.delenv("ODYSSEUS_OLLAMA_KEEP_ALIVE", raising=False)
        payload = llm_core._build_ollama_payload(
            "qwen3:14b", [{"role": "user", "content": "hi"}], 0.7, 100,
            stream=False, url="http://localhost:11434/api/chat",
        )
        assert "keep_alive" not in payload

    def test_native_duration_string(self, monkeypatch):
        monkeypatch.setenv("ODYSSEUS_OLLAMA_KEEP_ALIVE", "1h")
        payload = llm_core._build_ollama_payload(
            "qwen3:14b", [{"role": "user", "content": "hi"}], 0.7, 100,
            stream=False, url="http://localhost:11434/api/chat",
        )
        assert payload.get("keep_alive") == "1h"

    def test_native_numeric_values_are_ints(self, monkeypatch):
        # "-1" as a string makes Ollama return
        # `time: missing unit in duration "-1"`; it must be a JSON number.
        monkeypatch.setenv("ODYSSEUS_OLLAMA_KEEP_ALIVE", "-1")
        payload = llm_core._build_ollama_payload(
            "qwen3:14b", [{"role": "user", "content": "hi"}], 0.7, 100,
            stream=False, url="http://localhost:11434/api/chat",
        )
        assert payload.get("keep_alive") == -1
        assert isinstance(payload.get("keep_alive"), int)

    def test_apply_helper_numeric_is_int(self, monkeypatch):
        monkeypatch.setenv("ODYSSEUS_OLLAMA_KEEP_ALIVE", "0")
        payload = {}
        llm_core._apply_ollama_keep_alive(payload, "http://localhost:11434/api/chat")
        assert payload.get("keep_alive") == 0

    def test_cloud_native_does_not_send_keep_alive(self, monkeypatch):
        monkeypatch.setenv("ODYSSEUS_OLLAMA_KEEP_ALIVE", "30m")
        payload = llm_core._build_ollama_payload(
            "gpt-oss:120b", [{"role": "user", "content": "hi"}], 0.7, 100,
            stream=False, url="https://ollama.com/api/chat",
        )
        assert "keep_alive" not in payload

    def test_cloud_apply_helper_is_a_noop(self, monkeypatch):
        monkeypatch.setenv("ODYSSEUS_OLLAMA_KEEP_ALIVE", "30m")
        payload = {}
        llm_core._apply_ollama_keep_alive(payload, "https://ollama.com/api")
        assert "keep_alive" not in payload


class TestOllamaThinkingEffort:
    """Settings-selected thinking effort maps per transport surface."""

    def test_auto_keeps_existing_suppression(self, monkeypatch):
        monkeypatch.setattr(llm_core, "_ollama_thinking_effort", lambda: "auto")
        payload = _capture_payload(
            monkeypatch, "http://127.0.0.1:11434/v1/chat/completions", "qwen3:14b"
        )
        assert payload.get("think") is False
        assert "reasoning_effort" not in payload

    def test_off_uses_reasoning_effort_none_on_v1(self, monkeypatch):
        # think:false is ignored by current Ollama /v1 builds; "none" is the
        # effective off switch and must not be pre-empted by think:false.
        monkeypatch.setattr(llm_core, "_ollama_thinking_effort", lambda: "off")
        payload = _capture_payload(
            monkeypatch, "http://127.0.0.1:11434/v1/chat/completions", "qwen3:14b"
        )
        assert payload.get("reasoning_effort") == "none"
        assert "think" not in payload

    def test_level_is_sent_on_v1(self, monkeypatch):
        monkeypatch.setattr(llm_core, "_ollama_thinking_effort", lambda: "low")
        payload = _capture_payload(
            monkeypatch, "http://127.0.0.1:11434/v1/chat/completions", "gemma4:12b"
        )
        assert payload.get("reasoning_effort") == "low"
        assert "think" not in payload

    def test_effort_never_leaks_to_cloud(self, monkeypatch):
        monkeypatch.setattr(llm_core, "_ollama_thinking_effort", lambda: "high")
        payload = _capture_payload(
            monkeypatch, "https://api.openai.com/v1/chat/completions", "gpt-4o"
        )
        assert "reasoning_effort" not in payload
        assert "think" not in payload

    def test_native_payload_level(self, monkeypatch):
        monkeypatch.setattr(llm_core, "_ollama_thinking_effort", lambda: "high")
        payload = llm_core._build_ollama_payload(
            "qwen3:14b", [{"role": "user", "content": "hi"}], 0.7, 100,
            stream=False, url="http://localhost:11434/api/chat",
        )
        assert payload.get("think") == "high"

    def test_native_payload_off(self, monkeypatch):
        monkeypatch.setattr(llm_core, "_ollama_thinking_effort", lambda: "off")
        payload = llm_core._build_ollama_payload(
            "qwen3:14b", [{"role": "user", "content": "hi"}], 0.7, 100,
            stream=False, url="http://localhost:11434/api/chat",
        )
        assert payload.get("think") is False

    def test_unsupported_model_gets_no_effort_compat(self, monkeypatch):
        # Switching to a non-thinking model must not carry the setting over —
        # Ollama rejects thinking controls for unsupported models.
        monkeypatch.setattr(llm_core, "_ollama_thinking_effort", lambda: "low")
        monkeypatch.setattr(llm_core, "_route_supports_thinking", lambda url, model: False)
        payload = _capture_payload(
            monkeypatch, "http://127.0.0.1:11434/v1/chat/completions", "some-text-model"
        )
        assert "reasoning_effort" not in payload
        assert "think" not in payload

    def test_unsupported_model_gets_no_effort_native(self, monkeypatch):
        monkeypatch.setattr(llm_core, "_ollama_thinking_effort", lambda: "high")
        monkeypatch.setattr(llm_core, "_route_supports_thinking", lambda url, model: False)
        payload = llm_core._build_ollama_payload(
            "some-text-model", [{"role": "user", "content": "hi"}], 0.7, 100,
            stream=False, url="http://localhost:11434/api/chat",
        )
        assert "think" not in payload

    def test_native_payload_auto_omits_think(self, monkeypatch):
        monkeypatch.setattr(llm_core, "_ollama_thinking_effort", lambda: "auto")
        payload = llm_core._build_ollama_payload(
            "qwen3:14b", [{"role": "user", "content": "hi"}], 0.7, 100,
            stream=False, url="http://localhost:11434/api/chat",
        )
        assert "think" not in payload


# Captured before the autouse fixture patches it, so the settings tests can
# exercise the real accessor.
_real_thinking_effort = llm_core._ollama_thinking_effort


class TestOllamaThinkingEffortSetting:
    def _set(self, monkeypatch, raw):
        import src.settings as settings_mod

        monkeypatch.setattr(settings_mod, "load_settings", lambda: {"local_thinking_effort": raw})

    def test_normalizes_case_and_whitespace(self, monkeypatch):
        self._set(monkeypatch, " LOW ")
        assert _real_thinking_effort() == "low"

    def test_rejects_unknown_values(self, monkeypatch):
        self._set(monkeypatch, "banana")
        assert _real_thinking_effort() == "auto"

    def test_missing_key_is_auto(self, monkeypatch):
        import src.settings as settings_mod

        monkeypatch.setattr(settings_mod, "load_settings", lambda: {})
        assert _real_thinking_effort() == "auto"

    def test_settings_failure_is_auto(self, monkeypatch):
        import src.settings as settings_mod

        def boom():
            raise RuntimeError("settings unavailable")

        monkeypatch.setattr(settings_mod, "load_settings", boom)
        assert _real_thinking_effort() == "auto"

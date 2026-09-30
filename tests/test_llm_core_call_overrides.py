"""Per-call ``think`` / ``response_schema`` overrides and the utility workload."""
import asyncio

import pytest

from src import llm_core

SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
}


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    monkeypatch.setattr("src.ollama_capabilities.supports_thinking", lambda url, model: True)
    monkeypatch.setattr(llm_core, "_ollama_thinking_effort", lambda: "auto")
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda u: False)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "get_context_length", lambda u, m: 32768)
    monkeypatch.setattr(llm_core, "_local_model_gate_enabled", lambda: False)
    llm_core._response_cache.clear()


class _Resp:
    status_code = 200
    is_success = True
    text = ""

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class _Client:
    def __init__(self, data):
        self.payload = None
        self._data = data

    async def post(self, url, **kw):
        self.payload = kw.get("json")
        return _Resp(self._data)


def _run_async(monkeypatch, url, **kwargs):
    native = "/v1" not in url
    data = (
        {"model": "qwen3:8b", "message": {"content": "{\"title\": \"x\"}"}}
        if native
        else {"model": "qwen3:8b", "choices": [{"message": {"content": "{\"title\": \"x\"}"}}]}
    )
    client = _Client(data)
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: client)

    async def run():
        return await llm_core.llm_call_async(
            url, "qwen3:8b", [{"role": "user", "content": "hi"}], **kwargs
        )

    result = asyncio.run(run())
    return result, client.payload


def test_native_ollama_think_false_and_format(monkeypatch):
    result, payload = _run_async(
        monkeypatch, "http://localhost:11434", think=False, response_schema=SCHEMA,
    )
    assert result == "{\"title\": \"x\"}"
    assert payload["think"] is False
    assert payload["format"] == SCHEMA


def test_native_ollama_without_overrides_sends_nothing(monkeypatch):
    _, payload = _run_async(monkeypatch, "http://localhost:11434")
    assert "think" not in payload
    assert "format" not in payload


def test_ollama_v1_think_false_uses_reasoning_effort_none(monkeypatch):
    _, payload = _run_async(
        monkeypatch, "http://localhost:11434/v1", think=False, response_schema=SCHEMA,
    )
    assert payload["reasoning_effort"] == "none"
    assert "think" not in payload
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": SCHEMA},
    }


def test_think_level_override(monkeypatch):
    _, payload = _run_async(monkeypatch, "http://localhost:11434", think="low")
    assert payload["think"] == "low"


def test_think_override_ignored_for_non_thinking_model(monkeypatch):
    monkeypatch.setattr("src.ollama_capabilities.supports_thinking", lambda url, model: False)
    _, payload = _run_async(monkeypatch, "http://localhost:11434", think=False)
    assert "think" not in payload


def test_cloud_openai_ignores_response_schema(monkeypatch):
    payload = {"model": "gpt"}
    llm_core._apply_openai_compat_response_schema(payload, "https://api.openai.com/v1", SCHEMA)
    assert "response_format" not in payload


def test_cache_key_partitions_overrides():
    base = llm_core._get_cache_key("http://h", "m", [{"role": "user", "content": "x"}], 0.1, 10)
    think_off = llm_core._get_cache_key(
        "http://h", "m", [{"role": "user", "content": "x"}], 0.1, 10, think=False,
    )
    schema = llm_core._get_cache_key(
        "http://h", "m", [{"role": "user", "content": "x"}], 0.1, 10, response_schema=SCHEMA,
    )
    assert len({base, think_off, schema}) == 3


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), (True, None), (False, "off"), ("off", "off"), ("HIGH", "high"), ("auto", None), ("bogus", None)],
)
def test_normalize_think_override(value, expected):
    assert llm_core._normalize_think_override(value) == expected


def test_utility_workload_is_recognized():
    assert llm_core._gate_workload("utility") == "utility"
    assert llm_core._gate_workload("background") == "background"
    assert llm_core._gate_workload("nonsense") == "foreground"


def test_utility_workload_is_not_cancelled_by_foreground(monkeypatch):
    monkeypatch.setattr(llm_core, "_local_model_gate_enabled", lambda: True)
    monkeypatch.setattr(llm_core, "is_local_endpoint", lambda url: True)
    monkeypatch.setattr("src.interactive_gate.has_foreground_activity", lambda: True)

    async def run():
        order = []
        utility_started = asyncio.Event()

        async def utility():
            async with llm_core._local_model_slot("http://localhost:11434", "m", "utility"):
                order.append("utility-start")
                utility_started.set()
                await asyncio.sleep(0.05)
                order.append("utility-end")

        async def foreground():
            await utility_started.wait()
            async with llm_core._local_model_slot("http://localhost:11434", "m", "foreground"):
                order.append("foreground")

        await asyncio.gather(utility(), foreground())
        return order

    # Utility ignores browser activity (would block "background") and finishes
    # before the queued foreground request instead of being cancelled.
    assert asyncio.run(run()) == ["utility-start", "utility-end", "foreground"]

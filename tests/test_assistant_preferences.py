import asyncio

import pytest

from src.assistant_preferences import (
    AssistantPreferences, assistant_turn, current_preferences, parse_preferences,
    web_enabled, context_limit, merge_sources, record_search_report, search_reports,
)
from src import llm_core


@pytest.mark.parametrize("message", ["latest local models", "compare monitors", "Busca precios en España", "verify this claim"])
def test_auto_web_recognizes_research(message):
    assert web_enabled(AssistantPreferences(), message)


def test_web_off_and_explicit_denial_win():
    assert not web_enabled(AssistantPreferences(web_mode="off"), "search today")
    assert not web_enabled(AssistantPreferences(web_mode="on"), "search", explicitly_denied=True)
    assert not web_enabled(AssistantPreferences(), "Explain recursion")
    assert web_enabled(AssistantPreferences(web_mode="on"), "Explain recursion")


@pytest.mark.parametrize("data", [
    {"profile": "unknown"}, {"web_mode": "maybe"}, {"thinking": "yes"},
    {"instructions": "x" * 10001}, {"context_limits": {"model": 0}},
    {"context_limits": {"model": 999999}}, {"unexpected": True},
])
def test_invalid_preferences_are_rejected(data):
    with pytest.raises(ValueError):
        parse_preferences(data)


def test_concurrent_conversations_keep_thinking_and_search_reports_separate(monkeypatch):
    monkeypatch.setattr(llm_core, "_route_supports_thinking", lambda *args: True)
    async def run():
        both_started = asyncio.Event()
        started = 0
        async def turn(effort):
            nonlocal started
            with assistant_turn(AssistantPreferences(thinking=effort), "http://localhost:11434"):
                started += 1
                if started == 2:
                    both_started.set()
                await both_started.wait()
                record_search_report({"state": effort})
                await asyncio.sleep(0)
                return llm_core._ollama_think_value("http://localhost:11434/api/chat", "qwen"), search_reports()
        results = await asyncio.gather(turn("off"), turn("high"))
        assert results == [(False, [{"state": "off"}]), ("high", [{"state": "high"}])]
        assert current_preferences().profile == "legacy"
        assert search_reports() == []
    asyncio.run(run())


def test_local_context_cap_applies_to_payload_and_is_model_specific(monkeypatch):
    monkeypatch.setattr(llm_core, "_ollama_think_value", lambda *args: None)
    monkeypatch.setattr(llm_core, "is_local_endpoint", lambda *args: True)
    prefs = AssistantPreferences(context_limits={"big": 65536})
    with assistant_turn(prefs, "http://localhost:11434/v1"):
        payload = llm_core._build_ollama_payload("big", [], 0.2, 100, num_ctx=262144, url="http://localhost:11434/api/chat")
        assert payload["options"]["num_ctx"] == 65536
        assert context_limit("http://localhost:11434", "other") == 32768
        assert context_limit("http://other:11434", "big") == 32768
        smaller = llm_core._build_ollama_payload("big", [], 0.2, 100, num_ctx=8192, url="http://localhost:11434/api/chat")
        assert smaller["options"]["num_ctx"] == 8192


def test_discovered_128k_maximum_still_bounds_a_larger_local_cap(monkeypatch):
    monkeypatch.setattr(llm_core, "_ollama_think_value", lambda *args: None)
    monkeypatch.setattr(llm_core, "is_local_endpoint", lambda *args: True)
    with assistant_turn(AssistantPreferences(context_limits={"m": 262144}), "http://localhost:11434"):
        payload = llm_core._build_ollama_payload("m", [], 0.2, 100, num_ctx=131072, url="http://localhost:11434/api/chat")
    assert payload["options"]["num_ctx"] == 131072


def test_cache_key_separates_thinking_settings():
    with assistant_turn(AssistantPreferences(thinking="off")):
        first = llm_core._get_cache_key("http://localhost:11434", "m", [], 1, 100)
    with assistant_turn(AssistantPreferences(thinking="high")):
        second = llm_core._get_cache_key("http://localhost:11434", "m", [], 1, 100)
    assert first != second


def test_search_rounds_preserve_sources_and_upgrade_fetched_pages():
    first = [{"url": "https://a.example", "read_status": "snippet"}]
    second = [{"url": "https://b.example", "read_status": "read"}, {"url": "https://a.example", "read_status": "read"}]
    merged = merge_sources(first, second)
    assert [s["url"] for s in merged] == ["https://a.example", "https://b.example"]
    assert all(s["read_status"] == "read" for s in merged)
    assert merge_sources(merged, first) == merged


def test_foreground_overtakes_queued_research_without_cancelling_active_research(monkeypatch):
    monkeypatch.setattr(llm_core, "is_local_endpoint", lambda url: True)
    monkeypatch.setattr(llm_core, "_local_model_gate_enabled", lambda: True)
    monkeypatch.setattr(llm_core, "_LOCAL_MODEL_CURRENT", {})
    monkeypatch.setattr(llm_core, "_LOCAL_MODEL_WAITING_FOREGROUND", 0)
    async def run():
        monkeypatch.setattr(llm_core, "_LOCAL_MODEL_LOCK", asyncio.Lock())
        entered = asyncio.Event()
        release = asyncio.Event()
        order = []
        async def active():
            async with llm_core._local_model_slot("http://localhost", "m", "research"):
                entered.set()
                await release.wait()
                order.append("active finished")
        async def queued(kind):
            async with llm_core._local_model_slot("http://localhost", "m", kind):
                order.append(kind)
        first = asyncio.create_task(active())
        await entered.wait()
        research = asyncio.create_task(queued("research"))
        await asyncio.sleep(0)
        foreground = asyncio.create_task(queued("foreground"))
        await asyncio.sleep(0)
        assert llm_core._LOCAL_MODEL_WAITING_FOREGROUND == 1
        release.set()
        await asyncio.wait_for(asyncio.gather(first, research, foreground), 2)
        assert order == ["active finished", "foreground", "research"]
        assert llm_core._LOCAL_MODEL_WAITING_FOREGROUND == 0
        assert not llm_core._LOCAL_MODEL_LOCK.locked()
    asyncio.run(run())

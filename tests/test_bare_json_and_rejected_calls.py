"""Bare JSON tool calls, rejected native-call repair and the fenced salvage
prompt swap."""
import asyncio
import json

import pytest

import src.agent_loop as al
from src.agent_tools import parse_tool_blocks, strip_tool_blocks
from src.tool_schemas import function_call_rejection_reason


# ---------------------------------------------------------------------------
# Bare {"name", "arguments"|"parameters"} JSON
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    '{"name": "manage_notes", "arguments": {"action": "add", "title": "milk"}}',
    'I will add it.\n```json\n{"name": "manage_notes", "parameters": {"action": "add", "title": "milk"}}\n```',
])
def test_bare_json_call_executes_for_registered_tools_in_text_mode(text):
    blocks = parse_tool_blocks(text)
    assert [b.tool_type for b in blocks] == ["manage_notes"]
    assert json.loads(blocks[0].content) == {"action": "add", "title": "milk"}
    assert "manage_notes" not in strip_tool_blocks(text)


@pytest.mark.parametrize("text", [
    '{"name": "Alice", "arguments": {"age": 3}}',                 # not a tool
    '{"name": "run", "arguments": {"command": "ls"}}',            # alias, not a registered name
    '{"name": "bash", "arguments": "ls -la"}',                    # args not an object
    '{"name": "bash", "arguments": {"command": "ls"}, "id": 1}',  # extra keys
    '{"steps": [{"name": "bash", "arguments": {"command": "ls"}}]}',  # nested data
    'Here is the JSON you asked for: {"name": "Widget", "price": 3}',
])
def test_ordinary_json_is_never_executed(text):
    assert parse_tool_blocks(text) == []


def test_native_routes_accept_bare_json_only_as_the_whole_response():
    call = '{"name": "web_search", "arguments": {"query": "python 3.14 release"}}'
    assert [b.tool_type for b in parse_tool_blocks(call, skip_fenced=True)] == ["web_search"]
    assert [b.tool_type for b in parse_tool_blocks("```json\n" + call + "\n```", skip_fenced=True)] == ["web_search"]
    example = "To search you would send " + call + " to the API."
    assert parse_tool_blocks(example, skip_fenced=True) == []
    assert strip_tool_blocks(example, skip_fenced=True) == example


# ---------------------------------------------------------------------------
# Rejected native calls are fed back once
# ---------------------------------------------------------------------------

def test_rejection_reasons_explain_the_problem():
    assert "no tool named" in function_call_rejection_reason("frobnicate", "{}")
    assert "not valid JSON" in function_call_rejection_reason("bash", "{not json")
    assert "'query'" in function_call_rejection_reason("web_search", "{}")
    assert function_call_rejection_reason("bash", '{"command": "ls"}') is None


def _collect(gen):
    async def _run():
        return [c async for c in gen]
    return asyncio.run(_run())


def _patch_loop(monkeypatch, rounds):
    monkeypatch.setattr(al, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(al, "get_mcp_manager", lambda: None, raising=False)

    async def _fake_exec(block, *a, **k):
        return (block.tool_type, {"output": "ok", "exit_code": 0})

    monkeypatch.setattr(al, "execute_tool_block", _fake_exec, raising=False)
    calls = []

    async def _fake_stream(_candidates, messages, **kwargs):
        calls.append({"messages": [dict(m) for m in messages], "tools": kwargs.get("tools")})
        step = rounds[min(len(calls) - 1, len(rounds) - 1)]
        if step.get("tool_calls"):
            yield f'data: {json.dumps({"type": "tool_calls", "calls": step["tool_calls"]})}\n\n'
        if step.get("text") is not None:
            yield f'data: {json.dumps({"delta": step["text"]})}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)
    return calls


def test_rejected_native_call_is_fed_back_once(monkeypatch):
    bad = {"id": "c1", "name": "frobnicate", "arguments": "{}"}
    calls = _patch_loop(monkeypatch, [{"tool_calls": [bad]}, {"tool_calls": [bad]}, {"text": "sorry"}])
    _collect(al.stream_agent_loop(
        "http://x/v1", "qwen3:14b",
        [{"role": "user", "content": "search the web for the latest news"}],
        max_rounds=4, relevant_tools={"web_search"}, _is_teacher_run=True,
    ))
    assert len(calls) >= 2
    tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[-1]["tool_call_id"] == "c1"
    assert "NOT EXECUTED" in tool_msgs[-1]["content"] and "no tool named" in tool_msgs[-1]["content"]
    assert "web_search" in tool_msgs[-1]["content"]
    if len(calls) > 2:
        # One-shot: the second rejection is not fed back again.
        assert len([m for m in calls[2]["messages"] if m.get("role") == "tool"]) == len(tool_msgs)


def test_empty_native_round_swaps_in_the_fenced_prompt(monkeypatch):
    calls = _patch_loop(monkeypatch, [{"text": ""}, {"text": "Done."}])
    _collect(al.stream_agent_loop(
        "http://x/v1", "qwen3:14b",
        [{"role": "user", "content": "search the web for the latest news"}],
        max_rounds=3, relevant_tools={"web_search"}, _is_teacher_run=True,
    ))
    first_system = "\n".join(m["content"] or "" for m in calls[0]["messages"] if m.get("role") == "system")
    retry_prompt = next(m["content"] for m in calls[1]["messages"] if m.get("_agent_injected") in {"prompt", "merged_prompt"})
    assert "native tool/function calling" in first_system
    assert "native tool/function calling" not in retry_prompt
    assert "```web_search" in retry_prompt
    assert not calls[1]["tools"]

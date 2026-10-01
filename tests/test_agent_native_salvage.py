"""Behavioral tests for the native-tool empty-response salvage (#1567).

When a native (schema-based) round returns no content at all — some local
models stop after a token when tool schemas are present — the loop retries
once in fenced mode. That retry must:

  1. send NO tool schemas (built-in or MCP), because the model is switching to
     the textual channel, and
  2. replace the native-only prompt (which forbids fenced tool syntax) with
     explicit textual fenced-tool instructions.
"""
import asyncio
import json

import src.agent_loop as al


def _collect(gen):
    async def _run():
        return [c async for c in gen]
    return asyncio.run(_run())


def _types(chunks):
    out = []
    for c in chunks:
        if c.startswith("data: ") and not c.startswith("data: [DONE]"):
            try:
                out.append(json.loads(c[6:]))
            except Exception:
                pass
    return out


def _patch_common(monkeypatch):
    monkeypatch.setattr(al, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(al, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(al, "estimate_tokens", lambda *a, **k: 10, raising=False)

    async def _fake_exec(block, *a, **k):
        return (block.tool_type, {"output": "ok", "exit_code": 0})

    monkeypatch.setattr(al, "execute_tool_block", _fake_exec, raising=False)


def _run_loop(monkeypatch, texts, model="qwen3:14b", max_rounds=3):
    calls = []

    async def _fake_stream(_candidates, messages, **kwargs):
        calls.append({
            "tools": kwargs.get("tools"),
            "system": [m.get("content") for m in messages if m.get("role") == "system"],
            "last": dict(messages[-1]),
        })
        index = len(calls) - 1
        text = texts[index] if index < len(texts) else texts[-1]
        yield f'data: {json.dumps({"delta": text})}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)
    gen = al.stream_agent_loop(
        "http://x/v1",
        model,
        [{"role": "user", "content": "search the web for the latest news"}],
        max_rounds=max_rounds,
        relevant_tools={"web_search"},
    )
    return _types(_collect(gen)), calls


def test_empty_native_round_retries_without_schemas_and_with_fenced_prompt(monkeypatch):
    _patch_common(monkeypatch)
    events, calls = _run_loop(monkeypatch, ["", "Done: nothing needed."])

    assert len(calls) >= 2, events
    assert calls[0]["tools"], "round 1 should carry native schemas"
    assert not calls[1]["tools"], "retry must send no schemas (built-in or MCP)"

    # The retry note is a trailing user-role harness message: a system-role
    # note would be merged into the leading system prompt and invalidate the
    # backend's cached prompt prefix.
    note = calls[1]["last"]
    assert note["role"] == "user"
    assert note["content"].startswith("[Odysseus] TOOL MODE CHANGE")
    assert "fenced code block" in note["content"]
    assert "```web_search" in note["content"]
    joined = "\n".join(c for c in calls[1]["system"] if c)
    assert "TOOL MODE CHANGE" not in joined


def test_salvage_fires_only_once(monkeypatch):
    # Two consecutive empty rounds: the second must NOT trigger another retry
    # (the guard is one-shot), and rounds keep running until the cap.
    _patch_common(monkeypatch)
    _events, calls = _run_loop(monkeypatch, ["", "", ""], max_rounds=4)
    assert len(calls) <= 3, "salvage must not loop on repeated empty rounds"


def test_non_empty_native_round_does_not_retry(monkeypatch):
    _patch_common(monkeypatch)
    _events, calls = _run_loop(monkeypatch, ["Here is the answer."])
    assert len(calls) == 1

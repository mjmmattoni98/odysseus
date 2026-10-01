"""Native Ollama /api/chat: real timings, actionable errors and stream parity.

The native NDJSON path must not lose anything the OpenAI-compatible path has
(repetition guard, template-artifact stripping, inline <think> routing) and
must report Ollama's own durations so the UI shows real generation speed and
model (re)load time.
"""
import asyncio
import json
import threading

import httpx
import pytest

from src import llm_core

_NATIVE_URL = "http://localhost:11434"


class _Response:
    def __init__(self, lines, status_code=200, body=b""):
        self._lines = lines
        self.status_code = status_code
        self._body = body

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._body


class _StreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    async def __aexit__(self, *args):
        return False


class _Client:
    def __init__(self, response):
        self._response = response
        self.payloads = []

    def stream(self, method, url, **kwargs):
        self.payloads.append(kwargs.get("json"))
        return _StreamContext(self._response)


def _stream(monkeypatch, response, *, url=_NATIVE_URL, model="qwen3:8b"):
    client = _Client(response)
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: client)
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda url: False)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "_mark_host_dead", lambda *a, **k: False)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "get_context_length", lambda u, m: 32768)
    monkeypatch.setattr(llm_core, "_route_supports_thinking", lambda u, m: False)

    async def run():
        return [
            chunk
            async for chunk in llm_core._stream_llm_inner(
                url, model, [{"role": "user", "content": "hi"}],
            )
        ]

    return asyncio.run(run())


def _lines(*chunks):
    return [json.dumps(chunk) for chunk in chunks]


def _data(chunks, event_type):
    return [
        json.loads(chunk[6:])
        for chunk in chunks
        if chunk.startswith("data: ") and f'"type": "{event_type}"' in chunk
    ]


def _deltas(chunks):
    out = []
    for chunk in chunks:
        if chunk.startswith("data: {") and '"delta"' in chunk:
            payload = json.loads(chunk[6:])
            out.append((payload["delta"], bool(payload.get("thinking"))))
    return out


def _error(chunks):
    errors = [chunk for chunk in chunks if chunk.startswith("event: error")]
    assert len(errors) == 1, chunks
    return json.loads(errors[0].split("data: ", 1)[1])


def _done(**extra):
    chunk = {"message": {"content": ""}, "done": True}
    chunk.update(extra)
    return chunk


# ── Timings ──


def test_native_usage_reports_real_speed_and_load_time(monkeypatch):
    chunks = _stream(monkeypatch, _Response(_lines(
        {"message": {"content": "ok"}, "done": False},
        _done(
            prompt_eval_count=900,
            eval_count=120,
            load_duration=18_200_000_000,
            prompt_eval_duration=1_500_000_000,
            eval_duration=3_000_000_000,
            total_duration=22_800_000_000,
        ),
    )))

    usage = _data(chunks, "usage")[0]["data"]
    assert usage["input_tokens"] == 900
    assert usage["output_tokens"] == 120
    assert usage["gen_tps"] == 40.0
    assert usage["prefill_tps"] == 600.0
    assert usage["load_ms"] == 18200.0
    assert usage["prefill_ms"] == 1500.0
    assert usage["gen_ms"] == 3000.0
    assert "finish_reason" not in usage


def test_native_usage_flags_replies_cut_by_the_token_limit(monkeypatch):
    chunks = _stream(monkeypatch, _Response(_lines(
        {"message": {"content": "partial"}, "done": False},
        _done(prompt_eval_count=5, eval_count=64, done_reason="length"),
    )))

    assert _data(chunks, "usage")[0]["data"]["finish_reason"] == "length"


def test_native_usage_ignores_malformed_durations(monkeypatch):
    chunks = _stream(monkeypatch, _Response(_lines(
        {"message": {"content": "ok"}, "done": False},
        _done(prompt_eval_count=5, eval_count=3, eval_duration="fast", load_duration=-1),
    )))

    usage = _data(chunks, "usage")[0]["data"]
    assert usage == {"input_tokens": 5, "output_tokens": 3}


# ── Stream parity with the OpenAI-compatible path ──


def test_native_inline_think_block_streams_to_thinking_channel(monkeypatch):
    chunks = _stream(monkeypatch, _Response(_lines(
        {"message": {"content": "<thi"}, "done": False},
        {"message": {"content": "nk>weighing options</th"}, "done": False},
        {"message": {"content": "ink>The answer"}, "done": False},
        {"message": {"content": " is 4."}, "done": False},
        _done(),
    )))

    deltas = _deltas(chunks)
    thinking = "".join(text for text, flag in deltas if flag)
    visible = "".join(text for text, flag in deltas if not flag)
    assert thinking == "weighing options"
    assert visible == "The answer is 4."


def test_native_think_tags_inside_the_reply_stay_visible(monkeypatch):
    chunks = _stream(monkeypatch, _Response(_lines(
        {"message": {"content": "Use <think> tags like this."}, "done": False},
        _done(),
    )))

    assert _deltas(chunks) == [("Use <think> tags like this.", False)]


def test_native_strips_leaked_chat_template_artifacts(monkeypatch):
    chunks = _stream(monkeypatch, _Response(_lines(
        {"message": {"content": "Hello<|im_end|>"}, "done": False},
        {"message": {"content": "<|im_end|>"}, "done": False},
        _done(),
    )))

    assert "".join(text for text, _ in _deltas(chunks)) == "Hello"


def test_native_repetition_loop_aborts_the_stream(monkeypatch):
    chunks = _stream(monkeypatch, _Response(_lines(
        *({"message": {"content": "Var "}, "done": False} for _ in range(80)),
        _done(prompt_eval_count=1, eval_count=80),
    )))

    error = _error(chunks)
    assert error["status"] == 502
    assert error["fallback_eligible"] is False
    assert "repeating" in error["text"]
    assert "data: [DONE]\n\n" not in chunks


# ── Actionable errors ──


def test_native_out_of_memory_is_a_fallback_eligible_status(monkeypatch):
    chunks = _stream(monkeypatch, _Response(_lines(
        {"error": "model requires more system memory (24.1 GiB) than is available (12.0 GiB)"},
    )), model="gemma4:26b")

    error = _error(chunks)
    assert error["status"] == 507
    assert error["text"] == error["error"]
    assert "ran out of memory" in error["text"]
    assert "gemma4:26b" in error["text"]


def test_native_runner_crash_is_a_fallback_eligible_status(monkeypatch):
    chunks = _stream(monkeypatch, _Response(_lines(
        {"error": "llama runner process has terminated: signal: killed"},
    )))

    error = _error(chunks)
    assert error["status"] == 502
    assert "runner stopped unexpectedly" in error["text"]


def test_foreground_policy_treats_ollama_resource_failures_as_eligible():
    from src.foreground_model_routing import FOREGROUND_AVAILABILITY_STATUSES

    assert llm_core._ollama_error_status("CUDA error: out of memory") in FOREGROUND_AVAILABILITY_STATUSES
    assert llm_core._ollama_error_status("llama runner process has terminated: exit status 2") in FOREGROUND_AVAILABILITY_STATUSES
    assert llm_core._ollama_error_status("invalid JSON body") is None


def test_native_missing_model_suggests_ollama_pull(monkeypatch):
    chunks = _stream(
        monkeypatch,
        _Response([], status_code=404, body=b'{"error":"model \\"qwen9:1b\\" not found, try pulling it first"}'),
        model="qwen9:1b",
    )

    error = _error(chunks)
    assert error["status"] == 404
    assert "`ollama pull qwen9:1b`" in error["text"]


def test_ollama_v1_errors_name_ollama_not_a_local_endpoint():
    body = '{"error":{"message":"model \\"llama9\\" not found, try pulling it first","type":"api_error"}}'
    message = llm_core._format_upstream_error(404, body, "http://host.docker.internal:11434/v1/chat/completions")

    assert message.startswith("Ollama does not have model 'llama9'")
    assert "`ollama pull llama9`" in message
    assert llm_core._provider_label("http://host.docker.internal:11434/v1") == "Ollama"


def test_local_ollama_server_error_is_not_called_an_outage():
    message = llm_core._format_upstream_error(500, '{"error":"boom"}', "http://localhost:11434/api/chat")

    assert "outage" not in message
    assert "Ollama server log" in message


def test_non_ollama_local_servers_keep_neutral_errors():
    message = llm_core._format_upstream_error(404, '{"error":"model not found"}', "http://localhost:8080/v1/chat/completions")

    assert "ollama pull" not in message
    assert message.startswith("local endpoint returned 404")
    assert llm_core._ollama_unreachable_hint("http://localhost:8080/v1") == ""


def test_ollama_cloud_missing_model_does_not_suggest_pull():
    message = llm_core._format_upstream_error(404, '{"error":"model \\"nope\\" not found"}', "https://ollama.com/api/chat")

    assert "ollama pull" not in message
    assert "Ollama Cloud does not offer model 'nope'" in message


@pytest.mark.parametrize(
    ("url", "in_container", "expected"),
    [
        (_NATIVE_URL, True, "host.docker.internal:11434"),
        ("http://host.docker.internal:11434", True, "OLLAMA_HOST=0.0.0.0"),
        (_NATIVE_URL, False, "ollama serve"),
    ],
)
def test_unreachable_ollama_explains_how_to_fix_it(monkeypatch, url, in_container, expected):
    from src import host_docker_access

    monkeypatch.setattr(host_docker_access, "running_in_container", lambda: in_container)
    chunks = _stream(monkeypatch, httpx.ConnectError("refused"), url=url)

    error = _error(chunks)
    assert error["status"] == 503
    assert error["text"].startswith("Cannot reach ")
    assert expected in error["text"]


# ── Tool results carry tool_name ──


def test_native_tool_results_carry_the_executed_tool_name():
    payload = llm_core._build_ollama_payload("qwen3:8b", [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "call_b", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_b", "content": "file"},
        {"role": "tool", "tool_call_id": "call_a", "content": "sunny"},
    ], temperature=0.0, max_tokens=0)

    tools = [m for m in payload["messages"] if m["role"] == "tool"]
    assert [m["tool_name"] for m in tools] == ["read_file", "web_search"]
    assert tools[0]["tool_call_id"] == "call_b"


def test_harmony_tool_results_use_the_alias_the_model_saw():
    payload = llm_core._build_ollama_payload("gpt-oss:20b", [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "python", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_0", "content": "42"},
    ], temperature=0.0, max_tokens=0, tools=[
        {"type": "function", "function": {"name": "python", "parameters": {}}},
    ])

    assert payload["tools"][0]["function"]["name"] == "run_python_code"
    assert payload["messages"][0]["tool_calls"][0]["function"]["name"] == "run_python_code"
    assert payload["messages"][1]["tool_name"] == "run_python_code"


# ── Event loop ──


def _record_thread(calls):
    def fake_context_length(url, model):
        calls.append(threading.current_thread() is threading.main_thread())
        return 32768
    return fake_context_length


def test_stream_context_lookup_runs_off_the_event_loop(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda url: True)
    monkeypatch.setattr(llm_core, "get_context_length", _record_thread(calls))
    monkeypatch.setattr(llm_core, "_route_supports_thinking", lambda u, m: False)

    async def run():
        return [chunk async for chunk in llm_core._stream_llm_inner(_NATIVE_URL, "m", [{"role": "user", "content": "x"}])]

    asyncio.run(run())
    assert calls == [False]


def test_async_call_context_lookup_runs_off_the_event_loop(monkeypatch):
    calls = []
    llm_core._response_cache.clear()
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda url: True)
    monkeypatch.setattr(llm_core, "get_context_length", _record_thread(calls))
    monkeypatch.setattr(llm_core, "_route_supports_thinking", lambda u, m: False)

    async def run():
        with pytest.raises(Exception):
            await llm_core.llm_call_async(_NATIVE_URL, "m", [{"role": "user", "content": "x"}])

    asyncio.run(run())
    assert calls == [False]

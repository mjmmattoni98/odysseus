"""Memory suggestion calls pass response_schema and still parse replies from
providers that ignore it (thinking tags / prose around the JSON array)."""
import asyncio
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import UploadFile

import routes.memory_routes as mr


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def _router(monkeypatch, reply, calls):
    monkeypatch.setattr(mr, "get_current_user", lambda request: "alice", raising=False)
    monkeypatch.setattr(mr, "require_user", lambda request: "alice", raising=False)
    monkeypatch.setattr("src.auth_helpers.require_privilege", lambda request, privilege: "alice")
    monkeypatch.setattr(mr, "resolve_task_endpoint",
                        lambda *a, **k: ("http://localhost:11434", "m", {}))

    async def fake_llm(url, model, messages, **kwargs):
        calls.append(kwargs)
        return reply

    monkeypatch.setattr(mr, "llm_call_async", fake_llm)
    sm = MagicMock()
    sm.sessions = {}
    sm.get_session = lambda sid: SimpleNamespace(
        owner="alice", name="s", endpoint_url="http://x", model="m", headers={},
        history=[], get_context_messages=lambda: [{"role": "user", "content": "I live in Lisbon"}],
    )
    mem = MagicMock()
    mem.load = lambda owner=None: []
    return mr.setup_memory_routes(mem, sm)


def test_extract_passes_schema_and_parses_prose_wrapped_array(monkeypatch):
    calls = []
    reply = ('<think>look for facts</think>Here are the facts:\n'
             '[{"text": "Alice lives in Lisbon"}, {"text": "Alice works at Acme"}]\nDone.')
    router = _router(monkeypatch, reply, calls)
    extract = _route(router, "/api/memory/extract", "POST")
    out = asyncio.run(extract(request=None, session="s1"))
    assert out == {"suggestions": ["Alice lives in Lisbon", "Alice works at Acme"]}
    assert calls[0]["response_schema"] == mr._MEMORY_SUGGESTIONS_SCHEMA
    assert calls[0]["think"] is False


def test_extract_falls_back_to_lines_without_json(monkeypatch):
    calls = []
    router = _router(monkeypatch, "Alice lives in Lisbon\nAlice works at Acme", calls)
    extract = _route(router, "/api/memory/extract", "POST")
    out = asyncio.run(extract(request=None, session="s1"))
    assert out["suggestions"] == ["Alice lives in Lisbon", "Alice works at Acme"]


def test_import_passes_schema_and_parses_prose_wrapped_array(monkeypatch):
    calls = []
    reply = ('Sure! Extracted facts:\n'
             '[{"text": "Project Phoenix uses Python", "category": "project"}]\nLet me know.')
    router = _router(monkeypatch, reply, calls)
    import_route = _route(router, "/api/memory/import", "POST")
    upload = UploadFile(filename="notes.txt", file=io.BytesIO(b"Project Phoenix is written in Python."))
    out = asyncio.run(import_route(request=None, session=None, file=upload))
    assert out["suggestions"] == [{"text": "Project Phoenix uses Python", "category": "project"}]
    schema = calls[0]["response_schema"]
    assert schema == mr._MEMORY_IMPORT_SCHEMA
    assert schema["items"]["properties"]["category"]["enum"] == [
        "identity", "preference", "fact", "contact", "project", "goal",
    ]


def test_parse_json_list_prefers_last_array():
    text = 'Example: ["x"]\nAnswer: [{"text": "real"}]'
    assert mr._parse_json_list(text) == [{"text": "real"}]
    assert mr._parse_json_list("no json here") is None

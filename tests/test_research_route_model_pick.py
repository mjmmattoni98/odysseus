"""Research routes pick an endpoint's model with the shared chat-model picker.

routes/research kept its own name-only ``_first_chat_model`` copy, so a
research launch (panel endpoint without a model) or a research spin-off with
nothing else configured could auto-pick a model the Ollama server reports as
unable to chat, such as the QA audit's ``clef-flash:latest``
(``["decision", "vision"]``). Both call sites now use
``src.endpoint_resolver._first_chat_model`` with the endpoint URL, which reads
cached capability reports and never probes. No network: ``/api/show`` is
stubbed only to fill the cache the way a model-list refresh does.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

import routes.research.research_routes as rr
import src.ollama_capabilities as oc

OLLAMA = "http://localhost:11434"
SHOW = {
    "clef-flash:latest": ["decision", "vision"],
    "gemma4:12b": ["completion", "vision", "tools", "thinking"],
}
MODELS = ["clef-flash:latest", "gemma4:12b"]


class _Resp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.is_success = True

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def show_probes(monkeypatch):
    """Fill the capability cache as a model-list refresh would, then record
    any further /api/show probe (selection must not make one)."""
    oc.reset_cache()
    probes = []

    def fake_post(url, json=None, timeout=None, **kwargs):
        if not url.endswith("/api/show"):
            pytest.fail(f"unexpected request to {url}")
        probes.append(json["model"])
        return _Resp({"capabilities": SHOW[json["model"]]})

    monkeypatch.setattr(oc.httpx, "post", fake_post)
    monkeypatch.setattr(oc, "_registered_kind", lambda url: None)
    for model in MODELS:
        oc.capability_tokens(OLLAMA, model)
    probes.clear()
    yield probes
    oc.reset_cache()


def _endpoint():
    return SimpleNamespace(id="ep-1", base_url=OLLAMA, api_key="", provider_auth_id=None,
                           cached_models=json.dumps(MODELS), is_enabled=True, owner="alice")


def test_panel_endpoint_without_a_model_skips_a_reported_non_chat_model(show_probes):
    _url, model, _headers = rr._resolve_endpoint_runtime(_endpoint(), owner="alice")

    assert model == "gemma4:12b"
    assert show_probes == []  # selection reads the cache only


def test_spinoff_fallback_endpoint_skips_a_reported_non_chat_model(monkeypatch, tmp_path, show_probes):
    import src.database as database

    path = tmp_path / "rp-spinmodel01.json"
    path.write_text(json.dumps({"owner": "alice", "result": "saved report", "query": "q"}), encoding="utf-8")
    monkeypatch.setattr(rr, "get_current_user", lambda request: request.state.current_user)
    monkeypatch.setattr(rr, "_find_owned_research_path", lambda session_id, user: path)
    monkeypatch.setattr(rr, "resolve_endpoint", lambda *a, **k: ("", "", {}))
    monkeypatch.setattr(rr, "_owned_enabled_endpoint", lambda db, owner, endpoint_id=None: _endpoint())
    monkeypatch.setattr(database, "SessionLocal", lambda: SimpleNamespace(close=lambda: None), raising=False)
    monkeypatch.setattr("src.event_bus.fire_event", lambda *a, **k: None)

    created = {}

    class Sessions:
        def get_session(self, session_id):
            raise KeyError(session_id)

        def create_session(self, **kwargs):
            created.update(kwargs)
            return SimpleNamespace(headers={}, add_message=lambda message: None)

        def save_sessions(self):
            pass

    handler = SimpleNamespace(_active_tasks={}, get_result=lambda sid: None, get_sources=lambda sid: [])
    router = rr.setup_research_routes(handler, session_manager=Sessions())
    spinoff = next(r.endpoint for r in router.routes
                   if r.path == "/api/research/spinoff/{session_id}" and "POST" in r.methods)

    asyncio.run(spinoff(session_id="rp-spinmodel01",
                        request=SimpleNamespace(state=SimpleNamespace(current_user="alice"))))

    assert created["model"] == "gemma4:12b"
    assert show_probes == []

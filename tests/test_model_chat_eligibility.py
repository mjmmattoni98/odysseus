"""Chat eligibility from provider capability reports (QA audit 2026-10-12).

A local Ollama model that reports capabilities without ``completion`` (the
audit's ``clef-flash:latest``: ``["decision", "vision"]``) answers chat with
HTTP 400 "does not support chat". It must not be listed as a chat model,
picked as a default, or reported ready by the probe. A model with no report
stays listed as unknown. No network: ``/api/show`` and generation calls are
stubbed.
"""
import json
from types import SimpleNamespace

import pytest

import src.ollama_capabilities as oc
from src import endpoint_resolver

OLLAMA = "http://localhost:11434"

# Real /api/show capability lists from the audit machine.
SHOW = {
    "clef-flash:latest": ["decision", "vision"],
    "all-minilm:l6-v2": ["embedding"],
    "gemma4:12b": ["completion", "vision", "audio", "tools", "thinking"],
    "qwen3.8:27b": ["completion", "vision", "tools", "thinking"],
}


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.is_success = 200 <= status < 300

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _ollama_show(monkeypatch):
    """Serve SHOW from /api/show; models not in SHOW get no capability list."""
    oc.reset_cache()
    calls = []

    def fake_post(url, json=None, timeout=None, **kwargs):
        if not url.endswith("/api/show"):
            pytest.fail(f"unexpected generation request to {url}")
        calls.append(json["model"])
        caps = SHOW.get(json["model"])
        return _Resp({"capabilities": caps} if caps is not None else {"details": {}})

    monkeypatch.setattr(oc.httpx, "post", fake_post)
    monkeypatch.setattr(oc, "_registered_kind", lambda url: None)
    yield calls
    oc.reset_cache()


class TestChatCapability:
    def test_report_without_completion_is_not_chat(self):
        assert oc.chat_capability(frozenset({"decision", "vision"})) is False
        assert oc.chat_capability(frozenset({"embedding"})) is False
        assert oc.chat_capability(frozenset({"completion", "vision"})) is True

    def test_missing_or_empty_report_is_unknown(self):
        assert oc.chat_capability(None) is None
        assert oc.chat_capability(frozenset()) is None
        assert oc.supports_chat(OLLAMA, "mystery:1b") is None
        # Non-Ollama servers have no capability source at all.
        assert oc.supports_chat("http://localhost:1234/v1", "gemma4:12b") is None

    def test_reason_names_the_reported_capabilities(self):
        assert oc.non_chat_reason(frozenset({"decision", "vision"})) == (
            "Model does not support chat (capabilities: decision, vision)"
        )
        assert oc.non_chat_reason(frozenset({"embedding"})) == "Embedding model (no chat)"

    def test_cached_non_chat_models_never_probes(self, _ollama_show):
        assert oc.cached_non_chat_models(OLLAMA) == {}
        assert _ollama_show == []
        oc.capability_tokens(OLLAMA, "clef-flash:latest")
        oc.capability_tokens(OLLAMA, "gemma4:12b")
        oc.capability_tokens(OLLAMA, "mystery:1b")
        assert oc.cached_non_chat_models(OLLAMA + "/v1") == {
            "clef-flash:latest": "Model does not support chat (capabilities: decision, vision)",
        }


class TestModelListFilter:
    def test_known_non_chat_models_are_dropped_and_unknown_kept(self):
        from routes import model_routes as mr
        models = ["clef-flash:latest", "all-minilm:l6-v2", "mystery:1b", "gemma4:12b"]
        assert mr._drop_ollama_non_chat_models(OLLAMA, models) == ["mystery:1b", "gemma4:12b"]


class TestDefaultSelection:
    """``_first_chat_model(url=...)`` ranks from capability reports already cached
    (a model-list refresh runs ``_drop_ollama_non_chat_models`` first)."""

    @staticmethod
    def _refresh(models):
        from routes import model_routes as mr
        mr._drop_ollama_non_chat_models(OLLAMA, models)

    def test_verified_chat_model_beats_unknown_and_non_chat(self):
        models = ["clef-flash:latest", "mystery:1b", "gemma4:12b"]
        self._refresh(models)
        assert endpoint_resolver._first_chat_model(models, url=OLLAMA) == "gemma4:12b"

    def test_unknown_model_is_used_when_nothing_is_verified(self):
        models = ["clef-flash:latest", "mystery:1b"]
        self._refresh(models)
        assert endpoint_resolver._first_chat_model(models, url=OLLAMA) == "mystery:1b"

    def test_known_non_chat_model_is_never_picked(self):
        models = ["clef-flash:latest", "all-minilm:l6-v2"]
        self._refresh(models)
        assert endpoint_resolver._first_chat_model(models, url=OLLAMA) is None

    def test_reports_cached_under_the_chat_url_count_too(self):
        # Resolvers pass the endpoint base; the report is keyed by server root.
        self._refresh(["clef-flash:latest"])
        assert endpoint_resolver._first_chat_model(["clef-flash:latest", "mystery:1b"], url=OLLAMA + "/api/chat") == "mystery:1b"

    def test_selection_reads_cached_reports_and_never_probes(self, _ollama_show):
        # Nothing cached yet: every model is unknown, so the name heuristics
        # decide. Selection runs on request paths and must not call /api/show.
        assert endpoint_resolver._first_chat_model(["clef-flash:latest", "gemma4:12b"], url=OLLAMA) == "clef-flash:latest"
        assert _ollama_show == []

    def test_without_a_url_name_heuristics_are_unchanged(self, _ollama_show):
        assert endpoint_resolver._first_chat_model(["clef-flash:latest", "gemma4:12b"]) == "clef-flash:latest"
        assert _ollama_show == []

    def test_single_model_mode_treats_reported_non_chat_as_ineligible(self):
        assert endpoint_resolver._can_chat(OLLAMA, "clef-flash:latest") is False
        assert endpoint_resolver._can_chat(OLLAMA, "gemma4:12b") is True


class _Column:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return (self.name, value)

    __hash__ = object.__hash__


class _EndpointTable:
    """``ModelEndpoint`` stand-in: only the columns the resolver filters on."""
    id = _Column("id")
    is_enabled = _Column("is_enabled")


class _Query:
    def __init__(self, rows):
        self.rows = list(rows)

    def filter(self, *conditions):
        for field, value in conditions:
            self.rows = [row for row in self.rows if getattr(row, field) == value]
        return self

    def first(self):
        return self.rows[0] if self.rows else None


class _Db:
    def __init__(self, rows):
        self.rows = rows

    def query(self, _table):
        return _Query(self.rows)

    def close(self):
        pass


class TestAutomaticResolution:
    """Default/Utility routes and fallback entries with no model configured.

    Both resolver paths pick a model from the endpoint's cached list. The
    audit endpoint listed ``clef-flash:latest`` first; once a model-list
    refresh has cached its ``["decision", "vision"]`` report, neither path
    may pick it, and neither may probe ``/api/show`` to decide.
    """

    @pytest.fixture
    def ollama_endpoint(self, monkeypatch):
        import src.settings as settings_mod

        endpoint = SimpleNamespace(
            id="local", name="Local Ollama", base_url=OLLAMA, api_key=None,
            cached_models=json.dumps(["clef-flash:latest", "gemma4:12b"]),
            hidden_models=None, pinned_models=None, is_enabled=True,
        )
        settings = {"default_endpoint_id": "local", "default_model": ""}
        monkeypatch.setattr(settings_mod, "load_settings", lambda: settings)
        monkeypatch.setattr(
            settings_mod, "get_user_setting",
            lambda key, owner="", default=None: settings.get(key, default),
        )
        monkeypatch.setattr(endpoint_resolver, "ModelEndpoint", _EndpointTable)
        monkeypatch.setattr(endpoint_resolver, "SessionLocal", lambda: _Db([endpoint]))
        monkeypatch.setattr(endpoint_resolver, "resolve_url", lambda url: url)
        return endpoint

    @staticmethod
    def _refreshed(probes, *models):
        """Cache capability reports the way a model-list refresh does."""
        for model in models:
            oc.capability_tokens(OLLAMA, model)
        probes.clear()

    def test_default_route_skips_a_cached_non_chat_model_without_probing(self, ollama_endpoint, _ollama_show):
        self._refreshed(_ollama_show, "clef-flash:latest", "gemma4:12b")

        url, model, _headers = endpoint_resolver.resolve_endpoint("default")

        assert (url, model) == (OLLAMA + "/api/chat", "gemma4:12b")
        assert _ollama_show == []

    def test_fallback_entry_without_model_skips_a_cached_non_chat_model(self, ollama_endpoint, _ollama_show):
        self._refreshed(_ollama_show, "clef-flash:latest", "gemma4:12b")

        route = endpoint_resolver.resolve_endpoint_by_id("local")

        assert route[:2] == (OLLAMA + "/api/chat", "gemma4:12b")
        assert _ollama_show == []

    def test_unknown_model_beats_a_known_non_chat_model(self, ollama_endpoint, _ollama_show):
        # Only clef-flash has a cached report; gemma4 is unknown, not rejected.
        self._refreshed(_ollama_show, "clef-flash:latest")

        assert endpoint_resolver.resolve_endpoint("default")[1] == "gemma4:12b"
        assert endpoint_resolver.resolve_endpoint_by_id("local")[1] == "gemma4:12b"
        assert _ollama_show == []

    def test_no_model_is_resolved_when_every_listed_model_is_known_non_chat(self, ollama_endpoint, _ollama_show):
        ollama_endpoint.cached_models = json.dumps(["clef-flash:latest", "all-minilm:l6-v2"])
        self._refreshed(_ollama_show, "clef-flash:latest", "all-minilm:l6-v2")

        assert endpoint_resolver.resolve_endpoint("default")[1] is None
        assert endpoint_resolver.resolve_endpoint_by_id("local") is None
        assert _ollama_show == []


class TestProbe:
    def test_non_chat_model_fails_without_generating(self):
        from routes import model_routes as mr
        result = mr._probe_single_model(OLLAMA, None, "clef-flash:latest", timeout=8)
        assert result["status"] == "fail"
        assert result["incompatible"] is True
        assert result["error"] == "Model does not support chat (capabilities: decision, vision)"
        assert result["capabilities"] == ["decision", "vision"]

    def test_verified_chat_model_is_ok(self):
        from routes import model_routes as mr
        result = mr._probe_single_model(OLLAMA, None, "gemma4:12b", timeout=8)
        assert result["status"] == "ok"
        assert result["method"] == "capabilities"

    def test_unknown_model_is_verified_by_generation_and_400_is_incompatible(self, monkeypatch):
        from routes import model_routes as mr
        sent = []

        def fake_post(url, json=None, timeout=None, **kwargs):
            if url.endswith("/api/show"):
                return _Resp({"details": {}})  # older Ollama: no capability list
            sent.append(url)
            return _Resp({"error": '"mystery:1b" does not support chat'}, status=400)

        monkeypatch.setattr(oc.httpx, "post", fake_post)
        result = mr._probe_single_model(OLLAMA, None, "mystery:1b", timeout=8)
        assert sent == [OLLAMA + "/api/chat"]
        assert result["status"] == "fail"
        assert result["incompatible"] is True
        assert "does not support chat" in result["error"]

    @pytest.mark.parametrize("status, text, expected", [
        (400, '"clef-flash:latest" does not support chat', True),
        (404, "This is not a chat model and thus not supported in the v1/chat/completions endpoint.", True),
        (400, "messages: field required", False),
        (500, "does not support chat", False),
        (429, "rate limited", False),
    ])
    def test_incompatible_error_classification(self, status, text, expected):
        from routes import model_routes as mr
        assert mr._is_incompatible_model_error(status, text) is expected


def test_capability_reader_does_not_call_a_decision_model_chat():
    from src import model_capabilities as mc
    from src.model_capability_readers import ollama as reader
    record = reader.record_from_show_payload("clef-flash:latest", {"capabilities": ["decision", "vision"]})
    assert record.capability.family != mc.FAMILY_CHAT
    assert "chat" not in mc.display_surfaces_for(record.capability)

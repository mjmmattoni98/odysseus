"""Local model configuration & residency (Ollama detection, native
registration, vision capabilities, default context cap, single-model mode,
task model resolution, probe side effects). No network: every HTTP call and
fingerprint is stubbed."""
import asyncio
import types

import pytest

import src.ollama_capabilities as oc
from src import endpoint_resolver, llm_core


OLLAMA_ROOTS = {"http://localhost:11435", "http://127.0.0.1:11435"}


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.is_success = 200 <= status < 300

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """Fingerprint: only OLLAMA_ROOTS answer /api/version; no DB endpoint kinds."""
    oc.reset_cache()
    probes = []

    def fake_probe(root, timeout):
        probes.append(root)
        return "0.34.4" if root in OLLAMA_ROOTS else None

    monkeypatch.setattr(oc, "_probe_version", fake_probe)
    monkeypatch.setattr(oc, "_registered_kind", lambda url: None)
    monkeypatch.setattr(endpoint_resolver, "resolve_url", lambda url: url)
    yield probes
    oc.reset_cache()


# ── Item 2: one "is this Ollama?" answer for every surface ──────────────


@pytest.mark.parametrize("url, native, compat", [
    ("http://localhost:11434", True, False),
    ("http://host.docker.internal:11434/api/chat", True, False),
    ("http://localhost:11434/v1", False, True),
    ("http://localhost:11434/v1/chat/completions", False, True),
    # LM Studio, llama.cpp and vLLM are never Ollama, bare or /v1.
    ("http://localhost:1234", False, False),
    ("http://localhost:1234/v1", False, False),
    ("http://localhost:8080", False, False),
    ("http://127.0.0.1:8080/v1/chat/completions", False, False),
    ("http://localhost:8000/v1", False, False),
    # A fingerprinted Ollama on a custom port (OLLAMA_HOST / Cookbook).
    ("http://localhost:11435", True, False),
    ("http://127.0.0.1:11435/v1", False, True),
    ("https://ollama.com", True, False),
    ("https://ollama.com/api/chat", True, False),
    ("https://api.openai.com/v1", False, False),
])
def test_url_classification_matrix(url, native, compat):
    assert llm_core._is_ollama_native_url(url) is native
    assert llm_core._is_ollama_openai_compat_url(url) is compat


def test_endpoint_kind_ollama_marks_custom_port_without_probe(monkeypatch, _hermetic):
    monkeypatch.setattr(oc, "_registered_kind", lambda url: "ollama" if ":9999" in url else None)
    assert llm_core._is_ollama_native_url("http://gpu-box.example:9999")
    assert llm_core._is_ollama_openai_compat_url("http://gpu-box.example:9999/v1")
    assert _hermetic == []


def test_api_and_proxy_kinds_are_never_probed(monkeypatch, _hermetic):
    monkeypatch.setattr(oc, "_registered_kind", lambda url: "proxy")
    assert oc.is_ollama_url("http://100.117.136.97:34521/v1") is False
    assert _hermetic == []


def test_public_hosts_are_never_fingerprinted(_hermetic):
    assert oc.is_ollama_url("https://llm.example.com:8443/v1") is False
    assert _hermetic == []


def test_fingerprint_is_negative_cached(_hermetic):
    assert oc.is_ollama_url("http://localhost:1234") is False
    assert oc.is_ollama_url("http://localhost:1234/v1/chat/completions") is False
    assert _hermetic == ["http://localhost:1234"]


def test_event_loop_cache_miss_never_blocks(monkeypatch, _hermetic):
    warmed = []
    monkeypatch.setattr(oc, "_warm_version_in_background", warmed.append)

    async def classify():
        return oc.is_ollama_url("http://localhost:11435")

    assert asyncio.run(classify()) is False
    assert warmed == ["http://localhost:11435"]
    assert _hermetic == []  # no synchronous probe on the loop thread
    # Once warmed, the loop sees the answer.
    oc._store_version_answer("http://localhost:11435", True)
    assert asyncio.run(classify()) is True


def test_capability_probe_works_on_detected_non_default_port(monkeypatch):
    posts = []

    def fake_post(url, json=None, timeout=None):
        posts.append(url)
        return _Resp({"capabilities": ["completion", "vision", "tools", "thinking"]})

    monkeypatch.setattr(oc.httpx, "post", fake_post)
    assert oc.supports_thinking("http://localhost:11435/api/chat", "qwen3.8:27b") is True
    assert oc.supports_vision("http://localhost:11435", "qwen3.8:27b") is True
    assert posts == ["http://localhost:11435/api/show"]
    # A non-Ollama local server is never asked for /api/show.
    assert oc.supports_vision("http://localhost:1234/v1", "qwen3.8:27b") is None
    assert posts == ["http://localhost:11435/api/show"]


def test_embedding_capability(monkeypatch):
    monkeypatch.setattr(oc.httpx, "post", lambda url, json=None, timeout=None: _Resp(
        {"capabilities": ["embedding"]} if json["model"].startswith("all-minilm") else {"capabilities": ["completion"]}
    ))
    assert oc.supports_embedding("http://localhost:11434", "all-minilm:l6-v2") is True
    assert oc.is_embedding_only(oc.capability_tokens("http://localhost:11434", "all-minilm:l6-v2"))
    assert not oc.is_embedding_only(oc.capability_tokens("http://localhost:11434", "qwen3.8:27b"))


def test_is_local_endpoint_honors_ollama_kind(monkeypatch):
    from src import model_context
    monkeypatch.setattr(model_context, "_configured_endpoint_kind", lambda url: "ollama")
    assert model_context.is_local_endpoint("http://gpu-box.example:11435/api/chat") is True
    assert model_context.is_local_endpoint("https://ollama.com/api/chat") is False


# ── Item 1: native registration ─────────────────────────────────────────


def test_url_builders_for_native_and_local_openai_servers():
    assert endpoint_resolver.build_chat_url("http://localhost:11434") == "http://localhost:11434/api/chat"
    assert endpoint_resolver.build_models_url("http://localhost:11434") == "http://localhost:11434/api/tags"
    assert endpoint_resolver.build_chat_url("http://localhost:11435") == "http://localhost:11435/api/chat"
    assert endpoint_resolver.build_models_url("http://localhost:11435") == "http://localhost:11435/api/tags"
    # /v1 stays fully supported.
    assert endpoint_resolver.build_chat_url("http://localhost:11434/v1") == "http://localhost:11434/v1/chat/completions"
    # A bare local OpenAI-compatible server gets /v1 for chat, like models.
    assert endpoint_resolver.build_chat_url("http://localhost:1234") == "http://localhost:1234/v1/chat/completions"
    assert endpoint_resolver.build_models_url("http://localhost:1234") == "http://localhost:1234/v1/models"


class TestDiscoveryFingerprint:
    def _discovery(self, monkeypatch, responses):
        from src import model_discovery

        def fake_get(url, timeout=None):
            for suffix, payload in responses.items():
                if url.endswith(suffix):
                    return payload if isinstance(payload, _Resp) else _Resp(payload)
            return _Resp({}, status=404)

        monkeypatch.setattr(model_discovery.httpx, "get", fake_get)
        return model_discovery.ModelDiscovery(default_host="localhost")

    def test_ollama_is_registered_natively(self, monkeypatch):
        d = self._discovery(monkeypatch, {
            "/v1/models": {"data": [{"id": "qwen3.8:27b"}, {"id": "gemma4:12b"}]},
            "/api/version": {"version": "0.34.4"},
        })
        item = d._check_port("host.docker.internal", 11434)
        assert item["provider"] == "ollama"
        assert item["url"] == item["base_url"] == "http://host.docker.internal:11434"
        assert item["models"] == ["qwen3.8:27b", "gemma4:12b"]

    def test_older_ollama_without_v1_models_uses_native_tags(self, monkeypatch):
        d = self._discovery(monkeypatch, {
            "/api/version": {"version": "0.1.20"},
            "/api/tags": {"models": [{"name": "llama3:8b"}]},
        })
        item = d._check_port("localhost", 11435)
        assert (item["url"], item["models"]) == ("http://localhost:11435", ["llama3:8b"])

    def test_openai_compatible_server_keeps_v1_url(self, monkeypatch):
        d = self._discovery(monkeypatch, {"/v1/models": {"data": [{"id": "Qwen/Qwen3-14B"}]}})
        item = d._check_port("localhost", 8000)
        assert item["url"] == "http://localhost:8000/v1/chat/completions"
        assert item["provider"] is None

    def test_warmup_pings_ollama_version(self):
        from src.model_discovery import ModelDiscovery
        d = ModelDiscovery(default_host="localhost")
        d.discover_models = lambda: {"items": [
            {"url": "http://h:11434", "base_url": "http://h:11434", "provider": "ollama"},
            {"url": "http://h:8000/v1/chat/completions", "provider": None},
        ]}
        assert d.warmup_ping_urls() == ["http://h:11434/api/version", "http://h:8000/v1/models"]


class TestRegistrationHelpers:
    def test_custom_port_ollama_is_recorded_as_ollama_kind(self):
        from routes import model_routes as mr
        assert mr._detect_ollama_kind("http://localhost:11435", "local") == "ollama"
        assert mr._detect_ollama_kind("http://localhost:11435", "auto") == "ollama"
        assert mr._detect_ollama_kind("http://localhost:11435", "api") == "api"
        assert mr._detect_ollama_kind("http://localhost:1234/v1", "local") == "local"
        assert mr._detect_ollama_kind("http://localhost:11434", "local") == "local"

    def test_ollama_kind_is_valid_and_local(self):
        from routes import model_routes as mr
        assert mr._normalize_endpoint_kind("ollama") == "ollama"
        assert mr._classify_endpoint("http://gpu-box.example:11435", "ollama") == "local"

    def test_switch_to_native_target_for_ollama_v1_only(self):
        from routes import model_routes as mr
        assert mr._native_ollama_switch_target("http://host.docker.internal:11434/v1") == "http://host.docker.internal:11434"
        assert mr._native_ollama_switch_target("http://gpu:11435/v1", "ollama") == "http://gpu:11435"
        assert mr._native_ollama_switch_target("http://host.docker.internal:11434") is None
        assert mr._native_ollama_switch_target("http://localhost:1234/v1") is None

    def test_hints_recommend_native_url(self):
        from routes import model_routes as mr
        text = mr._model_endpoint_error_message("http://localhost:11434", {"error": "refused"})
        assert "http://localhost:11434 (native" in text
        assert "host.docker.internal:11434 when" in text

    def test_embedding_models_are_dropped_from_ollama_lists(self, monkeypatch):
        from routes import model_routes as mr
        caps = {"all-minilm:l6-v2": frozenset({"embedding"}), "qwen3.8:27b": frozenset({"completion", "vision"})}
        monkeypatch.setattr(oc, "capability_tokens", lambda url, model, **kw: caps.get(model))
        models = ["qwen3.8:27b", "all-minilm:l6-v2", "unknown:1b"]
        assert mr._drop_ollama_embedding_models("http://localhost:11434", models) == ["qwen3.8:27b", "unknown:1b"]
        # Non-Ollama lists are untouched.
        assert mr._drop_ollama_embedding_models("http://localhost:1234/v1", models) == models


# ── Item 9: probes don't load models ────────────────────────────────────


class TestOllamaModelProbe:
    def test_capabilities_replace_generation(self, monkeypatch):
        from routes import model_routes as mr
        monkeypatch.setattr(oc, "capability_tokens", lambda url, model, **kw: frozenset({"completion", "tools"}))
        monkeypatch.setattr(mr.httpx, "post", lambda *a, **k: pytest.fail("generation probe must not run"))
        result = mr._probe_single_model("http://localhost:11434", None, "qwen3.8:27b", timeout=8, with_tools=True)
        assert result["status"] == "ok"
        assert result["method"] == "capabilities"

    def test_embedding_and_missing_tools_fail_without_loading(self, monkeypatch):
        from routes import model_routes as mr
        caps = {"e": frozenset({"embedding"}), "t": frozenset({"completion"})}
        monkeypatch.setattr(oc, "capability_tokens", lambda url, model, **kw: caps[model])
        monkeypatch.setattr(mr.httpx, "post", lambda *a, **k: pytest.fail("generation probe must not run"))
        assert mr._probe_single_model("http://localhost:11434/v1", None, "e")["status"] == "fail"
        assert mr._probe_single_model("http://localhost:11434", None, "t", with_tools=True)["status"] == "fail"

    def test_unknown_capabilities_fall_back_with_cold_load_timeout_and_num_ctx(self, monkeypatch):
        from routes import model_routes as mr
        import src.assistant_preferences as ap
        monkeypatch.setattr(oc, "capability_tokens", lambda url, model, **kw: None)
        monkeypatch.setattr(ap, "default_context_limit", lambda: 24576)
        sent = {}

        def fake_post(url, headers=None, json=None, timeout=None, verify=None):
            sent.update(url=url, json=json, timeout=timeout)
            return _Resp({"message": {"content": "OK"}})

        monkeypatch.setattr(mr.httpx, "post", fake_post)
        result = mr._probe_single_model("http://localhost:11434", None, "qwen3.8:27b", timeout=8)
        assert result["status"] == "ok"
        assert sent["url"] == "http://localhost:11434/api/chat"
        assert sent["timeout"] >= mr._OLLAMA_COLD_LOAD_TIMEOUT
        assert sent["json"]["options"]["num_ctx"] == 24576


# ── Item 3: vision from /api/show ───────────────────────────────────────


class TestVisionCapability:
    URL = "http://host.docker.internal:11434/v1"

    def test_reported_vision_wins_over_name_list(self, monkeypatch):
        from src import chat_helpers
        monkeypatch.setattr(oc, "capability_tokens", lambda url, model, **kw: frozenset({"completion", "vision"}))
        monkeypatch.setattr(chat_helpers, "lmstudio_supports_vision", lambda *a: pytest.fail("no LM Studio probe for Ollama"))
        for model in ("qwen3.8:27b", "muse-glimmer:30b", "ornith-1.5:9b"):
            assert chat_helpers.is_vision_model(model) is False
            assert chat_helpers.model_supports_vision(model, self.URL) is True

    def test_reported_text_only_model(self, monkeypatch):
        from src import chat_helpers
        monkeypatch.setattr(oc, "capability_tokens", lambda url, model, **kw: frozenset({"completion"}))
        assert chat_helpers.model_supports_vision("gemma4:12b", self.URL) is False

    def test_unknown_capability_falls_back_to_names(self, monkeypatch):
        from src import chat_helpers
        monkeypatch.setattr(oc, "capability_tokens", lambda url, model, **kw: None)
        monkeypatch.setattr(chat_helpers, "lmstudio_supports_vision", lambda *a: pytest.fail("no LM Studio probe for Ollama"))
        assert chat_helpers.model_supports_vision("gemma4:12b", self.URL) is True
        assert chat_helpers.model_supports_vision("qwen3.8:27b", self.URL) is False


# ── Item 4: configurable default local context cap ──────────────────────


def _settings(monkeypatch, **values):
    import src.settings as settings_mod
    merged = {**settings_mod.DEFAULT_SETTINGS, **values}
    monkeypatch.setattr(settings_mod, "load_settings", lambda: merged)


class TestDefaultContextLimit:
    @pytest.mark.parametrize("raw, expected", [
        (16384, 16384), ("65536", 65536), (100, 1024), (10**7, 262144),
        ("nope", 32768), (True, 32768), (None, 32768),
    ])
    def test_setting_is_validated(self, monkeypatch, raw, expected):
        from src.assistant_preferences import default_context_limit
        _settings(monkeypatch, local_context_limit_default=raw)
        assert default_context_limit() == expected

    def test_background_payload_uses_configured_cap(self, monkeypatch):
        _settings(monkeypatch, local_context_limit_default=16384)
        from src import model_context
        monkeypatch.setattr(model_context, "_configured_endpoint_kind", lambda url: None)
        payload = llm_core._build_ollama_payload(
            "qwen3.8:27b", [{"role": "user", "content": "title"}], 0.2, 64,
            url="http://host.docker.internal:11434/api/chat",
        )
        assert payload["options"]["num_ctx"] == 16384

    def test_turn_without_model_limit_uses_configured_cap(self, monkeypatch):
        _settings(monkeypatch, local_context_limit_default=49152)
        from src.assistant_preferences import AssistantPreferences, assistant_turn, context_limit
        url = "http://localhost:11434"
        with assistant_turn(AssistantPreferences(context_limits={"big": 131072}), url):
            assert context_limit(url, "big") == 131072
            assert context_limit(url, "other") == 49152


# ── Item 5: health threshold ────────────────────────────────────────────


class TestHealthContextThreshold:
    SHOW = {"capabilities": ["completion"], "model_info": {"qwen35.context_length": 262144}}

    def _issues(self, monkeypatch, base_url, serving, cap=32768):
        from src import service_health as sh
        monkeypatch.setattr(sh, "_default_context_cap", lambda: cap)
        ps = {"models": [{"name": "qwen3.8:27b", "size": 1, "size_vram": 1, "context_length": serving}]}
        out = sh.local_models_health(
            [{"name": "ollama", "base_url": base_url}],
            get_json=lambda url: ps, post_json=lambda url, payload: self.SHOW,
        )
        return out["meta"]["endpoints"][0]["issues"]

    def test_native_at_default_cap_is_clean(self, monkeypatch):
        assert self._issues(monkeypatch, "http://localhost:11434", 32768) == []

    def test_native_below_cap_reports_reload(self, monkeypatch):
        issues = self._issues(monkeypatch, "http://localhost:11434", 8192)
        assert any("reload" in i for i in issues)

    def test_v1_server_default_above_cap_is_clean(self, monkeypatch):
        # The user's server: OLLAMA_CONTEXT_LENGTH=65536 with the 32768 default cap.
        assert self._issues(monkeypatch, "http://localhost:11434/v1", 65536) == []

    def test_v1_below_cap_points_at_server_setting(self, monkeypatch):
        issues = self._issues(monkeypatch, "http://localhost:11434/v1", 16384)
        assert any("OLLAMA_CONTEXT_LENGTH" in i for i in issues)


def test_slots_probe_url_is_the_server_root():
    from src import model_context
    assert model_context._slots_base("http://localhost:8080") == "http://localhost:8080"
    assert model_context._slots_base("http://localhost:8080/v1/chat/completions") == "http://localhost:8080"
    assert model_context._slots_base("http://gpu/llm/v1") == "http://gpu/llm"


# ── Item 7: single-resident-model mode ──────────────────────────────────


LOCAL = "http://host.docker.internal:11434/api/chat"
CLOUD = "https://api.openai.com/v1/chat/completions"


@pytest.fixture
def single_mode(monkeypatch):
    from src import model_context
    _settings(monkeypatch, local_single_model_mode=True)
    monkeypatch.setattr(model_context, "is_local_endpoint", lambda url: "openai.com" not in url)
    caps = {
        "qwen3.8:27b": frozenset({"completion", "vision", "tools", "thinking"}),
        "gemma4:12b": frozenset({"completion", "vision"}),
        "text-only:8b": frozenset({"completion"}),
        "all-minilm:l6-v2": frozenset({"embedding"}),
    }
    monkeypatch.setattr(oc, "capability_tokens", lambda url, model, **kw: caps.get(model))
    state = {"loaded": ["qwen3.8:27b"], "default": (LOCAL, "qwen3.8:27b", {})}
    monkeypatch.setattr(oc, "loaded_model_names", lambda url: list(state["loaded"]))
    real = endpoint_resolver.resolve_endpoint

    def fake_resolve(prefix, *args, **kwargs):
        if prefix == "default":
            return state["default"]
        return real(prefix, *args, **kwargs)

    monkeypatch.setattr(endpoint_resolver, "resolve_endpoint", fake_resolve)
    return state


class TestSingleModelMode:
    def test_loaded_model_replaces_other_local_model(self, single_mode):
        route = endpoint_resolver.apply_single_model_mode("utility", (LOCAL, "gemma4:12b", {"h": "1"}))
        assert route == (LOCAL, "qwen3.8:27b", {"h": "1"})

    def test_configured_model_already_loaded_is_kept(self, single_mode):
        single_mode["loaded"] = ["gemma4:12b"]
        route = (LOCAL, "gemma4:12b", {})
        assert endpoint_resolver.apply_single_model_mode("task", route) == route

    def test_mode_off_changes_nothing(self, single_mode, monkeypatch):
        _settings(monkeypatch, local_single_model_mode=False)
        route = (LOCAL, "gemma4:12b", {})
        assert endpoint_resolver.apply_single_model_mode("utility", route) == route

    def test_chat_roles_are_never_rewritten(self, single_mode):
        route = (LOCAL, "gemma4:12b", {})
        assert endpoint_resolver.apply_single_model_mode("default", route) == route

    def test_cloud_role_route_is_untouched(self, single_mode):
        route = (CLOUD, "gpt-4o-mini", {})
        assert endpoint_resolver.apply_single_model_mode("research", route) == route

    def test_nothing_loaded_uses_session_model_but_never_cloud(self, single_mode):
        single_mode["loaded"] = []
        session = (LOCAL, "qwen3.8:27b", {})
        assert endpoint_resolver.apply_single_model_mode(
            "utility", (LOCAL, "gemma4:12b", {}), chat_route=session) == session
        route = (LOCAL, "gemma4:12b", {})
        assert endpoint_resolver.apply_single_model_mode(
            "utility", route, chat_route=(CLOUD, "gpt-4o", {})) == route

    def test_nothing_loaded_falls_back_to_default_chat_model(self, single_mode):
        single_mode["loaded"] = []
        assert endpoint_resolver.apply_single_model_mode("compaction", (LOCAL, "gemma4:12b", {}))[1] == "qwen3.8:27b"

    def test_embedding_model_loaded_is_not_used(self, single_mode):
        single_mode["loaded"] = ["all-minilm:l6-v2"]
        assert endpoint_resolver.apply_single_model_mode("utility", (LOCAL, "gemma4:12b", {}))[1] == "qwen3.8:27b"

    def test_vision_only_moves_to_a_vision_capable_model(self, single_mode):
        single_mode["loaded"] = ["text-only:8b"]
        single_mode["default"] = (LOCAL, "text-only:8b", {})
        route = (LOCAL, "gemma4:12b", {})
        assert endpoint_resolver.apply_single_model_mode("vision", route) == route
        single_mode["loaded"] = ["qwen3.8:27b"]
        assert endpoint_resolver.apply_single_model_mode("vision", route)[1] == "qwen3.8:27b"

    def test_resolve_endpoint_applies_mode_with_session_fallback(self, single_mode, monkeypatch):
        single_mode["loaded"] = []
        monkeypatch.setattr(endpoint_resolver, "_resolve_configured_endpoint",
                            lambda *a: (LOCAL, "gemma4:12b", {}))
        session = (LOCAL, "muse-glimmer:30b", {})
        from src import chat_helpers
        monkeypatch.setattr(chat_helpers, "model_supports_vision", lambda m, u: True)
        assert endpoint_resolver.resolve_endpoint("utility", *session) == session


# ── Item 8: scheduler honors the task model ─────────────────────────────


class TestSchedulerDefaults:
    def _scheduler(self):
        from src.task_scheduler import TaskScheduler
        return TaskScheduler.__new__(TaskScheduler)

    def test_task_setting_wins_over_recent_session(self, monkeypatch):
        import src.task_endpoint as te
        monkeypatch.setattr(te, "resolve_task_endpoint", lambda **kw: ("http://t/v1/chat/completions", "task-model", {}))
        s = self._scheduler()
        monkeypatch.setattr(s, "_recent_session_route", lambda db, owner: pytest.fail("session fallback not needed"))
        assert s._resolve_defaults(None, "alice") == ("http://t/v1/chat/completions", "task-model")

    def test_default_model_then_recent_session(self, monkeypatch):
        import src.task_endpoint as te
        monkeypatch.setattr(te, "resolve_task_endpoint", lambda **kw: (None, None, None))
        calls = []

        def fake_default(prefix, **kw):
            calls.append(prefix)
            return ("http://d/api/chat", "default-model", {}) if len(calls) == 1 else (None, None, None)

        monkeypatch.setattr(endpoint_resolver, "resolve_endpoint", fake_default)
        s = self._scheduler()
        monkeypatch.setattr(s, "_recent_session_route", lambda db, owner: ("http://recent", "recent-model"))
        assert s._resolve_defaults(None, None) == ("http://d/api/chat", "default-model")
        assert s._resolve_defaults(None, None) == ("http://recent", "recent-model")
        assert calls == ["default", "default"]


# ── Residency helpers ───────────────────────────────────────────────────


def test_loaded_model_names_most_recent_first(monkeypatch):
    ps = {"models": [
        {"name": "gemma4:12b", "expires_at": "2026-09-30T10:00:00Z"},
        {"name": "qwen3.8:27b", "expires_at": "2026-09-30T10:14:00Z"},
    ]}
    calls = []
    monkeypatch.setattr(oc.httpx, "get", lambda url, timeout=None: calls.append(url) or _Resp(ps))
    assert oc.loaded_model_names("http://localhost:11434/v1") == ["qwen3.8:27b", "gemma4:12b"]
    assert oc.loaded_model_names("http://localhost:11434/api/chat") == ["qwen3.8:27b", "gemma4:12b"]
    assert calls == ["http://localhost:11434/api/ps"]  # cached briefly
    assert oc.loaded_model_names("http://localhost:1234/v1") is None


def test_known_only_classification_never_probes(_hermetic):
    assert oc.is_ollama_url("http://localhost:11435", probe=False) is False
    assert _hermetic == []
    assert oc.is_ollama_url("http://localhost:11434/v1", probe=False) is True


def test_session_runtime_reports_default_cap(monkeypatch):
    from types import SimpleNamespace
    from fastapi import APIRouter, FastAPI
    from fastapi.testclient import TestClient
    from routes import session_routes
    import src.assistant_preferences as ap
    from src import model_context

    monkeypatch.setattr(session_routes, "router", APIRouter(prefix="/api"))
    monkeypatch.setattr(session_routes, "_verify_session_owner", lambda request, sid: None, raising=False)
    monkeypatch.setattr(ap, "load_preferences", lambda sid: ap.LEGACY_PREFERENCES)
    monkeypatch.setattr(ap, "default_context_limit", lambda: 49152)
    monkeypatch.setattr(oc, "supports_thinking", lambda url, model: True)
    monkeypatch.setattr(oc, "model_context_window", lambda url, model: 262144)
    monkeypatch.setattr(model_context, "_ollama_ps_context", lambda url, model: None)
    monkeypatch.setattr(model_context, "is_local_endpoint", lambda url: True)
    manager = SimpleNamespace(get_session=lambda sid: SimpleNamespace(
        model="qwen3.8:27b", endpoint_url="http://localhost:11434/api/chat"))
    session_routes.setup_session_routes(manager, {})
    app = FastAPI()
    app.include_router(session_routes.router)
    with TestClient(app) as client:
        runtime = client.get("/api/session/s1/assistant").json()["runtime"]
    assert runtime["native"] is True
    assert runtime["context_limit"] == 49152
    assert runtime["default_context_limit"] == 49152

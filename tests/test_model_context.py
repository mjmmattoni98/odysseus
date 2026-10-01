"""Tests for model_context.py — local endpoint detection, token estimation, known model lookup."""

import sys
import types

import pytest

import src.model_context as model_context
from src.model_context import is_local_endpoint, estimate_tokens, _lookup_known


class _Column:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return ("eq", self.name, value)


class _ModelEndpoint:
    is_enabled = _Column("is_enabled")


class _Query:
    def __init__(self, rows):
        self.rows = list(rows)

    def filter(self, *conditions):
        for condition in conditions:
            if isinstance(condition, tuple) and condition[0] == "eq":
                _, field, value = condition
                self.rows = [row for row in self.rows if getattr(row, field) == value]
        return self

    def all(self):
        return list(self.rows)


class _Db:
    def __init__(self, rows):
        self.rows = rows

    def query(self, model):
        return _Query(self.rows)

    def close(self):
        pass


def _install_endpoint_db(monkeypatch, rows):
    mod = types.ModuleType("core.database")
    mod.ModelEndpoint = _ModelEndpoint
    mod.SessionLocal = lambda: _Db(rows)
    monkeypatch.setitem(sys.modules, "core.database", mod)


class TestIsLocalEndpoint:
    def test_localhost(self):
        assert is_local_endpoint("http://localhost:5000/v1/chat/completions") is True

    def test_loopback_ipv4(self):
        assert is_local_endpoint("http://127.0.0.1:8080/v1/chat/completions") is True

    def test_private_192_168(self):
        assert is_local_endpoint("http://192.168.1.1:11434/v1/chat/completions") is True

    def test_private_10(self):
        assert is_local_endpoint("http://10.0.0.5:8000/v1/chat/completions") is True

    @pytest.mark.parametrize("host", [
        "10.example-cloud.com",
        "172.16.example-cloud.com",
        "192.168.example-cloud.com",
    ])
    def test_private_prefix_dns_names_are_remote(self, host):
        assert is_local_endpoint(f"https://{host}/v1/chat/completions") is False

    def test_tailscale_100(self):
        # 100.64.0.0/10 is the CGNAT range Tailscale uses.
        assert is_local_endpoint("http://100.64.0.1:5000/v1/chat/completions") is True

    def test_configured_tailscale_proxy_is_remote(self, monkeypatch):
        _install_endpoint_db(monkeypatch, [
            types.SimpleNamespace(
                base_url="http://100.117.136.97:34521/v1",
                endpoint_kind="proxy",
                api_key="fake-key",
                is_enabled=True,
            )
        ])

        assert is_local_endpoint("http://100.117.136.97:34521/v1/chat/completions") is False

    def test_openai_is_remote(self):
        assert is_local_endpoint("https://api.openai.com/v1/chat/completions") is False

    def test_anthropic_is_remote(self):
        assert is_local_endpoint("https://api.anthropic.com/v1/messages") is False

    def test_empty_url(self):
        assert is_local_endpoint("") is False

    def test_malformed_url(self):
        assert is_local_endpoint("not-a-url") is False


class TestEstimateTokens:
    def test_empty_list(self):
        assert estimate_tokens([]) == 0

    def test_single_short_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        tokens = estimate_tokens(messages)
        # 4 overhead + int(5 * 0.3) = 4 + 1 = 5
        assert tokens == 5

    def test_multiple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi there"},
        ]
        tokens = estimate_tokens(messages)
        assert tokens > 0
        # Each message adds 4 overhead + chars * 0.3
        assert tokens == 4 + int(16 * 0.3) + 4 + int(8 * 0.3)

    def test_multimodal_content_list(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        tokens = estimate_tokens(messages)
        # 4 overhead + int(19 * 0.3) for the text item; image_url is ignored
        assert tokens == 4 + int(19 * 0.3)

    def test_missing_content_key(self):
        messages = [{"role": "assistant"}]
        tokens = estimate_tokens(messages)
        # 4 overhead + 0 content
        assert tokens == 4

    def test_scales_with_length(self):
        short = estimate_tokens([{"role": "user", "content": "short"}])
        long_text = "a" * 10000
        long = estimate_tokens([{"role": "user", "content": long_text}])
        assert long > short * 10


class TestLookupKnown:
    def test_claude_sonnet(self):
        assert _lookup_known("claude-sonnet-4-5") == 200000

    def test_gpt4o(self):
        assert _lookup_known("gpt-4o") == 128000

    def test_deepseek_r1(self):
        assert _lookup_known("deepseek-r1") == 64000

    def test_gemini_pro(self):
        assert _lookup_known("gemini-2.5-pro") == 1048576

    def test_unknown_model(self):
        assert _lookup_known("totally-unknown-model-xyz") is None

    def test_namespaced_model(self):
        """Models prefixed with provider/ should still match."""
        result = _lookup_known("openrouter/deepseek-r1")
        assert result == 64000

    def test_model_with_tag(self):
        """Models with :free or :extended suffixes should still match."""
        result = _lookup_known("deepseek-r1:free")
        assert result == 64000

    def test_o1_mini_not_shadowed_by_o1(self):
        """'o1' (200k) precedes 'o1-mini' (128k) in the table; longest match wins."""
        assert _lookup_known("o1-mini") == 128000

    def test_o1_full(self):
        assert _lookup_known("o1") == 200000

    def test_gpt4o_mini_not_shadowed_by_gpt4(self):
        assert _lookup_known("gpt-4o-mini") == 128000

    def test_gpt4_base(self):
        assert _lookup_known("gpt-4") == 8192

    def test_ollama_style_id_matches_hyphenated_key(self):
        """gemma4:26b must resolve against the 'gemma-4' table entry."""
        assert _lookup_known("gemma4:26b") == 262144
        assert _lookup_known("gemma4:12b") == 262144

    def test_local_family_windows(self):
        assert _lookup_known("granite4.2:8b") == 131072
        assert _lookup_known("lfm2.5:8b") == 128000
        assert _lookup_known("ornith-1.5:9b") == 262144
        assert _lookup_known("laguna-xs-2.1:q4_K_M") == 262144

    def test_lfm2_tag_does_not_shadow_lfm2_5(self):
        assert _lookup_known("lfm2.5:8b") == 128000
        assert _lookup_known("lfm2:8b") == 32768

    def test_hyphenated_model_still_matches_known_key(self):
        assert _lookup_known("gemma-4-31b") == 262144


class _FakeResp:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.is_success = ok

    def json(self):
        return self._payload


class TestGetContextLength:
    def setup_method(self):
        model_context._context_cache.clear()
        model_context._catalog_ctx_cache.clear()
        model_context._local_context_cache.clear()
        model_context._serving_context_seen.clear()
        import src.ollama_capabilities as oc
        oc.reset_cache()

    def test_local_endpoint_requeries_same_model_after_restart(self, monkeypatch):
        # Local answers live only for a short TTL: a restarted server with a
        # new --max-model-len is picked up once it expires, while requests in
        # between skip the database + probe round trip.
        calls = []
        now = [1000.0]

        def fake_query(endpoint_url, model):
            calls.append((endpoint_url, model))
            return (8192, True) if len(calls) == 1 else (27000, True)

        monkeypatch.setattr(model_context, "_query_context_length", fake_query)
        monkeypatch.setattr(model_context, "_clock", lambda: now[0])

        endpoint = "http://127.0.0.1:8000/v1/chat/completions"
        model = "Qwen/Qwen3-14B"

        first = model_context.get_context_length(endpoint, model)
        cached = model_context.get_context_length(endpoint, model)
        now[0] += model_context._LOCAL_CONTEXT_TTL_SECONDS + 1
        after_restart = model_context.get_context_length(endpoint, model)

        assert (first, cached, after_restart) == (8192, 8192, 27000)
        assert len(calls) == 2

    def test_remote_endpoint_keeps_cached_context(self, monkeypatch):
        calls = []

        def fake_query(endpoint_url, model):
            calls.append((endpoint_url, model))
            return (200000, True) if len(calls) == 1 else (12345, True)

        monkeypatch.setattr(model_context, "_query_context_length", fake_query)

        endpoint = "https://api.openai.com/v1/chat/completions"
        model = "gpt-5"

        first = model_context.get_context_length(endpoint, model)
        second = model_context.get_context_length(endpoint, model)

        assert first == 200000
        assert second == 200000
        assert len(calls) == 1

    def _proxy_db(self, monkeypatch):
        _install_endpoint_db(monkeypatch, [
            types.SimpleNamespace(
                base_url="http://100.117.136.97:34521/v1",
                endpoint_kind="proxy",
                api_key="fake-key",
                is_enabled=True,
            )
        ])

    def test_configured_proxy_known_model_skips_model_listing(self, monkeypatch):
        # A model covered by the known-context table must still resolve without
        # touching /models — the cheap path the proxy short-circuit exists for.
        self._proxy_db(monkeypatch)

        def fake_get(*args, **kwargs):
            raise AssertionError("/models must not be queried for a known proxy model")

        monkeypatch.setattr(model_context.httpx, "get", fake_get)

        endpoint = "http://100.117.136.97:34521/v1/chat/completions"
        assert model_context.get_context_length(endpoint, "gpt-4o") == 128000

    def test_configured_proxy_unknown_model_reads_catalog_context(self, monkeypatch):
        # A model missing from the known table (e.g. a new OpenRouter model)
        # must report the catalog's real window, not the bare default (#4886).
        # The catalog is fetched once per endpoint and reused for other models.
        self._proxy_db(monkeypatch)
        fetches = []

        def fake_get(url, *args, **kwargs):
            fetches.append(url)
            return _FakeResp({"data": [
                {"id": "owl-alpha", "context_length": 1048576},
                {"id": "tiny-proxy-model", "context_length": 8192},
            ]})

        monkeypatch.setattr(model_context.httpx, "get", fake_get)

        endpoint = "http://100.117.136.97:34521/v1/chat/completions"
        assert model_context.get_context_length(endpoint, "owl-alpha") == 1048576
        # A second unknown model on the same endpoint reuses the cached catalog.
        assert model_context.get_context_length(endpoint, "tiny-proxy-model") == 8192
        assert len(fetches) == 1

    def test_configured_proxy_unknown_model_falls_back_to_default(self, monkeypatch):
        # If the catalog can be read but doesn't list the model, keep the
        # conservative default rather than guessing.
        self._proxy_db(monkeypatch)

        def fake_get(url, *args, **kwargs):
            return _FakeResp({"data": [{"id": "some-other-model", "context_length": 4096}]})

        monkeypatch.setattr(model_context.httpx, "get", fake_get)

        endpoint = "http://100.117.136.97:34521/v1/chat/completions"
        assert model_context.get_context_length(endpoint, "absent-model") == model_context.DEFAULT_CONTEXT

    def test_configured_proxy_catalog_fetch_failure_uses_default(self, monkeypatch):
        # A failed/unreachable catalog must not raise — fall back to the default.
        self._proxy_db(monkeypatch)

        def fake_get(url, *args, **kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr(model_context.httpx, "get", fake_get)

        endpoint = "http://100.117.136.97:34521/v1/chat/completions"
        assert model_context.get_context_length(endpoint, "unknown-proxy-model") == model_context.DEFAULT_CONTEXT


class TestOllamaServingContext:
    """Ollama /api/ps reports the window actually allocated for a loaded model."""

    def setup_method(self):
        model_context._context_cache.clear()
        model_context._catalog_ctx_cache.clear()
        model_context._local_context_cache.clear()
        model_context._serving_context_seen.clear()
        import src.ollama_capabilities as oc
        oc.reset_cache()

    def test_loaded_model_uses_ollama_serving_window(self, monkeypatch):
        def fake_get(url, *args, **kwargs):
            if url.endswith("/api/ps"):
                return _FakeResp({"models": [
                    {"name": "qwen3.8:27b", "model": "qwen3.8:27b", "context_length": 32768},
                ]})
            return _FakeResp({}, ok=False)

        monkeypatch.setattr(model_context.httpx, "get", fake_get)

        ctx, known = model_context._query_context_length(
            "http://127.0.0.1:11434/v1/chat/completions", "qwen3.8:27b"
        )
        assert (ctx, known) == (32768, True)

    def test_loaded_model_matches_by_model_field(self, monkeypatch):
        def fake_get(url, *args, **kwargs):
            if url.endswith("/api/ps"):
                return _FakeResp({"models": [
                    {"name": "alias", "model": "gemma4:12b", "context_length": 65536},
                ]})
            return _FakeResp({}, ok=False)

        monkeypatch.setattr(model_context.httpx, "get", fake_get)

        ctx, _ = model_context._query_context_length(
            "http://127.0.0.1:11434/v1", "gemma4:12b"
        )
        assert ctx == 65536

    def _no_show(self, monkeypatch):
        def fake_post(*args, **kwargs):
            raise RuntimeError("no /api/show in this test")

        monkeypatch.setattr(model_context.httpx, "post", fake_post)

    def test_unloaded_v1_model_is_capped_by_default_limit_not_name_table(self, monkeypatch):
        # Ollama /v1 cannot carry num_ctx: the server picks the window when it
        # loads the model, so an unloaded model budgets at most the configured
        # default cap instead of the 131072 name-table maximum.
        def fake_get(url, *args, **kwargs):
            return _FakeResp({"models": []})

        monkeypatch.setattr(model_context.httpx, "get", fake_get)
        self._no_show(monkeypatch)
        import src.assistant_preferences as ap
        monkeypatch.setattr(ap, "default_context_limit", lambda: 40000)

        ctx, known = model_context._query_context_length(
            "http://127.0.0.1:11434/v1/chat/completions", "qwen3.8:27b"
        )
        assert (ctx, known) == (40000, True)

    def test_unloaded_v1_model_keeps_smaller_known_window(self, monkeypatch):
        monkeypatch.setattr(model_context.httpx, "get", lambda *a, **k: _FakeResp({"models": []}))
        self._no_show(monkeypatch)
        import src.assistant_preferences as ap
        monkeypatch.setattr(ap, "default_context_limit", lambda: 32768)

        ctx, _ = model_context._query_context_length("http://127.0.0.1:11434/v1", "phi-4:14b")
        assert ctx == 16000

    def test_unloaded_v1_model_reuses_last_serving_window(self, monkeypatch):
        # The user's server allocates 65536 (OLLAMA_CONTEXT_LENGTH); once
        # /api/ps has shown that, it wins after the model is unloaded.
        loaded = {"models": [{"name": "qwen3.8:27b", "model": "qwen3.8:27b", "context_length": 65536}]}
        responses = [loaded, {"models": []}]
        monkeypatch.setattr(model_context.httpx, "get", lambda url, *a, **k: _FakeResp(responses.pop(0)))
        self._no_show(monkeypatch)

        endpoint = "http://127.0.0.1:11434/v1/chat/completions"
        assert model_context._query_context_length(endpoint, "qwen3.8:27b") == (65536, True)
        assert model_context._query_context_length(endpoint, "qwen3.8:27b") == (65536, True)

    def test_ps_probe_is_direct_and_tolerant(self, monkeypatch):
        calls = []

        def fake_get(url, *args, **kwargs):
            calls.append(url)
            if url.endswith("/api/ps"):
                raise RuntimeError("connection refused")
            return _FakeResp({}, ok=False)

        monkeypatch.setattr(model_context.httpx, "get", fake_get)
        self._no_show(monkeypatch)
        import src.assistant_preferences as ap
        monkeypatch.setattr(ap, "default_context_limit", lambda: 32768)

        ctx, known = model_context._query_context_length(
            "http://127.0.0.1:11434/v1", "gemma4:26b"
        )
        assert (ctx, known) == (32768, True)
        assert any(url.endswith("/api/ps") for url in calls)
        # Ollama has no llama.cpp /slots endpoint; it is not probed.
        assert not any(url.endswith("/slots") for url in calls)

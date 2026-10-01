"""Cookbook Ollama management API (routes/ollama_routes.py + src/ollama_admin.py).

All Ollama HTTP traffic goes through an httpx.MockTransport installed via the
``ollama_admin._make_client`` seam — no network.
"""
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from routes.ollama_routes import setup_ollama_routes
from src import ollama_admin

ROOT = "http://127.0.0.1:11434"

SHOW_QWEN = {
    "details": {"parameter_size": "27.3B", "quantization_level": "Q4_K_M", "family": "qwen35"},
    "capabilities": ["completion", "vision", "tools", "thinking"],
    "parameters": "temperature                    1\ntop_k                          20\nnum_ctx                        65536\nstop                           \"<|im_end|>\"",
    "model_info": {
        "general.architecture": "qwen35",
        "qwen35.block_count": 65,
        "qwen35.full_attention_interval": 4,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.context_length": 262144,
    },
}
TAGS = {
    "models": [
        {"name": "qwen3.8:27b", "digest": "d1", "size": 18_000_000_000,
         "details": {"parameter_size": "27.3B", "quantization_level": "Q4_K_M"}},
        {"name": "all-minilm:l6-v2", "digest": "d2", "size": 45_000_000,
         "details": {"parameter_size": "23M"}, "capabilities": ["embedding"]},
    ]
}


class FakeOllama:
    """Records requests; answers like a small Ollama server."""

    def __init__(self):
        self.calls = []
        self.pull_lines = [
            {"status": "pulling manifest"},
            {"status": "pulling abc", "digest": "sha256:abc", "total": 100, "completed": 40},
            {"status": "pulling abc", "digest": "sha256:abc", "total": 100, "completed": 100},
            {"status": "verifying sha256 digest"},
            {"status": "success"},
        ]
        self.show_payloads = {"qwen3.8:27b": SHOW_QWEN, "all-minilm:l6-v2": {"capabilities": ["embedding"]}}

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}") if request.content else {}
        self.calls.append((request.method, str(request.url), body))
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.34.4"})
        if path == "/api/tags":
            return httpx.Response(200, json=TAGS)
        if path == "/api/show":
            return httpx.Response(200, json=self.show_payloads.get(body.get("model"), {}))
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [
                {"name": "qwen3.8:27b", "size": 20_000, "size_vram": 20_000, "context_length": 65536,
                 "expires_at": "2026-09-30T10:15:00Z", "details": {"quantization_level": "Q4_K_M"}},
                {"name": "gemma4:12b", "size": 10_000, "size_vram": 6_000, "expires_at": ""},
            ]})
        if path == "/api/pull":
            text = "".join(json.dumps(line) + "\n" for line in self.pull_lines)
            return httpx.Response(200, content=text.encode(), headers={"content-type": "application/x-ndjson"})
        if path == "/api/delete":
            return httpx.Response(200, json={})
        if path == "/api/generate":
            return httpx.Response(200, json={"model": body.get("model"), "done": True, "done_reason": "unload"})
        if path == "/api/create":
            return httpx.Response(200, json={"status": "success"})
        return httpx.Response(404, json={"error": "not found"})

    def paths(self, method=None):
        return [urlparse_path(url) for m, url, _ in self.calls if method is None or m == method]


def urlparse_path(url):
    return httpx.URL(url).path


@pytest.fixture
def fake(monkeypatch):
    server = FakeOllama()
    transport = httpx.MockTransport(server.handler)
    monkeypatch.setattr(
        ollama_admin, "_make_client",
        lambda timeout=None: httpx.AsyncClient(transport=transport, timeout=timeout),
    )
    ollama_admin.reset_cache()
    # One allowlisted target: a registered native Ollama endpoint.
    monkeypatch.setattr(ollama_admin, "_load_endpoint_rows", lambda: [
        SimpleNamespace(id="ep1", name="Host Ollama", base_url=ROOT + "/v1", endpoint_kind="auto",
                        model_type="llm", api_key=None),
    ])
    monkeypatch.setattr(ollama_admin, "local_candidate_roots", lambda: [])
    monkeypatch.setattr("src.settings.load_settings", lambda: {})
    yield server
    ollama_admin.reset_cache()


class _User(BaseHTTPMiddleware):
    def __init__(self, app, user):
        super().__init__(app)
        self.user = user

    async def dispatch(self, request, call_next):
        request.state.current_user = self.user
        return await call_next(request)


def _client(monkeypatch, *, admin=True):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    app = FastAPI()
    app.state.auth_manager = SimpleNamespace(is_configured=True, is_admin=lambda user: admin and user == "alice")
    app.add_middleware(_User, user="alice")
    app.include_router(setup_ollama_routes())
    return TestClient(app)


SERVER = "ep:ep1"


def test_non_admin_is_refused_everywhere(monkeypatch, fake):
    client = _client(monkeypatch, admin=False)
    responses = [
        client.get("/api/cookbook/ollama/servers"),
        client.get("/api/cookbook/ollama/models", params={"server": SERVER}),
        client.get("/api/cookbook/ollama/running", params={"server": SERVER}),
        client.post("/api/cookbook/ollama/pull", json={"server": SERVER, "model": "qwen3:8b"}),
        client.delete("/api/cookbook/ollama/models", params={"server": SERVER, "model": "qwen3.8:27b"}),
        client.post("/api/cookbook/ollama/unload", json={"server": SERVER, "model": "qwen3.8:27b"}),
        client.post("/api/cookbook/ollama/create", json={"server": SERVER, "name": "x:y", "from": "qwen3.8:27b"}),
    ]
    assert [r.status_code for r in responses] == [403] * len(responses)
    assert fake.calls == []


@pytest.mark.parametrize("server", ["http://169.254.169.254", "local:http://10.0.0.5:11434", "ep:missing"])
def test_unlisted_server_is_refused_without_any_request(monkeypatch, fake, server):
    client = _client(monkeypatch)
    r = client.get("/api/cookbook/ollama/models", params={"server": server})
    assert r.status_code == 404
    r = client.post("/api/cookbook/ollama/pull", json={"server": server, "model": "qwen3:8b"})
    assert r.status_code == 404
    assert fake.calls == []


def test_servers_lists_endpoint_targets_with_version(monkeypatch, fake):
    client = _client(monkeypatch)
    servers = client.get("/api/cookbook/ollama/servers").json()["servers"]
    assert servers == [{
        "id": SERVER, "label": "Host Ollama", "url": ROOT, "host": "127.0.0.1", "source": "endpoint",
        "endpoint_id": "ep1", "reachable": True, "version": "0.34.4",
    }]


def test_local_candidates_are_listed_only_when_reachable(monkeypatch, fake):
    monkeypatch.setattr(ollama_admin, "_load_endpoint_rows", lambda: [])
    monkeypatch.setattr(ollama_admin, "local_candidate_roots", lambda: [ROOT, "http://host.docker.internal:11434"])
    original = fake.handler

    def handler(request):
        if request.url.host == "host.docker.internal":
            raise httpx.ConnectError("down", request=request)
        return original(request)

    monkeypatch.setattr(ollama_admin, "_make_client",
                        lambda timeout=None: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    servers = _client(monkeypatch).get("/api/cookbook/ollama/servers").json()["servers"]
    assert [s["id"] for s in servers] == [f"local:{ROOT}"]


def test_installed_merges_tags_with_cached_show_details(monkeypatch, fake):
    client = _client(monkeypatch)
    data = client.get("/api/cookbook/ollama/models", params={"server": SERVER}).json()
    qwen, minilm = sorted(data["models"], key=lambda m: m["name"], reverse=True)
    assert qwen["name"] == "qwen3.8:27b"
    assert qwen["capabilities"] == ["completion", "vision", "tools", "thinking"]
    assert qwen["context_length"] == 262144
    assert qwen["size"] == 18_000_000_000
    assert qwen["quantization"] == "Q4_K_M" and qwen["parameter_size"] == "27.3B"
    assert qwen["parameters"]["num_ctx"] == 65536 and qwen["parameters"]["stop"] == ["<|im_end|>"]
    # every 4th of 65 layers (16) has attention: 16 × 4 kv heads × (256 + 256) × 2 bytes
    assert qwen["kv_bytes_per_token"] == 16 * 4 * 512 * 2
    assert qwen["kv_swa_bytes_per_token"] == 0 and qwen["kv_sliding_window"] == 0
    assert minilm["capabilities"] == ["embedding"]
    assert isinstance(data["default_context"], int) and data["default_context"] > 0

    shows = fake.paths().count("/api/show")
    client.get("/api/cookbook/ollama/models", params={"server": SERVER})
    assert fake.paths().count("/api/show") == shows, "unchanged digests must hit the show cache"


def test_running_reports_vram_residency_and_context(monkeypatch, fake):
    models = _client(monkeypatch).get("/api/cookbook/ollama/running", params={"server": SERVER}).json()["models"]
    assert models[0]["fully_on_gpu"] is True and models[0]["context_length"] == 65536
    assert models[0]["expires_at"] == "2026-09-30T10:15:00Z"
    assert models[1]["fully_on_gpu"] is False and models[1]["gpu_fraction"] == 0.6


def _sse_events(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_pull_streams_ndjson_progress_as_sse(monkeypatch, fake):
    r = _client(monkeypatch).post("/api/cookbook/ollama/pull", json={"server": SERVER, "model": "qwen3:8b"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)
    assert events[0] == ("start", {"model": "qwen3:8b", "server": SERVER})
    progress = [d for e, d in events if e == "progress"]
    assert progress[1] == {"status": "pulling abc", "digest": "sha256:abc", "total": 100, "completed": 40}
    assert events[-1] == ("done", {"model": "qwen3:8b", "status": "success"})
    assert ("POST", f"{ROOT}/api/pull", {"model": "qwen3:8b", "stream": True}) in fake.calls


def test_pull_upstream_error_becomes_error_event(monkeypatch, fake):
    fake.pull_lines = [{"status": "pulling manifest"}, {"error": "pull model manifest: file does not exist"}]
    events = _sse_events(_client(monkeypatch).post(
        "/api/cookbook/ollama/pull", json={"server": SERVER, "model": "nope:1b"}).text)
    assert events[-1] == ("error", {"error": "pull model manifest: file does not exist"})


def test_pull_rejects_unsafe_model_names(monkeypatch, fake):
    r = _client(monkeypatch).post("/api/cookbook/ollama/pull", json={"server": SERVER, "model": "x; rm -rf /"})
    assert r.status_code == 400
    assert fake.calls == []


@pytest.mark.asyncio
async def test_closing_the_pull_stream_stops_reading_upstream(fake):
    target = ollama_admin.resolve_target(SERVER)
    stream = ollama_admin.stream_pull(target, "qwen3:8b")
    first = await stream.__anext__()
    await stream.aclose()  # what a client disconnect does to the route generator
    assert first == {"status": "pulling manifest"}
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


def test_delete_refuses_models_in_use_without_force(monkeypatch, fake):
    monkeypatch.setattr("src.settings.load_settings", lambda: {
        "default_model": "qwen3.8:27b", "default_endpoint_id": "ep1",
        "utility_model_fallbacks": [{"endpoint_id": "", "model": "qwen3.8:27b"}],
    })
    client = _client(monkeypatch)
    r = client.delete("/api/cookbook/ollama/models", params={"server": SERVER, "model": "qwen3.8:27b"})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert [u["setting"] for u in detail["in_use"]] == ["default_model", "utility_model_fallbacks"]
    assert "/api/delete" not in fake.paths()

    r = client.delete("/api/cookbook/ollama/models", params={"server": SERVER, "model": "qwen3.8:27b", "force": "true"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert ("DELETE", f"{ROOT}/api/delete", {"model": "qwen3.8:27b"}) in fake.calls


def test_delete_ignores_same_model_name_on_another_endpoint(monkeypatch, fake):
    monkeypatch.setattr(ollama_admin, "_load_endpoint_rows", lambda: [
        SimpleNamespace(id="ep1", name="Host Ollama", base_url=ROOT, endpoint_kind="ollama", model_type="llm", api_key=None),
        SimpleNamespace(id="ep2", name="GPU box", base_url="http://gpu-box:11434", endpoint_kind="ollama", model_type="llm", api_key=None),
    ])
    monkeypatch.setattr("src.settings.load_settings", lambda: {"default_model": "qwen3.8:27b", "default_endpoint_id": "ep2"})
    r = _client(monkeypatch).delete("/api/cookbook/ollama/models", params={"server": SERVER, "model": "qwen3.8:27b"})
    assert r.status_code == 200


def test_unload_sends_keep_alive_zero_without_prompt(monkeypatch, fake):
    r = _client(monkeypatch).post("/api/cookbook/ollama/unload", json={"server": SERVER, "model": "qwen3.8:27b"})
    assert r.status_code == 200 and r.json()["done_reason"] == "unload"
    method, url, body = fake.calls[-1]
    assert (method, url) == ("POST", f"{ROOT}/api/generate")
    assert body == {"model": "qwen3.8:27b", "keep_alive": 0, "stream": False}


@pytest.mark.parametrize("value,sent", [(-1, -1), ("30m", "30m"), ("3600", 3600)])
def test_keep_alive_passes_normalized_duration(monkeypatch, fake, value, sent):
    r = _client(monkeypatch).post("/api/cookbook/ollama/keep-alive",
                                  json={"server": SERVER, "model": "qwen3.8:27b", "keep_alive": value})
    assert r.status_code == 200
    assert fake.calls[-1][2] == {"model": "qwen3.8:27b", "keep_alive": sent, "stream": False}


def test_keep_alive_rejects_garbage(monkeypatch, fake):
    r = _client(monkeypatch).post("/api/cookbook/ollama/keep-alive",
                                  json={"server": SERVER, "model": "qwen3.8:27b", "keep_alive": "forever; ls"})
    assert r.status_code == 400
    assert fake.calls == []


def test_create_preset_sends_from_parameters_and_system(monkeypatch, fake):
    r = _client(monkeypatch).post("/api/cookbook/ollama/create", json={
        "server": SERVER, "name": "qwen3.8:27b-64k", "from": "qwen3.8:27b",
        "parameters": {"num_ctx": 65536, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0,
                       "repeat_penalty": 1.05, "num_predict": -1, "stop": ["<|im_end|>"]},
        "system": "Be brief.",
    })
    assert r.status_code == 200, r.text
    method, url, body = fake.calls[-1]
    assert (method, url) == ("POST", f"{ROOT}/api/create")
    assert body == {
        "model": "qwen3.8:27b-64k", "from": "qwen3.8:27b", "stream": False, "system": "Be brief.",
        "parameters": {"num_ctx": 65536, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                       "repeat_penalty": 1.05, "num_predict": -1, "stop": ["<|im_end|>"]},
    }


@pytest.mark.parametrize("body,fragment", [
    ({"name": "Qwen Big!", "from": "qwen3.8:27b"}, "Preset name"),
    ({"name": "a:b:c", "from": "qwen3.8:27b"}, "Preset name"),
    ({"name": "x" * 90, "from": "qwen3.8:27b"}, "Preset name"),
    ({"name": "qwen3.8:27b", "from": "qwen3.8:27b"}, "differ"),
    ({"name": "p:1", "from": "qwen3.8:27b", "parameters": {"num_ctx": 100}}, "num_ctx"),
    ({"name": "p:1", "from": "qwen3.8:27b", "parameters": {"temperature": 3}}, "temperature"),
    ({"name": "p:1", "from": "qwen3.8:27b", "parameters": {"top_k": 1.5}}, "top_k"),
    ({"name": "p:1", "from": "qwen3.8:27b", "parameters": {"top_p": "nan"}}, "top_p"),
    ({"name": "p:1", "from": "qwen3.8:27b", "parameters": {"mirostat": 1}}, "Unsupported"),
    ({"name": "p:1", "from": "qwen3.8:27b", "parameters": {"stop": ["x"] * 9}}, "stop"),
    ({"name": "p:1", "from": "qwen3.8 27b"}, "model name"),
])
def test_create_preset_validation(monkeypatch, fake, body, fragment):
    r = _client(monkeypatch).post("/api/cookbook/ollama/create", json={"server": SERVER, **body})
    assert r.status_code == 400
    assert fragment.lower() in r.json()["detail"].lower()
    assert "/api/create" not in fake.paths()


def test_create_preset_refuses_overwrite_and_missing_base(monkeypatch, fake):
    client = _client(monkeypatch)
    r = client.post("/api/cookbook/ollama/create", json={"server": SERVER, "name": "all-minilm:l6-v2", "from": "qwen3.8:27b"})
    assert r.status_code == 409
    r = client.post("/api/cookbook/ollama/create", json={"server": SERVER, "name": "p:1", "from": "llama9:70b"})
    assert r.status_code == 404
    assert "/api/create" not in fake.paths()
    r = client.post("/api/cookbook/ollama/create",
                    json={"server": SERVER, "name": "all-minilm:l6-v2", "from": "qwen3.8:27b", "overwrite": True})
    assert r.status_code == 200


def test_unreachable_server_is_a_shaped_502(monkeypatch, fake):
    def down(request):
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(ollama_admin, "_make_client",
                        lambda timeout=None: httpx.AsyncClient(transport=httpx.MockTransport(down)))
    r = _client(monkeypatch).get("/api/cookbook/ollama/running", params={"server": SERVER})
    assert r.status_code == 502
    assert "not reachable" in r.json()["detail"]


def test_endpoint_detection_excludes_cloud_and_non_ollama_rows():
    assert ollama_admin.endpoint_looks_like_ollama("http://host.docker.internal:11434/v1")
    assert ollama_admin.endpoint_looks_like_ollama("http://gpu-box:11500", "ollama")
    assert not ollama_admin.endpoint_looks_like_ollama("http://localhost:8080/v1")
    assert not ollama_admin.endpoint_looks_like_ollama("https://ollama.com/api", "ollama")
    assert ollama_admin.ollama_root("http://0.0.0.0:11434/v1/") == "http://127.0.0.1:11434"


def test_kv_estimate_handles_per_layer_arrays_and_missing_fields():
    info = {"g.block_count": 4, "g.attention.head_count": 8, "g.attention.head_count_kv": [2, 2, 4, 4],
            "g.embedding_length": 512}
    # key/value length fall back to embedding / heads = 64
    assert ollama_admin.kv_cache_bytes_per_token(info) == (2 + 2 + 4 + 4) * (64 + 64) * 2
    assert ollama_admin.kv_cache_bytes_per_token({"g.attention.head_count": 8}) is None


def test_kv_estimate_splits_sliding_window_layers():
    # Gemma-style: 5 sliding layers then 1 global layer, repeated.
    info = {
        "gemma4.block_count": 12, "gemma4.attention.head_count": 16,
        "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 1] * 2,
        "gemma4.attention.key_length": 512, "gemma4.attention.value_length": 512,
        "gemma4.attention.key_length_swa": 256, "gemma4.attention.value_length_swa": 256,
        "gemma4.attention.sliding_window": 1024,
        "gemma4.attention.sliding_window_pattern": [True] * 5 + [False] + [True] * 5 + [False],
    }
    assert ollama_admin.kv_cache_profile(info) == {
        "per_token": 2 * 1 * 1024 * 2,
        "swa_per_token": 10 * 8 * 512 * 2,
        "sliding_window": 1024,
    }

"""Cookbook Ollama serves: reuse a running daemon, native endpoint registration,
and the agent tool's port inference. No network: Ollama probes go through an
httpx.MockTransport, tmux launches and endpoint probes are stubbed."""
import json

import httpx
import pytest
from starlette.requests import Request

import routes.cookbook_routes as cookbook_routes
from core.database import Base, ModelEndpoint
from routes.cookbook_helpers import ServeRequest
from src import ollama_admin
from src.tools import cookbook as cookbook_tool
from tests.helpers.sqlite_db import make_temp_sqlite


# ── _infer_serve_port (agent tool) ──────────────────────────────────────────

@pytest.mark.parametrize("cmd,port", [
    ("llama-server --model m.gguf --port 8081", 8081),
    ("vllm serve org/m --port=8082", 8082),
    ("OLLAMA_HOST=0.0.0.0:11435 ollama serve", 11435),
    ("OLLAMA_HOST='[::]:11436' ollama serve", 11436),
    ("ollama serve", 11434),
    ("llama-server --model m.gguf", 8080),
])
def test_infer_serve_port_reads_custom_ports(cmd, port):
    assert cookbook_tool._infer_serve_port(cmd) == port


@pytest.mark.asyncio
async def test_agent_tool_registers_ollama_natively(monkeypatch):
    posted = []

    def handler(request):
        posted.append(dict(httpx.QueryParams(request.content.decode())))
        return httpx.Response(200, json={"id": "ep-new"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), timeout=kw.get("timeout")))
    result = await cookbook_tool._ensure_served_endpoint(
        model="qwen3:8b", cmd="OLLAMA_HOST=0.0.0.0:11435 ollama serve", host="gpu-box")
    assert result["base_url"] == "http://gpu-box:11435"
    assert posted[0]["base_url"] == "http://gpu-box:11435"
    assert posted[0]["endpoint_kind"] == "ollama"

    posted.clear()
    result = await cookbook_tool._ensure_served_endpoint(model="org/m", cmd="vllm serve org/m --port 8001", host="")
    assert result["base_url"] == "http://localhost:8001/v1"
    assert "endpoint_kind" not in posted[0]


# ── Registration URL ────────────────────────────────────────────────────────

@pytest.mark.parametrize("cmd,remote,url,is_ollama", [
    ("ollama serve", None, "http://localhost:11434", True),
    ("OLLAMA_HOST=0.0.0.0:11435 ollama serve", "me@gpu-box", "http://gpu-box:11435", True),
    ("docker exec ollama-rocm ollama show qwen3:8b", None, "http://host.docker.internal:11434", True),
    ("vllm serve org/m --port 8001", "gpu-box", "http://gpu-box:8001/v1", False),
    ("llama-server --model m.gguf --port=8090", None, "http://localhost:8090/v1", False),
])
def test_llm_endpoint_base_url(cmd, remote, url, is_ollama):
    assert cookbook_routes._llm_endpoint_base_url(cmd, remote) == (url, is_ollama)


def test_llm_endpoint_base_url_prefers_reused_daemon():
    assert cookbook_routes._llm_endpoint_base_url(
        "ollama serve", None, ollama_url="http://host.docker.internal:11434/"
    ) == ("http://host.docker.internal:11434", True)


# ── Existing daemon detection ───────────────────────────────────────────────

def _mock_ollama(monkeypatch, live_hosts):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.host in live_hosts and request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.34.4"})
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(ollama_admin, "_make_client",
                        lambda timeout=None: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return seen


@pytest.mark.asyncio
async def test_detects_host_daemon_from_container(monkeypatch):
    _mock_ollama(monkeypatch, {"host.docker.internal"})
    monkeypatch.setattr(cookbook_routes, "running_in_container", lambda: True)
    found = await cookbook_routes._detect_existing_ollama(None, None, 11434)
    assert found == {"url": "http://host.docker.internal:11434", "version": "0.34.4"}


@pytest.mark.asyncio
async def test_no_daemon_means_none(monkeypatch):
    _mock_ollama(monkeypatch, set())
    monkeypatch.setattr(cookbook_routes, "running_in_container", lambda: False)
    assert await cookbook_routes._detect_existing_ollama(None, None, 11434) is None


@pytest.mark.asyncio
async def test_remote_loopback_only_daemon_is_seen_over_ssh(monkeypatch):
    _mock_ollama(monkeypatch, set())
    commands = []

    class _Proc:
        async def communicate(self):
            return (b'{"version":"0.34.4"}', b"")

    async def fake_exec(*argv, **kwargs):
        commands.append(argv)
        return _Proc()

    monkeypatch.setattr(cookbook_routes.asyncio, "create_subprocess_exec", fake_exec)
    found = await cookbook_routes._detect_existing_ollama("me@gpu-box", "2222", 11434)
    assert found == {"url": "", "version": "0.34.4", "loopback_only": True}
    assert commands[0][:1] == ("ssh",) and "-p" in commands[0] and "me@gpu-box" in commands[0]


# ── model_serve with an existing daemon ─────────────────────────────────────

def _serve_endpoint():
    router = cookbook_routes.setup_cookbook_routes()
    for route in router.routes:
        if route.path == "/api/model/serve" and "POST" in route.methods:
            return route.endpoint
    raise AssertionError("POST /api/model/serve route not found")


def _request():
    request = Request({"type": "http", "method": "POST", "path": "/api/model/serve", "headers": [], "state": {}})
    request.state.current_user = "admin"
    return request


@pytest.fixture
def serve_env(monkeypatch, tmp_path):
    SessionLocal, engine, tmpfile = make_temp_sqlite(Base.metadata)
    monkeypatch.setattr("core.database.SessionLocal", SessionLocal)
    monkeypatch.setattr("routes.model_routes._probe_endpoint", lambda *a, **k: [])
    launched = []

    class _Proc:
        returncode = 0

        async def wait(self):
            return None

    async def launch(command, **kwargs):
        launched.append(command)
        return _Proc()

    async def always(*args, **kwargs):
        return True

    monkeypatch.setattr(cookbook_routes, "require_admin", lambda request: None)
    monkeypatch.setattr(cookbook_routes, "_binary_available", always)
    monkeypatch.setattr(cookbook_routes, "TMUX_LOG_DIR", tmp_path)
    monkeypatch.setattr(cookbook_routes, "load_stored_hf_token", lambda **kwargs: "")
    monkeypatch.setattr(cookbook_routes.asyncio, "create_subprocess_shell", launch)
    real_create_task = cookbook_routes.asyncio.create_task

    def create_task(coro, **kwargs):
        # Skip only the crash watchdog (it sleeps for minutes).
        if getattr(coro, "__name__", "") == "_serve_crash_watchdog":
            coro.close()
            return None
        return real_create_task(coro, **kwargs)

    monkeypatch.setattr(cookbook_routes.asyncio, "create_task", create_task)
    monkeypatch.setattr(cookbook_routes, "IS_WINDOWS", False)
    yield SessionLocal, tmp_path, launched
    engine.dispose()


@pytest.mark.asyncio
async def test_serve_reuses_running_daemon_instead_of_starting_another(monkeypatch, serve_env):
    SessionLocal, tmp_path, launched = serve_env

    async def existing(remote, ssh_port, port):
        assert (remote, port) == (None, 11434)
        return {"url": "http://host.docker.internal:11434", "version": "0.34.4"}

    monkeypatch.setattr(cookbook_routes, "_detect_existing_ollama", existing)
    response = await _serve_endpoint()(_request(), ServeRequest(repo_id="qwen3.8:27b", cmd="ollama serve"))

    assert response["ok"] is True
    assert response["reused_ollama"] == {"url": "http://host.docker.internal:11434", "version": "0.34.4"}
    runner = next(tmp_path.glob("serve-*_run.sh")).read_text(encoding="utf-8")
    assert "already running at http://host.docker.internal:11434" in runner
    assert "Ollama API ready on port 11434: http://host.docker.internal:11434" in runner
    assert "ollama serve" not in runner
    assert launched, "the task still gets a tmux session so Stop/unload work"

    db = SessionLocal()
    try:
        rows = db.query(ModelEndpoint).all()
        assert [(r.base_url, r.endpoint_kind) for r in rows] == [("http://host.docker.internal:11434", "ollama")]
        assert response["endpoint_id"] == rows[0].id
    finally:
        db.close()


@pytest.mark.asyncio
async def test_serve_reuse_keeps_existing_v1_registration(monkeypatch, serve_env):
    SessionLocal, tmp_path, launched = serve_env
    db = SessionLocal()
    db.add(ModelEndpoint(id="user-ollama", name="My Ollama", base_url="http://host.docker.internal:11434/v1",
                         is_enabled=True, model_type="llm", endpoint_kind="auto"))
    db.commit()
    db.close()

    async def existing(remote, ssh_port, port):
        return {"url": "http://host.docker.internal:11434", "version": "0.34.4"}

    monkeypatch.setattr(cookbook_routes, "_detect_existing_ollama", existing)
    response = await _serve_endpoint()(_request(), ServeRequest(repo_id="qwen3.8:27b", cmd="ollama serve"))

    db = SessionLocal()
    try:
        rows = db.query(ModelEndpoint).all()
        assert len(rows) == 1, "no duplicate endpoint for the same Ollama server"
        assert rows[0].id == "user-ollama" == response["endpoint_id"]
        assert rows[0].name == "My Ollama"
        assert rows[0].base_url == "http://host.docker.internal:11434/v1"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_serve_starts_daemon_when_none_is_running(monkeypatch, serve_env):
    SessionLocal, tmp_path, launched = serve_env

    async def none(remote, ssh_port, port):
        return None

    monkeypatch.setattr(cookbook_routes, "_detect_existing_ollama", none)
    response = await _serve_endpoint()(_request(), ServeRequest(repo_id="qwen3:8b", cmd="OLLAMA_HOST=127.0.0.1:11440 ollama serve"))
    assert response["ok"] is True and "reused_ollama" not in response
    runner = next(tmp_path.glob("serve-*_run.sh")).read_text(encoding="utf-8")
    assert 'OLLAMA_HOST="${ODYSSEUS_OLLAMA_HOST}:${ODYSSEUS_OLLAMA_PORT}" ollama serve' in runner
    db = SessionLocal()
    try:
        assert [(r.base_url, r.endpoint_kind) for r in db.query(ModelEndpoint).all()] == [
            ("http://localhost:11440", "ollama")]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_serve_refuses_second_daemon_next_to_unreachable_remote_one(monkeypatch, serve_env):
    SessionLocal, tmp_path, launched = serve_env

    async def loopback(remote, ssh_port, port):
        return {"url": "", "version": "0.34.4", "loopback_only": True}

    monkeypatch.setattr(cookbook_routes, "_detect_existing_ollama", loopback)
    monkeypatch.setattr(cookbook_routes, "validate_remote_host", lambda host: host)
    response = await _serve_endpoint()(
        _request(), ServeRequest(repo_id="qwen3:8b", cmd="ollama serve", remote_host="gpu-box"))
    assert response["ok"] is False
    assert "already running on gpu-box" in response["error"]
    assert "OLLAMA_HOST=0.0.0.0" in response["error"]
    assert launched == []

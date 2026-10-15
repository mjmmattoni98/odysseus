import json

import httpx
import pytest

from src import builtin_actions


class _FakeServeResponse:
    content = b"{}"

    def json(self):
        return {"ok": True, "session_id": "tmux-123"}


async def _fake_post(self, *_args, **_kwargs):
    return _FakeServeResponse()


async def _run_scheduled_serve(tmp_path, monkeypatch, server):
    state_path = tmp_path / "cookbook_state.json"
    state_path.write_text(
        json.dumps({"env": {"servers": [server]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(builtin_actions, "COOKBOOK_STATE_FILE", str(state_path))
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    message, ok = await builtin_actions.action_cookbook_serve(
        owner="alice",
        task_name="test-serve",
        command=json.dumps({
            "repo_id": "org/model",
            "cmd": "llama-server --port 8080",
            "host": "gpu-box",
            "end_after_min": 30,
        }),
    )

    assert ok is True, message
    tasks = json.loads(state_path.read_text(encoding="utf-8"))["tasks"]
    assert len(tasks) == 1
    return tasks[0]


@pytest.mark.asyncio
async def test_scheduled_serve_preserves_server_ssh_port_and_platform(tmp_path, monkeypatch):
    task = await _run_scheduled_serve(
        tmp_path,
        monkeypatch,
        {"name": "gpu-box", "host": "gpu-box", "port": "2222", "platform": "windows"},
    )

    assert task["sshPort"] == "2222"
    assert task["platform"] == "windows"
    assert task["remoteHost"] == "gpu-box"
    assert task["payload"]["_cmd"] == "llama-server --port 8080"


@pytest.mark.asyncio
async def test_scheduled_serve_uses_task_state_fallbacks_without_server_metadata(
    tmp_path,
    monkeypatch,
):
    task = await _run_scheduled_serve(
        tmp_path,
        monkeypatch,
        {"name": "gpu-box", "host": "gpu-box"},
    )

    assert task["sshPort"] == ""
    assert task["platform"] == "linux"


class _FakeServeWithEndpoint(_FakeServeResponse):
    def json(self):
        return {"ok": True, "session_id": "tmux-123", "endpoint_id": "served"}


async def _serve_and_read_defaults(tmp_path, monkeypatch, cached_models, reports):
    """Run a scheduled serve that sets defaults; return (settings, prefs)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    import core.database as cdb
    import routes.prefs_routes as prefs_routes
    import src.ollama_capabilities as caps
    import src.settings as settings_mod

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    cdb.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        db.add(cdb.ModelEndpoint(id="served", owner="alice", name="served", is_enabled=True,
                                 base_url="http://127.0.0.1:11434", cached_models=json.dumps(cached_models)))
        db.commit()
    monkeypatch.setattr(cdb, "SessionLocal", factory)
    monkeypatch.setattr(caps, "ollama_api_root", lambda url, probe=True: "http://127.0.0.1:11434/api")
    monkeypatch.setattr(caps, "cached_capability_tokens", lambda url, model: reports.get(model))
    saved = {"default_endpoint_id": "working", "default_model": "gemma4:26b"}
    prefs = {"default_endpoint_id": "working", "default_model": "gemma4:26b"}
    monkeypatch.setattr(settings_mod, "load_settings", lambda: dict(saved))
    monkeypatch.setattr(settings_mod, "save_settings", lambda value: saved.update(value))
    monkeypatch.setattr(prefs_routes, "_load_for_user", lambda owner: dict(prefs))
    monkeypatch.setattr(prefs_routes, "_save_for_user", lambda owner, value: prefs.update(value))

    async def _post(self, *_args, **_kwargs):
        return _FakeServeWithEndpoint()

    state_path = tmp_path / "cookbook_state.json"
    state_path.write_text(json.dumps({"env": {"servers": [{"name": "gpu-box", "host": "gpu-box"}]}}), encoding="utf-8")
    monkeypatch.setattr(builtin_actions, "COOKBOOK_STATE_FILE", str(state_path))
    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    message, ok = await builtin_actions.action_cookbook_serve(
        owner="alice", task_name="test-serve",
        command=json.dumps({"repo_id": "org/model", "cmd": "llama-server --port 8080", "host": "gpu-box"}),
    )
    assert ok is True, message
    engine.dispose()
    return saved, prefs


@pytest.mark.asyncio
async def test_scheduled_serve_keeps_working_defaults_when_no_listed_model_can_chat(tmp_path, monkeypatch):
    saved, prefs = await _serve_and_read_defaults(
        tmp_path, monkeypatch, ["clef-flash:latest"], {"clef-flash:latest": frozenset({"decision", "vision"})},
    )

    assert (saved["default_endpoint_id"], saved["default_model"]) == ("working", "gemma4:26b")
    assert (prefs["default_endpoint_id"], prefs["default_model"]) == ("working", "gemma4:26b")
    assert "task_model" not in saved and "utility_model" not in saved


@pytest.mark.asyncio
async def test_scheduled_serve_default_skips_a_listed_model_that_cannot_chat(tmp_path, monkeypatch):
    saved, prefs = await _serve_and_read_defaults(
        tmp_path, monkeypatch, ["clef-flash:latest", "gemma4:12b"],
        {"clef-flash:latest": frozenset({"decision", "vision"}), "gemma4:12b": frozenset({"completion"})},
    )

    assert (saved["default_endpoint_id"], saved["default_model"]) == ("served", "gemma4:12b")
    assert prefs["default_model"] == "gemma4:12b"

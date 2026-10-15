"""HTTP surface of server-side backups: admin-only, and the happy path.

Every /api/backups route must refuse non-admins and the agent's loopback
identity (internal-tool token), because a backup holds every user's data and
the app key, and a restore replaces all of it.
"""
import errno
import os
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from core.middleware import INTERNAL_TOOL_HEADER, INTERNAL_TOOL_TOKEN
from routes.backup_routes import setup_backup_routes
from src import backup_service as service

NAME = "odysseus-backup-20261012T030000Z-manual.tar.gz"
ROUTES = [
    ("GET", "/api/backups", None),
    ("PUT", "/api/backups/schedule", {"frequency": "daily", "time": "03:00"}),
    ("GET", "/api/backups/operation", None),
    ("POST", "/api/backups", None),
    ("POST", f"/api/backups/{NAME}/verify", None),
    ("GET", f"/api/backups/{NAME}/download", None),
    ("DELETE", f"/api/backups/{NAME}", None),
    ("GET", f"/api/backups/{NAME}/restore-preview", None),
    ("POST", f"/api/backups/{NAME}/restore", {"confirm": NAME}),
    ("DELETE", "/api/backups/restore/pending", None),
]


class _User(BaseHTTPMiddleware):
    def __init__(self, app, user):
        super().__init__(app)
        self.user = user

    async def dispatch(self, request, call_next):
        request.state.current_user = self.user
        return await call_next(request)


def _client(monkeypatch, *, admin):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    app = FastAPI()
    app.state.auth_manager = SimpleNamespace(is_configured=True, is_admin=lambda user: admin and user == "alice")
    app.add_middleware(_User, user="alice")
    app.include_router(setup_backup_routes(MagicMock(), MagicMock(), MagicMock()))
    return TestClient(app)


@pytest.fixture
def instance(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "uploads").mkdir(parents=True)
    db = data / "app.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO notes DEFAULT VALUES")
    conn.commit()
    conn.close()
    (data / "uploads" / "a.txt").write_text("a", encoding="utf-8")
    paths = service.BackupPaths(
        data_dir=str(data), backups_dir=str(data / "backups"), staging_dir=str(data / ".restore-staging"),
        previous_dir=str(data / ".restore-previous"), state_file=str(data / "backups" / "backup-state.json"),
        db_path=str(db),
    )
    monkeypatch.setattr(service, "default_paths", lambda: paths)
    monkeypatch.setattr(service, "load_schedule", lambda: service.normalize_schedule({}))
    return paths


def _send(client, method, url, body=None, headers=None):
    return client.request(method, url, json=body, headers=headers or {})


@pytest.mark.parametrize("method,url,body", ROUTES)
def test_non_admin_is_refused(monkeypatch, instance, method, url, body):
    response = _send(_client(monkeypatch, admin=False), method, url, body)
    assert response.status_code == 403, (method, url, response.text)


@pytest.mark.parametrize("method,url,body", ROUTES)
def test_agent_loopback_token_is_refused_even_for_admin_routes(monkeypatch, instance, method, url, body):
    client = _client(monkeypatch, admin=True)
    response = _send(client, method, url, body, headers={INTERNAL_TOOL_HEADER: INTERNAL_TOOL_TOKEN})
    assert response.status_code == 403, (method, url, response.text)


def _wait_for_operation(client):
    for _ in range(200):
        status = client.get("/api/backups/operation").json()
        if not status["current"]:
            return status["last"]
        time.sleep(0.05)
    raise AssertionError("backup operation did not finish")


def test_admin_backup_verify_download_preview_and_delete(monkeypatch, instance):
    client = _client(monkeypatch, admin=True)

    started = client.post("/api/backups")
    assert started.status_code == 200 and started.json()["operation"]["state"] == "running"
    last = _wait_for_operation(client)
    assert last["ok"] is True, last
    name = last["result"]["name"]

    listed = client.get("/api/backups").json()
    assert [b["name"] for b in listed["backups"]] == [name]
    assert listed["backups"][0]["verification"]["ok"] is True
    assert "path" not in listed["backups"][0]

    assert client.post(f"/api/backups/{name}/verify").status_code == 200
    assert _wait_for_operation(client)["ok"] is True

    download = client.get(f"/api/backups/{name}/download")
    assert download.status_code == 200 and download.content[:2] == b"\x1f\x8b"

    preview = client.get(f"/api/backups/{name}/restore-preview").json()
    assert preview["tables"] == [{"name": "notes", "backup": 1, "current": 1}]
    assert preview["summary"] == {"replace": 1, "add": 0, "set_aside": 0}

    wrong = client.post(f"/api/backups/{name}/restore", json={"confirm": "yes"})
    assert wrong.status_code == 400 and "exactly" in wrong.json()["detail"]

    assert client.delete(f"/api/backups/{name}").status_code == 200
    assert client.get("/api/backups").json()["backups"] == []


def test_incomplete_backup_is_reported_as_failed_and_flagged_in_the_list(monkeypatch, instance):
    # A permission failure on uploads/, patched because chmod is moot as root.
    blocked = os.path.join(instance.data_dir, "uploads")
    real_scandir = os.scandir

    def scandir(path="."):
        if not isinstance(path, int) and os.path.abspath(path) == blocked:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    client = _client(monkeypatch, admin=True)

    assert client.post("/api/backups").status_code == 200
    last = _wait_for_operation(client)

    assert last["ok"] is False and "incomplete" in last["message"], last
    listed = client.get("/api/backups").json()["backups"]
    assert listed[0]["verification"]["ok"] is True
    assert listed[0]["verification"]["complete"] is False
    assert listed[0]["verification"]["unreadable"] == ["data/uploads: Permission denied"]


@pytest.mark.parametrize("bad", ["..%2F..%2Fapp.db", "app.db", "odysseus-backup-x.tar.gz"])
def test_names_that_are_not_backups_are_refused(monkeypatch, instance, bad):
    client = _client(monkeypatch, admin=True)
    assert client.get(f"/api/backups/{bad}/download").status_code in (400, 404)


def test_schedule_is_validated(monkeypatch, instance):
    saved = {}
    monkeypatch.setattr("src.settings.load_settings", lambda: {})
    monkeypatch.setattr("src.settings.save_settings", lambda s: saved.update(s))
    client = _client(monkeypatch, admin=True)

    bad = client.put("/api/backups/schedule", json={"frequency": "hourly"})
    good = client.put("/api/backups/schedule", json={"frequency": "weekly", "time": "04:30", "weekday": 2,
                                                     "retention": 5, "timezone": "UTC"})

    assert bad.status_code == 400
    assert good.status_code == 200
    assert saved["backup_schedule"]["frequency"] == "weekly" and good.json()["next_run"]

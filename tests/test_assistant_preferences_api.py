"""Exercise conversation settings at the HTTP and database boundaries."""
from types import SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import core.database as database
from routes import session_routes


@pytest.fixture
def client(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", factory)
    monkeypatch.setattr(session_routes, "SessionLocal", factory)
    monkeypatch.setattr(session_routes, "router", APIRouter(prefix="/api"))
    monkeypatch.setattr(session_routes, "effective_user", lambda request: request.headers.get("X-Test-User", "alice"))
    with factory() as db:
        db.add_all([database.Session(id=owner, name=owner, owner=owner, model="test", endpoint_url="https://provider.example/v1") for owner in ("alice", "bob")])
        db.commit()
    manager = SimpleNamespace(get_session=lambda sid: SimpleNamespace(model="test", endpoint_url="https://provider.example/v1"))
    session_routes.setup_session_routes(manager, {})
    app = FastAPI()
    app.include_router(session_routes.router)
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()


def test_existing_conversation_keeps_legacy_and_new_preferences_round_trip(client):
    assert client.get("/api/session/alice/assistant").json()["preferences"]["profile"] == "legacy"
    options = {"profile": "research", "web_mode": "on", "thinking": "high", "instructions": "Compare costs", "context_limits": {"local-model": 16384}}
    saved = client.put("/api/session/alice/assistant", json=options)
    assert saved.status_code == 200
    assert client.get("/api/session/alice/assistant").json()["preferences"] == options


def test_preferences_are_owner_scoped_and_invalid_updates_do_not_change_them(client):
    assert client.get("/api/session/bob/assistant").status_code == 404
    assert client.put("/api/session/bob/assistant", json={"profile": "actions"}).status_code == 404
    assert client.get("/api/session/missing/assistant").status_code == 404
    assert client.put("/api/session/alice/assistant", json={"context_limits": {"test": -1}}).status_code == 422
    assert client.put("/api/session/alice/assistant", content="{").status_code == 422
    assert client.get("/api/session/alice/assistant").json()["preferences"]["profile"] == "legacy"


def test_upgrade_preserves_existing_sessions_and_is_idempotent(monkeypatch):
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE sessions (id TEXT PRIMARY KEY, name TEXT)"))
        connection.execute(text("INSERT INTO sessions VALUES ('old', 'My chat')"))
    monkeypatch.setattr(database, "engine", engine)
    database._migrate_assistant_preferences()
    database._migrate_assistant_preferences()
    assert "assistant_preferences" in {c["name"] for c in inspect(engine).get_columns("sessions")}
    with engine.connect() as connection:
        assert connection.execute(text("SELECT name, assistant_preferences FROM sessions")).one() == ("My chat", None)
    engine.dispose()

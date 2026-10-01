"""Per-model performance summary: aggregation and owner scoping."""
import json
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import core.database as cdb
from core.database import ChatMessage as DbChatMessage, Session as DbSession
import routes.model_performance_routes as mpr


def _row(meta, *, model="qwen3:27b", endpoint_url="http://host.docker.internal:11434", stamp=None):
    return {
        "metadata": meta,
        "timestamp": stamp or datetime(2026, 9, 1, 12, 0, 0),
        "endpoint_url": endpoint_url,
        "model": model,
    }


def test_aggregates_speed_ttft_and_reloads_per_endpoint_and_model():
    rows = [
        _row({"model": "qwen3:27b", "endpoint_id": "ep1", "endpoint_label": "Ollama",
              "time_to_first_token": ttft, "tokens_per_second": tps, "tps_source": "backend",
              "prefill_tps": 600.0, "load_ms": load})
        for ttft, tps, load in [(2.0, 40.0, 80.0), (20.0, 38.0, 18000.0), (3.0, 42.0, 60.0), (4.0, 41.0, 900.0)]
    ]
    rows.append(_row({"model": "gemma4:26b", "endpoint_id": "ep1", "endpoint_label": "Ollama",
                      "time_to_first_token": 1.0, "tokens_per_second": 70.0, "tps_source": "backend"}))

    summary = mpr.aggregate_model_performance(rows)

    assert [item["model"] for item in summary] == ["qwen3:27b", "gemma4:26b"]
    qwen = summary[0]
    assert qwen["messages"] == 4
    assert qwen["endpoint_label"] == "Ollama"
    assert qwen["ttft_median_s"] == 3.5
    assert qwen["ttft_p90_s"] == 20.0
    assert qwen["gen_tps_median"] == 40.5
    assert qwen["gen_tps_source"] == "backend"
    assert qwen["prompt_tps_median"] == 600.0
    assert qwen["load_median_s"] == 0.49
    assert qwen["load_avg_s"] == 4.76
    assert qwen["reloads"] == 2
    assert qwen["last_used"] == "2026-09-01T12:00:00Z"


def test_wall_clock_speed_is_only_used_when_backend_never_reported_one():
    summary = mpr.aggregate_model_performance([
        _row({"model": "cloud-model", "tokens_per_second": 12.0, "tps_source": "computed"}),
        _row({"model": "mixed", "tokens_per_second": 9.0, "tps_source": "computed"}),
        _row({"model": "mixed", "gen_tps": 50.0}),
    ])
    by_model = {item["model"]: item for item in summary}

    assert by_model["cloud-model"]["gen_tps_median"] == 12.0
    assert by_model["cloud-model"]["gen_tps_source"] == "computed"
    assert by_model["mixed"]["gen_tps_median"] == 50.0
    assert by_model["mixed"]["gen_tps_source"] == "backend"


def test_plain_chat_ttft_falls_back_to_backend_load_plus_prompt_time():
    summary = mpr.aggregate_model_performance([
        _row({"model": "m", "gen_tps": 30.0, "load_ms": 1500.0, "prefill_ms": 500.0}),
    ])

    assert summary[0]["ttft_median_s"] == 2.0


def test_failed_stopped_and_metric_less_replies_are_skipped():
    summary = mpr.aggregate_model_performance([
        _row({"model": "m", "failed": True, "tokens_per_second": 1.0}),
        _row({"model": "m", "stopped": True, "tokens_per_second": 1.0}),
        _row({"model": "m", "total_time": 0}),
        _row(None),
    ])

    assert summary == []


def test_endpoint_label_falls_back_to_host_without_credentials():
    summary = mpr.aggregate_model_performance([
        _row({"gen_tps": 10.0}, endpoint_url="http://user:secret@10.0.0.5:11434/v1?key=x"),
    ])

    assert summary[0]["endpoint_label"] == "10.0.0.5:11434"
    assert summary[0]["model"] == "qwen3:27b"


@pytest.fixture
def db(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    cdb.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(mpr, "SessionLocal", factory)
    return factory


def _seed(factory, owner, model, meta, *, age_days=1, role="assistant"):
    session = factory()
    try:
        sid = str(uuid.uuid4())
        session.add(DbSession(id=sid, owner=owner, name="chat", endpoint_url="http://localhost:11434", model=model))
        session.add(DbChatMessage(
            id=str(uuid.uuid4()), session_id=sid, role=role, content="x",
            meta_data=json.dumps(meta), timestamp=cdb.utcnow_naive() - timedelta(days=age_days),
        ))
        session.commit()
    finally:
        session.close()


def _endpoint():
    router = mpr.setup_model_performance_routes()
    return next(r.endpoint for r in router.routes if r.path == "/api/model-performance")


def test_route_only_aggregates_the_callers_recent_assistant_replies(db, monkeypatch):
    _seed(db, "alice", "alice-model", {"gen_tps": 40.0})
    _seed(db, "alice", "old-model", {"gen_tps": 40.0}, age_days=90)
    _seed(db, "alice", "user-turn", {"gen_tps": 40.0}, role="user")
    _seed(db, "bob", "bob-model", {"gen_tps": 10.0})
    _seed(db, None, "shared-model", {"gen_tps": 10.0})
    monkeypatch.setattr(mpr, "require_user", lambda request: "alice")

    result = _endpoint()(SimpleNamespace(), days=30, limit=100)

    assert result["messages_scanned"] == 1
    assert [item["model"] for item in result["models"]] == ["alice-model"]


def test_route_refuses_unidentified_callers_unless_auth_is_disabled(db, monkeypatch):
    _seed(db, "bob", "bob-model", {"gen_tps": 10.0})
    monkeypatch.setattr(mpr, "require_user", lambda request: "")
    monkeypatch.setattr(mpr, "auth_disabled", lambda: False)

    with pytest.raises(HTTPException) as exc:
        _endpoint()(SimpleNamespace(), days=30, limit=100)
    assert exc.value.status_code == 401

    monkeypatch.setattr(mpr, "auth_disabled", lambda: True)
    result = _endpoint()(SimpleNamespace(), days=30, limit=100)
    assert [item["model"] for item in result["models"]] == ["bob-model"]

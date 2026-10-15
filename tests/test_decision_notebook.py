"""Decision notebook: scoring, revisit, snapshots, validation and owner scoping.

The scoring/comparison cases live in tests/fixtures/decision_scoring_cases.json
and are asserted here against src/decisions.py and in
tests/decisionScoring.test.mjs against the browser mirror, so the live editor
and the server cannot drift apart.

Route tests drive the ASGI app through ``httpx.ASGITransport`` with a
pure-ASGI identity shim and a non-loopback peer, the same harness as
test_notes_fail_closed_auth.py (see its docstring for why not TestClient).
"""
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import core.database as cdb
from core.database import Decision, DecisionSnapshot
from routes.decision import decision_routes as dr
from src import decisions as dm
from src.url_safety import check_link_url

CASES = json.loads((Path(__file__).parent / "fixtures" / "decision_scoring_cases.json").read_text())
_PEER = ("203.0.113.7", 54321)
TODAY = "2026-10-12"


# ---------------------------------------------------------------------------
# Pure domain logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES["scoring"], ids=lambda c: c["name"])
def test_weighted_scoring_matches_shared_cases(case):
    result = dm.score_options(case["body"])
    expect = case["expect"]
    assert result["weights"] == expect["weights"]
    assert {oid: r["total"] for oid, r in result["options"].items()} == expect["totals"]
    assert result["ranking"] == expect["ranking"]
    assert result["leader_id"] == expect["leader_id"]
    assert result["tied"] == expect["tied"]
    for oid, violations in expect.get("violations", {}).items():
        assert result["options"][oid]["violations"] == violations
        assert result["options"][oid]["eligible"] is (not violations)
    for oid, unscored in expect.get("unscored", {}).items():
        assert result["options"][oid]["unscored"] == unscored


@pytest.mark.parametrize("case", CASES["comparison"], ids=lambda c: c["name"])
def test_since_decision_comparison_matches_shared_cases(case):
    result = dm.compare_to_snapshot(case["body"], case["snapshot"])
    for key, value in case["expect"].items():
        assert result[key] == value, key


def test_built_snapshot_carries_the_fields_the_comparison_reads():
    case = CASES["comparison"][0]
    body = dm.normalize_body(CASES["scoring"][0]["body"])
    snap = dm.build_snapshot(body, title="Laptop", chosen_option_id="a", rationale="cheaper",
                             decided_at="2026-10-01T09:00:00Z", revisit_on="2027-01-01")
    for key in ("chosen_option_id", "chosen_option_name", "ranking", "leader_id"):
        assert snap[key] == case["snapshot"][key]
    assert [(c["id"], c["share"]) for c in snap["criteria"]] == [("cost", 0.75), ("speed", 0.25)]
    assert {o["id"]: o["total"] for o in snap["options"]} == {"a": 7, "b": 6.25}


def test_snapshot_is_a_deep_copy_of_the_body():
    body = dm.normalize_body(CASES["scoring"][0]["body"])
    snap = dm.build_snapshot(body, title="t", chosen_option_id="a", rationale="",
                             decided_at="2026-10-01T09:00:00Z", revisit_on=None)
    body["options"][0]["scores"]["cost"] = 0
    body["criteria"][0]["weight"] = 0
    assert snap["options"][0]["scores"]["cost"] == 8
    assert snap["criteria"][0]["weight"] == 3


@pytest.mark.parametrize("revisit_on,status,due", [
    ("2026-10-12", "decided", True),    # today
    ("2026-09-30", "open", True),       # past
    ("2026-10-13", "decided", False),   # future
    ("2026-09-30", "archived", False),  # archived never nags
    (None, "decided", False),
])
def test_revisit_due_is_today_or_past_and_not_archived(revisit_on, status, due):
    assert dm.is_revisit_due(revisit_on, status, TODAY) is due
    assert dm.effective_status(status, revisit_on, TODAY) == ("revisit-due" if due else status)


@pytest.mark.parametrize("url", [
    "javascript:alert(1)", "data:text/html,hi", "ftp://example.com/x", "file:///etc/passwd",
    "https://trusted.example@evil.example/", "https://", "example.com", "https://exa mple.com",
    "https://example.com/\x07", "",
])
def test_link_urls_reject_non_http_and_ambiguous_forms(url):
    ok, _ = check_link_url(url)
    assert not ok


def test_link_urls_accept_http_and_https_without_dns():
    # Unresolvable host on purpose: stored links are never fetched.
    assert check_link_url("https://does-not-resolve.invalid/path?q=1#f") == (True, "ok")
    assert check_link_url("http://192.168.1.10:8080/")[0]


def test_normalize_drops_scores_and_markers_for_deleted_columns():
    body = dm.normalize_body({
        "constraints": [{"id": "m", "text": "x", "must_have": True}],
        "criteria": [{"id": "k", "name": "K", "weight": 2}],
        "options": [{"id": "o", "name": "O", "scores": {"k": 5, "gone": 9}, "fails": ["m", "gone"]}],
    })
    assert body["options"][0]["scores"] == {"k": 5.0}
    assert body["options"][0]["fails"] == ["m"]


@pytest.mark.parametrize("bad", [
    {"criteria": [{"id": "k", "name": "K", "weight": 101}]},
    {"criteria": [{"id": "k", "name": "K", "weight": float("nan")}]},
    {"criteria": [{"id": "k", "name": "K"}], "options": [{"id": "o", "scores": {"k": 11}}]},
    {"criteria": [{"id": "k", "name": "K"}], "options": [{"id": "o", "scores": {"k": True}}]},
    {"criteria": [{"id": "k"}, {"id": "k"}]},
    {"criteria": [{"id": "bad id!"}]},
    {"question": "x" * (dm.MAX_QUESTION + 1)},
    {"options": [{"id": f"o{i}"} for i in range(dm.MAX_OPTIONS + 1)]},
    {"sources": [{"title": "x", "url": "javascript:alert(1)"}]},
    {"options": [{"id": "o", "links": [{"title": "x", "url": "data:text/html,1"}]}]},
])
def test_normalize_rejects_invalid_bodies(bad):
    with pytest.raises(dm.DecisionValidationError):
        dm.normalize_body(bad)


def test_markdown_export_escapes_table_cells_and_wraps_links():
    decision = {
        "title": "Pick | a laptop",
        **dm.normalize_body({
            "question": "Which one?",
            "criteria": [{"id": "c", "name": "Cost|Value", "weight": 1}],
            "options": [{"id": "a", "name": "A|1", "scores": {"c": 7},
                         "links": [{"title": "Review [1]", "url": "https://example.com/a_(b)"}]}],
        }),
        "chosen_option_id": "a",
        "rationale": "Best value.",
    }
    full = dm.to_markdown(decision)
    assert "| Cost\\|Value |" in full
    assert "| 1 | A\\|1 | 7.00 | 7 |" in full
    assert "[Review \\[1\\]](<https://example.com/a_(b)>)" in full
    brief = dm.to_markdown(decision, brief=True)
    assert "Current choice: A|1 - Best value." in brief
    assert "1. A|1 - 7.00" in brief


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class _Identity:
    """Pure-ASGI stand-in for the auth middleware's request.state fields."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            state = scope.setdefault("state", {})
            user = headers.get(b"x-test-user")
            if user:
                state["current_user"] = user.decode()
            if headers.get(b"x-test-api-token"):
                state["current_user"] = "api"
                state["api_token"] = True
        await self.app(scope, receive, send)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Configured auth (AUTH_ENABLED=true, no loopback bypass) over a temp DB."""
    engine = create_engine(f"sqlite:///{tmp_path / 'decisions.db'}",
                           connect_args={"check_same_thread": False}, poolclass=NullPool)
    cdb.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(dr, "SessionLocal", factory)
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.delenv("LOCALHOST_BYPASS", raising=False)
    app = FastAPI()
    app.state.auth_manager = SimpleNamespace(is_configured=True)
    app.include_router(dr.setup_decision_routes())
    return _Identity(app), factory


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=_PEER),
                             base_url="http://decisions.test")


ALICE = {"x-test-user": "alice"}
BOB = {"x-test-user": "bob"}
LAPTOP = {
    "title": "Laptop",
    "question": "Which laptop should I buy?",
    "constraints": [{"id": "budget", "text": "Under 1500", "must_have": True}],
    "criteria": [{"id": "cost", "name": "Cost", "weight": 3}, {"id": "speed", "name": "Speed", "weight": 1}],
    "options": [
        {"id": "a", "name": "Air", "scores": {"cost": 8, "speed": 4},
         "pros": ["light"], "cons": ["slow"], "links": [{"title": "Review", "url": "https://example.com/air"}]},
        {"id": "b", "name": "Pro", "scores": {"cost": 5, "speed": 10}},
    ],
    "sources": [{"title": "Prices", "url": "https://example.com/prices"}],
}


async def _create(c, headers=ALICE, **overrides):
    r = await c.post(f"/api/decisions?today={TODAY}", json={**LAPTOP, **overrides}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


async def test_crud_round_trip_with_computed_scoring(env):
    app, _ = env
    async with _client(app) as c:
        created = await _create(c)
        did = created["id"]
        assert created["status"] == "open"
        assert created["scoring"]["leader_id"] == "a"
        assert created["scoring"]["options"]["a"]["total"] == 7
        assert created["history"] == [] and created["since_decision"] is None

        r = await c.put(f"/api/decisions/{did}?today={TODAY}", headers=ALICE,
                        json={"title": "New laptop", "options": [{"id": "b", "name": "Pro", "scores": {"cost": 9}}]})
        assert r.status_code == 200
        updated = r.json()
        assert updated["title"] == "New laptop"
        assert [o["id"] for o in updated["options"]] == ["b"]
        assert updated["criteria"] == created["criteria"]  # untouched fields survive a partial update

        got = (await c.get(f"/api/decisions/{did}", headers=ALICE)).json()
        assert got["title"] == "New laptop"

        assert (await c.delete(f"/api/decisions/{did}", headers=ALICE)).json() == {"ok": True}
        assert (await c.get(f"/api/decisions/{did}", headers=ALICE)).status_code == 404


async def test_owner_isolation_hides_other_accounts_decisions(env):
    app, _ = env
    async with _client(app) as c:
        mine = await _create(c, ALICE)
        theirs = await _create(c, BOB, title="Bob's")
        listed = (await c.get("/api/decisions", headers=ALICE)).json()["decisions"]
        assert [d["id"] for d in listed] == [mine["id"]]
        tid = theirs["id"]
        assert (await c.get(f"/api/decisions/{tid}", headers=ALICE)).status_code == 404
        assert (await c.put(f"/api/decisions/{tid}", json={"title": "pwn"}, headers=ALICE)).status_code == 404
        assert (await c.post(f"/api/decisions/{tid}/decide", json={"option_id": "a"}, headers=ALICE)).status_code == 404
        assert (await c.get(f"/api/decisions/{tid}/export", headers=ALICE)).status_code == 404
        assert (await c.delete(f"/api/decisions/{tid}", headers=ALICE)).status_code == 404
        still = (await c.get(f"/api/decisions/{tid}", headers=BOB)).json()
        assert still["title"] == "Bob's" and still["status"] == "open"


async def test_no_identity_fails_closed_and_api_tokens_are_rejected(env):
    app, factory = env
    async with _client(app) as c:
        did = (await _create(c))["id"]
        assert (await c.get("/api/decisions")).status_code == 401
        assert (await c.post("/api/decisions", json={"title": "ghost"})).status_code == 401
        assert (await c.get(f"/api/decisions/{did}")).status_code == 401
        assert (await c.put(f"/api/decisions/{did}", json={"title": "pwn"})).status_code == 401
        assert (await c.post(f"/api/decisions/{did}/decide", json={"option_id": "a"})).status_code == 401
        assert (await c.delete(f"/api/decisions/{did}")).status_code == 401
        assert (await c.get("/api/decisions", headers={"x-test-api-token": "1"})).status_code == 403
    db = factory()
    try:
        assert [(d.title, d.status) for d in db.query(Decision).all()] == [("Laptop", "open")]
    finally:
        db.close()


async def test_decide_snapshot_is_immutable_and_redeciding_appends(env):
    app, factory = env
    async with _client(app) as c:
        did = (await _create(c))["id"]
        r = await c.post(f"/api/decisions/{did}/decide?today={TODAY}", headers=ALICE,
                         json={"option_id": "a", "rationale": "Cost matters most", "revisit_on": "2027-01-01"})
        assert r.status_code == 200, r.text
        decided = r.json()
        assert decided["status"] == "decided" and decided["chosen_option_id"] == "a"
        assert decided["revisit_on"] == "2027-01-01"
        first = decided["history"][0]
        assert first["seq"] == 1 and first["chosen_option_name"] == "Air"
        assert decided["since_decision"]["changed"] is False
        assert decided["since_decision"]["chosen_still_wins"] is True

        # Re-weight speed and rescore: the working copy changes, the snapshot does not.
        r = await c.put(f"/api/decisions/{did}", headers=ALICE, json={
            "criteria": [{"id": "cost", "name": "Cost", "weight": 1}, {"id": "speed", "name": "Speed", "weight": 3}],
            "options": [{"id": "a", "name": "Air", "scores": {"cost": 8, "speed": 2}},
                        {"id": "b", "name": "Pro", "scores": {"cost": 5, "speed": 10}}],
        })
        after = r.json()
        assert after["history"][0] == first
        since = after["since_decision"]
        assert since["chosen_still_wins"] is False
        assert since["leader_after"] == "Pro"
        assert since["leader_under_original_weights"] == "Air"
        assert {"option": "Air", "criterion": "Speed", "before": 4.0, "after": 2.0} in since["score_changes"]

        r = await c.post(f"/api/decisions/{did}/decide", headers=ALICE,
                         json={"option_id": "b", "rationale": "Speed now matters"})
        again = r.json()
        assert [h["seq"] for h in again["history"]] == [1, 2]
        assert again["history"][0] == first
        assert again["history"][1]["chosen_option_name"] == "Pro"
        assert again["revisit_on"] == "2027-01-01"  # omitted revisit_on keeps the date
        assert again["since_decision"]["changed"] is False

    db = factory()
    try:
        assert db.query(DecisionSnapshot).filter(DecisionSnapshot.decision_id == did).count() == 2
    finally:
        db.close()


async def test_deleting_a_decision_removes_its_snapshots(env):
    app, factory = env
    async with _client(app) as c:
        did = (await _create(c))["id"]
        await c.post(f"/api/decisions/{did}/decide", headers=ALICE, json={"option_id": "a"})
        await c.delete(f"/api/decisions/{did}", headers=ALICE)
    db = factory()
    try:
        assert db.query(DecisionSnapshot).count() == 0
    finally:
        db.close()


async def test_list_surfaces_revisit_due_first_with_count(env):
    app, _ = env
    async with _client(app) as c:
        later = await _create(c, title="Later", revisit_on="2026-12-01")
        due = await _create(c, title="Due", revisit_on="2026-10-12")
        await _create(c, title="Undated")
        archived = await _create(c, title="Archived", revisit_on="2026-01-01")
        await c.put(f"/api/decisions/{archived['id']}", headers=ALICE, json={"status": "archived"})

        data = (await c.get(f"/api/decisions?today={TODAY}", headers=ALICE)).json()
        assert data["revisit_due_count"] == 1
        assert data["decisions"][0]["id"] == due["id"]
        assert data["decisions"][0]["effective_status"] == "revisit-due"
        assert data["decisions"][-1]["title"] == "Archived"
        assert data["decisions"][-1]["revisit_due"] is False

        # The client's "today" decides: the same row becomes due a day later.
        tomorrow = (await c.get("/api/decisions?today=2026-12-01", headers=ALICE)).json()
        assert tomorrow["revisit_due_count"] == 2
        assert {d["id"] for d in tomorrow["decisions"][:2]} == {due["id"], later["id"]}
        assert (await c.get("/api/decisions?today=yesterday", headers=ALICE)).status_code == 400


@pytest.mark.parametrize("payload", [
    {"sources": [{"title": "x", "url": "javascript:alert(1)"}]},
    {"options": [{"id": "a", "links": [{"title": "x", "url": "ftp://example.com"}]}]},
    {"revisit_on": "2026-02-30"},
    {"revisit_on": "next week"},
    {"criteria": [{"id": "c", "name": "C", "weight": -1}]},
    {"title": "t" * (dm.MAX_TITLE + 1)},
])
async def test_invalid_input_is_a_400_and_not_stored(env, payload):
    app, factory = env
    async with _client(app) as c:
        r = await c.post("/api/decisions", headers=ALICE, json=payload)
        assert r.status_code == 400, r.text
        did = (await _create(c))["id"]
        r = await c.put(f"/api/decisions/{did}", headers=ALICE, json=payload)
        assert r.status_code == 400, r.text
        assert (await c.get(f"/api/decisions/{did}", headers=ALICE)).json()["title"] == "Laptop"


async def test_decide_and_status_guards(env):
    app, _ = env
    async with _client(app) as c:
        did = (await _create(c))["id"]
        r = await c.put(f"/api/decisions/{did}", headers=ALICE, json={"status": "decided"})
        assert r.status_code == 400
        r = await c.put(f"/api/decisions/{did}", headers=ALICE, json={"status": "revisit-due"})
        assert r.status_code == 400
        r = await c.post(f"/api/decisions/{did}/decide", headers=ALICE, json={"option_id": "nope"})
        assert r.status_code == 400
        assert (await c.get(f"/api/decisions/{did}", headers=ALICE)).json()["history"] == []


async def test_export_markdown_full_brief_and_download(env):
    app, _ = env
    async with _client(app) as c:
        did = (await _create(c))["id"]
        await c.post(f"/api/decisions/{did}/decide", headers=ALICE,
                     json={"option_id": "a", "rationale": "Cheaper", "revisit_on": "2027-01-01"})
        full = await c.get(f"/api/decisions/{did}/export", headers=ALICE)
        assert full.status_code == 200
        assert full.headers["content-type"].startswith("text/markdown")
        assert full.text.startswith("# Laptop\n")
        assert "| 1 | Air | 7.00 | 8 | 4 |  |" in full.text
        assert "**Must-have:** Under 1500" in full.text
        assert "## History" in full.text and "chose **Air** (7.00/10)" in full.text
        assert "content-disposition" not in full.headers

        brief = await c.get(f"/api/decisions/{did}/export?brief=true", headers=ALICE)
        assert "Current choice: Air - Cheaper" in brief.text
        assert "## History" not in brief.text

        dl = await c.get(f"/api/decisions/{did}/export?download=true", headers=ALICE)
        assert dl.headers["content-disposition"] == 'attachment; filename="laptop.md"'


async def test_auth_disabled_keeps_single_user_mode_working(env, monkeypatch):
    app, _ = env
    monkeypatch.setenv("AUTH_ENABLED", "false")
    async with _client(app) as c:
        created = (await c.post("/api/decisions", json={"title": "Solo"})).json()
        assert created["title"] == "Solo"
        listed = (await c.get("/api/decisions")).json()["decisions"]
        assert [d["id"] for d in listed] == [created["id"]]

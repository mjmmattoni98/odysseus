"""Email poller JSON calls pass response_schema and still parse replies from
providers that ignore it (prose-wrapped JSON)."""
import json
import sqlite3

import pytest


RAW_EMAIL = (
    b"From: Sam <sam@example.com>\r\n"
    b"To: Alice <alice@example.com>\r\n"
    b"Subject: Call on Friday\r\n"
    b"Message-ID: <call@example.com>\r\n"
    b"Date: Tue, 01 Jan 2026 12:00:00 +0000\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    + (b"Can we have a call on Friday at 10am to review the contract? " * 4)
)


class _FakeImap:
    def select(self, folder, readonly=True):
        return ("OK", []) if "INBOX" in str(folder) else ("NO", [])

    def uid(self, command, *args):
        if command == "SEARCH":
            return "OK", [b"1"]
        if command == "FETCH":
            return "OK", [(b"1 (RFC822)", RAW_EMAIL)]
        raise AssertionError(f"unexpected uid command: {command!r}")

    def logout(self):
        pass


@pytest.mark.asyncio
async def test_classification_and_calendar_calls_pass_schemas(tmp_path, monkeypatch):
    import core.database as database
    import routes.email_helpers as email_helpers
    import routes.email_pollers as email_pollers
    import src.tool_implementations as tool_impl

    db_path = tmp_path / "scheduled_emails.db"
    monkeypatch.setattr(email_helpers, "SCHEDULED_DB", db_path)
    monkeypatch.setattr(email_pollers, "SCHEDULED_DB", db_path)
    email_helpers._init_scheduled_db()

    monkeypatch.setattr(email_pollers, "_load_settings",
                        lambda: {"email_auto_tag": True, "email_auto_calendar": True})
    monkeypatch.setattr(email_pollers, "_owner_for_email_account", lambda _a: "alice")
    monkeypatch.setattr(email_pollers, "_imap_connect", lambda account_id=None, owner="": _FakeImap())
    monkeypatch.setattr(email_pollers, "_get_email_config",
                        lambda account_id=None, owner="": {"from_address": "alice@example.com"})
    monkeypatch.setattr(email_pollers, "resolve_task_candidates",
                        lambda owner=None: [("http://localhost:11434", "m", {})])
    monkeypatch.setattr(database, "get_upcoming_events", lambda *a, **k: [])

    calendar_ops = []

    async def fake_manage_calendar(args, owner=None):
        calendar_ops.append(json.loads(args))
        return {"exit_code": 0, "uid": "evt-1"}

    monkeypatch.setattr(tool_impl, "do_manage_calendar", fake_manage_calendar)

    calls = []

    async def fake_task_llm(messages, **kwargs):
        system = messages[0]["content"]
        calls.append((system, kwargs))
        # Providers that ignore the schema wrap the JSON in prose.
        if "Classify the email" in system:
            return 'Sure, here you go: {"tags": ["work", "promo"], "spam": false, "reason": "client call"} Thanks!'
        if "calendar assistant" in system:
            return ('Here are the operations:\n'
                    '[{"action": "create", "title": "Call with Sam", "date": "2026-01-02T10:00:00"}]')
        raise AssertionError("unexpected LLM call")

    monkeypatch.setattr(email_pollers, "task_llm_call_async", fake_task_llm)

    async def no_sleep(_s):
        return None

    monkeypatch.setattr("asyncio.sleep", no_sleep)

    await email_pollers._auto_summarize_pass_single(account_id="acct-alice")

    by_kind = {("class" if "Classify the email" in s else "cal"): kw for s, kw in calls}
    assert by_kind["class"]["response_schema"] == email_pollers._EMAIL_CLASS_SCHEMA
    assert by_kind["class"]["think"] is False
    assert by_kind["cal"]["response_schema"] == email_pollers._CAL_OPS_SCHEMA
    assert "think" not in by_kind["cal"]  # long extraction keeps configured thinking

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT tags, spam_verdict FROM email_tags WHERE message_id=?",
                           ("<call@example.com>",)).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert json.loads(row[0]) == ["work", "marketing"]
    assert row[1] == 0
    assert calendar_ops and calendar_ops[0]["action"] == "create_event"
    assert calendar_ops[0]["summary"] == "Call with Sam"


def test_email_schemas_match_parsers():
    import routes.email_pollers as email_pollers

    cal = email_pollers._CAL_OPS_SCHEMA
    assert cal["type"] == "array"
    assert cal["items"]["properties"]["action"]["enum"] == ["create", "update", "cancel", "noop"]
    urgency = email_pollers._URGENCY_SCHEMA["properties"]["urgency"]["enum"]
    assert urgency == ["critical", "high", "medium", "low", "none"]
    cls = email_pollers._EMAIL_CLASS_SCHEMA["properties"]
    assert cls["spam"]["type"] == "boolean"
    assert "promo" in cls["tags"]["items"]["enum"]
    text = json.dumps([cal, email_pollers._URGENCY_SCHEMA, email_pollers._EMAIL_CLASS_SCHEMA])
    for banned in ("oneOf", "anyOf", "$ref"):
        assert banned not in text

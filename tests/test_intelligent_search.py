"""Decision transport, overrides and actual chat-route search permissions."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from core.models import ChatMessage
from src import intelligent_search
from src.assistant_preferences import AssistantPreferences
from src.intelligent_search import SearchDecision, decide_search_for_turn, decision_payload, parse_decision, previous_search_evidence


def response_data(search=0.96):
    return {"answers": {"search": {
        "type": "choice", "choice": "search" if search >= 0.5 else "no_search",
        "probabilities": {"search": search, "no_search": 1 - search},
    }}}


@pytest.mark.parametrize("message", ["Did you search the web?", "Which sources did you use?", "Show me the sources", "¿Has buscado en internet?"])
@pytest.mark.asyncio
async def test_search_record_questions_reuse_evidence_without_another_lookup(monkeypatch, session, message):
    sources = [{"url": "https://news.example/article", "read_status": "failed", "snippet_available": True}]
    session.history = [ChatMessage("assistant", "News summary", metadata={
        "web_sources": sources, "search_reports": [{"state": "ok", "pages_read": 0, "results": 1}],
    })]
    monkeypatch.setattr(intelligent_search, "_decision_endpoint", lambda *a: pytest.fail("A saved search record does not need d1"))
    decision = await decide_search_for_turn(AssistantPreferences(), session, message)
    assert decision.needs_search is False
    evidence = previous_search_evidence(message, session.history)
    assert evidence["searched"] is True
    assert evidence["pages_read"] == 0
    assert evidence["sources"] == sources


def test_new_lookup_request_is_not_treated_as_a_question_about_previous_searches():
    assert previous_search_evidence("Did you search the web for current laptop prices? Search again.", []) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chat", "agent"])
async def test_search_confirmation_is_answered_without_exposing_search_tools(monkeypatch, mode):
    from routes import chat_routes
    from tests.test_foreground_model_routing import _RouteRequest, _chat_stream_endpoint

    captured = {}
    history = [ChatMessage("assistant", "News", metadata={"search_reports": [{"state": "ok", "results": 2, "pages_read": 0}]})]
    endpoint = _chat_stream_endpoint(monkeypatch, mode, captured, capture_context=True, session_history=history)
    monkeypatch.setattr(chat_routes, "load_preferences", lambda sid: AssistantPreferences(search_mode="auto"))
    monkeypatch.setattr(chat_routes, "coerce_message_and_session", lambda *a, **k: ("Did you search the web?", "session-1"))
    monkeypatch.setattr(intelligent_search, "_decision_endpoint", lambda *a: pytest.fail("No new lookup needed"))
    request = _RouteRequest(mode)
    request._form.pop("allow_web_search", None)
    response = await endpoint(request)
    async for _ in response.body_iterator:
        pass
    assert captured["build_context"]["use_web"] is False
    assert "chat" in captured and "agent" not in captured


def test_decision_includes_bounded_recent_context_and_latest_message():
    history = [ChatMessage("system", "secret")] + [ChatMessage("user", "x" * 2000)] * 8
    payload = decision_payload("y" * 10000, history)
    assert payload["model"] == "d1:3b"
    assert len(payload["state"]["recent_conversation"]) == 6
    assert all(len(item["content"]) == 1200 for item in payload["state"]["recent_conversation"])
    assert payload["state"]["latest_user_message"] == "y" * 6000
    assert "secret" not in json.dumps(payload)


@pytest.mark.parametrize(("probability", "expected"), [(0.98, True), (0.02, False), (0.52, None)])
def test_probability_decisions_and_uncertainty(probability, expected):
    decision = parse_decision(response_data(probability))
    assert decision.needs_search is expected
    assert decision.probability == probability


@pytest.mark.parametrize("probabilities", [
    {"search": True, "no_search": False}, {"search": "0.9", "no_search": 0.1},
    {"search": float("nan"), "no_search": 0.1}, {"search": 0.9, "no_search": 0.9},
    {"search": 1.1, "no_search": -0.1}, {"search": 0.1, "no_search": 0.9},
])
def test_malformed_or_inconsistent_probabilities_are_rejected(probabilities):
    data = response_data()
    data["answers"]["search"]["probabilities"] = probabilities
    with pytest.raises(ValueError):
        parse_decision(data)


@pytest.fixture
def session():
    return SimpleNamespace(endpoint_url="http://localhost:11434/v1", headers={}, history=[ChatMessage("user", "Prior topic")])


@pytest.mark.asyncio
@pytest.mark.parametrize(("kwargs", "expected"), [
    ({"preferences": AssistantPreferences(search_mode="off")}, False),
    ({"preferences": AssistantPreferences(search_mode="off"), "use_web": True, "allow_web_search": True}, False),
    ({"preferences": AssistantPreferences(search_mode="always")}, True),
    ({"allowed": False}, None),
    ({"allow_web_search": False}, False), ({"allow_web_search": "false", "use_web": True}, False),
    ({"allow_web_search": True}, None), ({"use_web": True}, None),
])
async def test_off_always_and_explicit_api_flags_do_not_invoke_d1(monkeypatch, session, kwargs, expected):
    def unexpected(*args):
        pytest.fail("d1 must not be invoked for this turn")
    monkeypatch.setattr(intelligent_search, "_decision_endpoint", unexpected)
    preferences = kwargs.pop("preferences", AssistantPreferences(search_mode="auto"))
    result = await decide_search_for_turn(preferences, session, "question", **kwargs)
    if expected is None:
        assert result is None
    else:
        assert result.needs_search is expected


def mock_client(monkeypatch, handler):
    client = httpx.AsyncClient
    monkeypatch.setattr(intelligent_search.httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(intelligent_search, "_decision_endpoint", lambda *args: ("http://localhost:11434", {"Authorization": "Bearer local-only"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("include_history", [True, False])
async def test_native_request_and_incognito_context(monkeypatch, session, include_history):
    def handler(request):
        assert str(request.url) == "http://localhost:11434/v1/systemone"
        assert request.headers["Authorization"] == "Bearer local-only"
        payload = json.loads(request.content)
        assert payload["state"]["latest_user_message"] == "And in Barcelona?"
        assert bool(payload["state"]["recent_conversation"]) is include_history
        return httpx.Response(200, json=response_data())
    mock_client(monkeypatch, handler)
    result = await decide_search_for_turn(AssistantPreferences(search_mode="auto"), session, "And in Barcelona?", include_history=include_history)
    assert result.needs_search is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing_model", "timeout", "malformed", "uncertain"])
async def test_provider_failures_preserve_standard_routing(monkeypatch, session, failure):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if failure == "missing_model":
            return httpx.Response(404, json={"error": "model missing"})
        return httpx.Response(200, json={} if failure == "malformed" else response_data(0.51))
    mock_client(monkeypatch, handler)
    result = await decide_search_for_turn(AssistantPreferences(search_mode="auto"), session, "question")
    assert result.needs_search is None


@pytest.mark.asyncio
async def test_cancelled_turn_cancels_classifier(monkeypatch, session):
    started = asyncio.Event()
    async def handler(request):
        started.set()
        await asyncio.Event().wait()
    mock_client(monkeypatch, handler)
    task = asyncio.create_task(decide_search_for_turn(AssistantPreferences(search_mode="auto"), session, "question"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_total_deadline_cancels_slow_classifier(monkeypatch, session):
    async def handler(request):
        await asyncio.Event().wait()
    mock_client(monkeypatch, handler)
    monkeypatch.setattr(intelligent_search, "DECISION_TIMEOUT", 0.02)
    result = await decide_search_for_turn(AssistantPreferences(search_mode="auto"), session, "question")
    assert result.needs_search is None


@pytest.mark.asyncio
async def test_total_deadline_includes_model_gate_wait(monkeypatch, session):
    from contextlib import asynccontextmanager
    from src import llm_core
    @asynccontextmanager
    async def busy(*args):
        await asyncio.Event().wait()
        yield
    def unexpected(request):
        pytest.fail("HTTP must not start while the model gate is busy")
    mock_client(monkeypatch, unexpected)
    monkeypatch.setattr(llm_core, "_local_model_slot", busy)
    monkeypatch.setattr(intelligent_search, "DECISION_TIMEOUT", 0.02)
    result = await decide_search_for_turn(AssistantPreferences(search_mode="auto"), session, "question")
    assert result.needs_search is None


def test_fallback_endpoint_is_enabled_owner_visible_and_has_its_own_credentials(monkeypatch, session):
    import core.database as database
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine("sqlite://")
    database.ModelEndpoint.__table__.create(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", factory)
    monkeypatch.delenv("ODYSSEUS_SEARCH_DECISION_URL", raising=False)
    with factory() as db:
        db.add_all([
            database.ModelEndpoint(id="a", name="Other owner", base_url="http://localhost:11434", owner="bob"),
            database.ModelEndpoint(id="b", name="Disabled", base_url="http://localhost:11434", is_enabled=False),
            database.ModelEndpoint(id="c", name="Own", base_url="http://127.0.0.1:11434/v1", owner="alice", api_key="own-test-key"),
        ])
        db.commit()
    session.endpoint_url = "https://provider.example/v1"
    session.headers = {"Authorization": "Bearer must-not-leak"}
    try:
        root, headers = intelligent_search._decision_endpoint(session, "alice")
        assert root == "http://127.0.0.1:11434"
        assert headers["Authorization"] == "Bearer own-test-key"
    finally:
        engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("needs_search", [True, False])
async def test_nonstream_chat_applies_and_persists_decision(monkeypatch, needs_search):
    from routes import chat_routes
    from src import llm_core
    from src.request_models import ChatRequest
    from tests.test_foreground_model_routing import _RouteRequest, _chat_endpoint
    endpoint, saved = _chat_endpoint(monkeypatch)
    monkeypatch.setattr(chat_routes, "load_preferences", lambda sid: AssistantPreferences(search_mode="auto"))
    build_context = chat_routes.build_chat_context
    observed = []
    async def capture(*args, **kwargs):
        observed.append(kwargs["use_web"])
        return await build_context(*args, **kwargs)
    async def decide(*args, **kwargs):
        return SearchDecision(needs_search)
    async def answer(*args, **kwargs):
        return "answer"
    monkeypatch.setattr(chat_routes, "build_chat_context", capture)
    monkeypatch.setattr(chat_routes, "decide_search_for_turn", decide)
    monkeypatch.setattr(llm_core, "llm_call_async", answer)
    result = await endpoint(_RouteRequest("chat"), ChatRequest(message="Question", session="session-1"))
    assert result["response"] == "answer"
    assert observed == [needs_search]
    assert saved[-1].metadata["search_reports"][0]["needs_search"] is needs_search


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "needs_search", "message"), [
    ("chat", True, "Is this a good laptop to buy?"),
    ("chat", False, "Explain recursion"),
    ("agent", False, "Rewrite this paragraph about today's news"),
])
async def test_chat_route_applies_d1_decision_to_tool_contract(monkeypatch, mode, needs_search, message):
    from routes import chat_routes
    from tests.test_foreground_model_routing import _RouteRequest, _chat_stream_endpoint
    captured = {}
    endpoint = _chat_stream_endpoint(monkeypatch, mode, captured)
    monkeypatch.setattr(chat_routes, "load_preferences", lambda sid: AssistantPreferences(search_mode="auto"))
    monkeypatch.setattr(chat_routes, "coerce_message_and_session", lambda *args, **kwargs: (message, "session-1"))
    calls = []
    async def decide(*args, **kwargs):
        calls.append(kwargs)
        return SearchDecision(needs_search, 0.99 if needs_search else 0.01)
    monkeypatch.setattr(chat_routes, "decide_search_for_turn", decide)
    async def agent(*args, **kwargs):
        captured["disabled"] = kwargs["disabled_tools"]
        captured["contract"] = kwargs.get("turn_contract")
        captured["authority"] = kwargs["request_authority"]
        yield 'data: {"delta":"done"}\n\n'
        yield "data: [DONE]\n\n"
    monkeypatch.setattr(chat_routes, "stream_agent_loop", agent)
    request = _RouteRequest(mode)
    request._form.pop("allow_web_search", None)
    request._form["message"] = message
    response = await endpoint(request)
    events = "".join([part async for part in response.body_iterator])
    assert len(calls) == 1
    assert '"needs_search": ' + str(needs_search).lower() in events
    if needs_search:
        from src.agent_runtime.authority import ExactOperation
        assert "web_search" not in captured["disabled"]
        assert captured["contract"].permits("web_search")
        assert captured["authority"].permits(ExactOperation.normalize("web_search", '{"query":"laptop reviews"}'))
        assert not captured["authority"].permits(ExactOperation.normalize("send_email", '{}'))
        assert not captured["authority"].permits(ExactOperation.normalize("private_browser", '{"action":"click","selector":"button"}'))
    elif "disabled" in captured:
        from src.agent_runtime.authority import ExactOperation
        assert "web_search" in captured["disabled"]
        assert not captured["contract"].permits("web_search")
        assert not captured["authority"].permits(ExactOperation.normalize("web_search", '{"query":"news"}'))
    else:
        assert "chat" in captured


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chat", "agent"])
@pytest.mark.parametrize("saved_mode", ["off", "always"])
async def test_saved_mode_controls_prefetch_and_request_authority(monkeypatch, mode, saved_mode):
    from routes import chat_routes
    from src.agent_runtime.authority import ExactOperation
    from src.tool_policy import WEB_ACCESS_TOOL_NAMES
    from tests.test_foreground_model_routing import _RouteRequest, _chat_stream_endpoint

    captured = {}
    endpoint = _chat_stream_endpoint(monkeypatch, mode, captured, capture_context=True)
    monkeypatch.setattr(chat_routes, "load_preferences", lambda sid: AssistantPreferences(search_mode=saved_mode))
    message = ("Open https://example.com in the browser and search the web for its latest news"
               if saved_mode == "off" else "What is 2 + 2?")
    monkeypatch.setattr(chat_routes, "coerce_message_and_session", lambda *args, **kwargs: (message, "session-1"))
    monkeypatch.setattr(intelligent_search, "_decision_endpoint", lambda *args: pytest.fail("Explicit modes must bypass d1"))

    async def agent(*args, **kwargs):
        captured["disabled"] = kwargs["disabled_tools"]
        captured["authority"] = kwargs["request_authority"]
        yield 'data: {"delta":"done"}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(chat_routes, "stream_agent_loop", agent)
    request = _RouteRequest(mode)
    request._form["message"] = message
    if saved_mode == "off":
        # Even contradictory legacy API flags cannot bypass saved Off.
        request._form.update({"use_web": "true", "allow_web_search": "true"})
    else:
        request._form.pop("allow_web_search", None)
    response = await endpoint(request)
    events = "".join([part async for part in response.body_iterator])
    enabled = saved_mode == "always"
    assert captured["build_context"]["use_web"] is enabled
    assert '"mode": "' + saved_mode + '"' in events
    authority = captured["authority"]
    assert authority.permits(ExactOperation.normalize("web_search", '{"query":"news"}')) is enabled
    if not enabled:
        assert WEB_ACCESS_TOOL_NAMES <= captured["disabled"]
        for tool, arguments in [("web_fetch", '{"url":"https://example.com"}'),
                                ("private_browser", '{"action":"open","url":"https://example.com"}')]:
            assert not authority.permits(ExactOperation.normalize(tool, arguments))

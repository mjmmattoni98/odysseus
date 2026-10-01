"""Agent context budget, local tool-payload cap, sticky tool sets and the
small-context tier (workstream 3).

Behavior under test:
- native tool schemas count toward the per-round trim budget and compaction;
- local routes get a token-budgeted tool payload that keeps pinned tools and
  lists exactly the sent tools in the prompt;
- the tool payload stays byte-identical across turns of one conversation;
- small windows get fewer retrieved tools, core domain tools only and short
  (still valid) schemas;
- text-mode Ollama routes get fenced tool syntax and no schemas.
"""
import asyncio
import json

import jsonschema
import pytest

import src.agent_loop as al
import src.model_context as model_context
import src.tool_index as tool_index
from src.context_budget import (
    estimate_tool_tokens,
    is_small_context_window,
    tool_output_char_cap,
    tool_schema_token_budget,
)
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS, small_context_tool_schema

LOCAL_URL = "http://localhost:11434/v1"
LOCAL_MODEL = "local-model:latest"


def _collect(gen):
    async def _run():
        return [chunk async for chunk in gen]

    return asyncio.run(_run())


def _names(tools):
    return [tool["function"]["name"] for tool in tools or []]


def _system_text(messages):
    return "\n".join(m.get("content") or "" for m in messages if m.get("role") == "system")


class _FakeIndex:
    """Deterministic stand-in for the embedding index."""

    def __init__(self, retrieved=()):
        self.retrieved = list(retrieved)
        self.calls = []

    def index_mcp_tools(self, *args, **kwargs):
        pass

    def get_tools_for_query(self, query, k=8, min_score=None):
        self.calls.append({"query": query, "k": k, "min_score": min_score})
        from src.tool_index import ALWAYS_AVAILABLE, ToolIndex

        return set(ALWAYS_AVAILABLE) | set(self.retrieved[:k]) | ToolIndex.keyword_tools_for_query(query)


@pytest.fixture
def local_route(monkeypatch):
    """Patch a native-tool local Ollama /v1 route with a configurable window."""
    state = {"window": 32768, "native": True, "index": _FakeIndex(), "requests": []}
    al._sticky_tool_sets.clear()
    monkeypatch.setattr(al, "get_mcp_manager", lambda: None)
    monkeypatch.setattr(al, "blocked_tools_for_owner", lambda owner: set())
    monkeypatch.setattr(
        al,
        "_agent_route_tool_mode",
        lambda url, model, owner=None, headers=None: (state["native"], False, True),
    )
    monkeypatch.setattr(
        model_context,
        "budget_context_for_model",
        lambda url, model, fallback=0: state["window"],
    )
    monkeypatch.setattr(tool_index, "get_tool_index", lambda: state["index"])

    async def fake_stream(candidates, messages, **kwargs):
        request = await kwargs["candidate_request_factory"](0, *candidates[0])
        state["requests"].append(request)
        yield 'data: {"delta": "ok"}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(al, "stream_llm_with_fallback", fake_stream)
    yield state
    al._sticky_tool_sets.clear()


def _run_turn(state, messages, **kwargs):
    kwargs.setdefault("session_id", "sess-ws3")
    kwargs.setdefault("max_rounds", 1)
    kwargs.setdefault("_is_teacher_run", True)
    _collect(al.stream_agent_loop(LOCAL_URL, LOCAL_MODEL, messages, **kwargs))
    request = state["requests"][-1]
    return request["kwargs"].get("tools") or [], request["messages"]


# ---------------------------------------------------------------------------
# Budget helpers
# ---------------------------------------------------------------------------

def test_tool_tokens_count_the_serialized_payload():
    small = [FUNCTION_TOOL_SCHEMAS[0]]
    big = [s for s in FUNCTION_TOOL_SCHEMAS if s["function"]["name"] == "ui_control"]
    assert estimate_tool_tokens([]) == 0
    assert estimate_tool_tokens(None) == 0
    assert estimate_tool_tokens(big) > estimate_tool_tokens(small) > 0
    assert estimate_tool_tokens(small + big) == estimate_tool_tokens(small) + estimate_tool_tokens(big)


def test_schema_budget_scales_with_window_within_bounds():
    assert tool_schema_token_budget(32768) == int(32768 * 0.15)
    assert tool_schema_token_budget(1_000_000) == 10000
    assert tool_schema_token_budget(4096) == 1200
    # Unknown local windows are budgeted like the default 32K allocation.
    assert tool_schema_token_budget(0) == tool_schema_token_budget(32768)


def test_small_tier_and_output_cap_only_for_known_small_windows():
    assert is_small_context_window(8192) and is_small_context_window(16384)
    assert not is_small_context_window(0)
    assert not is_small_context_window(32768)
    assert tool_output_char_cap(32768) is None
    # ~15% of the window, in characters (chars * 0.3 ~= tokens).
    assert 3900 < tool_output_char_cap(8192) <= int(8192 * 0.15 / 0.3)


def test_format_tool_result_scales_to_max_chars():
    from src.tool_execution import format_tool_result

    result = {"output": "x" * 20000 + "TAIL-ERROR", "exit_code": 1}
    capped = format_tool_result("bash: run", result, max_chars=3000)
    assert len(capped) < 3200
    assert "truncated to fit the model context" in capped
    assert capped.endswith("**exit_code:** 1")
    assert format_tool_result("bash: run", result) == format_tool_result("bash: run", result, max_chars=None)


# ---------------------------------------------------------------------------
# Item 1: schemas count in the trim budget and in compaction
# ---------------------------------------------------------------------------

def test_per_round_trim_reserves_the_tool_payload(local_route, monkeypatch):
    import src.context_compactor as context_compactor

    reserves = []

    def fake_trim(messages, context_length, reserve_tokens=512):
        reserves.append(reserve_tokens)
        return list(messages)

    monkeypatch.setattr(context_compactor, "trim_for_context", fake_trim)
    tools, _messages = _run_turn(
        local_route,
        [{"role": "user", "content": "search the web for the latest python release"}],
        max_tokens=1024,
    )
    assert tools
    assert reserves and reserves[0] == 1024 + estimate_tool_tokens(tools)


def test_compaction_counts_agent_overhead():
    import src.context_compactor as cc

    messages = [{"role": "system", "content": "sys"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "m" * 300}
        for i in range(8)
    ]
    calls = []

    async def fake_llm(*args, **kwargs):
        calls.append(1)
        return "summary"

    original = (cc.get_context_length, cc.llm_call_async, cc.resolve_endpoint)
    cc.get_context_length = lambda url, model: 1000
    cc.llm_call_async = fake_llm
    cc.resolve_endpoint = lambda *a, **k: (None, None, None)
    try:
        _m, _ctx, compacted = asyncio.run(cc.maybe_compact(None, "u", "m", messages, persist=False))
        assert compacted is False  # the history alone is under 85%
        _m, _ctx, compacted = asyncio.run(
            cc.maybe_compact(None, "u", "m", messages, persist=False, overhead_tokens=400)
        )
        assert compacted is True and calls
    finally:
        cc.get_context_length, cc.llm_call_async, cc.resolve_endpoint = original


# ---------------------------------------------------------------------------
# Item 2: token cap for local routes; prompt lists exactly the sent tools
# ---------------------------------------------------------------------------

def test_local_cap_keeps_pinned_and_forced_tools_and_rebuilds_tool_list(local_route):
    local_route["window"] = 20000  # budget 3000 tokens
    many = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS}
    tools, messages = _run_turn(
        local_route,
        [{"role": "user", "content": "do everything"}],
        relevant_tools=many,
        forced_tools={"web_search", "web_fetch"},
    )
    names = _names(tools)
    assert {"web_search", "web_fetch", "ask_user", "manage_memory", "update_plan"} <= set(names)
    assert len(names) <= al._LOCAL_TOOL_SCHEMA_COUNT_CEILING
    pinned_cost = estimate_tool_tokens(
        [t for t in tools if t["function"]["name"] in {"web_search", "web_fetch", "ask_user", "manage_memory", "update_plan"}]
    )
    assert estimate_tool_tokens(tools) <= max(tool_schema_token_budget(20000), pinned_cost)
    system = _system_text(messages)
    listed = [
        line[3:-1] for line in system.split("## Available tools\n", 1)[1].split("\n\n", 1)[0].splitlines()
    ]
    assert listed == names


def test_cloud_routes_are_not_capped(local_route, monkeypatch):
    local_route["window"] = 20000
    many = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS}
    _collect(al.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-test",
        [{"role": "user", "content": "do everything"}],
        relevant_tools=many, max_rounds=1, _is_teacher_run=True,
    ))
    tools = local_route["requests"][-1]["kwargs"].get("tools")
    assert len(tools) > al._LOCAL_TOOL_SCHEMA_COUNT_CEILING


# ---------------------------------------------------------------------------
# Item 3: sticky tool set per conversation
# ---------------------------------------------------------------------------

def test_sticky_tool_set_is_byte_identical_across_turns(local_route):
    history = [{"role": "user", "content": "check my inbox for unread emails"}]
    first, _ = _run_turn(local_route, history)
    history += [
        {"role": "assistant", "content": "You have two unread emails."},
        {"role": "user", "content": "thanks"},
    ]
    second, _ = _run_turn(local_route, history)
    history += [
        {"role": "assistant", "content": "Anything else?"},
        {"role": "user", "content": "ok, and the second one?"},
    ]
    third, _ = _run_turn(local_route, history)
    assert first and "list_emails" in _names(first)
    assert json.dumps(second) == json.dumps(first)
    assert json.dumps(third) == json.dumps(first)


def test_sticky_tool_set_appends_newly_needed_tools(local_route):
    history = [{"role": "user", "content": "remind me to call mom tomorrow"}]
    first, _ = _run_turn(local_route, history)
    history += [
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "also search the web for the weather in Madrid"},
    ]
    second, _ = _run_turn(local_route, history)
    first_names, second_names = _names(first), _names(second)
    assert "web_search" not in first_names and "web_search" in second_names
    # Earlier tools keep their positions; the new ones are appended.
    kept = [name for name in second_names if name in first_names]
    assert second_names[: len(kept)] == kept


def test_sticky_store_is_bounded():
    store = al._StickyToolSets(max_entries=2)
    store.put(("a",), ["x"])
    store.put(("b",), ["y"])
    store.get(("a",))
    store.put(("c",), ["z"])
    assert store.get(("b",)) is None
    assert store.get(("a",)) == ("x",) and store.get(("c",)) == ("z",)


# ---------------------------------------------------------------------------
# Item 4: small-context tier
# ---------------------------------------------------------------------------

def test_small_tier_limits_retrieval_and_skips_domain_packs(local_route):
    local_route["window"] = 8192
    local_route["index"] = _FakeIndex(
        retrieved=["bash", "python", "grep", "glob", "ls", "todowrite", "apply_patch", "read_file"]
    )
    tools, messages = _run_turn(
        local_route, [{"role": "user", "content": "check my inbox emails"}], session_id=None,
    )
    call = local_route["index"].calls[-1]
    assert call["k"] <= 4 and call["min_score"] is not None
    names = set(_names(tools))
    assert len(names) <= al._SMALL_CONTEXT_TOOL_COUNT_CEILING + 2  # + pinned
    assert "list_emails" in names
    assert not names & {"bulk_email", "archive_email", "mark_email_read", "unsubscribe_email", "manage_session"}
    assert "update_plan" not in names  # no approved plan
    assert estimate_tool_tokens(tools) <= tool_schema_token_budget(8192) + 500


def test_small_tier_skips_admin_expansion(local_route):
    local_route["window"] = 8192
    tools, _ = _run_turn(local_route, [{"role": "user", "content": "rename this session"}], session_id=None)
    assert not set(_names(tools)) & {"manage_endpoints", "manage_mcp", "manage_webhooks", "manage_tokens"}


def test_short_schemas_stay_valid_json_schema_and_keep_arguments():
    for schema in FUNCTION_TOOL_SCHEMAS:
        short = small_context_tool_schema(schema)
        params = short["function"].get("parameters")
        if params is not None:
            jsonschema.Draft202012Validator.check_schema(params)
            original = schema["function"]["parameters"]
            assert set(params.get("properties", {})) == set(original.get("properties", {}))
            assert params.get("required") == original.get("required")
        assert short["function"]["name"] == schema["function"]["name"]
    for name in ("ui_control", "manage_calendar", "manage_tasks", "manage_notes", "manage_skills", "ask_user"):
        full = next(s for s in FUNCTION_TOOL_SCHEMAS if s["function"]["name"] == name)
        assert estimate_tool_tokens([small_context_tool_schema(full)]) < 0.8 * estimate_tool_tokens([full])


def test_small_tier_caps_prefetched_search_context():
    from src.prompt_security import untrusted_context_message

    web = untrusted_context_message("web search results", "r" * 20000)
    other = {"role": "user", "content": "q" * 20000}
    capped = al._cap_presearch_context([web, other], 4000)
    assert len(capped[0]["content"]) < 4200
    assert capped[1] is other


# ---------------------------------------------------------------------------
# Item 5: whole-word admin keywords
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "restart the docker container",
    "open my jupyter notebook",
    "which tokenizer does qwen use",
    "is chatgpt better",
    "explain project management",
])
def test_admin_keywords_do_not_match_inside_words(text):
    assert al._detect_admin_intent([{"role": "user", "content": text}]) is False


@pytest.mark.parametrize("text", ["show my notes", "list my docs", "add an MCP server", "rename the chat"])
def test_admin_keywords_match_whole_words(text):
    assert al._detect_admin_intent([{"role": "user", "content": text}]) is True


# ---------------------------------------------------------------------------
# Item 6: trim order
# ---------------------------------------------------------------------------

def test_trim_drops_old_turns_before_cutting_the_system_prompt():
    from src.context_compactor import trim_for_context

    system = "## Tools\n" + ("tool rules " * 300) + "\n\n## FINAL-RULE\nnever cut me"
    messages = [{"role": "system", "content": system}]
    messages += [{"role": "user" if i % 2 == 0 else "assistant", "content": f"old-{i} " + "x" * 800} for i in range(20)]
    messages.append({"role": "user", "content": "latest"})
    trimmed = trim_for_context(messages, context_length=3000, reserve_tokens=200)
    assert trimmed[0]["content"] == system
    joined = "\n".join(str(m.get("content")) for m in trimmed)
    assert "old-0 " not in joined and trimmed[-1]["content"] == "latest"


def test_trim_stubs_consumed_tool_results_before_the_newest_one():
    from src.context_compactor import trim_for_context

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "OLD-RESULT " + "o" * 6000},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "b", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": "NEW-RESULT " + "n" * 3000},
    ]
    trimmed = trim_for_context(messages, context_length=1600, reserve_tokens=200)
    old = next(m for m in trimmed if m.get("tool_call_id") == "a")
    new = next(m for m in trimmed if m.get("tool_call_id") == "b")
    assert "OLD-RESULT" not in old["content"] and "omitted" in old["content"]
    assert new["content"] == messages[-1]["content"]


def test_system_prompt_is_cut_at_a_section_boundary():
    from src.context_compactor import _truncate_system_at_boundary

    text = "## A\nalpha rule\n\n## B\nbeta rule\n\n## C\n" + "gamma " * 500
    cut = _truncate_system_at_boundary(text, 110)
    assert cut.startswith("## A\nalpha rule\n\n## B\nbeta rule")
    assert "gamma" not in cut and "truncated" in cut


# ---------------------------------------------------------------------------
# Item 7: text mode on Ollama
# ---------------------------------------------------------------------------

def test_non_native_ollama_route_gets_fenced_prompt_and_no_schemas(local_route, monkeypatch):
    local_route["native"] = False

    class FakeMcp:
        def get_all_openai_schemas(self, disabled_map):
            return [{"type": "function", "function": {"name": "mcp__browser__navigate", "parameters": {}}}]

        def get_all_tools(self):
            return []

        def get_tool_descriptions_for_prompt(self, disabled_map):
            return ""

    monkeypatch.setattr(al, "get_mcp_manager", lambda: FakeMcp())
    monkeypatch.setattr(al, "_load_mcp_disabled_map", lambda: {})
    tools, messages = _run_turn(
        local_route, [{"role": "user", "content": "open the browser website and search the web for news"}],
    )
    system = _system_text(messages)
    assert tools == []
    assert "native tool/function calling" not in system
    assert "```web_search" in system


def test_schema_only_tools_have_fenced_syntax():
    schema_names = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS}
    assert schema_names <= set(al.TOOL_SECTIONS)
    slim = al._assemble_prompt({"grep", "api_call", "ui_control"}, set(), slim=True)
    assert "```grep```" in slim and "```api_call```" in slim


def test_rule_lines_for_disabled_tools_are_dropped():
    prompt = al._assemble_prompt(
        {"web_search", "web_fetch", "ask_user"},
        {"manage_memory", "trigger_research"},
        compact=True,
    )
    assert "manage_memory" not in prompt and "trigger_research" not in prompt
    assert "## Web rules" in prompt


def test_local_machine_rules_only_for_file_capable_turns():
    plain, _ = al._build_system_prompt(
        [{"role": "user", "content": "best espresso machine?"}], "m", None, None,
        relevant_tools={"web_search", "ask_user"}, compact=True, suppress_skills=True,
    )
    files, _ = al._build_system_prompt(
        [{"role": "user", "content": "list files on this computer"}], "m", None, None,
        relevant_tools={"ls", "read_file", "ask_user"}, compact=True, suppress_skills=True,
    )
    assert "local-machine mode" not in _system_text(plain)
    assert "local-machine mode" in _system_text(files)

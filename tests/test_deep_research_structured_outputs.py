"""DeepResearcher structured outputs, thinking overrides and context fitting.

JSON steps pass a ``response_schema`` (local servers constrain decoding to it)
and still parse replies from providers that ignore it. Mechanical steps carry
the preset's ``think`` override; synthesis/final report keep the configured
thinking. Prompts embedding the growing report/findings fit the known window.
"""
import asyncio
import json
import sys
import types

import pytest

import src.deep_research as dr
from src.deep_research import DeepResearcher
from src.model_context import estimate_tokens


def _researcher(**kwargs):
    kwargs.setdefault("mechanical_think", False)
    return DeepResearcher(llm_endpoint="http://localhost:11434", llm_model="m", **kwargs)


def _recording(researcher, reply):
    calls = []

    async def fake_llm(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return reply(len(calls)) if callable(reply) else reply

    researcher._llm = fake_llm
    researcher._emit = lambda **k: None
    return calls


def test_plan_passes_schema_and_parses_prose_wrapped_json():
    r = _researcher()
    calls = _recording(r, 'Here is the plan:\n```json\n{"sub_questions": ["a?", "b?"], '
                          '"key_topics": ["x"], "success_criteria": "done"}\n```\nGood luck!')
    plan = asyncio.run(r._create_plan("q"))
    assert calls[0]["response_schema"] == dr.RESEARCH_PLAN_SCHEMA
    assert calls[0]["think"] is False
    assert "Sub-questions: a?; b?" in plan and "Success: done" in plan


def test_query_generation_passes_schema_caps_count_and_parses_prose():
    r = _researcher(queries_first_round=3, query_max_tokens=768)
    calls = _recording(r, 'Sure! ["one", "two", "three", "four", "five"] hope this helps')
    queries = asyncio.run(r._generate_queries("q", "", 1))
    assert queries == ["one", "two", "three"]
    assert calls[0]["response_schema"] == dr.QUERY_LIST_SCHEMA
    assert calls[0]["think"] is False
    assert calls[0]["max_tokens"] == 768
    assert "Generate 3 focused search queries" in calls[0]["messages"][0]["content"]


def test_later_rounds_use_queries_per_round():
    r = _researcher(queries_per_round=2)
    _recording(r, '["a", "b", "c"]')
    assert asyncio.run(r._generate_queries("q", "report", 2)) == ["a", "b"]


@pytest.mark.parametrize("reply, expected", [
    ('"comparison"', "comparison"),          # schema-constrained JSON string
    ("The category is product.", "product"),  # provider ignored the schema
    ('"general"', None),
])
def test_category_uses_enum_schema(reply, expected):
    r = _researcher()
    calls = _recording(r, reply)
    assert asyncio.run(r._classify_category("q")) == expected
    schema = calls[0]["response_schema"]
    assert schema == dr.CATEGORY_SCHEMA
    assert schema["type"] == "string" and set(schema["enum"]) == set(dr.CATEGORY_PROMPTS) | {"general"}
    assert calls[0]["think"] is False


@pytest.mark.parametrize("reply, expected", [
    ('{"answer": "YES", "reason": "covered"}', True),
    ('{"answer": "NO", "reason": "gap"}', False),
    ("**YES** — all aspects covered", True),
    ("NO — still missing prices", False),
])
def test_stop_check_schema_and_parsing(reply, expected):
    r = _researcher()
    calls = _recording(r, reply)
    assert asyncio.run(r._should_stop("q", "report", 3)) is expected
    assert calls[0]["response_schema"] == dr.STOP_SCHEMA
    assert calls[0]["think"] is False


def test_stop_prompt_distinguishes_auto_and_explicit_rounds():
    auto = _researcher(max_rounds=6, auto_rounds=True)
    calls = _recording(auto, "NO")
    asyncio.run(auto._should_stop("q", "report", 2))
    prompt = calls[0]["messages"][0]["content"]
    assert "automatic mode" in prompt and "prefer continuing" not in prompt

    explicit = _researcher(max_rounds=5)
    calls = _recording(explicit, "NO")
    asyncio.run(explicit._should_stop("q", "report", 2))
    prompt = calls[0]["messages"][0]["content"]
    assert "2 of 5" in prompt and "prefer continuing" in prompt


def _fake_page(monkeypatch, content):
    search_mod = types.ModuleType("src.search")
    search_mod.fetch_webpage_content = lambda url, timeout: {
        "success": True, "content": content, "title": "Page", "og_image": "",
    }
    monkeypatch.setitem(sys.modules, "src.search", search_mod)

    async def immediate_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)


def test_extraction_schema_and_prose_wrapped_reply(monkeypatch):
    _fake_page(monkeypatch, "useful page content")
    r = _researcher(extraction_max_tokens=768)
    calls = _recording(r, 'Result:\n{"rational": "r", "evidence": "e", "summary": "Useful summary."}')
    finding = asyncio.run(r._fetch_and_extract("https://ex.test", "q", "Title"))
    assert finding["summary"] == "Useful summary."
    assert calls[0]["response_schema"] == dr.EXTRACTION_SCHEMA
    assert calls[0]["think"] is False
    assert calls[0]["max_tokens"] == 768


def test_truncated_extraction_json_is_salvaged(monkeypatch):
    _fake_page(monkeypatch, "useful page content")
    r = _researcher()
    _recording(r, '{"rational": "relevant", "evidence": "Quote one. Quote two, cut of')
    finding = asyncio.run(r._fetch_and_extract("https://ex.test", "q", "Title"))
    assert finding["evidence"].startswith("Quote one.")
    assert finding["summary"].startswith("Quote one.")
    assert '"rational"' not in finding["evidence"]


def test_extraction_page_is_bounded_by_small_window(monkeypatch):
    _fake_page(monkeypatch, ("word " * 40 + "\n\n") * 400)  # ~80K chars
    r = _researcher(context_window=4096, extraction_max_tokens=768, max_content_chars=15000)
    calls = _recording(r, '{"rational": "r", "evidence": "e", "summary": "Useful summary."}')
    asyncio.run(r._fetch_and_extract("https://ex.test", "q", "Title"))
    assert estimate_tokens(calls[0]["messages"]) + calls[0]["max_tokens"] <= 4096


def _long_report(sections=30, body_chars=4000):
    return "\n".join(
        f"## Section {i}\n" + ("Detail sentence about topic %d. " % i) * (body_chars // 32)
        for i in range(sections)
    )


def _findings(n=6, chars=5000):
    return [{"url": f"https://ex.test/{i}", "title": f"T{i}", "summary": "s" * chars} for i in range(n)]


def test_synthesis_prompt_fits_known_window_and_keeps_structure():
    r = _researcher(context_window=8192, synthesis_max_tokens=2048, synthesis_window=5)
    calls = _recording(r, "updated report")
    report = _long_report()
    assert asyncio.run(r._synthesize("q", _findings(), report)) == "updated report"
    call = calls[0]
    prompt = call["messages"][0]["content"]
    assert estimate_tokens(call["messages"]) + call["max_tokens"] <= 8192
    assert all(f"## Section {i}" in prompt for i in range(30))
    assert "https://ex.test/5" in prompt and "https://ex.test/0" not in prompt  # last 5 findings
    # Synthesis keeps the configured thinking and free-form output.
    assert call.get("think") is None and call.get("response_schema") is None


def test_synthesis_unbounded_when_window_unknown():
    r = _researcher(synthesis_max_tokens=2048)
    calls = _recording(r, "updated")
    report = _long_report(sections=3, body_chars=1000)
    asyncio.run(r._synthesize("q", _findings(n=2, chars=300), report))
    assert report in calls[0]["messages"][0]["content"]


def test_final_report_fits_window_and_uses_preset_length():
    r = _researcher(context_window=8192, max_report_tokens=16384, report_min_words=600,
                    expand_below_words=250)
    calls = _recording(r, "word " * 700)
    out = asyncio.run(r._final_report("q", _long_report()))
    assert out.startswith("word")
    call = calls[0]
    assert call["max_tokens"] <= 4096
    assert estimate_tokens(call["messages"]) + call["max_tokens"] <= 8192
    assert "MINIMUM 600 words" in call["messages"][0]["content"]
    assert call.get("think") is None and call.get("response_schema") is None
    assert len(calls) == 1  # 700 words ≥ 250: no expansion call


def test_final_report_default_prompt_unchanged():
    r = _researcher()
    calls = _recording(r, "word " * 500)
    asyncio.run(r._final_report("q", "short report"))
    assert "MINIMUM 1500 words" in calls[0]["messages"][0]["content"]
    assert calls[0]["max_tokens"] == r.max_report_tokens


def test_generate_plan_passes_schema_and_parses_prose(monkeypatch):
    import src.llm_core as llm_core
    import src.research_handler as rh

    seen = {}

    async def fake_call(**kwargs):
        seen.update(kwargs)
        return 'Plan below.\n{"sub_questions": ["a?"], "key_topics": ["t"], "success_criteria": "ok"}'

    monkeypatch.setattr(llm_core, "llm_call_async", fake_call)
    handler = rh.ResearchHandler.__new__(rh.ResearchHandler)
    plan = asyncio.run(handler.generate_plan("q", "http://localhost:11434", "m"))
    assert seen["response_schema"] == dr.RESEARCH_PLAN_SCHEMA
    assert plan["sub_questions"] == ["a?"] and plan["success_criteria"] == "ok"


def test_llm_helper_forwards_think_and_schema(monkeypatch):
    import src.llm_core as llm_core

    seen = {}

    async def fake_call(**kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(llm_core, "llm_call_async", fake_call)
    r = _researcher()
    asyncio.run(r._llm([{"role": "user", "content": "x"}], think=False, response_schema={"type": "string"}))
    assert seen["think"] is False
    assert seen["response_schema"] == {"type": "string"}
    assert seen["workload"] == "research"


def test_schemas_stay_grammar_friendly():
    schemas = [dr.RESEARCH_PLAN_SCHEMA, dr.QUERY_LIST_SCHEMA, dr.EXTRACTION_SCHEMA,
               dr.CATEGORY_SCHEMA, dr.STOP_SCHEMA]
    text = json.dumps(schemas)
    for banned in ("oneOf", "anyOf", "allOf", "$ref"):
        assert banned not in text

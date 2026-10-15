"""Deep Research honors requested answer lengths.

The 2026-10-12 QA audit asked for "a brief summary" and got a 1,810-word,
133-second report. A length now comes from the launch option or the
question's wording, reshapes the work done (Brief runs fewer rounds/queries
and skips per-round synthesis), and constrains the final writer through its
prompt, its token budget, and a condense pass.
"""
import asyncio
from types import SimpleNamespace

import pytest

import src.research_presets as rp
from src.deep_research import DeepResearcher
from src.research_navigator import ResearchAction
from src.research_presets import (
    BRIEF_DEFAULT_WORDS, PRESETS, apply_report_length, count_report_words, detect_length_request,
    limited_report_tokens, resolve_report_length, trim_report_to_words,
)

AUDIT_QUESTION = ("Find NASA's official Mars Exploration Program page and summarize its "
                  "purpose briefly using only official NASA sources.")


@pytest.mark.parametrize("question, expected", [
    (AUDIT_QUESTION, ("brief", None, False)),
    ("Explain transformers in 200 words", ("brief", 200, False)),
    ("Write a 300-word overview of RISC-V", ("brief", 300, False)),
    ("In one paragraph, what is CRISPR?", ("brief", 150, False)),
    ("Answer in two sentences: is coffee healthy?", ("brief", 50, False)),
    ("Explain the euro crisis in 1000 words", ("standard", 1000, False)),
    ("At least 2500 words on the history of Rome", ("detailed", 2500, True)),
    ("Give me a detailed analysis of lithium supply", ("detailed", None, False)),
    ("Short-term rentals in Lisbon", (None, None, False)),
    ("What is the best laptop for students?", (None, None, False)),
])
def test_detect_length_request(question, expected):
    assert detect_length_request(question) == expected


def test_launch_choice_beats_question_wording_but_word_count_is_the_target():
    assert resolve_report_length("detailed", AUDIT_QUESTION).name == "detailed"
    plan = resolve_report_length("standard", "Summarize this in 120 words")
    assert (plan.name, plan.word_target, plan.word_limit, plan.source) == ("standard", 120, 150, "setting")


def test_auto_follows_question_and_defaults_to_standard():
    brief = resolve_report_length("auto", AUDIT_QUESTION)
    assert (brief.name, brief.word_target, brief.source) == ("brief", BRIEF_DEFAULT_WORDS, "question")
    assert brief.word_limit > brief.word_target
    default = resolve_report_length(None, "History of the printing press")
    assert (default.name, default.word_target, default.word_limit, default.source) == ("standard", None, None, "default")


def test_brief_reduces_work_and_caps_output():
    medium = PRESETS["medium"]
    plan = resolve_report_length("brief", "anything")
    brief = apply_report_length(medium, plan)

    assert brief.max_rounds == 2 and brief.min_rounds == 1
    assert brief.queries_first_round <= 3 and brief.queries_per_round <= 2
    assert brief.urls_per_query <= 2
    assert brief.synthesis_findings <= 6
    assert brief.expand_below_words == 0
    assert brief.report_min_words == BRIEF_DEFAULT_WORDS
    assert brief.report_max_tokens == limited_report_tokens(plan.word_limit)
    assert brief.report_max_tokens < medium.report_max_tokens


def test_standard_without_target_keeps_the_preset():
    medium = PRESETS["medium"]
    assert apply_report_length(medium, resolve_report_length("standard", "x")) == medium


def test_detailed_raises_the_minimum_length():
    medium = PRESETS["medium"]
    detailed = apply_report_length(medium, resolve_report_length("detailed", "x"))
    assert detailed.report_min_words > medium.report_min_words


# ── Final writer ──

def _researcher(**kwargs):
    kwargs.setdefault("mechanical_think", False)
    return DeepResearcher(llm_endpoint="http://localhost:11434", llm_model="m", **kwargs)


def _recording(researcher, replies):
    calls = []

    async def fake_llm(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return replies[min(len(calls), len(replies)) - 1]

    researcher._llm = fake_llm
    researcher._emit = lambda **k: None
    return calls


def test_limited_final_report_states_the_ceiling_and_uses_its_token_budget():
    r = _researcher(report_min_words=250, report_word_limit=312, report_length="brief",
                    max_report_tokens=limited_report_tokens(312), expand_below_words=0)
    calls = _recording(r, ["word " * 240])
    out = asyncio.run(r._final_report("q", "evidence"))

    prompt = calls[0]["messages"][0]["content"]
    assert len(out.split()) == 240
    assert len(calls) == 1
    assert "about 250 words" in prompt and "Never exceed 312 words" in prompt
    assert "long, detailed, comprehensive" not in prompt
    assert calls[0]["max_tokens"] == limited_report_tokens(312)
    assert calls[0]["think"] is False  # brief follows the preset's mechanical thinking


def test_overlong_limited_report_is_condensed():
    r = _researcher(report_min_words=250, report_word_limit=312, report_length="brief",
                    max_report_tokens=limited_report_tokens(312))
    calls = _recording(r, ["word " * 900, "short " * 280])
    out = asyncio.run(r._final_report("q", "evidence"))

    assert len(calls) == 2
    assert "at most 312 words" in calls[1]["messages"][-1]["content"]
    assert len(out.split()) == 280


def test_brief_run_skips_per_round_synthesis():
    r = _researcher(report_length="brief", report_word_limit=312, report_min_words=250,
                    max_rounds=2, min_rounds=1)
    r._emit = lambda **k: None

    async def plan(question):
        return "plan"

    async def no_category(question, research_plan=""):
        return None

    async def actions(question, report, round_num):
        return [ResearchAction("web_search", {"query": f"q{round_num}"})]

    async def execute(actions, question):
        found = [
            {"url": f"https://www.nasa.gov/{i}", "title": str(i), "summary": "Useful official evidence.",
             "source_kind": "official", "source_score": 90}
            for i in range(2)
        ]
        r.analyzed_urls.extend({"url": f["url"], "title": f["title"]} for f in found)
        return found

    async def must_not_run(*args, **kwargs):
        raise AssertionError("brief reports must not run per-round synthesis or stop checks")

    written = {}

    async def final(question, report):
        written["report"] = report
        return "brief answer"

    r._create_plan = plan
    r._classify_category = no_category
    r._plan_research_actions = actions
    r._execute_research_actions = execute
    r._synthesize = must_not_run
    r._should_stop = must_not_run
    r._final_report = final

    assert asyncio.run(r.research("q")) == "brief answer"
    assert r.round_count == 1
    assert "https://www.nasa.gov/0" in written["report"]


# ── Handler and route wiring ──

class _FakeResearcher:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.findings = []
        self.evolving_report = ""
        self.category = None
        _FakeResearcher.instances.append(self)

    async def research(self, query, **kwargs):
        self.research_kwargs = kwargs
        return "report"

    def get_stats(self):
        return {"Rounds": 1}


def _run_handler(monkeypatch, question, report_length=None):
    import src.deep_research as dr
    import src.research_handler as rh

    _FakeResearcher.instances.clear()
    monkeypatch.setattr(dr, "DeepResearcher", _FakeResearcher)

    async def ok_probe(*a, **k):
        return None

    monkeypatch.setattr(rh.ResearchHandler, "_probe_endpoint", staticmethod(ok_probe))
    profile = rp.ResearchProfile("auto", PRESETS["medium"], 32768, None, "test")
    monkeypatch.setattr(rp, "resolve_research_profile", lambda url, model, **kw: profile)
    handler = rh.ResearchHandler.__new__(rh.ResearchHandler)
    handler._legacy_engine = None
    handler._active_tasks = {}
    entry = {}
    asyncio.run(handler.call_research_service(
        question, "http://localhost:11434", "m", _task_entry=entry, report_length=report_length,
    ))
    return _FakeResearcher.instances[-1].kwargs, entry


def test_handler_applies_brief_length_from_question(monkeypatch):
    kw, entry = _run_handler(monkeypatch, AUDIT_QUESTION)

    assert kw["report_length"] == "brief"
    assert kw["report_word_limit"] == resolve_report_length(None, AUDIT_QUESTION).word_limit
    assert kw["max_rounds"] == 2 and kw["min_rounds"] == 1
    assert kw["queries_first_round"] <= 3
    assert kw["max_report_tokens"] < PRESETS["medium"].report_max_tokens
    assert entry["report_length"]["name"] == "brief"
    assert entry["report_length"]["description"].startswith("Brief (~250 words)")


def test_handler_launch_option_overrides_question(monkeypatch):
    kw, _entry = _run_handler(monkeypatch, AUDIT_QUESTION, report_length="detailed")
    assert kw["report_length"] == "detailed"
    assert kw["report_word_limit"] is None
    assert kw["report_min_words"] > PRESETS["medium"].report_min_words


def _start_route(monkeypatch, handler):
    import routes.research.research_routes as rr

    monkeypatch.setattr(rr, "get_current_user", lambda request: "alice")
    monkeypatch.setattr(rr, "resolve_endpoint",
                        lambda purpose, owner=None, **k: ("http://localhost:11434", "m", {})
                        if purpose == "research" else ("", "", {}))
    monkeypatch.setattr("src.auth_helpers.require_privilege", lambda request, privilege: "alice")
    router = rr.setup_research_routes(handler)
    for route in router.routes:
        if route.path == "/api/research/start" and "POST" in route.methods:
            return route.endpoint
    raise AssertionError("start route missing")


def _body(**overrides):
    values = dict(query="q", max_rounds=0, search_provider=None, endpoint_id=None, model=None,
                  max_time=300, extraction_timeout=None, extraction_concurrency=None,
                  category=None, origin_chat_id=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_start_route_passes_length_and_persists_settings(monkeypatch):
    started = {}

    class Handler:
        _active_tasks = {}

        def start_research(self, **kwargs):
            started.update(kwargs)
            return {"source_rules": {}, "report_length": {"name": "brief"}}

    start = _start_route(monkeypatch, Handler())
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    out = asyncio.run(start(body=_body(report_length="brief"), request=request))

    assert started["report_length"] == "brief"
    assert started["run_settings"]["report_length"] == "brief"
    assert out["report_length"] == {"name": "brief"}


# ── The word limit is a hard ceiling on every return path (review finding 10) ──

def _limited(**kwargs):
    kwargs.setdefault("report_length", "brief")
    return _researcher(report_min_words=250, report_word_limit=312,
                       max_report_tokens=limited_report_tokens(312), **kwargs)


def test_report_words_count_prose_not_link_targets_or_a_sources_list():
    report = (
        "## Answer\n\nMars has [two small moons](https://nasa.gov/moons) today.\n\n"
        "- Phobos orbits fast.\n\n## Sources\n\n- [NASA](https://nasa.gov/moons)\n- [ESA](https://esa.int/x)\n"
    )
    assert count_report_words(report) == 10  # Answer + 6 + 3
    assert count_report_words("word " * 240) == 240


def test_draft_over_the_limit_is_never_returned():
    r = _limited()
    calls = _recording(r, ["word " * 400, "word " * 400])
    out = asyncio.run(r._final_report("q", "evidence"))
    assert len(calls) == 2  # asked to condense, but the rewrite did not help
    assert count_report_words(out) <= 312


def test_condensed_rewrite_still_over_the_limit_is_trimmed():
    r = _limited()
    _recording(r, ["word " * 900, "short " * 500])
    out = asyncio.run(r._final_report("q", "evidence"))
    assert 0 < count_report_words(out) <= 312
    assert set(out.rstrip("\u2026").split()) == {"short"}  # the trimmed text is the condensed rewrite


def test_failed_condense_pass_still_caps_the_draft():
    r = _limited()
    replies = iter(["word " * 400])

    async def flaky(messages, **kwargs):
        try:
            return next(replies)
        except StopIteration:
            raise RuntimeError("condense timed out")

    r._llm = flaky
    r._emit = lambda **k: None
    out = asyncio.run(r._final_report("q", "evidence"))
    assert 0 < count_report_words(out) <= 312


def test_writer_failure_falls_back_to_a_capped_report():
    r = _limited()

    async def down(messages, **kwargs):
        raise RuntimeError("model unavailable")

    r._llm = down
    r._emit = lambda **k: None
    evidence = "\n\n".join(f"**Finding {i}** — [Page {i}](https://nasa.gov/{i})\n" + "Evidence sentence. " * 30
                           for i in range(10))
    out = asyncio.run(r._final_report("q", evidence))
    assert 0 < count_report_words(out) <= 312
    assert "[Page 0](https://nasa.gov/0)" in out


def test_trim_cuts_at_a_sentence_and_keeps_links_and_the_sources_list():
    report = (
        "## Answer\n\nMars is cold [NASA facts](https://nasa.gov/mars). It has a thin atmosphere. "
        + "Filler sentence here. " * 20 + "\n\n## Details\n\n" + "Detail sentence. " * 40
        + "\n\n## Sources\n\n- [NASA facts](https://nasa.gov/mars)\n"
    )
    out = trim_report_to_words(report, 30)
    body, sources = out.split("## Sources")

    assert count_report_words(out) <= 30
    assert out.startswith("## Answer\n\nMars is cold [NASA facts](https://nasa.gov/mars). It has a thin atmosphere.")
    assert body.rstrip().endswith("Filler sentence here.")
    assert "## Details" not in out  # no heading left without its section
    assert sources.strip() == "- [NASA facts](https://nasa.gov/mars)"
    assert trim_report_to_words("Short answer.", 30) == "Short answer."


LIST_AND_CODE = (
    "Intro sentence here.\n\n"
    "- First point [ESA](https://esa.int/a).\n- Second point is here.\n- Third point is also here.\n\n"
    "```\nline one two\n\nline three four\n```\n\n"
    "Closing sentence."
)


def test_trim_keeps_whole_list_items():
    assert trim_report_to_words(LIST_AND_CODE, 10) == (
        "Intro sentence here.\n\n- First point [ESA](https://esa.int/a).\n- Second point is here."
    )


def test_trim_keeps_code_blocks_whole_or_drops_them():
    kept = trim_report_to_words(LIST_AND_CODE, 21)
    assert kept.endswith("```\nline one two\n\nline three four\n```")
    dropped = trim_report_to_words(LIST_AND_CODE, 18)
    assert "```" not in dropped and dropped.endswith("- Third point is also here.")


def test_trim_cuts_an_overlong_first_sentence_at_a_word():
    out = trim_report_to_words("A very [long link text](https://x.org/a) sentence " * 20, 10)
    assert count_report_words(out) <= 10
    assert out.endswith("\u2026")
    assert out.count("[") == out.count("](")  # links are never cut in half


def test_research_fallback_report_respects_the_limit():
    r = _researcher(report_length="standard", report_word_limit=150, report_min_words=120,
                    max_rounds=1, min_rounds=1)
    r._emit = lambda **k: None

    async def plan(question):
        return "plan"

    async def no_category(question, research_plan=""):
        return None

    async def actions(question, report, round_num):
        return [ResearchAction("web_search", {"query": "q"})]

    async def execute(actions, question):
        return [{"url": f"https://www.nasa.gov/{i}", "title": str(i),
                 "summary": "Useful official evidence. " * 20} for i in range(6)]

    async def no_synthesis(question, findings, report):
        return ""

    r._create_plan = plan
    r._classify_category = no_category
    r._plan_research_actions = actions
    r._execute_research_actions = execute
    r._synthesize = no_synthesis

    out = asyncio.run(r.research("Explain the Mars program in 120 words"))
    assert "Automatic synthesis did not complete" in out
    assert count_report_words(out) <= 150


def test_fast_fact_report_respects_the_limit():
    r = DeepResearcher(llm_endpoint="http://localhost:11434", llm_model="qwen2.5:3b",
                       report_length="brief", report_word_limit=60, report_min_words=50)
    r._emit = lambda **k: None

    async def plan(question):
        return "plan"

    async def no_category(question, research_plan=""):
        return None

    async def actions(question, report, round_num):
        return [ResearchAction("web_search", {"query": "q"})]

    async def execute(actions, question):
        return [{"url": f"https://www.nasa.gov/{i}", "title": str(i),
                 "summary": "Useful official evidence. " * 15} for i in range(4)]

    r._create_plan = plan
    r._classify_category = no_category
    r._plan_research_actions = actions
    r._execute_research_actions = execute

    out = asyncio.run(r.research("What is the Mars Exploration Program?"))
    assert r.fast_fact_mode
    assert 0 < count_report_words(out) <= 60


@pytest.mark.parametrize("failure", ["timeout", "error"])
def test_partial_results_saved_after_timeout_or_error_respect_the_limit(monkeypatch, tmp_path, failure):
    import src.research_handler as rh
    monkeypatch.setattr(rh, "RESEARCH_DATA_DIR", tmp_path)
    handler = rh.ResearchHandler.__new__(rh.ResearchHandler)
    handler._legacy_engine = None
    handler._active_tasks = {}
    handler._save_result = lambda *a, **k: None
    partial = _limited()
    partial.evolving_report = "Partial evidence sentence. " * 300

    async def stalled(*args, _task_entry=None, **kwargs):
        _task_entry["researcher"] = partial
        if failure == "error":
            raise RuntimeError("engine crashed")
        await asyncio.sleep(5)

    handler.call_research_service = stalled

    async def run():
        handler.start_research(session_id="rp-partial00001", query="q", llm_endpoint="http://x",
                               llm_model="m", hard_timeout=0.05)
        await handler._active_tasks["rp-partial00001"]["task"]

    asyncio.run(run())
    result = handler._active_tasks["rp-partial00001"]["result"]
    # _format_research_report puts the stats header first, then the report.
    body = result.split("## Research Summary", 1)[1].split("---", 1)[1]
    assert 0 < count_report_words(body) <= 312

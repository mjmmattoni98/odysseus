"""Deep Research hardware presets and round limits (src/research_presets.py).

Covers the Auto round fix (the model's stop check used to be skipped until
round 18 because min_rounds was max(2, max_rounds - 2) with max_rounds=20),
the Auto preset choice from context window / parameter size, the legacy
"custom" fallback for installs that tuned research_max_tokens, and that the
handler threads the resolved preset into DeepResearcher.
"""
import asyncio

import pytest

import src.research_presets as rp
from src.research_presets import PRESETS, research_round_limits


# ── Round limits ──

def test_auto_rounds_consult_stop_check_after_small_minimum():
    for name, preset in PRESETS.items():
        max_rounds, min_rounds, auto = research_round_limits(0, preset)
        assert auto is True
        assert max_rounds == preset.max_rounds
        assert min_rounds == preset.min_rounds <= 3, name


def test_custom_auto_keeps_old_cap_but_stops_early():
    # Pre-fix: Auto → max_rounds=20, min_rounds=18. Now the cap stays 20 but
    # the stop decision is consulted from round 2.
    assert research_round_limits(0, PRESETS["custom"]) == (20, 2, True)


@pytest.mark.parametrize("requested, expected", [
    (1, (1, 1, False)),
    (2, (2, 2, False)),
    (5, (5, 3, False)),
    (20, (20, 18, False)),
])
def test_explicit_round_count_means_about_that_many_rounds(requested, expected):
    assert research_round_limits(requested, PRESETS["medium"]) == expected


# ── Auto choice ──

@pytest.mark.parametrize("window, params, expected", [
    (8192, None, "small"),
    (16384, 70.0, "small"),
    (32768, 9.0, "small"),
    (32768, 27.3, "medium"),
    (65536, None, "medium"),
    (131072, 27.3, "large"),
    (None, None, "medium"),
    (None, 3.8, "small"),
])
def test_auto_preset_name(window, params, expected):
    assert rp.auto_preset_name(window, params) == expected


@pytest.mark.parametrize("raw, expected", [
    ("27.3B", 27.3), ("9.0B", 9.0), ("23M", 0.023), ("1.2T", 1200.0),
    ("", None), ("unknown", None), (None, None),
])
def test_parse_parameter_size(raw, expected):
    got = rp.parse_parameter_size(raw)
    assert got == pytest.approx(expected) if expected is not None else got is None


def test_parameter_probe_skips_non_ollama_endpoints(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise AssertionError("must not probe")

    monkeypatch.setattr(httpx, "post", boom)
    assert rp.ollama_parameter_size_b("https://api.openai.com/v1", "gpt-4o") is None
    assert rp.ollama_parameter_size_b("http://localhost:8080/v1", "llama") is None


def test_parameter_probe_reads_api_show_details(monkeypatch):
    import httpx

    class Resp:
        is_success = True

        def json(self):
            return {"details": {"parameter_size": "27.3B"}}

    calls = []
    monkeypatch.setattr(httpx, "post", lambda url, **k: calls.append((url, k["json"])) or Resp())
    monkeypatch.setattr(rp, "_param_cache", {})
    assert rp.ollama_parameter_size_b("http://localhost:11434", "qwen3.8:27b") == pytest.approx(27.3)
    assert calls == [("http://localhost:11434/api/show", {"model": "qwen3.8:27b"})]
    # Cached: no second probe.
    assert rp.ollama_parameter_size_b("http://localhost:11434", "qwen3.8:27b") == pytest.approx(27.3)
    assert len(calls) == 1


# ── Configured choice ──

def _settings(monkeypatch, values):
    import src.settings as settings
    monkeypatch.setattr(settings, "get_setting", lambda key, default=None: values.get(key, default))


@pytest.mark.parametrize("values, expected", [
    ({}, "auto"),
    ({"research_preset": "", "research_max_tokens": 16384}, "auto"),
    ({"research_preset": "", "research_max_tokens": 32000}, "custom"),
    ({"research_preset": "small", "research_max_tokens": 32000}, "small"),
    ({"research_preset": "Large"}, "large"),
    ({"research_preset": "bogus"}, "auto"),
])
def test_configured_preset(monkeypatch, values, expected):
    _settings(monkeypatch, values)
    assert rp.configured_preset() == expected


# ── Profile resolution ──

def _context(monkeypatch, ctx, known):
    import src.model_context as mc
    monkeypatch.setattr(mc, "get_context_length_known", lambda url, model: (ctx, known))


def test_auto_profile_uses_window_and_parameter_size(monkeypatch):
    _context(monkeypatch, 32768, True)
    monkeypatch.setattr(rp, "ollama_parameter_size_b", lambda url, model: 27.3)
    profile = rp.resolve_research_profile("http://localhost:11434", "qwen3.8:27b", requested="auto")
    assert profile.preset.name == "medium"
    assert profile.context_window == 32768
    assert profile.parameter_size_b == pytest.approx(27.3)
    assert "32K context" in profile.reason and "27.3B" in profile.reason and "Medium" in profile.reason


def test_unknown_window_is_not_used_for_bounding(monkeypatch):
    _context(monkeypatch, 128000, False)
    monkeypatch.setattr(rp, "ollama_parameter_size_b", lambda url, model: None)
    profile = rp.resolve_research_profile("https://api.example.test/v1", "m", requested="auto")
    assert profile.context_window is None
    assert profile.preset.name == "medium"


def test_custom_profile_uses_research_max_tokens(monkeypatch):
    _context(monkeypatch, 131072, True)
    profile = rp.resolve_research_profile("http://x/v1", "m", requested="custom", max_report_tokens=9000)
    assert profile.preset.name == "custom"
    assert profile.preset.synthesis_max_tokens == 9000
    assert profile.preset.report_max_tokens == 9000
    assert profile.preset.mechanical_think is None


def test_probe_failure_degrades_to_unknown(monkeypatch):
    import src.model_context as mc

    def boom(url, model):
        raise RuntimeError("down")

    monkeypatch.setattr(mc, "get_context_length_known", boom)
    monkeypatch.setattr(rp, "ollama_parameter_size_b", lambda url, model: None)
    profile = rp.resolve_research_profile("http://localhost:11434", "m", requested="auto")
    assert profile.context_window is None and profile.preset.name == "medium"


def test_presets_disable_thinking_for_mechanical_steps_except_custom():
    assert all(PRESETS[n].mechanical_think is False for n in ("small", "medium", "large"))
    assert PRESETS["custom"].mechanical_think is None


# ── Handler wiring ──

class _FakeResearcher:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.findings = []
        self.evolving_report = ""
        _FakeResearcher.instances.append(self)

    async def research(self, query, **kwargs):
        return "report"

    def get_stats(self):
        return {"Rounds": 1}


def _run_handler(monkeypatch, preset_name, window, max_rounds):
    import src.deep_research as dr
    import src.research_handler as rh

    _FakeResearcher.instances.clear()
    monkeypatch.setattr(dr, "DeepResearcher", _FakeResearcher)

    async def ok_probe(*a, **k):
        return None

    monkeypatch.setattr(rh.ResearchHandler, "_probe_endpoint", staticmethod(ok_probe))
    profile = rp.ResearchProfile("auto", PRESETS[preset_name], window, None, "test")
    seen = {}

    def fake_resolve(url, model, **kwargs):
        seen["args"] = (url, model)
        return profile

    monkeypatch.setattr(rp, "resolve_research_profile", fake_resolve)
    handler = rh.ResearchHandler.__new__(rh.ResearchHandler)
    handler._legacy_engine = None
    handler._active_tasks = {}
    out = asyncio.run(handler.call_research_service("q", "http://localhost:11434", "m", max_rounds=max_rounds))
    assert "report" in out
    assert seen["args"] == ("http://localhost:11434", "m")
    return _FakeResearcher.instances[-1].kwargs


def test_handler_auto_rounds_use_preset_and_early_stop(monkeypatch):
    kw = _run_handler(monkeypatch, "small", 8192, max_rounds=0)
    small = PRESETS["small"]
    assert kw["max_rounds"] == small.max_rounds
    assert kw["min_rounds"] == 2
    assert kw["auto_rounds"] is True
    assert kw["mechanical_think"] is False
    assert kw["context_window"] == 8192
    assert kw["max_urls_per_round"] == small.urls_per_query
    assert kw["max_content_chars"] == small.page_chars
    assert kw["synthesis_window"] == small.synthesis_findings
    assert kw["queries_first_round"] == small.queries_first_round
    assert kw["report_min_words"] == small.report_min_words
    assert kw["preset_label"] == "Small (auto)"


def test_handler_explicit_rounds_stay_meaningful(monkeypatch):
    kw = _run_handler(monkeypatch, "medium", 32768, max_rounds=5)
    assert (kw["max_rounds"], kw["min_rounds"], kw["auto_rounds"]) == (5, 3, False)


# ── Routes ──

def _research_router(monkeypatch, handler):
    import routes.research.research_routes as rr

    monkeypatch.setattr(rr, "get_current_user", lambda request: "alice")
    monkeypatch.setattr(rr, "resolve_endpoint",
                        lambda purpose, owner=None, **k: ("http://localhost:11434", "qwen3.8:27b", {})
                        if purpose == "research" else ("", "", {}))
    return rr, rr.setup_research_routes(handler)


def _endpoint(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def test_preset_info_route_reports_auto_pick(monkeypatch):
    _settings(monkeypatch, {"research_preset": "auto"})
    _context(monkeypatch, 32768, True)
    monkeypatch.setattr(rp, "ollama_parameter_size_b", lambda url, model: 27.3)
    _rr, router = _research_router(monkeypatch, handler=object())
    info = asyncio.run(_endpoint(router, "/api/research/preset", "GET")(request=None))
    assert info["configured"] == "auto"
    assert info["resolved"] == "medium"
    assert info["model"] == "qwen3.8:27b"
    assert info["context_window"] == 32768
    assert [p["name"] for p in info["presets"]] == ["auto", "small", "medium", "large", "custom"]


def test_start_route_passes_auto_rounds_through(monkeypatch):
    from types import SimpleNamespace

    started = {}

    class Handler:
        _active_tasks = {}

        def start_research(self, **kwargs):
            started.update(kwargs)

    rr, router = _research_router(monkeypatch, Handler())
    monkeypatch.setattr("src.auth_helpers.require_privilege", lambda request, privilege: "alice")
    start = _endpoint(router, "/api/research/start", "POST")
    body = SimpleNamespace(query="q", max_rounds=0, search_provider=None, endpoint_id=None,
                           model=None, max_time=300, extraction_timeout=None,
                           extraction_concurrency=None, category=None)
    asyncio.run(start(body=body, request=None))
    assert started["max_rounds"] == 0  # Auto: the handler applies the preset cap

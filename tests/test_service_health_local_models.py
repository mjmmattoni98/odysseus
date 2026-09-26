"""Tests for local Ollama tuning checks (service_health.local_models_health)."""
from src import service_health as sh


def _endpoint(supports_tools=None):
    return {
        "name": "ollama",
        "base_url": "http://localhost:11434/v1",
        "api_key": None,
        "supports_tools": supports_tools,
        "model_type": "llm",
    }


def _run(endpoints, ps, show=None, calls=None):
    def get_json(url):
        if calls is not None:
            calls.append(("GET", url))
        return ps

    def post_json(url, payload):
        if calls is not None:
            calls.append(("POST", url, payload))
        return show

    return sh.local_models_health(endpoints, get_json=get_json, post_json=post_json)


def test_no_local_endpoints_is_disabled():
    out = _run([], {})
    assert out["status"] == sh.DISABLED


def test_non_ollama_endpoints_are_ignored():
    out = _run([{"name": "lmstudio", "base_url": "http://localhost:1234/v1"}], {})
    assert out["status"] == sh.DISABLED


def test_reachable_idle_endpoint_is_ok():
    out = _run([_endpoint()], {"models": []})
    assert out["status"] == sh.OK
    assert out["meta"]["endpoints"][0]["ok"] is True


def test_unreachable_endpoint_is_down():
    out = _run([_endpoint()], None)
    assert out["status"] == sh.DOWN
    assert out["meta"]["endpoints"][0]["error"] == "unreachable"


def test_cpu_spill_is_reported():
    ps = {"models": [{"name": "qwen3.8:27b", "size": 20_000, "size_vram": 13_000,
                      "context_length": 32768}]}
    out = _run([_endpoint()], ps, show=None)
    issues = out["meta"]["endpoints"][0]["issues"]
    assert any("spilling to CPU" in i for i in issues)
    assert out["status"] == sh.DEGRADED


def test_context_mismatch_is_reported():
    ps = {"models": [{"name": "qwen3.8:27b", "size": 20_000, "size_vram": 20_000,
                      "context_length": 32768}]}
    show = {"capabilities": ["completion", "tools"],
            "model_info": {"qwen35.context_length": 262144}}
    out = _run([_endpoint()], ps, show=show)
    issues = out["meta"]["endpoints"][0]["issues"]
    assert any("OLLAMA_CONTEXT_LENGTH" in i for i in issues)


def test_no_context_warning_when_serving_window_is_reasonable():
    ps = {"models": [{"name": "gemma4:12b", "size": 9_000, "size_vram": 9_000,
                      "context_length": 65536}]}
    show = {"capabilities": ["completion"], "model_info": {"gemma4.context_length": 262144}}
    out = _run([_endpoint()], ps, show=show)
    assert out["status"] == sh.OK


def test_tools_disabled_override_is_reported():
    ps = {"models": [{"name": "lfm2.5:8b", "size": 5_000, "size_vram": 5_000,
                      "context_length": 32768}]}
    show = {"capabilities": ["completion", "tools"], "model_info": {"lfm2moe.context_length": 32768}}
    out = _run([_endpoint(supports_tools=False)], ps, show=show)
    issues = out["meta"]["endpoints"][0]["issues"]
    assert any("tool calling is disabled" in i for i in issues)


def test_tools_enabled_override_is_clean():
    ps = {"models": [{"name": "lfm2.5:8b", "size": 5_000, "size_vram": 5_000,
                      "context_length": 32768}]}
    show = {"capabilities": ["completion", "tools"], "model_info": {"lfm2moe.context_length": 32768}}
    out = _run([_endpoint(supports_tools=True)], ps, show=show)
    assert out["status"] == sh.OK


def test_probes_use_native_api_root_not_v1():
    calls = []
    ps = {"models": [{"name": "lfm2.5:8b", "size": 5, "size_vram": 5, "context_length": 4096}]}
    show = {"capabilities": [], "model_info": {}}
    _run([_endpoint()], ps, show=show, calls=calls)
    assert ("GET", "http://localhost:11434/api/ps") in calls
    assert ("POST", "http://localhost:11434/api/show", {"model": "lfm2.5:8b"}) in calls


def test_cloud_endpoint_is_excluded_from_local_checks():
    # ollama.com /api/ps requires credentials; probing it unauthenticated made
    # a working cloud account look like a local outage.
    calls = []
    out = _run([{"name": "ollama cloud", "base_url": "https://ollama.com/api"}], {}, calls=calls)
    assert out["status"] == sh.DISABLED
    assert calls == []

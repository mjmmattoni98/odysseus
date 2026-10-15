"""Cookbook recommendations expose task and runtime facets separately from memory fit.

QA found a catalog mixing chat, embedding, speech, vision and time-series rows
where a "PERFECT" memory-fit label read as "a good chat model that runs on the
displayed engine". services.hwfit.facets classifies what each row is for and,
per engine, whether its files can go through that engine; rank_models and the
image/Ollama endpoints attach those facets and keep non-chat rows below chat
rows in chat-oriented views. Rows are synthetic; names only select heuristics.
"""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from services.hwfit.facets import RUNTIME_ENGINES, classify_model_facets
from src.ollama_capabilities import chat_capability

# Shared with tests/cookbook_facets.test.mjs so the JS labels match.
NATIVE_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "ollama_native_capability_cases.json").read_text()
)["installed"]


def _facets(**row):
    return classify_model_facets({"name": "test/model", **row})


@pytest.mark.parametrize(
    ("pipeline_tag", "task", "chat_capable"),
    [
        ("text-generation", "chat", True),
        ("image-text-to-text", "vision", True),
        ("audio-text-to-text", "audio_chat", True),
        ("feature-extraction", "embedding", False),
        ("sentence-similarity", "embedding", False),
        ("text-ranking", "reranking", False),
        ("automatic-speech-recognition", "speech_to_text", False),
        ("text-to-speech", "text_to_speech", False),
        ("time-series-forecasting", "time_series", False),
        ("zero-shot-image-classification", "classification", False),
        ("text-to-image", "image_generation", False),
    ],
)
def test_pipeline_tag_decides_task(pipeline_tag, task, chat_capable):
    facets = _facets(pipeline_tag=pipeline_tag)

    assert facets["task"] == task
    assert facets["task_source"] == "pipeline_tag"
    assert facets["chat_capable"] is chat_capable
    assert facets["task_label"]


def test_missing_metadata_is_explicitly_unknown_not_chat():
    facets = classify_model_facets({"name": "test/Mystery-9B", "use_case": "general"})

    assert facets["task"] == "unknown"
    assert facets["task_label"] == "Unknown"
    assert facets["chat_capable"] is None


@pytest.mark.parametrize(
    ("name", "task"),
    [
        ("test/Embedder-Embedding-0.6B-GGUF", "embedding"),
        ("test/timesfm-2.5-200m", "time_series"),
        ("test/Reranker-4B", "reranking"),
        ("test/whisper-large", "speech_to_text"),
        ("test/Voice-TTS-1.7B-Base", "text_to_speech"),
        ("test/Thing-VL-7B", "vision"),
        ("test/Thing-7B-Instruct", "chat"),
    ],
)
def test_name_heuristics_apply_without_pipeline_tag(name, task):
    assert classify_model_facets({"name": name})["task"] == task


def test_base_checkpoints_are_not_chat_models_but_t5_base_size_is_not_a_checkpoint():
    base = classify_model_facets({"name": "test/Model-8B-Base", "pipeline_tag": "text-generation"})
    t5 = classify_model_facets({"name": "test/flan-t5-base"})

    assert base["task"] == "base"
    assert base["chat_capable"] is False
    assert t5["task"] != "base"


def test_runtime_covers_every_displayable_engine():
    runtime = _facets(pipeline_tag="text-generation")["runtime"]

    assert set(runtime) == set(RUNTIME_ENGINES)
    for entry in runtime.values():
        assert entry["status"] in {"compatible", "unverified", "not_supported"}
        assert entry["reason"]


def test_safetensors_only_row_is_unverified_not_rejected_for_llamacpp_or_ollama():
    # Safetensors convert to GGUF and Ollama's library often packages the same
    # model (gemma-4 runs as gemma4:26b), so "not supported" would be false.
    runtime = _facets(pipeline_tag="text-generation", format="safetensors", quantization="BF16")["runtime"]

    assert runtime["llamacpp"]["status"] == "unverified"
    assert runtime["ollama"]["status"] == "unverified"
    assert "GGUF" in runtime["ollama"]["reason"]
    assert runtime["vllm"]["status"] == "unverified"
    assert runtime["sglang"]["status"] == "unverified"


def test_listed_gguf_build_is_unverified_not_compatible_for_llamacpp():
    runtime = _facets(
        pipeline_tag="text-generation",
        quantization="F16",
        gguf_sources=[{"repo": "test/model-GGUF", "file": "model-Q4_K_M.gguf"}],
    )["runtime"]

    assert runtime["llamacpp"]["status"] == "unverified"
    assert runtime["ollama"]["status"] == "unverified"


def test_awq_and_mlx_weights_are_not_supported_on_the_wrong_engines():
    awq = _facets(name="test/Model-AWQ", pipeline_tag="text-generation", quantization="AWQ-4bit")["runtime"]
    mlx = _facets(name="mlx-community/Model-4bit", pipeline_tag="text-generation", format="mlx")["runtime"]

    assert awq["llamacpp"]["status"] == "not_supported"
    assert mlx["vllm"]["status"] == "not_supported"
    assert mlx["llamacpp"]["status"] == "not_supported"
    assert mlx["mlx"]["status"] == "unverified"


def test_catalog_rows_are_never_compatible_without_evidence():
    rows = [
        {"pipeline_tag": "text-generation", "format": "safetensors"},
        {"pipeline_tag": "image-text-to-text", "is_gguf": True, "format": "gguf"},
        {"name": "mlx-community/Model-4bit", "pipeline_tag": "text-generation", "format": "mlx"},
    ]
    for row in rows:
        statuses = {entry["status"] for entry in _facets(**row)["runtime"].values()}
        assert "compatible" not in statuses


def test_official_vllm_recipe_is_compatibility_evidence_for_vllm_only():
    runtime = _facets(pipeline_tag="text-generation", format="safetensors", vllm_recipe=True)["runtime"]

    assert runtime["vllm"]["status"] == "compatible"
    assert runtime["sglang"]["status"] == "unverified"


def test_time_series_and_image_models_are_not_served_by_llm_engines():
    ts = _facets(pipeline_tag="time-series-forecasting", format="safetensors")["runtime"]
    image = _facets(pipeline_tag="text-to-image", format="safetensors")["runtime"]
    chat = _facets(pipeline_tag="text-generation", format="safetensors")["runtime"]

    for engine in ("llamacpp", "ollama", "vllm", "sglang", "mlx"):
        assert ts[engine]["status"] == "not_supported"
        assert image[engine]["status"] == "not_supported"
    assert image["diffusers"]["status"] == "unverified"
    assert chat["diffusers"]["status"] == "not_supported"


def test_ollama_library_entries_use_ollama_capabilities():
    embed = classify_model_facets({"name": "embedder"}, source="ollama_library", ollama_capabilities=["embedding"])
    vision = classify_model_facets({"name": "seer"}, source="ollama_library", ollama_capabilities=["vision", "tools"])
    plain = classify_model_facets({"name": "talker"}, source="ollama_library", ollama_capabilities=[])
    cloud = classify_model_facets({"name": "big"}, source="ollama_library", ollama_capabilities=["tools", "cloud"], cloud_only=True)

    assert (embed["task"], embed["chat_capable"]) == ("embedding", False)
    assert vision["task"] == "vision"
    assert plain["task"] == "chat"
    assert embed["runtime"]["ollama"]["status"] == "compatible"
    assert embed["runtime"]["vllm"]["status"] == "not_supported"
    assert cloud["runtime"]["ollama"]["status"] == "unverified"


@pytest.mark.parametrize("case", NATIVE_CASES, ids=[c["name"] for c in NATIVE_CASES])
def test_installed_ollama_models_follow_the_native_capability_report(case):
    # An installed model's /api/show report is evidence, not a hint: no
    # "completion" means Ollama rejects chat ("does not support chat").
    facets = classify_model_facets(
        {"name": "test/Talker-7B-Instruct"}, source="ollama_installed", ollama_capabilities=case["capabilities"],
    )

    assert {key: facets[key] for key in case["expect"]} == case["expect"]
    assert facets["task_source"] == "ollama"
    assert facets["runtime"]["ollama"]["status"] == "compatible"


@pytest.mark.parametrize("case", NATIVE_CASES, ids=[c["name"] for c in NATIVE_CASES])
def test_native_capability_cases_match_chat_eligibility(case):
    # The Cookbook label and model-list chat eligibility use one rule.
    caps = case["capabilities"]
    report = None if caps is None else frozenset(str(c).lower() for c in caps)

    assert chat_capability(report) is case["expect"]["chat_capable"]


def test_ollama_library_capability_chips_stay_hints():
    # ollama.com cards never list "completion"; their chips only hint at the
    # task, so a vision card is still a vision chat model.
    card = classify_model_facets({"name": "seer"}, source="ollama_library", ollama_capabilities=["vision"])

    assert (card["task"], card["chat_capable"]) == ("vision", True)


# ── rank_models / route integration ──

CATALOG = [
    {"name": "test/Talker-7B-Instruct", "pipeline_tag": "text-generation", "parameter_count": "7B",
     "parameters_raw": 7_000_000_000, "quantization": "Q4_K_M", "context_length": 4096,
     "release_date": "2026-01-01", "is_gguf": True, "format": "gguf"},
    {"name": "test/Forecaster-200M", "pipeline_tag": "time-series-forecasting", "parameter_count": "200M",
     "parameters_raw": 200_000_000, "quantization": "Q4_K_M", "context_length": 4096,
     "release_date": "2026-03-01", "format": "safetensors"},
    {"name": "test/Embedder-0.6B", "pipeline_tag": "feature-extraction", "parameter_count": "0.6B",
     "parameters_raw": 600_000_000, "quantization": "Q4_K_M", "context_length": 4096,
     "release_date": "2026-02-01", "format": "safetensors"},
    {"name": "test/Mystery-3B", "parameter_count": "3B", "parameters_raw": 3_000_000_000,
     "quantization": "Q4_K_M", "context_length": 4096, "release_date": "2026-01-15", "format": "safetensors"},
]

SYSTEM = {
    "has_gpu": True, "backend": "cuda", "gpu_name": "Synthetic GPU", "gpu_vram_gb": 24.0, "gpu_count": 1,
    "gpus": [{"index": 0, "name": "Synthetic GPU", "vram_gb": 24.0}],
    "gpu_groups": [{"name": "Synthetic GPU", "vram_each": 24.0, "count": 1, "indices": [0], "vram_total": 24.0}],
    "available_ram_gb": 64.0, "total_ram_gb": 64.0,
}


@pytest.fixture
def synthetic_catalog(monkeypatch):
    from services.hwfit import fit
    monkeypatch.setattr(fit, "get_models", lambda: deepcopy(CATALOG))


def test_rank_models_attaches_facets_and_lists_chat_rows_before_non_chat(synthetic_catalog):
    from services.hwfit.fit import rank_models

    # "newest" would otherwise put the tiny, perfectly-fitting forecaster and
    # embedder (newest release dates) on top of the chat recommendations.
    rows = rank_models(deepcopy(SYSTEM), sort="newest", limit=50)
    by_name = {r["name"]: r for r in rows}

    assert [r["name"] for r in rows] == [
        "test/Mystery-3B", "test/Talker-7B-Instruct", "test/Forecaster-200M", "test/Embedder-0.6B",
    ]
    assert by_name["test/Forecaster-200M"]["fit_level"] == "perfect"
    assert by_name["test/Forecaster-200M"]["task"] == "time_series"
    assert by_name["test/Forecaster-200M"]["runtime"]["llamacpp"]["status"] == "not_supported"
    assert by_name["test/Talker-7B-Instruct"]["chat_capable"] is True
    assert by_name["test/Mystery-3B"]["task"] == "unknown"


def test_rank_models_keeps_non_chat_rows_reachable_and_unpartitioned_when_requested(synthetic_catalog):
    from services.hwfit.fit import rank_models

    embedding_view = rank_models(deepcopy(SYSTEM), use_case="embedding", sort="newest", limit=50)
    top_one = rank_models(deepcopy(SYSTEM), sort="newest", limit=1)

    assert [r["name"] for r in embedding_view] == ["test/Embedder-0.6B"]
    assert top_one[0]["chat_capable"] is not False


def test_models_route_returns_facets(synthetic_catalog, monkeypatch):
    from routes.hwfit_routes import setup_hwfit_routes
    from services.hwfit import hardware, models

    monkeypatch.setattr(hardware, "detect_system", lambda **_: deepcopy(SYSTEM))
    monkeypatch.setattr(models, "get_models", lambda: deepcopy(CATALOG))
    router = setup_hwfit_routes()
    endpoint = next(r.endpoint for r in router.routes if getattr(r, "path", "").endswith("/models"))

    payload = endpoint(limit=50, sort="newest")

    assert payload["models"]
    for row in payload["models"]:
        assert {"fit_level", "task", "task_label", "chat_capable", "runtime"} <= set(row)


_OLLAMA_SEARCH_HTML = """
<a href="/library/embedder" class="group w-full"><p class="mt-1">Synthetic embedding model.</p>
  <span class="chip">embedding</span><span>0.3b</span></a>
<a href="/library/talker" class="group w-full"><p class="mt-1">Synthetic chat model.</p>
  <span class="chip">tools</span><span class="chip">thinking</span><span>8b</span></a>
<a href="/library/hosted" class="group w-full"><p class="mt-1">Synthetic hosted model.</p>
  <span class="chip">tools</span><span class="chip">cloud</span></a>
"""


class _FakeResponse:
    status_code = 200
    text = _OLLAMA_SEARCH_HTML


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, *args, **kwargs):
        return _FakeResponse()


@pytest.mark.asyncio
async def test_ollama_library_route_attaches_facets_from_capability_chips(monkeypatch):
    import httpx
    from routes.cookbook_routes import setup_cookbook_routes

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    router = setup_cookbook_routes()
    endpoint = next(r.endpoint for r in router.routes if getattr(r, "path", "") == "/api/cookbook/ollama/library")

    payload = await endpoint(refresh=1, request=None, owner="tester")
    by_name = {m["name"]: m for m in payload["models"]}

    assert by_name["embedder"]["task"] == "embedding"
    assert by_name["embedder"]["chat_capable"] is False
    assert by_name["embedder"]["runtime"]["ollama"]["status"] == "compatible"
    assert by_name["talker"]["task"] == "chat"
    assert by_name["hosted"]["runtime"]["ollama"]["status"] == "unverified"
    # Curated fallback entries merged in are classified too.
    assert all("task" in m and "runtime" in m for m in payload["models"])

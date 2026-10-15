"""Task and runtime facets for Cookbook model recommendations.

A recommendation row's ``fit_level`` only answers "do the weights + KV cache
fit in memory?". It says nothing about what the model is for (a time-series
or embedding model can fit perfectly and still be useless as a chat model) or
whether the engine the UI displays can actually load its files. This module
classifies those two facets separately so the UI can label all three.

Runtime statuses are deliberately conservative:

* ``compatible``    - there is explicit evidence: the model is published in
                      the Ollama library / installed on the Ollama server, or
                      the catalog row carries an official vLLM recipe.
* ``not_supported`` - the files or task cannot go through that engine (e.g.
                      safetensors-only repos on llama.cpp, MLX weights on vLLM,
                      time-series models on any LLM server).
* ``unverified``    - the format matches but nobody has tested this model on
                      that engine. Most rows land here; that is the honest
                      answer for metadata-only catalog entries.
"""

import re

from services.hwfit.models import is_prequantized
from src.ollama_capabilities import chat_capability

TASK_LABELS = {
    "chat": "Chat",
    "vision": "Vision chat",
    "audio_chat": "Audio chat",
    "base": "Base model",
    "embedding": "Embedding",
    "reranking": "Reranker",
    "speech_to_text": "Speech (STT)",
    "text_to_speech": "Voice (TTS)",
    "time_series": "Time series",
    "image_generation": "Image gen",
    "video_generation": "Video gen",
    "classification": "Classifier",
    "fill_mask": "Fill-mask",
    "translation": "Translation",
    "other": "Other",
    "unknown": "Unknown",
}

# Tasks a user can hold a conversation with through a chat-completions API.
CHAT_TASKS = frozenset({"chat", "vision", "audio_chat"})

RUNTIME_ENGINES = ("llamacpp", "ollama", "vllm", "sglang", "mlx", "diffusers", "mlx_image")
LLM_ENGINES = ("llamacpp", "ollama", "vllm", "sglang", "mlx")
ENGINE_LABELS = {
    "llamacpp": "llama.cpp", "ollama": "Ollama", "vllm": "vLLM", "sglang": "SGLang",
    "mlx": "MLX", "diffusers": "Diffusers", "mlx_image": "MLX Image",
}

COMPATIBLE = "compatible"
UNVERIFIED = "unverified"
NOT_SUPPORTED = "not_supported"

# Hugging Face pipeline tags are the model author's own declaration, so they
# win over every name heuristic below.
_PIPELINE_TASKS = {
    "text-generation": "chat",
    "conversational": "chat",
    "image-text-to-text": "vision",
    "visual-question-answering": "vision",
    "video-text-to-text": "vision",
    "any-to-any": "vision",
    "audio-text-to-text": "audio_chat",
    "feature-extraction": "embedding",
    "sentence-similarity": "embedding",
    "image-feature-extraction": "embedding",
    "text-ranking": "reranking",
    "automatic-speech-recognition": "speech_to_text",
    "text-to-speech": "text_to_speech",
    "text-to-audio": "text_to_speech",
    "time-series-forecasting": "time_series",
    "text-to-image": "image_generation",
    "image-to-image": "image_generation",
    "text-to-video": "video_generation",
    "image-to-video": "video_generation",
    "image-text-to-video": "video_generation",
    "text-classification": "classification",
    "token-classification": "classification",
    "zero-shot-classification": "classification",
    "image-classification": "classification",
    "zero-shot-image-classification": "classification",
    "audio-classification": "classification",
    "fill-mask": "fill_mask",
    "translation": "translation",
}

# "*-Base" checkpoints are raw completion models. T5/BERT "-base" is a size
# name (flan-t5-base is instruction-tuned), not a checkpoint stage.
_BASE_CHECKPOINT = re.compile(r"(?<!t5)(?<!bert)(^|[-_])base($|[-_])")

# Ordered name heuristics for rows without a usable pipeline tag. Specific
# non-chat families come first so "Qwen3-TTS-...-Base" reads as TTS, not base.
_NAME_RULES = (
    ("reranking", re.compile(r"rerank")),
    ("embedding", re.compile(r"embed|(^|[-_/])(bge|e5|gte)[-_]|minilm")),
    ("speech_to_text", re.compile(r"whisper|(^|[-_/])asr([-_]|$)|speech-recognition|transcri")),
    ("text_to_speech", re.compile(r"(^|[-_/])tts([-_]|$)|text-to-speech|kokoro|cosyvoice|parler")),
    ("time_series", re.compile(r"timesfm|time-series|chronos")),
    ("image_generation", re.compile(r"stable-diffusion|(^|[-_/])sdxl|flux\.?\d")),
    ("base", _BASE_CHECKPOINT),
    ("vision", re.compile(r"(^|[-_])vl($|[-_])|vision|llava|(^|[-_])vlm($|[-_])")),
    ("chat", re.compile(r"instruct|(^|[-_])it($|[-_])|chat|coder|(^|[-_])r1($|[-_])|reason|thinking")),
)

# Catalog `use_case` values written by discovery/maintenance scripts. "general"
# is their default for "no idea", so it is deliberately absent.
_USE_CASE_TASKS = {
    "chat": "chat", "coding": "chat", "reasoning": "chat",
    "multimodal": "vision", "embedding": "embedding",
    "tts": "text_to_speech", "stt": "speech_to_text", "image_gen": "image_generation",
}

# Tasks no LLM server (llama.cpp/Ollama/vLLM/SGLang/MLX-LM) can serve at all.
_NON_LLM_TASKS = frozenset({"time_series", "image_generation", "video_generation"})
_DIFFUSION_TASKS = frozenset({"image_generation", "video_generation"})


def _text(value):
    return value if isinstance(value, str) else ""


def _native_ollama_task(capabilities):
    """``(task, label)`` from an installed model's ``/api/show`` capabilities.

    The report is evidence, so it follows chat eligibility
    (:func:`src.ollama_capabilities.chat_capability`): ``completion`` means
    chat; a report without it is a model Ollama refuses to chat with, named
    from what it does report (``["decision", "vision"]`` is a vision
    classifier); no report is unknown, never assumed to be chat.
    """
    caps = frozenset(str(c).strip().lower() for c in capabilities or () if str(c).strip())
    can_chat = chat_capability(caps)
    if can_chat is None:
        return "unknown", TASK_LABELS["unknown"]
    if can_chat:
        task = "vision" if "vision" in caps else "chat"
        return task, TASK_LABELS[task]
    if "embedding" in caps:
        return "embedding", TASK_LABELS["embedding"]
    if "decision" in caps:
        return "classification", "Vision classifier" if "vision" in caps else TASK_LABELS["classification"]
    return "other", "No chat: " + ", ".join(sorted(caps))


def _classify_task(model, ollama_capabilities):
    if ollama_capabilities is not None:
        # ollama.com library cards: their chips (never "completion") only
        # hint at the task. Installed models go through _native_ollama_task.
        caps = {str(c).lower() for c in ollama_capabilities}
        if "embedding" in caps:
            return "embedding", "ollama"
        if "vision" in caps:
            return "vision", "ollama"
        if caps & {"completion", "tools", "thinking"}:
            return "chat", "ollama"
    pipeline = _text(model.get("pipeline_tag")).strip().lower()
    short_name = _text(model.get("name")).lower().rsplit("/", 1)[-1]
    task = _PIPELINE_TASKS.get(pipeline)
    if task == "chat" and _BASE_CHECKPOINT.search(short_name):
        # The pipeline tag says "text generation", but a *-Base checkpoint is a
        # raw completion model, not something tuned to follow a conversation.
        return "base", "name"
    if task:
        return task, "pipeline_tag"
    for rule_task, pattern in _NAME_RULES:
        if pattern.search(short_name):
            return rule_task, "name"
    task = _USE_CASE_TASKS.get(_text(model.get("use_case")).strip().lower())
    if task:
        return task, "catalog"
    if ollama_capabilities is not None:
        # Every Ollama library tag without an embedding capability is served
        # through Ollama's generate/chat API.
        return "chat", "ollama"
    return "unknown", "none"


def _weights(model):
    """Describe what files the row would download: (kind, has_gguf_build)."""
    name = _text(model.get("name")).lower()
    fmt = _text(model.get("format")).lower()
    quant = _text(model.get("quantization") or model.get("quant"))
    has_gguf = bool(model.get("is_gguf") or model.get("gguf_sources")) or fmt == "gguf" or "gguf" in name
    is_mlx = (
        fmt == "mlx" or bool(model.get("mlx_only"))
        or name.startswith("mlx-community/") or _text(model.get("provider")).lower() == "mlx-community"
        or quant.lower().startswith("mlx-")
    )
    if is_mlx:
        return "mlx", has_gguf
    if model.get("is_gguf") or fmt == "gguf" or (has_gguf and "gguf" in name):
        return "gguf", True
    if is_prequantized({"name": model.get("name"), "format": model.get("format"), "quantization": quant}):
        return "accel_quant", has_gguf
    return "hf", has_gguf


def _status(status, reason):
    return {"status": status, "reason": reason}


def _ollama_runtime(source, cloud_only):
    runtime = {}
    for engine in RUNTIME_ENGINES:
        runtime[engine] = _status(NOT_SUPPORTED, "Ollama library tags are pulled and served by Ollama.")
    if cloud_only:
        runtime["ollama"] = _status(UNVERIFIED, "Listed as an Ollama Cloud model; no local sizes are listed.")
    elif source == "ollama_installed":
        runtime["ollama"] = _status(COMPATIBLE, "Installed on your Ollama server.")
    else:
        runtime["ollama"] = _status(COMPATIBLE, "Published in the Ollama library (may need a recent Ollama).")
    return runtime


def _hf_runtime(model, task):
    kind, has_gguf = _weights(model)
    label = TASK_LABELS[task]
    runtime = {}
    for engine in RUNTIME_ENGINES:
        name = ENGINE_LABELS[engine]
        if engine in ("diffusers", "mlx_image"):
            if task not in _DIFFUSION_TASKS:
                runtime[engine] = _status(NOT_SUPPORTED, f"{name} serves image/video generation models only.")
            elif engine == "mlx_image" and task == "video_generation":
                runtime[engine] = _status(NOT_SUPPORTED, "MLX Image serves image generation models only.")
            elif engine == "diffusers" and kind == "mlx":
                runtime[engine] = _status(NOT_SUPPORTED, "MLX weights run only through MLX on Apple Silicon.")
            else:
                runtime[engine] = _status(UNVERIFIED, f"{name} pipeline support for this model has not been tested.")
            continue
        if task in _NON_LLM_TASKS:
            runtime[engine] = _status(NOT_SUPPORTED, f"{label} models are not served by {name}.")
            continue
        if engine in ("llamacpp", "ollama"):
            if kind == "mlx":
                runtime[engine] = _status(NOT_SUPPORTED, f"MLX weights; {name} needs a GGUF build.")
            elif has_gguf:
                runtime[engine] = _status(UNVERIFIED, f"GGUF build listed; {name} support for this architecture is not verified.")
            elif kind == "accel_quant":
                runtime[engine] = _status(NOT_SUPPORTED, f"GPU-only quantized weights; {name} needs a GGUF build.")
            else:
                # Plain safetensors usually convert to GGUF, and Ollama's own
                # library often packages the same model, so this is unknown,
                # not incompatible. The Cookbook itself only serves listed GGUFs.
                runtime[engine] = _status(UNVERIFIED, f"No GGUF files in this listing; {name} needs a GGUF conversion or a separate GGUF build.")
        elif engine in ("vllm", "sglang"):
            if kind == "mlx":
                runtime[engine] = _status(NOT_SUPPORTED, f"MLX weights run only through MLX; {name} needs safetensors.")
            elif kind == "gguf":
                runtime[engine] = _status(UNVERIFIED, f"GGUF-only repo; {name} GGUF loading is experimental.")
            elif engine == "vllm" and model.get("vllm_recipe"):
                runtime[engine] = _status(COMPATIBLE, "Has an official vLLM recipe.")
            else:
                runtime[engine] = _status(UNVERIFIED, f"Weight format matches {name}; architecture support is not verified.")
        else:  # mlx
            if kind == "mlx":
                runtime[engine] = _status(UNVERIFIED, "MLX build; mlx-lm support for this model is not verified.")
            elif kind == "gguf":
                runtime[engine] = _status(NOT_SUPPORTED, "GGUF-only repo; MLX needs MLX or safetensors weights.")
            else:
                runtime[engine] = _status(UNVERIFIED, "mlx-lm may convert these weights; not verified.")
    return runtime


def classify_model_facets(model, *, source="catalog", ollama_capabilities=None, cloud_only=False):
    """Classify one recommendation row's task and per-engine runtime support.

    ``model`` is a catalog row (name, pipeline_tag, use_case, format,
    quantization, is_gguf/gguf_sources, vllm_recipe...). ``source`` is
    ``"catalog"`` for Hugging Face catalog rows, ``"ollama_library"`` or
    ``"ollama_installed"`` for Ollama tags; Ollama rows may pass
    ``ollama_capabilities``. For ``"ollama_library"`` those are the
    ollama.com card chips (hints); for ``"ollama_installed"`` they are the
    server's own ``/api/show`` report, classified by the chat-eligibility
    rule (no ``completion`` means not chat; no report means unknown).

    Returns ``{"task", "task_label", "task_source", "chat_capable", "runtime"}``
    where ``runtime`` maps every engine key the Cookbook can display to
    ``{"status": compatible|unverified|not_supported, "reason": str}``.
    ``chat_capable`` is ``None`` when the task is unknown.
    """
    if not isinstance(model, dict):
        model = {}
    is_ollama = source in ("ollama_library", "ollama_installed")
    if source == "ollama_installed":
        task, task_label = _native_ollama_task(ollama_capabilities)
        task_source = "ollama"
    else:
        caps = (ollama_capabilities or []) if is_ollama else None
        task, task_source = _classify_task(model, caps)
        task_label = TASK_LABELS[task]
    runtime = _ollama_runtime(source, cloud_only) if is_ollama else _hf_runtime(model, task)
    return {
        "task": task,
        "task_label": task_label,
        "task_source": task_source,
        "chat_capable": None if task == "unknown" else task in CHAT_TASKS,
        "runtime": runtime,
    }


def chat_rank_tier(row):
    """Sort tier for chat-oriented views: definite non-chat rows go last.

    Unknown tasks stay with chat rows (their Task facet says "Unknown"), so a
    catalog row without metadata is never hidden behind embeddings/TTS.
    """
    return 1 if row.get("chat_capable") is False else 0

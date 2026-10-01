# src/research_presets.py
"""Deep Research presets sized to the research model's hardware.

A preset bundles every knob that scales how much work (and how many model
calls) one research run costs: rounds, queries, URLs, page size fed to
extraction, generation budgets, how many findings synthesis sees, and whether
mechanical steps run with thinking disabled. ``auto`` picks small/medium/large
from the research model's context window and, for Ollama, its parameter size;
``custom`` keeps the pre-preset behavior driven by the individual settings.

The table here is the single source of truth; ``specs/research.md`` documents
it and ``/api/research/preset`` exposes it to the settings UI.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, replace
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

PRESET_CHOICES = ("auto", "small", "medium", "large", "custom")

# Default of ``research_max_tokens`` in src/settings.py. A saved value that
# differs from it predates presets and is treated as explicit tuning.
_LEGACY_MAX_TOKENS_DEFAULT = 16384

# Same chars→tokens ratio as src.model_context.estimate_tokens (chars * 0.3).
TOKENS_PER_CHAR = 0.3

# Auto thresholds: small covers ≤16K windows and sub-10B models (Ollama reports
# e.g. "9.0B"/"9.2B" for the 9B class), medium covers windows up to 64K.
_SMALL_MAX_WINDOW = 16384
_SMALL_MAX_PARAMS_B = 10.0
_MEDIUM_MAX_WINDOW = 65536


@dataclass(frozen=True)
class ResearchPreset:
    name: str
    label: str
    description: str
    max_rounds: int             # cap used when the user picks Auto rounds
    min_rounds: int             # rounds before the stop check is consulted
    queries_first_round: int
    queries_per_round: int
    urls_per_query: int
    page_chars: int             # page text fed to one extraction call
    extraction_max_tokens: int
    query_max_tokens: int
    synthesis_max_tokens: int   # 0 → research_max_tokens setting
    report_max_tokens: int      # 0 → research_max_tokens setting
    report_min_words: int       # final report target length
    expand_below_words: int     # re-ask for a longer report below this
    synthesis_findings: int     # findings carried into each synthesis
    mechanical_think: Optional[bool]  # False → think=False on mechanical steps


PRESETS = {
    "small": ResearchPreset(
        name="small", label="Small",
        description="Small local models (≤16K context or under 10B parameters): "
                    "few short calls, compact prompts.",
        max_rounds=4, min_rounds=2,
        queries_first_round=3, queries_per_round=2, urls_per_query=2,
        page_chars=6000, extraction_max_tokens=768, query_max_tokens=768,
        synthesis_max_tokens=2048, report_max_tokens=3072,
        report_min_words=600, expand_below_words=250, synthesis_findings=5,
        mechanical_think=False,
    ),
    "medium": ResearchPreset(
        name="medium", label="Medium",
        description="Mid-size local models (up to 64K context): balanced depth "
                    "and call count.",
        max_rounds=6, min_rounds=2,
        queries_first_round=4, queries_per_round=3, urls_per_query=2,
        page_chars=12000, extraction_max_tokens=1536, query_max_tokens=1024,
        synthesis_max_tokens=4096, report_max_tokens=6144,
        report_min_words=1000, expand_below_words=300, synthesis_findings=8,
        mechanical_think=False,
    ),
    "large": ResearchPreset(
        name="large", label="Large",
        description="Large-context or cloud models (over 64K context): deeper "
                    "rounds, full pages and long reports.",
        max_rounds=10, min_rounds=3,
        queries_first_round=5, queries_per_round=3, urls_per_query=3,
        page_chars=20000, extraction_max_tokens=2048, query_max_tokens=2048,
        synthesis_max_tokens=8192, report_max_tokens=12288,
        report_min_words=1500, expand_below_words=400, synthesis_findings=12,
        mechanical_think=False,
    ),
    "custom": ResearchPreset(
        name="custom", label="Custom",
        description="Pre-preset behavior: Max Tokens and the other Deep Research "
                    "settings apply as configured.",
        max_rounds=20, min_rounds=2,
        queries_first_round=4, queries_per_round=3, urls_per_query=3,
        page_chars=15000, extraction_max_tokens=2048, query_max_tokens=4096,
        synthesis_max_tokens=0, report_max_tokens=0,
        report_min_words=1500, expand_below_words=400, synthesis_findings=10,
        mechanical_think=None,
    ),
}

AUTO_DESCRIPTION = ("Pick Small, Medium or Large from the research model's "
                    "context window and parameter size.")


@dataclass(frozen=True)
class ResearchProfile:
    """A preset resolved for one research model."""
    requested: str                    # configured choice (auto/small/…/custom)
    preset: ResearchPreset            # effective preset, token budgets filled
    context_window: Optional[int]     # known window, None when unknown
    parameter_size_b: Optional[float]  # billions of parameters, None when unknown
    reason: str


def chars_for_tokens(tokens: int) -> int:
    """Inverse of the estimate_tokens ratio: how many chars fit in ``tokens``."""
    return max(0, int(tokens / TOKENS_PER_CHAR))


def configured_preset() -> str:
    """The configured ``research_preset`` choice.

    An empty/unknown value means "never chosen": installs whose saved
    ``research_max_tokens`` differs from its default keep their explicit
    tuning as ``custom``; everyone else gets ``auto``.
    """
    from src.settings import get_setting
    raw = str(get_setting("research_preset", "") or "").strip().lower()
    if raw in PRESET_CHOICES:
        return raw
    try:
        tuned = int(get_setting("research_max_tokens", _LEGACY_MAX_TOKENS_DEFAULT)) != _LEGACY_MAX_TOKENS_DEFAULT
    except (TypeError, ValueError):
        tuned = False
    return "custom" if tuned else "auto"


def auto_preset_name(context_window: Optional[int], parameter_size_b: Optional[float]) -> str:
    """Map a model's known window / size to small, medium or large."""
    if context_window and context_window <= _SMALL_MAX_WINDOW:
        return "small"
    if parameter_size_b is not None and parameter_size_b < _SMALL_MAX_PARAMS_B:
        return "small"
    if context_window and context_window <= _MEDIUM_MAX_WINDOW:
        return "medium"
    if context_window:
        return "large"
    return "medium"  # nothing known: stay in the middle


def research_round_limits(max_rounds: int, preset: ResearchPreset) -> Tuple[int, int, bool]:
    """Return ``(max_rounds, min_rounds, auto)`` for a requested round count.

    ``max_rounds <= 0`` is Auto: the preset caps the run and the model's stop
    check runs after the preset's small minimum. An explicit count keeps its
    old meaning — roughly that many rounds (the stop check starts two rounds
    before the end, never before the preset minimum).
    """
    try:
        requested = int(max_rounds or 0)
    except (TypeError, ValueError):
        requested = 0
    if requested <= 0:
        cap = max(1, preset.max_rounds)
        return cap, min(preset.min_rounds, cap), True
    return requested, max(min(requested, preset.min_rounds), requested - 2), False


# ---------------------------------------------------------------------------
# Ollama parameter size (/api/show details.parameter_size)
# ---------------------------------------------------------------------------
_PARAM_CACHE_TTL = 600.0
_PARAM_FAILURE_TTL = 60.0
_PARAM_PROBE_TIMEOUT = 1.5
_param_cache: dict = {}
_param_lock = threading.Lock()
_PARAM_RE = re.compile(r"^\s*([\d.]+)\s*([KMBT])?\s*$", re.IGNORECASE)
_PARAM_SCALE_B = {"K": 1e-6, "M": 1e-3, "B": 1.0, "T": 1e3}


def parse_parameter_size(value) -> Optional[float]:
    """Parse Ollama's ``parameter_size`` ("27.3B", "23M") into billions."""
    match = _PARAM_RE.match(str(value or ""))
    if not match:
        return None
    try:
        number = float(match.group(1))
    except ValueError:
        return None
    unit = (match.group(2) or "B").upper()
    return number * _PARAM_SCALE_B[unit] if number > 0 else None


def ollama_parameter_size_b(url: str, model: str) -> Optional[float]:
    """Parameter size of a local Ollama model in billions, ``None`` if unknown.

    Read-only ``POST /api/show`` with a short timeout, cached per model.
    Non-Ollama and non-local endpoints are never probed.
    """
    from src.model_context import is_local_endpoint
    from src.ollama_capabilities import ollama_api_root

    root = ollama_api_root(url)
    model = str(model or "").strip()
    if not root or not model or not is_local_endpoint(url):
        return None
    key = (root, model)
    now = time.time()
    with _param_lock:
        cached = _param_cache.get(key)
    if cached is not None:
        ts, size = cached
        if now - ts < (_PARAM_FAILURE_TTL if size is None else _PARAM_CACHE_TTL):
            return size
    size = None
    try:
        import httpx
        response = httpx.post(f"{root}/api/show", json={"model": model}, timeout=_PARAM_PROBE_TIMEOUT)
        if response.is_success:
            details = (response.json() or {}).get("details") or {}
            size = parse_parameter_size(details.get("parameter_size"))
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        logger.debug("Ollama parameter-size probe failed for %s: %s", model, exc)
    with _param_lock:
        _param_cache[key] = (now, size)
    return size


def _fmt_window(tokens: Optional[int]) -> str:
    if not tokens:
        return "unknown context"
    return f"{tokens // 1024}K context" if tokens >= 1024 else f"{tokens}-token context"


def resolve_research_profile(
    url: str,
    model: str,
    *,
    requested: Optional[str] = None,
    max_report_tokens: Optional[int] = None,
) -> ResearchProfile:
    """Resolve the configured preset for ``model`` at ``url``.

    Blocking (may probe the endpoint) — call via ``asyncio.to_thread`` from
    async code. Probe failures degrade to "unknown"; never raises.
    """
    choice = (requested or configured_preset()).strip().lower()
    if choice not in PRESET_CHOICES:
        choice = "auto"

    window: Optional[int] = None
    try:
        from src.model_context import get_context_length_known
        ctx, known = get_context_length_known(url, model)
        window = int(ctx) if known and ctx else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Research context probe failed for %s: %s", model, exc)

    params: Optional[float] = None
    if choice == "auto":
        try:
            params = ollama_parameter_size_b(url, model)
        except Exception:  # noqa: BLE001
            params = None
        name = auto_preset_name(window, params)
        facts = [_fmt_window(window)]
        if params is not None:
            facts.append(f"{params:g}B parameters")
        reason = f"{', '.join(facts)} → {PRESETS[name].label}"
    else:
        name = choice
        reason = f"{PRESETS[name].label} selected"

    preset = PRESETS[name]
    if name == "custom":
        try:
            budget = int(max_report_tokens or _LEGACY_MAX_TOKENS_DEFAULT)
        except (TypeError, ValueError):
            budget = _LEGACY_MAX_TOKENS_DEFAULT
        preset = replace(preset, synthesis_max_tokens=budget, report_max_tokens=budget)
    return ResearchProfile(
        requested=choice,
        preset=preset,
        context_window=window,
        parameter_size_b=params,
        reason=reason,
    )


def preset_table() -> list:
    """Serializable preset table for the settings UI."""
    rows = [{"name": "auto", "label": "Auto", "description": AUTO_DESCRIPTION}]
    rows.extend(asdict(PRESETS[name]) for name in ("small", "medium", "large", "custom"))
    return rows

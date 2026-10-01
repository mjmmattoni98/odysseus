"""Adaptive input-token budget for the agent loop (#1170).

The agent soft-trims its input context to ``agent_input_token_budget`` (default
6000). The old computation was ``min(context_length or budget, budget)``, which
made the 6000 default a hard ceiling for *every* model — so a 128K or 1M context
model was silently capped at 6000 input tokens even though it can hold far more.

This derives the effective budget from the model's discovered context window when
the user has NOT set an explicit budget, while still honouring an explicit setting
exactly (clamped to the window). Pure and side-effect free so it is unit-testable.

It also budgets the native tools payload (``estimate_tool_tokens``,
``tool_schema_token_budget``) and defines the small-context tier.
"""

import json
from typing import Optional

# Generous ceiling so long-context models are unblocked without sending a
# pathologically large prompt every agent turn. Tunable; chosen to fully cover
# 128K models and give 1M models a large but bounded budget.
DEFAULT_HARD_MAX = 200_000
DEFAULT_BUDGET = 6000
DEFAULT_HEADROOM = 0.85


def _int_or_zero(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def compute_input_token_budget(
    configured: int,
    context_length: int,
    explicit: bool,
    *,
    default: int = DEFAULT_BUDGET,
    headroom: float = DEFAULT_HEADROOM,
    hard_max: int = DEFAULT_HARD_MAX,
) -> int:
    """Return the effective soft input-token budget.

    Args:
        configured: the value read from settings (may be the default).
        context_length: the model's discovered context window. Pass 0 when the
            window is unknown / only a bare fallback — auto-scaling then stays
            conservative instead of trusting an unproven window (review on #4122).
        explicit: True if the user set a NON-default budget. The default value is
            the "auto" sentinel (scale to the window); any other value is an
            explicit cap. (A deliberately-chosen default can't be distinguished
            from a materialized default by value, so the default reads as auto.)

    Rules:
        - Explicit user budget is honoured exactly, only clamped to the model's
          window when that window is known (the user's deliberate choice wins;
          ``hard_max`` is an auto-budget ceiling only — see #1230).
        - Otherwise (auto), scale to ``headroom`` of the context window, capped at
          ``hard_max`` — so long-context models use their capacity.
        - When the window is unknown (context_length <= 0), use the conservative
          ``default`` budget and do NOT scale off the fallback.
    """
    configured = _int_or_zero(configured)
    context_length = _int_or_zero(context_length)

    if explicit and configured > 0:
        return min(configured, context_length) if context_length > 0 else configured

    if context_length > 0:
        scaled = int(context_length * headroom)
        return max(1, min(scaled, hard_max))

    return configured if configured > 0 else default


def budget_is_explicit(configured: int, *, default: int = DEFAULT_BUDGET) -> bool:
    """Whether a configured agent_input_token_budget is a deliberate explicit cap.

    The default value is the "auto" sentinel (scale to the model's window), so only
    a NON-default positive value counts as explicit. This keys off the VALUE, not
    settings *presence* — the settings-save path materializes every default into
    settings.json, so a persisted default must still read as auto (the regression
    #4121 / #1230 are about). Centralised here so the materialized-default contract
    is unit-testable and can't silently regress to a presence check.
    """
    configured = int(configured or 0)
    return configured > 0 and configured != default


# ---------------------------------------------------------------------------
# Tool-schema budgeting and the small-context tier
# ---------------------------------------------------------------------------

# Same rough chars -> tokens convention as src.model_context.estimate_tokens.
TOKENS_PER_CHAR = 0.3

# Models whose known window is at or below this get the small-context tier:
# fewer retrieved tools, no whole domain packs, short schema descriptions and
# tool outputs scaled to the window.
SMALL_CONTEXT_TIER_MAX = 16384

# Window assumed when a local route does not report one. Matches the default
# local context allocation (src.assistant_preferences.DEFAULT_LOCAL_CONTEXT_LIMIT)
# so an unknown local model is budgeted like a default Ollama allocation rather
# than like a 128K cloud model. Unknown windows never enter the small tier.
DEFAULT_LOCAL_CONTEXT_WINDOW = 32768

# Local routes render every schema into the prompt prefix. Keep the tools
# payload to this share of the window, within an absolute floor/ceiling.
TOOL_SCHEMA_BUDGET_FRACTION = 0.15
TOOL_SCHEMA_BUDGET_FLOOR = 1200
TOOL_SCHEMA_BUDGET_CEILING = 10000


def estimate_tool_tokens(tools) -> int:
    """Rough token cost of a native ``tools`` payload.

    ``estimate_tokens`` only sees messages, but providers render every schema
    into the prompt (Ollama templates put them in the prefix), so budgets that
    ignore the payload under-count action turns by thousands of tokens. Uses
    the chars * 0.3 convention on the compact JSON serialization, plus a small
    per-tool wrapper overhead.
    """
    total = 0
    for tool in tools or ():
        try:
            text = json.dumps(tool, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            text = str(tool)
        total += 4 + int(len(text) * TOKENS_PER_CHAR)
    return total


def is_small_context_window(context_window) -> bool:
    """True for a KNOWN window at or below SMALL_CONTEXT_TIER_MAX."""
    window = _int_or_zero(context_window)
    return 0 < window <= SMALL_CONTEXT_TIER_MAX


def tool_schema_token_budget(context_window) -> int:
    """Token budget for the tools payload of a local route.

    ~15% of the window, bounded to [TOOL_SCHEMA_BUDGET_FLOOR,
    TOOL_SCHEMA_BUDGET_CEILING]; the floor never exceeds 30% of a tiny window.
    An unknown window (0) is budgeted as DEFAULT_LOCAL_CONTEXT_WINDOW.
    """
    window = _int_or_zero(context_window) or DEFAULT_LOCAL_CONTEXT_WINDOW
    floor = min(TOOL_SCHEMA_BUDGET_FLOOR, int(window * 0.3))
    scaled = int(window * TOOL_SCHEMA_BUDGET_FRACTION)
    return max(1, max(floor, min(scaled, TOOL_SCHEMA_BUDGET_CEILING)))


def tool_output_char_cap(context_window) -> Optional[int]:
    """Per-result character cap for tool output fed back to a small model.

    Returns None outside the small tier (the tools' own MAX_OUTPUT_CHARS caps
    apply unchanged). Inside it, one tool result may use about
    SMALL_CONTEXT_TOOL_OUTPUT_FRACTION of the window.
    """
    if not is_small_context_window(context_window):
        return None
    from src.constants import MIN_SCALED_TOOL_OUTPUT_CHARS, SMALL_CONTEXT_TOOL_OUTPUT_FRACTION

    tokens = int(_int_or_zero(context_window) * SMALL_CONTEXT_TOOL_OUTPUT_FRACTION)
    return max(MIN_SCALED_TOOL_OUTPUT_CHARS, int(tokens / TOKENS_PER_CHAR))

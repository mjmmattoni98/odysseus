"""Local-route native tool payload planning.

Local routes render every schema into the prompt prefix, so the payload is
capped by a token budget (a share of the model window) with a count ceiling as
a safety net. Pinned tools (always-on loop primitives, per-request forced
tools) survive the cap; lower-priority tools are dropped first; the order is
stable so an unchanged selection sends a byte-identical payload.
"""
import json

from src.agent_loop import (
    _LOCAL_TOOL_SCHEMA_COUNT_CEILING,
    _TOOL_RANK_PINNED,
    _TOOL_RANK_RETRIEVED,
    _plan_tool_schemas,
)
from src.context_budget import estimate_tool_tokens


def _schema(name, desc_chars=40):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "d" * desc_chars,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _names(schemas):
    return [s["function"]["name"] for s in schemas]


def test_small_sets_untouched():
    schemas = [_schema(f"t{i}") for i in range(5)]
    assert _plan_tool_schemas(schemas, token_budget=10_000) == schemas


def test_none_and_empty_are_safe():
    assert _plan_tool_schemas([]) == []
    assert _plan_tool_schemas(None) == []


def test_count_ceiling_is_a_safety_net_preserving_order():
    schemas = [_schema(f"t{i}") for i in range(60)]
    capped = _plan_tool_schemas(schemas, max_count=_LOCAL_TOOL_SCHEMA_COUNT_CEILING)
    assert _names(capped) == [f"t{i}" for i in range(_LOCAL_TOOL_SCHEMA_COUNT_CEILING)]


def test_token_budget_drops_lowest_priority_first_and_keeps_pinned():
    schemas = [_schema(f"t{i}", desc_chars=400) for i in range(10)]
    one = estimate_tool_tokens([schemas[0]])
    ranks = {"t9": _TOOL_RANK_PINNED, "t8": _TOOL_RANK_PINNED, "t0": _TOOL_RANK_RETRIEVED}
    # Budget for three tools: both pinned tools plus the best-ranked other.
    capped = _plan_tool_schemas(schemas, ranks=ranks, token_budget=one * 3)
    names = _names(capped)
    assert {"t8", "t9"} <= set(names)
    assert len(names) == 3
    # A retrieval-ranked tool outranks unranked (admin/other) tools.
    assert "t0" in names and "t1" not in names
    assert estimate_tool_tokens(capped) <= one * 3


def test_pinned_tools_survive_even_an_exhausted_budget():
    schemas = [_schema("big", desc_chars=5000), _schema("ask_user")]
    capped = _plan_tool_schemas(schemas, ranks={"ask_user": _TOOL_RANK_PINNED}, token_budget=10)
    assert _names(capped) == ["ask_user"]


def test_established_order_first_then_canonical_additions():
    schemas = [_schema(n) for n in ("a", "b", "c", "d")]
    planned = _plan_tool_schemas(schemas, established_order=["c", "a", "gone"])
    assert _names(planned) == ["c", "a", "b", "d"]


def test_same_inputs_give_byte_identical_payload():
    schemas = [_schema(f"t{i}", desc_chars=300) for i in range(30)]
    kwargs = dict(ranks={"t3": 0}, established_order=["t5", "t3"], token_budget=900, max_count=24)
    first = _plan_tool_schemas(schemas, **kwargs)
    second = _plan_tool_schemas(list(schemas), **kwargs)
    assert json.dumps(first) == json.dumps(second)

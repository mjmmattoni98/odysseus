"""Ollama native-tool schema capping.

Local models choke on the full ~70-schema catalog; Ollama routes are capped to
a curated head plus any explicitly relevant tools. Cloud routes are untouched.
"""
from src.agent_loop import (
    _OLLAMA_NATIVE_TOOL_SCHEMA_LIMIT,
    _cap_ollama_tool_schemas,
)


def _schema(name):
    return {"type": "function", "function": {"name": name}}


def test_small_sets_untouched():
    schemas = [_schema(f"t{i}") for i in range(5)]
    assert _cap_ollama_tool_schemas(schemas, set()) == schemas


def test_none_and_empty_are_safe():
    assert _cap_ollama_tool_schemas([], set()) == []
    assert _cap_ollama_tool_schemas(None, set()) is None


def test_large_set_is_capped_preserving_order():
    schemas = [_schema(f"t{i}") for i in range(60)]
    capped = _cap_ollama_tool_schemas(schemas, set())
    assert len(capped) == _OLLAMA_NATIVE_TOOL_SCHEMA_LIMIT
    assert [s["function"]["name"] for s in capped] == [
        f"t{i}" for i in range(_OLLAMA_NATIVE_TOOL_SCHEMA_LIMIT)
    ]


def test_relevant_tools_are_kept_first():
    schemas = [_schema(f"t{i}") for i in range(60)]
    capped = _cap_ollama_tool_schemas(schemas, {"t42", "t7"})
    names = [s["function"]["name"] for s in capped]
    assert names[:2] == ["t7", "t42"]
    assert len(names) == _OLLAMA_NATIVE_TOOL_SCHEMA_LIMIT
    assert "t0" in names

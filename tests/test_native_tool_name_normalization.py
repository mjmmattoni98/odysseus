"""Native function-call name normalization for decorated model output.

Local models occasionally decorate native tool names. Observed in the wild:
Ollama + gemma4 emits ``web_search:search``, which previously fell through
``_TOOL_NAME_MAP`` as an unknown name, logged "Unknown function call", and
was silently dropped from the agent round.

The normalizer must resolve decorated *known* tools, preserve namespaced MCP
tools (``mcp__server__tool``) untouched, and still fail closed on genuinely
unknown names.
"""
import src.agent_tools  # noqa: F401  (break agent_tools <-> tool_parsing import cycle)

from src.tool_schemas import function_call_to_tool_block, normalize_native_tool_name


class TestNormalizeNativeToolName:
    def test_colon_suffix_resolves_to_tool(self):
        assert normalize_native_tool_name("web_search:search") == "web_search"

    def test_colon_prefix_resolves_to_tool(self):
        assert normalize_native_tool_name("web_search:query") == "web_search"

    def test_dot_prefix_resolves_to_tool(self):
        assert normalize_native_tool_name("functions.web_search") == "web_search"
        assert normalize_native_tool_name("tools.web_search") == "web_search"

    def test_bare_name_unchanged(self):
        assert normalize_native_tool_name("web_search") == "web_search"

    def test_mcp_names_are_preserved(self):
        assert normalize_native_tool_name("mcp__rag__search_documents") == "mcp__rag__search_documents"
        assert normalize_native_tool_name("mcp__email__send_email") == "mcp__email__send_email"

    def test_unknown_decorated_name_passes_through(self):
        assert normalize_native_tool_name("frobnicate:run") == "frobnicate:run"

    def test_empty_name(self):
        assert normalize_native_tool_name("") == ""
        assert normalize_native_tool_name(None) == ""


class TestFunctionCallToToolBlockNamespacing:
    def test_colon_namespaced_web_search_converts(self):
        block = function_call_to_tool_block("web_search:search", '{"query": "amd rocm"}')
        assert block is not None
        assert block.tool_type == "web_search"
        assert block.content == "amd rocm"

    def test_dot_prefixed_web_fetch_converts(self):
        block = function_call_to_tool_block(
            "functions.web_fetch", '{"url": "https://example.com"}'
        )
        assert block is not None
        assert block.tool_type == "web_fetch"
        assert block.content == '{"url": "https://example.com"}'

    def test_time_filter_survives_normalization(self):
        block = function_call_to_tool_block(
            "web_search:search", '{"query": "amd rocm", "time_filter": "week"}'
        )
        assert block is not None
        assert block.tool_type == "web_search"
        assert block.content == '{"query": "amd rocm", "time_filter": "week"}'

    def test_empty_required_argument_still_rejected(self):
        assert function_call_to_tool_block("web_search:search", '{"query": ""}') is None

    def test_unknown_decorated_name_still_fails_closed(self):
        assert function_call_to_tool_block("frobnicate:run", '{"x": 1}') is None

    def test_mcp_tool_passes_through(self):
        block = function_call_to_tool_block("mcp__demo__do_thing", '{"a": 1}')
        assert block is not None
        assert block.tool_type == "mcp__demo__do_thing"
        assert block.content == '{"a": 1}'

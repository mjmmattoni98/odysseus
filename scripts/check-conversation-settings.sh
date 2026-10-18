#!/usr/bin/env bash
# Run against this checkout, optionally using an existing application's image.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
assistant_tests=(
  tests/test_assistant_preferences.py
  tests/test_assistant_preferences_api.py
  tests/test_search_evidence.py
  tests/test_llm_core_ollama.py
  tests/test_llm_core_ollama_thinking.py
  tests/test_ollama_capabilities.py
  tests/test_model_context.py
  tests/test_search_content_block_source_index.py
  tests/test_chat_processor_web_search.py
  tests/test_chat_preprocess_tool_policy.py
  tests/test_chat_route_tool_policy.py
  tests/test_web_fetch_size_caps.py
  tests/test_web_search_time_filter.py
  tests/test_session_endpoint_owner_scope.py
  tests/test_tool_policy.py
  tests/test_foreground_model_routing.py
  tests/test_external_context_tool_gate.py
  tests/test_deep_research_search_error.py
  tests/test_deep_research_synthesis_resilience.py
  tests/test_deep_research_extraction_controls.py
  tests/test_new_chat_model_preference.py
)
if [[ -n "${ODYSSEUS_TEST_IMAGE:-}" ]]; then
  docker run --rm --network none --entrypoint python -v "$PWD:/app:ro" -w /app \
    -e ODYSSEUS_DATA_DIR=/tmp/odysseus-test -e DATABASE_URL=sqlite:///:memory: \
    -e PYTHONDONTWRITEBYTECODE=1 "$ODYSSEUS_TEST_IMAGE" \
    -m pytest -q -p no:cacheprovider "${assistant_tests[@]}"
else
  ODYSSEUS_DATA_DIR="$(mktemp -d)" DATABASE_URL=sqlite:///:memory: \
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider "${assistant_tests[@]}"
fi
node --test tests/assistant_controls.test.mjs tests/search_evidence.test.mjs
for assistant_js in assistantControls searchEvidence chat chatRenderer sessions; do
  node --check "static/js/$assistant_js.js"
done
git diff --check

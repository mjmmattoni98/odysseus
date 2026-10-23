# Conversation settings

The sliders button beside the web toggle opens settings saved with the chat: custom instructions, thinking effort, and the local context cap. They do not change which tools a turn may use; the Chat/Agent switch and the web and shell toggles decide that, as described in [Chat](chat.md). The model picker still selects the model for the conversation. The sliders button is highlighted while the chat has instructions, a thinking override, or a context cap different from the default.

Custom instructions are appended to the system prompt. For example: “When helping me decide, compare total cost and maintenance, give your recommendation, and explain what would change your mind.” Each turn takes a snapshot; changing settings while an answer is streaming applies to later turns.

Assistant profiles (Everyday, Research, Actions) were removed. Conversations saved with one still load: the stored `profile` and `web_mode` keys are dropped when read and ignored when written, and the other settings keep their values.

## Web access and evidence

The magnifying-glass toggle is remembered separately for Chat and Agent mode (on by default in Agent, off in Chat). In Chat mode, on searches before answering, using a short query the model extracts from the message. In Agent mode, on offers the web tools to the model and off withholds them. Deep Research is a separate explicit workflow.

Sources show whether only a search snippet was available, page text was retrieved, retrieval failed, or the text was partial. Search status reports provider fallback, empty results, and failures. Evidence is stored in assistant-message metadata, including interrupted responses, so reloading history retains it. Citation numbers stay stable across multiple searches and fetches within one answer. Retrieval status describes available evidence; it does not certify a source's accuracy.

## Local model controls

Thinking controls appear only when the selected local Ollama model advertises thinking support. **Default** uses the Settings value (`local_thinking_effort`, reported as `runtime.default_thinking` and shown in the menu); any other choice overrides it for this conversation. Native Ollama requests have a default context allocation cap set by `local_context_limit_default` (Settings → AI Defaults → Local Model Context & Residency; 1,024–262,144, default 32,768; `src.assistant_preferences.default_context_limit()`). Background (non-turn) calls always use it. Each conversation can override that cap per model, from 1,024 to 262,144 tokens; the conversation settings show the default and the model maximum, offer **Use default**, and warn that a cap different from the default makes Ollama reload the model when calls switch sizes. A discovered model maximum further bounds the request. The UI distinguishes the model's advertised maximum from the allocation currently loaded by Ollama.

Ollama's OpenAI-compatible `/v1` interface retains server-controlled context allocation; the UI explains the server settings instead of presenting an ineffective local override. For budgeting, a `/v1` model that is not loaded uses the last `/api/ps` allocation seen for it, else the smaller of its advertised maximum and the default cap. Ollama Cloud does not receive local context caps. Fallback models do not inherit another model's custom cap.

When the existing local-inference gate is enabled, foreground chat takes priority over queued background/research calls. An active research inference finishes before yielding; periodic background calls retain their existing cancellation behavior. This queues inference calls, not webpage downloads, and does not unload models or stop research jobs.

## Persistence and API

`GET /api/session/{id}/assistant` returns `preferences` and model `runtime` information. `PUT` accepts:

```json
{
  "thinking": "auto",
  "instructions": "Keep answers concise; explain important tradeoffs.",
  "context_limits": {"my-local-model:latest": 32768}
}
```

Both routes verify session ownership. Invalid values return 422. Preferences live in the nullable `sessions.assistant_preferences` JSON column; startup adds the column to older databases without rewriting existing rows. A null value means the defaults above.

Core behavior is covered by `tests/test_assistant_preferences.py`, `tests/test_assistant_preferences_api.py`, `tests/test_search_evidence.py`, and the JavaScript controller/evidence tests. `scripts/check-conversation-settings.sh` runs the relevant regression suite using an installed Python environment, or an existing Odysseus image with `ODYSSEUS_TEST_IMAGE` set.

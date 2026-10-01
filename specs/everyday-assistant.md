# Everyday assistant

The conversation settings button beside the web toggle selects a profile and saves it with the chat. New browser chats start with Everyday, automatic web access, and thinking off. Existing conversations retain their previous tool and thinking behavior until a profile is selected. The model picker still selects the model for the conversation.

- **Everyday** supports questions, explanations, recommendations, and decisions. It permits search, page fetches, chat search, reference-file reading, and clarification questions. It blocks action tools and MCP execution.
- **Research and decisions** has the same tool restrictions, defaults web access to On and thinking to the model default, and asks for primary sources, conflicting evidence, and uncertainties. Deep Research remains a separate explicit workflow.
- **Actions and tools** enables the existing agent workflow, subject to the user's tool toggles, privileges, and approval rules. Choosing it does not automatically grant shell access.
- **Existing settings** retains the legacy Chat/Agent controls and global thinking setting.

Custom instructions are conversation-scoped. For example: “When helping me decide, compare total cost and maintenance, give your recommendation, and explain what would change your mind.” Each turn takes a snapshot; changing settings while an answer is streaming applies to later turns.

## Web access and evidence

**Off** disables chat web search, automatic URL fetching, and Deep Research dispatch from the composer. **Auto** performs an initial search for recognizable current-information, recommendation, comparison, or verification requests; web tools remain available for follow-up lookups. **On** searches before answering. An explicit per-request web denial overrides the saved setting. Everyday and Research use the user's question as the initial query instead of spending another local inference rewriting it.

Sources show whether only a search snippet was available, page text was retrieved, retrieval failed, or the text was partial. Search status reports provider fallback, empty results, and failures. Evidence is stored in assistant-message metadata, including interrupted responses, so reloading history retains it. Citation numbers stay stable across multiple searches and fetches within one answer. Retrieval status describes available evidence; it does not certify a source's accuracy.

## Local model controls

Thinking controls appear only when the selected local Ollama model advertises thinking support. Native Ollama requests have a default context allocation cap set by `local_context_limit_default` (Settings → AI Defaults → Local Model Context & Residency; 1,024–262,144, default 32,768; `src.assistant_preferences.default_context_limit()`). Background (non-turn) calls always use it. Each conversation can override that cap per model, from 1,024 to 262,144 tokens; the conversation settings note that a cap different from the default makes Ollama reload the model when calls switch sizes. A discovered model maximum further bounds the request. The UI distinguishes the model's advertised maximum from the allocation currently loaded by Ollama. The default allocation cap also applies to legacy native local requests.

Ollama's OpenAI-compatible `/v1` interface retains server-controlled context allocation; the UI explains the server settings instead of presenting an ineffective local override. For budgeting, a `/v1` model that is not loaded uses the last `/api/ps` allocation seen for it, else the smaller of its advertised maximum and the default cap. Ollama Cloud does not receive local context caps. Fallback models do not inherit another model's custom cap.

When the existing local-inference gate is enabled, foreground chat takes priority over queued background/research calls. An active research inference finishes before yielding; periodic background calls retain their existing cancellation behavior. This queues inference calls, not webpage downloads, and does not unload models or stop research jobs.

## Persistence and API

`GET /api/session/{id}/assistant` returns `preferences` and model `runtime` information. `PUT` accepts:

```json
{
  "profile": "everyday",
  "web_mode": "auto",
  "thinking": "off",
  "instructions": "Keep answers concise; explain important tradeoffs.",
  "context_limits": {"my-local-model:latest": 32768}
}
```

Both routes verify session ownership. Invalid values return 422. Preferences live in the nullable `sessions.assistant_preferences` JSON column; startup adds the column to older databases without rewriting existing rows. A null value means legacy behavior. API clients creating a session can opt into a profile with this PUT endpoint before sending their first message.

Core behavior is covered by `tests/test_assistant_preferences.py`, `tests/test_assistant_preferences_api.py`, `tests/test_search_evidence.py`, the agent execution tests in `tests/test_tool_policy.py`, and the JavaScript controller/evidence tests. `scripts/check-everyday-assistant.sh` runs the relevant regression suite using an installed Python environment, or an existing Odysseus image with `ODYSSEUS_TEST_IMAGE` set.

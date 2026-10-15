// ============================================
// Cookbook recommendation facets — pure helpers (no DOM, no imports) so they
// can be unit-tested under node (tests/cookbook_facets.test.mjs).
//
// Each Scan / Download row carries three independent facets:
//   memory fit  — fit_level from the hwfit ranker (do weights + KV fit?)
//   task        — task/task_label/chat_capable (services/hwfit/facets.py)
//   runtime     — runtime[engine].status/reason, per engine (same module)
// The backend does the classification; these helpers only label it.
// ============================================

// Labels say "memory" explicitly so a roomy fit never reads as "a good
// model for you": fit_level knows nothing about task or engine support.
const _MEMORY_FIT = {
  perfect: { text: 'ample', title: 'Memory: fits with plenty of headroom' },
  good: { text: 'fits', title: 'Memory: fits' },
  marginal: { text: 'tight', title: 'Memory: fits, but with little headroom' },
  too_tight: { text: 'no fit', title: 'Memory: does not fit the selected budget' },
  no_fit: { text: 'no fit', title: 'Memory: does not fit the selected budget' },
};

const _MEMORY_NOTE = 'Memory fit only; see Task and Engine for what the model does and whether it runs there.';

export function memoryFitFacet(fitLevel) {
  const hit = _MEMORY_FIT[fitLevel];
  if (hit) return { text: hit.text, title: `${hit.title}. ${_MEMORY_NOTE}` };
  return { text: '?', title: `Memory: not estimated. ${_MEMORY_NOTE}` };
}

export const RUNTIME_STATUS_LABELS = {
  compatible: 'compatible',
  unverified: 'unverified',
  not_supported: 'not supported',
};

// Runtime support of `model` on the engine the row displays. Rows without
// backend data (old cached scans) fall back to "unverified" — never to
// "compatible".
export function runtimeFacet(model, engine) {
  const entry = model && model.runtime && engine ? model.runtime[engine] : null;
  const status = entry && RUNTIME_STATUS_LABELS[entry.status] ? entry.status : 'unverified';
  const reason = entry && RUNTIME_STATUS_LABELS[entry.status]
    ? String(entry.reason || '')
    : 'No compatibility data for this engine; rescan to refresh.';
  return { status, label: RUNTIME_STATUS_LABELS[status], reason };
}

export function taskFacet(model) {
  const task = typeof model?.task === 'string' && model.task ? model.task : 'unknown';
  const label = typeof model?.task_label === 'string' && model.task_label ? model.task_label : 'Unknown';
  const chatCapable = typeof model?.chat_capable === 'boolean' ? model.chat_capable : null;
  let title = `Task: ${label}`;
  if (chatCapable === false) title += ' (not a chat model)';
  else if (chatCapable === null) title += ' (catalog has no task metadata for this model)';
  return { task, label, chatCapable, title };
}

// Stable partition for chat-oriented views: rows known NOT to be chat models
// (embeddings, speech, time series, base checkpoints...) go after the rest.
// Unknown-task rows stay up top; their Task cell says "Unknown". Returns
// { rows, firstOtherIndex } where firstOtherIndex is -1 when the list has no
// such boundary (all chat, or all non-chat).
export function partitionChatFirst(rows) {
  const list = Array.isArray(rows) ? rows : [];
  const chat = [];
  const other = [];
  for (const row of list) (row?.chat_capable === false ? other : chat).push(row);
  return {
    rows: chat.concat(other),
    firstOtherIndex: chat.length && other.length ? chat.length : -1,
  };
}

// Tokens in an Ollama capability report that mean "can chat" (the same set
// as src/ollama_capabilities.py _CHAT_TOKENS).
const _OLLAMA_CHAT_CAPS = ['completion', 'completions', 'chat'];

// Task facet for an installed Ollama model, from the capabilities the
// server reports (/api/show). The report is evidence, so it follows the
// backend chat-eligibility rule (services/hwfit/facets.py
// _native_ollama_task): "completion" means chat; a report without it is a
// model Ollama refuses to chat with, named from what it does report
// (["decision", "vision"] is a vision classifier); no report is unknown.
// Library card chips are only hints and are classified server-side.
// Cases shared with Python: tests/fixtures/ollama_native_capability_cases.json.
export function installedOllamaFacets(capabilities) {
  const caps = new Set((Array.isArray(capabilities) ? capabilities : [])
    .map(c => String(c).trim().toLowerCase()).filter(Boolean));
  let task = 'unknown';
  let label = 'Unknown';
  let chatCapable = null;
  if (caps.size) {
    chatCapable = _OLLAMA_CHAT_CAPS.some(c => caps.has(c));
    if (chatCapable) [task, label] = caps.has('vision') ? ['vision', 'Vision chat'] : ['chat', 'Chat'];
    else if (caps.has('embedding')) [task, label] = ['embedding', 'Embedding'];
    else if (caps.has('decision')) [task, label] = ['classification', caps.has('vision') ? 'Vision classifier' : 'Classifier'];
    else [task, label] = ['other', `No chat: ${[...caps].sort().join(', ')}`];
  }
  return {
    task,
    task_label: label,
    task_source: 'ollama',
    chat_capable: chatCapable,
    runtime: { ollama: { status: 'compatible', reason: 'Installed on your Ollama server.' } },
  };
}

// A library row whose tag is installed: the server's own report replaces
// the card hint. Without a report the library hint is all there is.
export function withInstalledOllamaFacets(row, capabilities) {
  if (!Array.isArray(capabilities) || !capabilities.some(c => String(c).trim())) return row;
  return { ...row, ...installedOllamaFacets(capabilities) };
}

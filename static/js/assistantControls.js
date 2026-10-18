import ui from './ui.js?v=20260916largetoolscroll1';

const defaults = () => ({ thinking: 'auto', instructions: '', context_limits: {} });
let preferences = defaults();
let runtime = {};
let sessionId = null;
let generation = 0;
let loading = Promise.resolve();
let loadError = null;
let dirty = true;
let revision = 0;
let saves = Promise.resolve();
const el = id => document.getElementById(id);

// Ollama reloads a model whenever num_ctx changes between calls, and
// background calls always use the default cap.
export function contextReloadNote(value) {
  const fallback = Number(runtime.default_context_limit);
  if (!runtime.native || !fallback || Number(value) === fallback) return '';
  return ` This differs from the default (${fallback.toLocaleString()} tokens): Ollama reloads the model when calls switch between the two sizes.`;
}

function runtimeText(contextValue) {
  return runtime.local_ollama
    ? `${runtime.model}. ${runtime.loaded_context ? `Loaded context: ${runtime.loaded_context.toLocaleString()} tokens.` : 'Model is not loaded.'} ${runtime.maximum_context ? `Model maximum: ${runtime.maximum_context.toLocaleString()} tokens.` : ''} ${runtime.native ? `Context limit applies to this model in this conversation.${contextReloadNote(contextValue)}` : 'Ollama controls context and keep-alive for this connection. Set OLLAMA_CONTEXT_LENGTH and OLLAMA_KEEP_ALIVE on the server.'}`
    : 'The selected model is saved with this conversation.';
}

function render() {
  if (!el('assistant-settings-btn')) return;
  el('assistant-thinking').value = preferences.thinking;
  el('assistant-instructions').value = preferences.instructions;
  el('assistant-thinking-row').hidden = runtime.supports_thinking !== true;
  el('assistant-context-row').hidden = !runtime.native;
  el('assistant-context').value = preferences.context_limits[runtime.model] || runtime.context_limit || runtime.default_context_limit || 32768;
  el('assistant-runtime').textContent = runtimeText(el('assistant-context').value);
}

export function loadSession(id) {
  const version = ++generation;
  sessionId = id;
  loadError = null;
  preferences = defaults();
  runtime = {};
  dirty = !id;
  render();
  loading = id ? fetch(`/api/session/${encodeURIComponent(id)}/assistant`).then(async response => {
    if (!response.ok) throw new Error('Could not load conversation settings');
    const data = await response.json();
    if (version !== generation) return;
    preferences = data.preferences;
    runtime = data.runtime;
    render();
  }).catch(error => {
    if (version === generation) { loadError = error; ui.showError(error.message); }
  }) : Promise.resolve();
  return loading;
}

export async function prepareTurn(id) {
  const version = generation;
  await loading;
  if (version !== generation) throw new Error('Conversation changed. Send again in the selected chat.');
  if (loadError) throw loadError;
  if (sessionId && sessionId !== id) throw new Error('Conversation changed. Send again in the selected chat.');
  const snapshot = structuredClone(preferences);
  const savedRevision = revision;
  if (dirty || !sessionId) {
    const save = saves.catch(() => {}).then(async () => {
      const response = await fetch(`/api/session/${encodeURIComponent(id)}/assistant`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(snapshot),
      });
      if (!response.ok) throw new Error('Could not save conversation settings');
    });
    saves = save;
    await save;
    if (version !== generation) throw new Error('Conversation changed. Send again in the selected chat.');
    sessionId = id;
    dirty = revision !== savedRevision;
  }
  return snapshot;
}

function init() {
  el('assistant-settings-btn')?.addEventListener('click', async () => {
    await loading;
    if (sessionId && !dirty) await loadSession(sessionId);
    if (loadError) return;
    render();
    el('assistant-settings-modal').classList.remove('hidden');
    el('assistant-instructions').focus();
  });
  el('assistant-context')?.addEventListener('input', event => {
    el('assistant-runtime').textContent = runtimeText(event.target.value);
  });
  el('assistant-close')?.addEventListener('click', () => el('assistant-settings-modal').classList.add('hidden'));
  el('assistant-save')?.addEventListener('click', async () => {
    const context = el('assistant-context');
    if (runtime.native && !context.reportValidity()) return;
    preferences = {
      ...preferences,
      thinking: el('assistant-thinking').value,
      instructions: el('assistant-instructions').value,
      context_limits: { ...preferences.context_limits },
    };
    if (runtime.native && runtime.model) preferences.context_limits[runtime.model] = Number(context.value);
    dirty = true;
    revision++;
    try {
      if (sessionId) await prepareTurn(sessionId);
      render();
      el('assistant-settings-modal').classList.add('hidden');
    } catch (error) { ui.showError(error.message); }
  });
  render();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
else init();

export default { loadSession, prepareTurn, contextReloadNote };

import ui from './ui.js?v=20260916largetoolscroll1';

const defaults = () => ({ search_mode: 'auto', thinking: 'auto', instructions: '', context_limits: {} });
const searchModes = {
  off: { label: 'Off', description: 'Answer without using the web' },
  auto: { label: 'Auto', description: 'Search only when needed' },
  always: { label: 'Always', description: 'Search before every answer' },
};
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

export function searchMode() { return preferences.search_mode; }

export function renderSearchMode() {
  const mode = searchModes[searchMode()];
  const button = el('web-toggle-btn');
  const label = el('web-mode-label');
  if (label) label.textContent = mode.label;
  if (button) {
    button.title = `Web: ${mode.label} — ${mode.description}`;
    button.setAttribute('aria-label', `Web: ${mode.label}. ${mode.description}`);
    button.classList.toggle('active', searchMode() !== 'off');
  }
  // Compatibility for Compare's shared pre-search. This checkbox is derived,
  // never an independent preference and never sent by the main composer.
  if (el('web-toggle')) el('web-toggle').checked = searchMode() === 'always';
  el('web-search-menu')?.querySelectorAll('[data-search-mode]').forEach(option => {
    option.setAttribute('aria-checked', String(option.dataset.searchMode === searchMode()));
  });
}

export async function setSearchMode(mode) {
  if (!Object.hasOwn(searchModes, mode)) throw new Error('Invalid web search mode');
  const version = generation;
  await loading;
  if (version !== generation) throw new Error('Conversation changed. Choose the web mode again.');
  if (loadError) throw loadError;
  preferences = { ...preferences, search_mode: mode };
  dirty = true;
  revision++;
  renderSearchMode();
  if (sessionId) await prepareTurn(sessionId);
}

const efforts = { auto: 'Auto', off: 'Off', low: 'Low', medium: 'Medium', high: 'High' };
const tokens = value => Number(value).toLocaleString();

// Ollama reloads a model whenever num_ctx changes between calls, and
// background calls always use the default cap.
export function contextReloadNote(value) {
  const fallback = Number(runtime.default_context_limit);
  if (!runtime.native || !fallback || Number(value) === fallback) return '';
  return `Differs from the default, so Ollama reloads the model when calls switch between ${tokens(value)} and ${tokens(fallback)} tokens.`;
}

function contextWarning(value) {
  const maximum = Number(runtime.maximum_context);
  if (maximum && Number(value) > maximum) return `Above this model's maximum; requests use ${tokens(maximum)} tokens.`;
  return contextReloadNote(value);
}

function runtimeNote() {
  if (!runtime.model) return 'Thinking and context options for local models appear after your first message.';
  if (!runtime.local_ollama) return 'Thinking and context limits apply to local Ollama models.';
  if (!runtime.native) return 'Ollama controls context and keep-alive for this connection. Set OLLAMA_CONTEXT_LENGTH and OLLAMA_KEEP_ALIVE on the server.';
  return '';
}

// Highlights the composer button while this chat differs from the defaults.
function isCustomized() {
  const limit = preferences.context_limits[runtime.model];
  return Boolean(preferences.instructions.trim()) || preferences.thinking !== 'auto'
    || Boolean(limit && runtime.default_context_limit && limit !== runtime.default_context_limit);
}

function renderInstructionsCount() {
  el('assistant-instructions-count').textContent = `${el('assistant-instructions').value.length.toLocaleString()} / 10,000`;
}

function renderContext() {
  const value = el('assistant-context').value;
  const fallback = runtime.default_context_limit;
  el('assistant-context-reset').hidden = !fallback || Number(value) === Number(fallback);
  el('assistant-context-hint').textContent = [
    fallback && `Default ${tokens(fallback)}`,
    runtime.maximum_context && `model maximum ${tokens(runtime.maximum_context)}`,
  ].filter(Boolean).join(' · ');
  const warning = contextWarning(value);
  el('assistant-context-warning').textContent = warning;
  el('assistant-context-warning').hidden = !warning;
}

function render() {
  renderSearchMode();
  const button = el('assistant-settings-btn');
  if (!button) return;
  const label = isCustomized() ? 'Conversation settings (customized)' : 'Conversation settings';
  button.classList.toggle('active', isCustomized());
  button.title = label;
  button.setAttribute('aria-label', label);
  el('assistant-thinking').value = preferences.thinking;
  const automatic = el('assistant-thinking').options?.[0];
  if (automatic) automatic.textContent = runtime.default_thinking ? `Settings default (${efforts[runtime.default_thinking]})` : 'Default (from Settings)';
  el('assistant-instructions').value = preferences.instructions;
  renderInstructionsCount();
  el('assistant-model-name').textContent = runtime.model || '';
  el('assistant-model-status').textContent = runtime.local_ollama
    ? (runtime.loaded_context ? `Loaded · ${tokens(runtime.loaded_context)} tokens` : 'Not loaded')
    : '';
  el('assistant-thinking-row').hidden = runtime.supports_thinking !== true;
  el('assistant-context-row').hidden = !runtime.native;
  el('assistant-context').value = preferences.context_limits[runtime.model] || runtime.context_limit || runtime.default_context_limit || 32768;
  renderContext();
  el('assistant-runtime').textContent = runtimeNote();
  el('assistant-runtime').hidden = !runtimeNote();
}

export function loadSession(id) {
  el('web-search-menu')?.classList.add('hidden');
  el('web-toggle-btn')?.setAttribute('aria-expanded', 'false');
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
    preferences = { ...defaults(), ...data.preferences };
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
  const webButton = el('web-toggle-btn');
  const webMenu = el('web-search-menu');
  const closeWebMenu = () => {
    webMenu?.classList.add('hidden');
    webButton?.setAttribute('aria-expanded', 'false');
  };
  const openWebMenu = () => {
    const rect = webButton.getBoundingClientRect();
    webMenu.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - 296))}px`;
    webMenu.style.bottom = `${window.innerHeight - rect.top + 8}px`;
    webMenu.classList.remove('hidden');
    webButton.setAttribute('aria-expanded', 'true');
    webMenu.querySelector('[aria-checked="true"]')?.focus();
  };
  webButton?.addEventListener('click', () => webMenu.classList.contains('hidden') ? openWebMenu() : closeWebMenu());
  webButton?.addEventListener('keydown', event => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); openWebMenu(); }
  });
  webMenu?.querySelectorAll('[data-search-mode]').forEach((option, index, options) => {
    option.addEventListener('click', async () => {
      closeWebMenu();
      webButton.focus();
      try { await setSearchMode(option.dataset.searchMode); }
      catch (error) { ui.showError(error.message); }
    });
    option.addEventListener('keydown', event => {
      if (event.key === 'Escape') { closeWebMenu(); webButton.focus(); }
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        options[(index + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length].focus();
      }
    });
  });
  document.addEventListener('click', event => {
    if (!webButton?.contains(event.target) && !webMenu?.contains(event.target)) closeWebMenu();
  });
  window.addEventListener('resize', closeWebMenu);
  const settingsModal = el('assistant-settings-modal');
  const closeSettings = () => {
    settingsModal.classList.add('hidden');
    el('assistant-settings-btn')?.focus();
  };
  const saveSettings = async () => {
    const context = el('assistant-context');
    const save = el('assistant-save');
    if (save.disabled) return;
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
    save.disabled = true;
    save.textContent = 'Saving…';
    try {
      if (sessionId) await prepareTurn(sessionId);
      render();
      closeSettings();
    } catch (error) { ui.showError(error.message); }
    finally {
      save.disabled = false;
      save.textContent = 'Save';
    }
  };
  el('assistant-settings-btn')?.addEventListener('click', async () => {
    await loading;
    if (sessionId && !dirty) await loadSession(sessionId);
    if (loadError) return;
    render();
    settingsModal.classList.remove('hidden');
    el('assistant-instructions').focus();
  });
  el('assistant-instructions')?.addEventListener('input', renderInstructionsCount);
  el('assistant-context')?.addEventListener('input', renderContext);
  el('assistant-context-reset')?.addEventListener('click', () => {
    el('assistant-context').value = runtime.default_context_limit;
    renderContext();
    el('assistant-context').focus();
  });
  // Ctrl/Cmd+Enter saves from anywhere in the dialog; plain Enter saves from
  // the context field, where it would otherwise do nothing.
  settingsModal?.addEventListener('keydown', event => {
    if (event.key !== 'Enter' || event.isComposing) return;
    if (!(event.ctrlKey || event.metaKey || event.target === el('assistant-context'))) return;
    event.preventDefault();
    saveSettings();
  });
  el('assistant-close')?.addEventListener('click', closeSettings);
  el('assistant-cancel')?.addEventListener('click', closeSettings);
  el('assistant-save')?.addEventListener('click', saveSettings);
  render();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
else init();

export default { loadSession, prepareTurn, contextReloadNote, searchMode, setSearchMode, renderSearchMode };

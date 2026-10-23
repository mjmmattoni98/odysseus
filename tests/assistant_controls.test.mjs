import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';

globalThis.window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800 };

const DEFAULTS = { search_mode: 'auto', thinking: 'auto', instructions: '', context_limits: {} };

// Exercise the actual settings controller with deferred network responses.
async function controller(document = { readyState: 'loading', addEventListener() {}, getElementById: () => null }) {
  globalThis.document = document;
  const source = (await readFile(new URL('../static/js/assistantControls.js', import.meta.url), 'utf8'))
    .replace(/import ui from '\.\/ui\.js(?:\?[^']*)?';/, 'const ui = { showError() {} };');
  return import(`data:text/javascript,${encodeURIComponent(source)}#${Math.random()}`);
}

test('switching chats during a save never attaches the previous chat to the new composer', async () => {
  const controls = await controller();
  const calls = [];
  let release;
  globalThis.fetch = (url, options) => {
    calls.push({ url, options });
    return new Promise(resolve => { release = () => resolve({ ok: true }); });
  };
  await controls.loadSession(null);
  const oldSave = controls.prepareTurn('old');
  await new Promise(resolve => setImmediate(resolve));
  await controls.loadSession(null);
  release();
  await assert.rejects(oldSave, /Conversation changed/);
  const newSave = controls.prepareTurn('new');
  await new Promise(resolve => setImmediate(resolve));
  release();
  assert.deepEqual(await newSave, DEFAULTS);
  assert.deepEqual(calls.map(c => c.url), ['/api/session/old/assistant', '/api/session/new/assistant']);
});

test('a stale settings fetch cannot overwrite the selected conversation', async () => {
  const controls = await controller();
  let release;
  globalThis.fetch = () => new Promise(resolve => { release = () => resolve({ ok: true, json: async () => ({ preferences: { ...DEFAULTS, thinking: 'high' }, runtime: {} }) }); });
  const oldLoad = controls.loadSession('old');
  await controls.loadSession(null);
  release();
  await oldLoad;
  globalThis.fetch = async () => ({ ok: true });
  assert.deepEqual(await controls.prepareTurn('new'), DEFAULTS);
});

function settingsDocument() {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) {
      const classes = new Set(id === 'web-search-menu' ? ['hidden'] : []);
      elements.set(id, {
        id, value: '', hidden: false, textContent: '', listeners: {}, attributes: {}, dataset: {}, style: {},
        classList: {
          add: name => classes.add(name), remove: name => classes.delete(name), contains: name => classes.has(name),
          toggle(name, active) { if (active) classes.add(name); else classes.delete(name); },
        },
        addEventListener(name, handler) { this.listeners[name] = handler; },
        setAttribute(name, value) { this.attributes[name] = value; },
        focus() { this.focused = true; }, reportValidity: () => true,
        getBoundingClientRect: () => ({ left: 100, top: 500 }), contains: () => false,
        querySelectorAll: () => id === 'web-search-menu' ? options : [],
        querySelector: () => options.find(option => option.attributes['aria-checked'] === 'true'),
      });
    }
    return elements.get(id);
  };
  const options = ['off', 'auto', 'always'].map(mode => {
    const option = element(`option-${mode}`);
    option.dataset.searchMode = mode;
    return option;
  });
  return { readyState: 'complete', addEventListener() {}, getElementById: element, elements, options };
}

test('the Web menu saves immediately, survives switching chats, and new chats default to Auto', async () => {
  const document = settingsDocument();
  const controls = await controller(document);
  const saved = new Map();
  globalThis.fetch = async (url, options) => {
    if (options?.method === 'PUT') saved.set(url, JSON.parse(options.body));
    return { ok: true, json: async () => ({ preferences: saved.get(url) || { ...DEFAULTS, search_mode: 'off' }, runtime: {} }) };
  };
  await controls.loadSession('conversation');
  assert.equal(document.getElementById('web-mode-label').textContent, 'Off');
  await controls.setSearchMode('always');
  assert.equal(saved.get('/api/session/conversation/assistant').search_mode, 'always');
  assert.equal(document.getElementById('web-toggle').checked, true);
  await controls.loadSession('other');
  assert.equal(controls.searchMode(), 'off');
  await controls.loadSession('conversation');
  assert.equal(controls.searchMode(), 'always');
  await controls.loadSession(null);
  assert.equal(document.getElementById('web-mode-label').textContent, 'Auto');
  assert.equal(document.getElementById('web-toggle').checked, false);
  for (const id of ['bash-toggle-btn', 'bash-toggle', 'mode-agent-btn']) assert.equal(document.elements.has(id), false);
});

test('the menu selects one mode and keeps it when conversation settings are saved', async () => {
  const document = settingsDocument();
  const controls = await controller(document);
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ preferences: DEFAULTS, runtime: {} }) });
  await controls.loadSession('s');
  document.getElementById('web-toggle-btn').listeners.click();
  assert.equal(document.getElementById('web-search-menu').classList.contains('hidden'), false);
  await document.options[0].listeners.click();
  assert.equal(controls.searchMode(), 'off');
  assert.equal(document.options.filter(option => option.attributes['aria-checked'] === 'true').length, 1);
  await document.getElementById('assistant-save').listeners.click();
  assert.equal((await controls.prepareTurn('s')).search_mode, 'off');
  assert.equal(document.getElementById('web-search-menu').classList.contains('hidden'), true);
});

test('failed mode saves prevent a turn from using stale server preferences', async () => {
  const controls = await controller();
  globalThis.fetch = async (url, options) => options ? { ok: false } : {
    ok: true, json: async () => ({ preferences: DEFAULTS, runtime: {} }),
  };
  await controls.loadSession('s');
  await assert.rejects(controls.setSearchMode('off'), /Could not save/);
  await assert.rejects(controls.prepareTurn('s'), /Could not save/);
});

test('the dialog explains the context cap and marks customized chats', async () => {
  const document = settingsDocument();
  const controls = await controller(document);
  const get = id => document.getElementById(id);
  globalThis.fetch = async () => ({ ok: true, json: async () => ({
    preferences: { ...DEFAULTS, context_limits: { m: 65536 } },
    runtime: { model: 'm', local_ollama: true, native: true, maximum_context: 131072, default_context_limit: 32768 },
  }) });
  await controls.loadSession('s');
  assert.equal(Number(get('assistant-context').value), 65536);
  assert.match(get('assistant-context-hint').textContent, /^Default 32\D?768 · model maximum 131\D?072$/);
  assert.match(get('assistant-context-warning').textContent, /Ollama reloads the model/);
  assert.equal(get('assistant-settings-btn').classList.contains('active'), true);
  get('assistant-context').value = '200000';
  get('assistant-context').listeners.input();
  assert.match(get('assistant-context-warning').textContent, /Above this model's maximum/);
  get('assistant-context-reset').listeners.click();
  assert.equal(Number(get('assistant-context').value), 32768);
  assert.equal(get('assistant-context-warning').hidden, true);
  assert.equal(get('assistant-context-reset').hidden, true);
});

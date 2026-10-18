import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';

const DEFAULTS = { thinking: 'auto', instructions: '', context_limits: {} };

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

test('conversation settings leave the web, shell, and Chat/Agent controls alone', async () => {
  const touched = new Set();
  const element = id => ({
    id, value: '', hidden: false, textContent: '',
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {}, focus() {}, closest: () => null, reportValidity: () => true,
  });
  const controls = await controller({
    readyState: 'complete',
    addEventListener() {},
    getElementById: id => { touched.add(id); return element(id); },
  });
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ preferences: DEFAULTS, runtime: {} }) });
  await controls.loadSession('s');
  for (const id of ['web-toggle-btn', 'web-toggle', 'bash-toggle-btn', 'bash-toggle', 'mode-agent-btn']) {
    assert.ok(!touched.has(id), `${id} must stay owned by the composer toggles`);
  }
});

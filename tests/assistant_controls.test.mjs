import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';

// Exercise the actual settings controller with deferred network responses.
async function controller() {
  globalThis.document = { readyState: 'loading', addEventListener() {}, getElementById: () => null };
  const source = (await readFile(new URL('../static/js/assistantControls.js', import.meta.url), 'utf8'))
    .replace("import ui from './ui.js';", 'const ui = { showError() {} };');
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
  assert.equal((await newSave).profile, 'everyday');
  assert.deepEqual(calls.map(c => c.url), ['/api/session/old/assistant', '/api/session/new/assistant']);
});

test('a stale settings fetch cannot overwrite the selected conversation', async () => {
  const controls = await controller();
  let release;
  globalThis.fetch = () => new Promise(resolve => { release = () => resolve({ ok: true, json: async () => ({ preferences: { profile: 'actions' }, runtime: {} }) }); });
  const oldLoad = controls.loadSession('old');
  await controls.loadSession(null);
  release();
  await oldLoad;
  globalThis.fetch = async () => ({ ok: true });
  assert.equal((await controls.prepareTurn('new')).profile, 'everyday');
});

// The composer button changes meaning with state (send / queue / stop / new
// chat / record voice / group send), and Compare reuses it. Its static
// aria-label used to stay "New chat" while only `title` followed the state, so
// screen readers announced the wrong action during generation. These tests run
// the shipped state-transition functions against a fake button and check that
// the accessible name always matches the tooltip and the current action.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import test from 'node:test';

import { setActionLabel } from '../static/js/a11yLabel.js';

function slice(file, startMarker, endMarker) {
  const source = readFileSync(new URL(file, import.meta.url), 'utf8');
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.ok(start >= 0 && end > start, `could not locate ${startMarker} in ${file}`);
  return source.slice(start, end);
}

function fakeButton() {
  const attrs = new Map();
  const classes = new Set();
  return {
    dataset: {},
    style: {},
    innerHTML: '',
    title: '',
    classList: {
      add: (...names) => names.forEach(name => classes.add(name)),
      remove: (...names) => names.forEach(name => classes.delete(name)),
      contains: name => classes.has(name),
    },
    setAttribute(name, value) { attrs.set(name, String(value)); },
    getAttribute(name) { return attrs.has(name) ? attrs.get(name) : null; },
    addEventListener() {},
    get offsetWidth() { return 0; },
  };
}

function assertAnnounces(btn, label) {
  assert.equal(btn.title, label);
  assert.equal(btn.getAttribute('aria-label'), label, `aria-label should follow title "${label}"`);
}

function timerQueue() {
  const pending = [];
  return {
    setTimeout: (fn) => { pending.push(fn); return pending.length; },
    flush: () => { while (pending.length) pending.shift()(); },
  };
}

test('setActionLabel sets title and accessible name together', () => {
  const btn = fakeButton();
  setActionLabel(btn, 'Stop generation');
  assertAnnounces(btn, 'Stop generation');
  setActionLabel(null, 'ignored'); // tolerates a missing control
});

const APP_SLICE = slice(
  '../static/app.js',
  '  function _updateStreamingSubmitButton() {',
  "  if (sendBtn) {\n    sendBtn.addEventListener('click'",
);

function appComposer({ welcome = false, stt = false, group = false } = {}) {
  const sendBtn = fakeButton();
  sendBtn.setAttribute('aria-label', 'New chat'); // the static index.html default
  const messageInput = { value: '' };
  const timers = timerQueue();
  const ctx = {
    sendBtn,
    messageInput,
    _sendIcon: 'SEND', _micIcon: 'MIC', _stopIcon: 'STOP', _newChatIcon: 'PLUS',
    _isSttEnabled: () => stt,
    _hasAttachments: () => false,
    groupModule: { isActive: () => group },
    document: {
      getElementById: id => (id === 'chat-container'
        ? { classList: { contains: name => name === 'welcome-active' && welcome } }
        : null),
    },
    setTimeout: timers.setTimeout,
    clearTimeout() {},
    setActionLabel,
  };
  runInNewContext(APP_SLICE, ctx);
  return { sendBtn, messageInput, timers, update: ctx._updateSendBtnIcon };
}

test('idle composer announces new chat, then send as soon as text is typed', () => {
  const { sendBtn, messageInput, timers, update } = appComposer();
  update();
  assert.equal(sendBtn.dataset.mode, 'newchat');
  assertAnnounces(sendBtn, 'New chat');

  messageInput.value = 'hello';
  update();
  assert.equal(sendBtn.dataset.mode, 'send');
  // The icon swap waits for its animation; the name must not lag behind.
  assertAnnounces(sendBtn, 'Send message');
  timers.flush();
  assertAnnounces(sendBtn, 'Send message');
});

test('fresh chat, voice, and group composers announce their own action', () => {
  const welcome = appComposer({ welcome: true });
  welcome.update();
  assertAnnounces(welcome.sendBtn, 'Send message');

  const voice = appComposer({ stt: true });
  voice.update();
  assert.equal(voice.sendBtn.dataset.mode, 'mic');
  assertAnnounces(voice.sendBtn, 'Record voice');

  const group = appComposer({ group: true });
  group.update();
  assertAnnounces(group.sendBtn, 'Send to group');
});

test('streaming composer switches between stop and queue as text changes', () => {
  const { sendBtn, messageInput, update } = appComposer();
  sendBtn.dataset.mode = 'streaming';
  update();
  assertAnnounces(sendBtn, 'Stop generation');

  messageInput.value = 'follow-up';
  update();
  assertAnnounces(sendBtn, 'Queue message');

  messageInput.value = '';
  update();
  assertAnnounces(sendBtn, 'Stop generation');
});

const CHAT_SLICE = slice(
  '../static/js/chat.js',
  '  function updateSubmitButton(state, submitBtn) {',
  '  // -----------------------------------------------------------------------\n  // Slash commands',
);

function chatSubmitButton({ queuedText = '' } = {}) {
  const timers = timerQueue();
  const ctx = {
    window: { _odysseusBtnIcons: { send: 'SEND' } },
    uiModule: { el: () => ({ value: queuedText }) },
    setTimeout: timers.setTimeout,
    isStreaming: false,
    _setForegroundChatBusy() {},
    _startStallWatchdog() {},
    _stopStallWatchdog() {},
    setActionLabel,
  };
  runInNewContext(CHAT_SLICE, ctx);
  return { timers, updateSubmitButton: ctx.updateSubmitButton };
}

test('starting a stream announces stop generation immediately', () => {
  const { timers, updateSubmitButton } = chatSubmitButton();
  const btn = fakeButton();
  btn.setAttribute('aria-label', 'New chat');
  updateSubmitButton('streaming', btn);
  assertAnnounces(btn, 'Stop generation');
  timers.flush();
  assertAnnounces(btn, 'Stop generation');
});

test('a stream started with queued text announces queue message after launch', () => {
  const { timers, updateSubmitButton } = chatSubmitButton({ queuedText: 'next' });
  const btn = fakeButton();
  updateSubmitButton('streaming', btn);
  timers.flush();
  assertAnnounces(btn, 'Queue message');
});

test('returning to idle without the global updater announces send', () => {
  const { updateSubmitButton } = chatSubmitButton();
  const btn = fakeButton();
  updateSubmitButton('streaming', btn);
  updateSubmitButton('idle', btn);
  assertAnnounces(btn, 'Send message');
});

test('Compare announces stop all models while streaming and send afterwards', () => {
  const btn = fakeButton();
  btn.setAttribute('aria-label', 'New chat');
  const ctx = {
    document: { querySelector: () => btn },
    SEND_SVG: 'SEND',
    setActionLabel,
  };
  runInNewContext(
    slice('../static/js/compare/index.js', 'function _setSendBtn(mode) {', '// ── handleCompareSubmit ──'),
    ctx,
  );
  ctx._setSendBtn('stop');
  assertAnnounces(btn, 'Stop all models');
  ctx._setSendBtn('send');
  assertAnnounces(btn, 'Send to all models');
});

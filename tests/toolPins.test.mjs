// Tool pins (static/js/section-management.js): everyday tools pinned to the
// top of the sidebar Tools section and the icon rail, the rest behind a
// collapsible "More tools" row. Runs the real controller against a small
// in-memory DOM so it needs nothing beyond plain node.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  DEFAULT_PINNED_TOOLS,
  initToolPins,
  isToolPinned,
  normalizeToolPinState,
  railIdForTool,
  setToolPinned,
  splitToolPins,
  toolKeyFromId,
} from '../static/js/section-management.js';

// ── Minimal DOM ────────────────────────────────────────────────────────────

class FakeEvent {
  constructor(type, init = {}) {
    Object.assign(this, { bubbles: true, clientX: 0, clientY: 0, key: '', shiftKey: false }, init);
    this.type = type;
    this.defaultPrevented = false;
    this.propagationStopped = false;
  }
  preventDefault() { this.defaultPrevented = true; }
  stopPropagation() { this.propagationStopped = true; }
}

function matchesSimple(el, sel) {
  const m = /^([a-z]+)?((?:[.#][\w-]+)*)$/i.exec(sel.trim());
  if (!m) throw new Error('unsupported selector ' + sel);
  if (m[1] && el.tagName !== m[1].toUpperCase()) return false;
  for (const part of m[2].match(/[.#][\w-]+/g) || []) {
    if (part[0] === '#' && el.id !== part.slice(1)) return false;
    if (part[0] === '.' && !el.classList.contains(part.slice(1))) return false;
  }
  return true;
}
const matches = (el, sel) => sel.split(',').some(s => matchesSimple(el, s));

class FakeElement {
  constructor(doc, tag) {
    this.ownerDocument = doc;
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.attrs = new Map();
    this.dataset = {};
    this.style = {};
    this.listeners = {};
    this._text = '';
    this._classes = new Set();
    const self = this;
    this.classList = {
      add: (...c) => c.forEach(x => self._classes.add(x)),
      remove: (...c) => c.forEach(x => self._classes.delete(x)),
      contains: (c) => self._classes.has(c),
      toggle: (c, force) => {
        const on = force === undefined ? !self._classes.has(c) : !!force;
        if (on) self._classes.add(c); else self._classes.delete(c);
        return on;
      },
    };
  }
  get id() { return this.attrs.get('id') || ''; }
  set id(v) { this.attrs.set('id', v); }
  get className() { return [...this._classes].join(' '); }
  set className(v) { this._classes = new Set(String(v).split(/\s+/).filter(Boolean)); }
  setAttribute(k, v) { if (k === 'class') this.className = v; else this.attrs.set(k, String(v)); }
  getAttribute(k) { return this.attrs.has(k) ? this.attrs.get(k) : null; }
  hasAttribute(k) { return this.attrs.has(k); }
  get title() { return this.getAttribute('title') || ''; }
  set title(v) { this.setAttribute('title', v); }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set textContent(v) { this.children = []; this._text = String(v); }
  set innerHTML(html) {
    this.children = [];
    this._text = '';
    for (const m of html.matchAll(/<(span|svg|div)\b([^>]*)>([\s\S]*?)<\/\1>/g)) {
      const child = this.ownerDocument.createElement(m[1]);
      const cls = /class="([^"]*)"/.exec(m[2]);
      if (cls) child.className = cls[1];
      if (!/</.test(m[3])) child._text = m[3];
      this.appendChild(child);
    }
  }
  get lastChild() { return this.children[this.children.length - 1] || null; }
  get nextSibling() {
    const sib = this.parentNode ? this.parentNode.children : [];
    return sib[sib.indexOf(this) + 1] || null;
  }
  get previousElementSibling() {
    const sib = this.parentNode ? this.parentNode.children : [];
    return sib[sib.indexOf(this) - 1] || null;
  }
  get isConnected() {
    let n = this;
    while (n.parentNode) n = n.parentNode;
    return n === this.ownerDocument.root;
  }
  appendChild(child) { return this.insertBefore(child, null); }
  insertBefore(child, ref) {
    if (child.parentNode) child.parentNode.children.splice(child.parentNode.children.indexOf(child), 1);
    child.parentNode = this;
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i === -1) this.children.push(child); else this.children.splice(i, 0, child);
    return child;
  }
  remove() {
    if (this.parentNode) this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1);
    this.parentNode = null;
  }
  contains(el) {
    for (let n = el; n; n = n.parentNode) if (n === this) return true;
    return false;
  }
  *descendants() { for (const c of this.children) { yield c; yield* c.descendants(); } }
  querySelectorAll(sel) { return [...this.descendants()].filter(el => matches(el, sel)); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  closest(sel) {
    for (let n = this; n; n = n.parentNode) if (n.tagName && matches(n, sel)) return n;
    return null;
  }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) {
    this.listeners[type] = (this.listeners[type] || []).filter(f => f !== fn);
  }
  dispatchEvent(evt) {
    evt.target ||= this;
    for (let n = this; n && !evt.propagationStopped; n = n.parentNode) {
      for (const fn of [...(n.listeners[evt.type] || [])]) fn(evt);
      if (!evt.bubbles) break;
    }
    return !evt.defaultPrevented;
  }
  click() { this.dispatchEvent(new FakeEvent('click')); }
  focus() { this.ownerDocument.activeElement = this; }
  getBoundingClientRect() { return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }; }
}

class FakeDocument {
  constructor() {
    this.root = new FakeElement(this, 'html');
    this.body = this.root.appendChild(new FakeElement(this, 'body'));
    this.activeElement = this.body;
    this.defaultView = { innerWidth: 1280, innerHeight: 800 };
    this.listeners = {};
  }
  createElement(tag) { return new FakeElement(this, tag); }
  getElementById(id) { return [...this.root.descendants()].find(el => el.id === id) || null; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) {
    this.listeners[type] = (this.listeners[type] || []).filter(f => f !== fn);
  }
}

// The shipped Tools section and icon rail, reduced to what the controller reads.
const TOOLS = [
  ['calendar', 'Calendar'], ['compare', 'Compare'], ['cookbook', 'Cookbook'],
  ['research', 'Deep Research'], ['gallery', 'Gallery'], ['library', 'Library'],
  ['memory', 'Memory'], ['notes', 'Notes'], ['skills', 'Skills'], ['tasks', 'Tasks'],
  ['theme', 'Theme'],
];

function addTool(doc, key, label) {
  const section = doc.getElementById('tools-section');
  const row = section.appendChild(doc.createElement('div'));
  row.id = `tool-${key}-btn`;
  row.className = 'list-item';
  const text = row.appendChild(doc.createElement('span'));
  text.className = 'grow';
  text.textContent = label;
  const rail = doc.getElementById('icon-rail');
  const launcher = doc.createElement('button');
  launcher.id = railIdForTool(key);
  launcher.className = 'icon-rail-btn';
  rail.insertBefore(launcher, doc.getElementById('rail-email').nextSibling);
  return row;
}

function buildDom() {
  const doc = new FakeDocument();
  globalThis.document = doc; // escMenuStack's outside-click listener
  const sidebar = doc.body.appendChild(doc.createElement('nav'));
  const section = sidebar.appendChild(doc.createElement('div'));
  section.id = 'tools-section';
  section.className = 'section';
  section.appendChild(doc.createElement('div')).className = 'section-header-flex';
  const rail = doc.body.appendChild(doc.createElement('div'));
  rail.id = 'icon-rail';
  for (const id of ['rail-search-btn', 'rail-new-session', 'rail-email']) {
    const b = rail.appendChild(doc.createElement('button'));
    b.id = id;
    b.className = 'icon-rail-btn';
  }
  rail.appendChild(doc.createElement('div')); // flex spacer
  const settings = rail.appendChild(doc.createElement('button'));
  settings.id = 'rail-settings';
  settings.className = 'icon-rail-btn';
  for (const [key, label] of TOOLS) addTool(doc, key, label);
  return doc;
}

function memoryStorage(initial = {}) {
  const data = { ...initial };
  return {
    KEYS: { TOOL_PINS: 'odysseus-tool-pins' },
    data,
    getJSON: (k, fallback) => (k in data ? JSON.parse(data[k]) : fallback),
    setJSON: (k, v) => { data[k] = JSON.stringify(v); },
  };
}

function recordingFetch(remote = null) {
  const calls = [];
  const impl = async (url, options = {}) => {
    calls.push({ url, method: options.method || 'GET', body: options.body ? JSON.parse(options.body) : null });
    return { ok: true, json: async () => ({ key: 'tool-pins', value: remote }) };
  };
  impl.calls = calls;
  return impl;
}

const sidebarOrder = (doc) => doc.getElementById('tools-section').children
  .filter(el => el.id).map(el => toolKeyFromId(el.id));
const railOrder = (doc) => doc.getElementById('icon-rail').children
  .filter(el => el.id).map(el => el.id.replace(/^rail-/, ''));

// ── Pure state ─────────────────────────────────────────────────────────────

test('default pins are the chat-centric everyday tools, including Decisions', () => {
  assert.deepEqual([...DEFAULT_PINNED_TOOLS], ['notes', 'calendar', 'research', 'memory', 'library', 'decisions']);
  const split = splitToolPins(['calendar', 'compare', 'research', 'notes', 'theme'], null);
  assert.deepEqual(split, { pinned: ['calendar', 'research', 'notes'], unpinned: ['compare', 'theme'] });
});

test('pins are stored as deltas against the defaults and garbage is dropped', () => {
  let s = normalizeToolPinState({ pin: ['compare', 'notes', 7, 'compare'], unpin: ['theme', 'memory'], moreExpanded: 'yes' });
  assert.deepEqual(s, { pin: ['compare'], unpin: ['memory'], moreExpanded: false });
  s = setToolPinned(s, 'memory', true);
  s = setToolPinned(s, 'notes', false);
  s = setToolPinned(s, 'compare', false);
  assert.deepEqual(s, { pin: [], unpin: ['notes'], moreExpanded: false });
  assert.equal(isToolPinned(s, 'memory'), true);
  assert.equal(isToolPinned(s, 'notes'), false);
  assert.deepEqual(normalizeToolPinState('nonsense'), { pin: [], unpin: [], moreExpanded: false });
});

test('tool keys and rail launchers derive from ids, with the Library exception', () => {
  assert.equal(toolKeyFromId('tool-decisions-btn'), 'decisions');
  assert.equal(railIdForTool('decisions'), 'rail-decisions');
  assert.equal(railIdForTool('library'), 'rail-archive');
  assert.equal(railIdForTool('custom', 'rail-elsewhere'), 'rail-elsewhere');
});

// ── Controller ─────────────────────────────────────────────────────────────

test('first run pins the everyday tools and collapses the rest, on the sidebar and rail', () => {
  const doc = buildDom();
  initToolPins(memoryStorage(), { doc, fetchImpl: null });

  assert.deepEqual(sidebarOrder(doc), [
    'calendar', 'research', 'library', 'memory', 'notes', 'tools-more-toggle',
    'compare', 'cookbook', 'gallery', 'skills', 'tasks', 'theme',
  ]);
  const section = doc.getElementById('tools-section');
  const toggle = doc.getElementById('tools-more-toggle');
  assert.ok(section.classList.contains('tools-more-collapsed'));
  assert.equal(toggle.getAttribute('aria-expanded'), 'false');
  assert.equal(toggle.getAttribute('role'), 'button');
  assert.equal(toggle.querySelector('.tools-more-count').textContent, '6');
  assert.ok(doc.getElementById('tool-compare-btn').classList.contains('tool-unpinned'));
  assert.ok(!doc.getElementById('tool-notes-btn').classList.contains('tool-unpinned'));

  assert.deepEqual(railOrder(doc), [
    'search-btn', 'new-session', 'calendar', 'research', 'archive', 'memory', 'notes', 'email',
    'more-tools', 'compare', 'cookbook', 'gallery', 'skills', 'tasks', 'theme', 'settings',
  ]);
  assert.ok(doc.getElementById('icon-rail').classList.contains('rail-more-collapsed'));
  assert.ok(doc.getElementById('rail-compare').classList.contains('tool-unpinned'));
});

test('the More tools expanded state is remembered across reloads', () => {
  const storage = memoryStorage();
  let doc = buildDom();
  initToolPins(storage, { doc, fetchImpl: null });
  doc.getElementById('tools-more-toggle').click();
  assert.equal(JSON.parse(storage.data['odysseus-tool-pins']).moreExpanded, true);
  assert.ok(!doc.getElementById('tools-section').classList.contains('tools-more-collapsed'));

  doc = buildDom();
  initToolPins(storage, { doc, fetchImpl: null });
  assert.equal(doc.getElementById('tools-more-toggle').getAttribute('aria-expanded'), 'true');
  assert.ok(!doc.getElementById('icon-rail').classList.contains('rail-more-collapsed'));

  doc.getElementById('rail-more-tools').click();
  assert.equal(JSON.parse(storage.data['odysseus-tool-pins']).moreExpanded, false);
});

test('right-click Pin to top moves a tool up, persists it, and syncs the preference', () => {
  const doc = buildDom();
  const storage = memoryStorage();
  const fetchImpl = recordingFetch();
  initToolPins(storage, { doc, fetchImpl });

  const label = doc.getElementById('tool-compare-btn').querySelector('.grow');
  const evt = new FakeEvent('contextmenu', { clientX: 40, clientY: 200, target: label });
  label.dispatchEvent(evt);
  assert.ok(evt.defaultPrevented, 'the browser context menu is replaced');
  const menu = doc.body.querySelector('.tool-pin-menu');
  assert.equal(menu.getAttribute('role'), 'menu');
  const item = menu.querySelector('.dropdown-item-compact');
  assert.equal(item.textContent, 'Pin to top');
  assert.equal(doc.activeElement, item, 'focus moves into the menu');

  item.click();
  assert.equal(doc.body.querySelector('.tool-pin-menu'), null);
  assert.deepEqual(sidebarOrder(doc).slice(0, 7), [
    'calendar', 'compare', 'research', 'library', 'memory', 'notes', 'tools-more-toggle',
  ]);
  assert.deepEqual(JSON.parse(storage.data['odysseus-tool-pins']).pin, ['compare']);
  const put = fetchImpl.calls.find(c => c.method === 'PUT');
  assert.equal(put.url, '/api/prefs/tool-pins');
  assert.deepEqual(put.body, { value: { pin: ['compare'], unpin: [] } });
  assert.ok(railOrder(doc).indexOf('compare') < railOrder(doc).indexOf('more-tools'));
});

test('Shift+F10 opens the same menu from the keyboard and Enter unpins', () => {
  const doc = buildDom();
  const storage = memoryStorage();
  initToolPins(storage, { doc, fetchImpl: null });
  const notes = doc.getElementById('tool-notes-btn');
  notes.dispatchEvent(new FakeEvent('keydown', { key: 'F10', shiftKey: true, target: notes }));
  const menu = doc.body.querySelector('.tool-pin-menu');
  assert.equal(menu.querySelector('.dropdown-item-compact').textContent, 'Unpin (move to More tools)');

  menu.dispatchEvent(new FakeEvent('keydown', { key: 'Enter', target: doc.activeElement }));
  assert.deepEqual(JSON.parse(storage.data['odysseus-tool-pins']).unpin, ['notes']);
  assert.ok(notes.classList.contains('tool-unpinned'));
  assert.equal(doc.activeElement, doc.getElementById('tools-more-toggle'),
    'focus lands on More tools when the unpinned row is now collapsed away');
});

test('a tool added later participates without code changes (Decisions is pinned by default)', () => {
  const doc = buildDom();
  const controller = initToolPins(memoryStorage(), { doc, fetchImpl: null });
  addTool(doc, 'decisions', 'Decisions');
  addTool(doc, 'widgets', 'Widgets');
  controller.apply();
  const order = sidebarOrder(doc);
  assert.ok(order.indexOf('decisions') < order.indexOf('tools-more-toggle'));
  assert.ok(order.indexOf('widgets') > order.indexOf('tools-more-toggle'));
  assert.ok(railOrder(doc).indexOf('decisions') < railOrder(doc).indexOf('email'));
});

test('More tools hides when Settings hides every unpinned tool; live indicators stay out', () => {
  const doc = buildDom();
  const controller = initToolPins(memoryStorage(), { doc, fetchImpl: null });
  const dot = doc.getElementById('tool-tasks-btn').appendChild(doc.createElement('span'));
  dot.className = 'sidebar-notif-dot';
  controller.apply();
  assert.ok(doc.getElementById('tool-tasks-btn').classList.contains('tool-attention'));
  dot.style.display = 'none';
  controller.apply();
  assert.ok(!doc.getElementById('tool-tasks-btn').classList.contains('tool-attention'));

  for (const key of ['compare', 'cookbook', 'gallery', 'skills', 'tasks', 'theme']) {
    doc.getElementById(`tool-${key}-btn`).style.display = 'none';
  }
  controller.apply();
  assert.ok(doc.getElementById('tools-more-toggle').classList.contains('tools-more-empty'));
  assert.ok(doc.getElementById('rail-more-tools').classList.contains('tools-more-empty'));
});

test('pins saved on the account are adopted, but never over a newer local edit', async () => {
  let doc = buildDom();
  let storage = memoryStorage();
  let controller = initToolPins(storage, { doc, fetchImpl: recordingFetch({ pin: ['tasks'], unpin: ['memory'] }) });
  await controller.ready;
  assert.equal(isToolPinned(controller.getState(), 'tasks'), true);
  assert.ok(doc.getElementById('tool-memory-btn').classList.contains('tool-unpinned'));
  assert.deepEqual(JSON.parse(storage.data['odysseus-tool-pins']).pin, ['tasks']);

  doc = buildDom();
  storage = memoryStorage();
  controller = initToolPins(storage, { doc, fetchImpl: recordingFetch({ pin: ['tasks'], unpin: [] }) });
  controller.setPinned('gallery', true);
  await controller.ready;
  assert.deepEqual(controller.getState().pin, ['gallery']);
});

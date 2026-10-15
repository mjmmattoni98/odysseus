/**
 * Decisions — a notebook for choices: options scored against weighted
 * criteria, must-have constraints, tradeoffs, sources, the recorded choice
 * and a date to revisit it.
 *
 * A standard `.modal` tool window (drag/resize on desktop, bottom sheet with
 * a visible close button on mobile), minimizable to a dock chip through
 * modalManager. Scoring runs live in the browser (decisionScoring.js mirrors
 * the server); edits autosave. Recording a choice stores an immutable
 * snapshot server-side, and the History section compares today's matrix
 * against it.
 */

import uiModule from './ui.js?v=20260916largetoolscroll1';
import * as Modals from './modalManager.js';
import { makeWindowDraggable } from './windowDrag.js';
import {
  scoreOptions, compareToSnapshot, localToday, isRevisitDue, formatShare, newRowId, idDict,
} from './decisionScoring.js?v=20261014audit1';

const API = '/api/decisions';
const MODAL_ID = 'decisions-modal';
const SAVE_DELAY_MS = 700;
const BADGE_REFRESH_MS = 30 * 60 * 1000;

const ICON_SCALE_PATHS = '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>';
const ICON_SCALE = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON_SCALE_PATHS}</svg>`;
const ICON_CHAT = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';
const ICON_EXPORT = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>';
const ICON_ARCHIVE = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2" y="3" width="20" height="5" rx="1"/><path d="M4 8v11a2 2 0 002 2h12a2 2 0 002-2V8"/><path d="M10 12h4"/></svg>';
const ICON_TRASH = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>';
const ICON_REMOVE = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
const ICON_BACK = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="15 18 9 12 15 6"/></svg>';

const STATUS_LABELS = { open: 'Open', decided: 'Decided', 'revisit-due': 'Revisit due', archived: 'Archived' };

let _open = false;
let _list = [];
let _dueCount = 0;
let _current = null;   // last server copy of the selected decision (history, ids, status)
let _draft = null;     // the editable working copy
let _query = '';
let _showArchived = false;
let _saveTimer = null;
let _saving = null;
let _dirty = false;
let _escHandler = null;
let _openSeq = 0;      // bumped whenever a different decision is asked for
const _calendarQueues = new Map();   // decision id -> tail of its calendar syncs

const esc = (s) => uiModule.esc(s == null ? '' : String(s));
const today = () => localToday();

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function _api(path, { method = 'GET', body } = {}) {
  const sep = path.includes('?') ? '&' : '?';
  const res = await fetch(`${API}${path}${sep}today=${today()}`, {
    method,
    credentials: 'same-origin',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json()).detail || ''; } catch (_) {}
    const err = new Error(typeof detail === 'string' && detail ? detail : `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

async function _loadList() {
  const data = await _api('');
  _list = data.decisions || [];
  _dueCount = data.revisit_due_count || 0;
  _renderBadges();
}

// What the server will accept as a stored link: http(s), a host, and no
// credentials (src/url_safety.check_link_url). Anything else is still being
// typed, so it stays in the draft but is held back from the save.
function _isSavableLink(link) {
  return /^https?:\/\/[^\s/@?#]+([/?#]\S*)?$/i.test((link.url || '').trim());
}

function _bodyPayload() {
  const d = _draft;
  return {
    title: d.title,
    question: d.question,
    constraints: d.constraints,
    criteria: d.criteria,
    options: d.options.map(o => ({
      ...o,
      pros: _lines(o.prosText),
      cons: _lines(o.consText),
      links: o.links.filter(_isSavableLink),
      prosText: undefined,
      consText: undefined,
    })),
    sources: d.sources.filter(_isSavableLink),
    revisit_on: d.revisit_on || '',
  };
}

function _lines(text) {
  return String(text || '').split('\n').map(s => s.trim()).filter(Boolean);
}

function _toDraft(decision) {
  return {
    id: decision.id,
    title: decision.title || '',
    question: decision.question || '',
    constraints: (decision.constraints || []).map(c => ({ ...c })),
    criteria: (decision.criteria || []).map(c => ({ ...c })),
    options: (decision.options || []).map(o => ({
      ...o,
      // Keyed by criterion id, which may be "__proto__" or "constructor".
      scores: Object.assign(idDict(), o.scores || {}),
      fails: [...(o.fails || [])],
      links: (o.links || []).map(l => ({ ...l })),
      prosText: (o.pros || []).join('\n'),
      consText: (o.cons || []).join('\n'),
    })),
    sources: (decision.sources || []).map(s => ({ ...s })),
    revisit_on: decision.revisit_on || '',
    // Decision-form state; sent only by "Record decision".
    choice: decision.chosen_option_id || '',
    rationale: decision.rationale || '',
    calendar: !!decision.revisit_event_uid,
  };
}

function _heldBackLinks() {
  const held = (link) => (link.url || '').trim() && !_isSavableLink(link);
  return _draft.sources.filter(held).length
    + _draft.options.reduce((n, o) => n + o.links.filter(held).length, 0);
}

function _scheduleSave() {
  _dirty = true;
  _setSaveState('Unsaved changes');
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => { _saveNow().catch(() => {}); }, SAVE_DELAY_MS);
}

async function _saveNow() {
  clearTimeout(_saveTimer);
  _saveTimer = null;
  if (_saving) await _saving.catch(() => {});
  if (!_dirty || !_draft) return;
  const id = _draft.id;
  const payload = _bodyPayload();
  const held = _heldBackLinks();
  _dirty = false;
  _setSaveState('Saving...');
  _saving = _api(`/${encodeURIComponent(id)}`, { method: 'PUT', body: payload })
    .then((saved) => {
      if (_draft && _draft.id === id) {
        _current = saved;
        if (held) _setSaveState(`Saved; ${held} link${held === 1 ? '' : 's'} need a full http(s) address`, true);
        else _setSaveState(_dirty ? 'Unsaved changes' : 'Saved');
        _refreshComputed();
      }
      _upsertSummary(saved);
    })
    .catch((err) => {
      // Only the decision these edits came from is unsaved.
      if (_draft && _draft.id === id) {
        _dirty = true;
        _setSaveState(`Not saved: ${err.message}`, true);
      }
      throw err;
    })
    .finally(() => { _saving = null; });
  return _saving;
}

// Save the open draft until nothing is pending: an in-flight save, edits
// still waiting on the autosave timer, and edits typed while a request was
// in flight. Run before the draft is replaced, so edits reach the decision
// they were typed into. False when a save failed (the draft stays open).
async function _flushEdits() {
  do {
    try { await _saveNow(); } catch (_) { return false; }
  } while (_dirty && _draft);
  return true;
}

function _upsertSummary(decision) {
  const keys = ['id', 'title', 'status', 'effective_status', 'revisit_due', 'revisit_on',
    'chosen_option_name', 'leader_name', 'option_count', 'decided_at', 'updated_at'];
  const summary = Object.fromEntries(keys.map(k => [k, decision[k]]));
  const idx = _list.findIndex(d => d.id === decision.id);
  if (idx >= 0) _list[idx] = summary; else _list.unshift(summary);
  _dueCount = _list.filter(d => d.revisit_due).length;
  _renderBadges();
  _renderList();
}

// ---------------------------------------------------------------------------
// Badges on the tool entry and rail button
// ---------------------------------------------------------------------------

function _renderBadges() {
  const label = `${_dueCount} decision${_dueCount === 1 ? '' : 's'} due for revisit`;
  const place = (btn, cls) => {
    if (!btn) return;
    let badge = btn.querySelector(`.${cls}`);
    if (_dueCount > 0) {
      if (!badge) {
        badge = document.createElement('span');
        badge.className = `rail-notes-badge fired ${cls}`;
        btn.appendChild(badge);
      }
      badge.textContent = _dueCount > 99 ? '99+' : String(_dueCount);
      badge.title = label;
      badge.setAttribute('aria-label', label);
    } else if (badge) {
      badge.remove();
    }
  };
  place(document.getElementById('rail-decisions'), 'decisions-rail-badge');
  place(document.getElementById('tool-decisions-btn'), 'decisions-due-count');
}

export async function refreshDecisionBadge() {
  try { await _loadList(); } catch (_) { /* signed out / offline: keep the last badge */ }
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

function _buildModal() {
  const modal = document.createElement('div');
  modal.className = 'modal';
  modal.id = MODAL_ID;
  modal.innerHTML = `
    <div class="modal-content decisions-modal-content" role="dialog" aria-labelledby="decisions-modal-title">
      <div class="modal-header">
        <h4 id="decisions-modal-title">${ICON_SCALE}<span>Decisions</span></h4>
        <button type="button" class="close-btn" id="decisions-close" aria-label="Close decisions"></button>
      </div>
      <div class="modal-body decisions-body">
        <div class="decisions-layout" id="decisions-layout">
          <aside class="decisions-sidebar" aria-label="Decision list">
            <div class="decisions-list-toolbar">
              <input type="search" id="decisions-search" class="memory-search-input" placeholder="Search decisions" aria-label="Search decisions" autocomplete="off">
              <button type="button" class="memory-toolbar-btn active" id="decisions-new-btn">New</button>
            </div>
            <div class="decisions-due-banner" id="decisions-due-banner" role="status" hidden></div>
            <div class="decisions-list" id="decisions-list"></div>
            <label class="decisions-check decisions-archived-toggle">
              <input type="checkbox" id="decisions-show-archived"> Show archived
            </label>
          </aside>
          <section class="decisions-editor" id="decisions-editor" aria-label="Decision editor"></section>
        </div>
      </div>
    </div>`;
  return modal;
}

export async function openDecisions(decisionId = null) {
  if (_open) {
    if (Modals.isMinimized(MODAL_ID)) Modals.restore(MODAL_ID);
    if (decisionId) await _select(decisionId);
    return;
  }
  _open = true;
  const modal = _buildModal();
  document.body.appendChild(modal);
  document.getElementById('tool-decisions-btn')?.classList.add('active');

  Modals.register(MODAL_ID, {
    railBtnId: 'rail-decisions',
    sidebarBtnId: 'tool-decisions-btn',
    label: 'Decisions',
    icon: ICON_SCALE,
    restoreFn: () => {},
    closeFn: () => _teardown(),
  });
  Modals.injectMinimizeButton(modal, MODAL_ID);
  const content = modal.querySelector('.modal-content');
  makeWindowDraggable(modal, { content, header: modal.querySelector('.modal-header') });

  document.getElementById('decisions-close').addEventListener('click', () => closeDecisions());
  modal.addEventListener('click', (e) => {
    if (uiModule.isTouchInsideModal()) return;
    if (e.target === modal) closeDecisions();
  });
  document.getElementById('decisions-search').addEventListener('input', (e) => {
    _query = e.target.value.trim().toLowerCase();
    _renderList();
  });
  document.getElementById('decisions-show-archived').addEventListener('change', (e) => {
    _showArchived = e.target.checked;
    _renderList();
  });
  document.getElementById('decisions-new-btn').addEventListener('click', _create);
  document.getElementById('decisions-list').addEventListener('click', (e) => {
    const item = e.target.closest('[data-decision-id]');
    if (item) _select(item.dataset.decisionId);
  });
  _wireEditor(document.getElementById('decisions-editor'));

  _escHandler = (e) => {
    if (e.key !== 'Escape' || e.defaultPrevented) return;
    const m = document.getElementById(MODAL_ID);
    if (!m || m.classList.contains('hidden')) return;
    if (document.getElementById('styled-confirm-overlay')) return;
    closeDecisions();
  };
  document.addEventListener('keydown', _escHandler);

  _renderList();
  _renderEditor();
  try {
    await _loadList();
  } catch (err) {
    uiModule.showError(`Could not load decisions: ${err.message}`);
  }
  _renderList();
  const first = decisionId || (_list.find(d => d.revisit_due) || {}).id;
  if (first) await _select(first);
  else if (window.innerWidth > 768 && _list.length) await _select(_list[0].id);
  else _renderEditor();
}

export async function closeDecisions() {
  if (!_open) return;
  try { await _saveNow(); } catch (_) { /* keep the window closing; the error was shown */ }
  if (Modals.isRegistered(MODAL_ID)) Modals.close(MODAL_ID);
  else _teardown();
}

// Closing from the dock chip runs synchronously through modalManager, so an
// edit still waiting on the autosave timer is sent without awaiting it.
function _flushOnClose() {
  if (!_dirty || !_draft) return;
  const payload = _bodyPayload();
  fetch(`${API}/${encodeURIComponent(_draft.id)}?today=${today()}`, {
    method: 'PUT',
    credentials: 'same-origin',
    keepalive: true,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).catch(() => {});
  _dirty = false;
}

function _teardown() {
  if (!_open) return;
  _open = false;
  _openSeq++;
  clearTimeout(_saveTimer);
  _flushOnClose();
  if (_escHandler) document.removeEventListener('keydown', _escHandler);
  _escHandler = null;
  _current = null;
  _draft = null;
  document.getElementById('tool-decisions-btn')?.classList.remove('active');
  const modal = document.getElementById(MODAL_ID);
  const content = modal?.querySelector('.modal-content');
  if (content) {
    content.classList.add('modal-closing');
    setTimeout(() => modal.remove(), 200);
  } else {
    modal?.remove();
  }
}

export function toggleDecisions() {
  if (Modals.toggle(MODAL_ID)) return;
  if (_open) closeDecisions(); else openDecisions();
}

export function isDecisionsOpen() { return _open; }

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

function _renderList() {
  const listEl = document.getElementById('decisions-list');
  if (!listEl) return;
  const banner = document.getElementById('decisions-due-banner');
  if (banner) {
    banner.hidden = _dueCount === 0;
    banner.textContent = _dueCount
      ? `${_dueCount} decision${_dueCount === 1 ? ' is' : 's are'} due for a second look.`
      : '';
  }
  const visible = _list.filter(d => (_showArchived || d.status !== 'archived')
    && (!_query || (d.title || '').toLowerCase().includes(_query)
      || (d.chosen_option_name || '').toLowerCase().includes(_query)));
  if (!visible.length) {
    listEl.innerHTML = `<p class="memory-desc decisions-empty">${_list.length
      ? 'No decisions match.'
      : 'No decisions yet. Start one to weigh options against what matters.'}</p>`;
    return;
  }
  const selected = _draft && _draft.id;
  listEl.innerHTML = visible.map((d) => {
    const status = d.effective_status || d.status;
    const sub = d.chosen_option_name
      ? `Chose ${esc(d.chosen_option_name)}`
      : (d.leader_name ? `Leading: ${esc(d.leader_name)}` : `${d.option_count || 0} options`);
    const when = d.revisit_on ? `<span class="decisions-list-date">Revisit ${esc(d.revisit_on)}</span>` : '';
    return `<button type="button" class="decisions-list-item${d.id === selected ? ' active' : ''}"
        data-decision-id="${esc(d.id)}" aria-current="${d.id === selected ? 'true' : 'false'}">
        <span class="decisions-list-title">${esc(d.title || 'Untitled decision')}</span>
        <span class="decisions-list-meta">
          <span class="decisions-status decisions-status-${esc(status)}">${esc(STATUS_LABELS[status] || status)}</span>
          <span class="decisions-list-sub">${sub}</span>
          ${when}
        </span>
      </button>`;
  }).join('');
}

async function _create() {
  const seq = ++_openSeq;
  if (!(await _flushEdits())) return;
  try {
    const created = await _api('', {
      method: 'POST',
      body: {
        title: '',
        criteria: [{ id: newRowId(), name: '', weight: 1 }],
        options: [{ id: newRowId(), name: '' }, { id: newRowId(), name: '' }],
      },
    });
    _upsertSummary(created);
    // Another decision picked while this one was being created wins.
    if (seq !== _openSeq) return;
    if (!(await _flushEdits()) || seq !== _openSeq) return;
    _setCurrent(created);
    document.getElementById('decision-title')?.focus();
  } catch (err) {
    uiModule.showError(`Could not create decision: ${err.message}`);
  }
}

async function _select(id) {
  const seq = ++_openSeq;
  if (_draft && _draft.id === id) {
    _setEditing(true);
    return;
  }
  if (!(await _flushEdits())) return;
  try {
    const decision = await _api(`/${encodeURIComponent(id)}`);
    // The last decision asked for wins over a slower, earlier click.
    if (seq !== _openSeq) return;
    // Edits typed into the open decision while this one was loading.
    if (!(await _flushEdits()) || seq !== _openSeq) return;
    _setCurrent(decision);
  } catch (err) {
    if (seq === _openSeq) uiModule.showError(`Could not open decision: ${err.message}`);
  }
}

function _setCurrent(decision) {
  _current = decision;
  _draft = _toDraft(decision);
  _dirty = false;
  _renderList();
  _renderEditor();
  _setEditing(true);
}

// Narrow screens show either the list or the editor, with a Back button.
function _setEditing(on) {
  document.getElementById('decisions-layout')?.classList.toggle('decisions-layout-editing', !!(on && _draft));
}

// ---------------------------------------------------------------------------
// Editor rendering
// ---------------------------------------------------------------------------

function _setSaveState(text, isError = false) {
  const el = document.getElementById('decisions-save-state');
  if (!el) return;
  el.textContent = text;
  el.classList.toggle('decisions-save-error', !!isError);
}

function _crit(id) { return _draft.criteria.find(c => c.id === id); }
function _opt(id) { return _draft.options.find(o => o.id === id); }
function _optLabel(o, i) { return o.name || `Option ${i + 1}`; }
function _critLabel(c, i) { return c.name || `Criterion ${i + 1}`; }

function _effectiveStatus() {
  const status = _current?.status || 'open';
  return isRevisitDue(_draft.revisit_on, status, today()) ? 'revisit-due' : status;
}

function _linkRows(links, kind, ownerId = '') {
  return links.map((l, i) => `
    <div class="decisions-row" data-row="${kind}" data-index="${i}"${ownerId ? ` data-option-id="${esc(ownerId)}"` : ''}>
      <input class="task-form-input" data-key="${kind}:${esc(ownerId)}:${i}:title" data-prop="title" maxlength="300"
        value="${esc(l.title)}" placeholder="Title" aria-label="Link ${i + 1} title">
      <input class="task-form-input" type="url" data-key="${kind}:${esc(ownerId)}:${i}:url" data-prop="url" maxlength="2048"
        value="${esc(l.url)}" placeholder="https://" aria-label="Link ${i + 1} URL">
      ${_isSavableLink(l) ? `<a class="memory-toolbar-btn decisions-open-link" href="${esc(l.url)}" target="_blank" rel="noopener noreferrer" aria-label="Open link ${i + 1}">Open</a>` : ''}
      <button type="button" class="doc-action-icon-btn decisions-remove-btn" data-act="remove-${kind}" aria-label="Remove link ${i + 1}">${ICON_REMOVE}</button>
    </div>`).join('');
}

function _section(id, title, inner, hint = '') {
  return `<section class="decisions-section" aria-labelledby="dsec-${id}">
    <h5 class="decisions-section-title" id="dsec-${id}">${title}</h5>
    ${hint ? `<p class="memory-desc">${hint}</p>` : ''}
    ${inner}
  </section>`;
}

function _renderEditor() {
  const editor = document.getElementById('decisions-editor');
  if (!editor) return;
  if (!_draft) {
    editor.innerHTML = `<div class="decisions-placeholder memory-desc">${_list.length
      ? 'Pick a decision from the list, or start a new one.'
      : 'A decision notebook keeps the options, what matters, and why one option won, so you can check the choice later against the same criteria.'}</div>`;
    return;
  }
  const d = _draft;
  const mustHaves = d.constraints.filter(c => c.must_have);
  const history = _current?.history || [];
  const archived = _current?.status === 'archived';
  const chosenId = _current?.chosen_option_id || '';
  const preselect = (d.choice && _opt(d.choice) ? d.choice : '') || scoreOptions(d).leader_id || '';

  const constraints = d.constraints.map((c, i) => `
    <div class="decisions-row" data-row="constraint" data-id="${esc(c.id)}">
      <input class="task-form-input" data-key="k:${esc(c.id)}:text" data-prop="text" maxlength="500"
        value="${esc(c.text)}" placeholder="e.g. Must arrive before March" aria-label="Constraint ${i + 1}">
      <label class="decisions-check"><input type="checkbox" data-key="k:${esc(c.id)}:must" data-prop="must_have"${c.must_have ? ' checked' : ''}> Must-have</label>
      <button type="button" class="doc-action-icon-btn decisions-remove-btn" data-act="remove-constraint" aria-label="Remove constraint ${i + 1}">${ICON_REMOVE}</button>
    </div>`).join('');

  const criteria = d.criteria.map((c, i) => `
    <div class="decisions-row" data-row="criterion" data-id="${esc(c.id)}">
      <input class="task-form-input" data-key="c:${esc(c.id)}:name" data-prop="name" maxlength="200"
        value="${esc(c.name)}" placeholder="Criterion ${i + 1}" aria-label="Criterion ${i + 1} name">
      <input class="task-form-input decisions-num" type="number" min="0" max="100" step="1"
        data-key="c:${esc(c.id)}:weight" data-prop="weight" value="${esc(c.weight)}"
        aria-label="Weight for ${esc(_critLabel(c, i))}">
      <span class="decisions-share" data-share="${esc(c.id)}"></span>
      <button type="button" class="doc-action-icon-btn decisions-remove-btn" data-act="remove-criterion" aria-label="Remove criterion ${i + 1}">${ICON_REMOVE}</button>
    </div>`).join('');

  const matrixHead = d.criteria.map((c, i) => `<th scope="col"><span>${esc(_critLabel(c, i))}</span>
    <span class="decisions-share" data-share="${esc(c.id)}"></span></th>`).join('');
  const matrixRows = d.options.map((o, i) => `
    <tr data-option-id="${esc(o.id)}">
      <th scope="row">
        <input class="task-form-input" data-key="o:${esc(o.id)}:name" data-prop="name" maxlength="200"
          value="${esc(o.name)}" placeholder="Option ${i + 1}" aria-label="Option ${i + 1} name">
        <span class="decisions-flag" data-flag="${esc(o.id)}" hidden>Fails must-have</span>
      </th>
      ${d.criteria.map((c, ci) => `<td><input class="task-form-input decisions-num" type="number" min="0" max="10" step="0.5"
        data-key="s:${esc(o.id)}:${esc(c.id)}" data-score="${esc(c.id)}" value="${esc(o.scores[c.id] ?? '')}"
        aria-label="Score for ${esc(_optLabel(o, i))} on ${esc(_critLabel(c, ci))}, 0 to 10"></td>`).join('')}
      <td class="decisions-total" data-total="${esc(o.id)}"></td>
      <td class="decisions-rank" data-rank="${esc(o.id)}"></td>
      <td><button type="button" class="doc-action-icon-btn decisions-remove-btn" data-act="remove-option" aria-label="Remove ${esc(_optLabel(o, i))}">${ICON_REMOVE}</button></td>
    </tr>`).join('');

  const tradeoffs = d.options.map((o, i) => `
    <details class="decisions-tradeoff" data-option-id="${esc(o.id)}">
      <summary><span>${esc(_optLabel(o, i))}</span><span class="decisions-total" data-total="${esc(o.id)}"></span></summary>
      <label class="task-form-label" for="dt-desc-${esc(o.id)}">Description</label>
      <textarea class="task-form-input task-form-textarea" id="dt-desc-${esc(o.id)}" rows="2" maxlength="4000"
        data-key="o:${esc(o.id)}:description" data-prop="description">${esc(o.description)}</textarea>
      <div class="decisions-two-col">
        <div>
          <label class="task-form-label" for="dt-pros-${esc(o.id)}">Pros (one per line)</label>
          <textarea class="task-form-input task-form-textarea" id="dt-pros-${esc(o.id)}" rows="3"
            data-key="o:${esc(o.id)}:pros" data-prop="prosText">${esc(o.prosText)}</textarea>
        </div>
        <div>
          <label class="task-form-label" for="dt-cons-${esc(o.id)}">Cons (one per line)</label>
          <textarea class="task-form-input task-form-textarea" id="dt-cons-${esc(o.id)}" rows="3"
            data-key="o:${esc(o.id)}:cons" data-prop="consText">${esc(o.consText)}</textarea>
        </div>
      </div>
      ${mustHaves.length ? `<fieldset class="decisions-fails">
        <legend class="task-form-label">Must-haves this option fails</legend>
        ${mustHaves.map(c => `<label class="decisions-check"><input type="checkbox" data-key="f:${esc(o.id)}:${esc(c.id)}"
          data-fail="${esc(c.id)}"${o.fails.includes(c.id) ? ' checked' : ''}> ${esc(c.text || 'Untitled constraint')}</label>`).join('')}
      </fieldset>` : ''}
      <div class="task-form-label">Links</div>
      ${_linkRows(o.links, 'option-link', o.id)}
      <button type="button" class="memory-toolbar-btn" data-act="add-option-link">Add link</button>
    </details>`).join('');

  const choiceOptions = d.options.map((o, i) =>
    `<option value="${esc(o.id)}"${o.id === preselect ? ' selected' : ''}>${esc(_optLabel(o, i))}</option>`).join('');
  const decidedLine = _current?.decided_at
    ? `<p class="memory-desc">Recorded ${esc(_current.decided_at.slice(0, 10))}: <strong>${esc(
      (_opt(chosenId) && _optLabel(_opt(chosenId), d.options.indexOf(_opt(chosenId)))) || history.at(-1)?.chosen_option_name || 'removed option')}</strong></p>`
    : '';

  editor.innerHTML = `
    <div class="decisions-editor-top">
      <button type="button" class="memory-toolbar-btn decisions-back-btn" data-act="back">${ICON_BACK} Back</button>
      <input id="decision-title" class="task-form-input decisions-title-input" data-key="title" data-field="title"
        maxlength="200" value="${esc(d.title)}" placeholder="What are you deciding?" aria-label="Decision title">
      <span class="decisions-status" id="decisions-editor-status"></span>
    </div>
    <div class="decisions-actions">
      <button type="button" class="memory-toolbar-btn" data-act="discuss">${ICON_CHAT} Discuss in chat</button>
      <button type="button" class="memory-toolbar-btn" data-act="export">${ICON_EXPORT} Export</button>
      <button type="button" class="memory-toolbar-btn" data-act="archive">${ICON_ARCHIVE} ${archived ? 'Unarchive' : 'Archive'}</button>
      <button type="button" class="memory-toolbar-btn danger" data-act="delete">${ICON_TRASH} Delete</button>
      <span class="decisions-save-state memory-count" id="decisions-save-state" aria-live="polite">Saved</span>
    </div>
    <div class="decisions-revisit-banner" id="decisions-revisit-banner" role="status" hidden></div>
    ${_section('question', 'Question', `<textarea class="task-form-input task-form-textarea" rows="3" maxlength="4000"
        data-key="question" data-field="question" aria-labelledby="dsec-question"
        placeholder="The context, what prompted this, and what a good outcome looks like">${esc(d.question)}</textarea>`)}
    ${_section('constraints', 'Constraints', `${constraints}
      <button type="button" class="memory-toolbar-btn" data-act="add-constraint">Add constraint</button>`,
      'Mark the ones that are must-haves: an option that fails one is ranked last and cannot win.')}
    ${_section('criteria', 'Criteria', `${criteria}
      <button type="button" class="memory-toolbar-btn" data-act="add-criterion">Add criterion</button>`,
      'Weights are relative (0 to 100); the percentage is each criterion\'s share of the total.')}
    ${_section('options', 'Options', `<div class="decisions-matrix-wrap">
        <table class="decisions-matrix">
          <thead><tr><th scope="col">Option</th>${matrixHead}<th scope="col">Total</th><th scope="col">Rank</th>
            <th scope="col"><span class="a11y-visually-hidden">Remove</span></th></tr></thead>
          <tbody>${matrixRows}</tbody>
        </table>
      </div>
      <button type="button" class="memory-toolbar-btn" data-act="add-option">Add option</button>
      <p class="decisions-leader" id="decisions-leader" aria-live="polite"></p>`,
      'Score each option from 0 to 10 on each criterion. The total is the weighted average.')}
    ${_section('tradeoffs', 'Tradeoffs', tradeoffs || '<p class="memory-desc">Add options above to note their pros and cons.</p>')}
    ${_section('sources', 'Sources', `${_linkRows(d.sources, 'source')}
      <button type="button" class="memory-toolbar-btn" data-act="add-source">Add source</button>`)}
    ${_section('decision', 'Decision', `
      <label class="task-form-label" for="decision-choice">Chosen option</label>
      <select class="task-form-input" id="decision-choice">${choiceOptions}</select>
      <label class="task-form-label" for="decision-rationale">Why this option wins</label>
      <textarea class="task-form-input task-form-textarea" id="decision-rationale" rows="3" maxlength="8000"
        placeholder="The reasons in your own words; they are saved with the scores">${esc(d.rationale)}</textarea>
      <label class="task-form-label" for="decision-revisit">Revisit on</label>
      <input class="task-form-input decisions-date" type="date" id="decision-revisit" data-key="revisit" data-field="revisit_on" value="${esc(d.revisit_on)}">
      <label class="decisions-check"><input type="checkbox" id="decision-calendar"${d.calendar ? ' checked' : ''}> Add the revisit date to my calendar</label>
      <div class="task-form-actions">
        <button type="button" class="memory-toolbar-btn active" data-act="decide">${history.length ? 'Record again' : 'Record decision'}</button>
      </div>
      ${decidedLine}`, 'Recording saves the criteria, weights, scores and rationale as they stand now. Later edits do not change it.')}
    ${_section('history', 'History', `<div id="decisions-since"></div>${_historyHtml(history)}`)}
  `;
  _refreshComputed();
}

function _historyHtml(history) {
  if (!history.length) return '<p class="memory-desc">Nothing recorded yet.</p>';
  return [...history].reverse().map((snap) => {
    const rows = [...(snap.options || [])].sort((a, b) => (a.rank || 0) - (b.rank || 0)).map(o =>
      `<li>${esc(o.name || 'Untitled')}: ${Number(o.total || 0).toFixed(2)}${o.eligible === false ? ' (fails a must-have)' : ''}</li>`).join('');
    const weights = (snap.criteria || []).map(c => `${esc(c.name || 'Untitled')} ${formatShare(c.share)}`).join(', ');
    return `<details class="decisions-snapshot">
      <summary>${esc((snap.decided_at || '').slice(0, 10))}: chose <strong>${esc(snap.chosen_option_name || '')}</strong></summary>
      ${snap.rationale ? `<p>${esc(snap.rationale)}</p>` : ''}
      ${weights ? `<p class="memory-desc">Weights: ${weights}</p>` : ''}
      <ol class="decisions-snapshot-ranking">${rows}</ol>
      ${snap.revisit_on ? `<p class="memory-desc">Revisit set for ${esc(snap.revisit_on)}</p>` : ''}
    </details>`;
  }).join('');
}

function _sinceHtml(since) {
  if (!since) return '';
  const list = (label, items) => (items.length ? `<li>${label}: ${items.map(esc).join(', ')}</li>` : '');
  const fmt = v => (v === null || v === undefined ? 'blank' : String(v));
  const verdict = since.chosen_removed
    ? `The chosen option (${esc(since.chosen_option_name)}) has been removed.`
    : since.chosen_still_wins
      ? `${esc(since.chosen_option_name)} still leads.`
      : `${esc(since.chosen_option_name)} no longer leads${since.leader_after ? `; ${esc(since.leader_after)} does` : ''}.`;
  const original = since.chosen_removed ? '' : (since.chosen_wins_under_original_weights
    ? ' Judged by the original criteria and weights it would still win.'
    : ` Judged by the original criteria and weights, ${esc(since.leader_under_original_weights || 'no option')} would lead.`);
  if (!since.changed) {
    return `<div class="decisions-since"><p><strong>Since the decision:</strong> nothing in the matrix has changed. ${verdict}</p></div>`;
  }
  return `<div class="decisions-since${since.chosen_still_wins ? '' : ' decisions-since-flipped'}">
    <p><strong>Since the decision</strong> (${esc((since.decided_at || '').slice(0, 10))}): ${verdict}${original}</p>
    <ul>
      ${list('Criteria added', since.criteria_added)}
      ${list('Criteria removed', since.criteria_removed)}
      ${since.weight_changes.length ? `<li>Weights: ${since.weight_changes.map(w =>
        `${esc(w.criterion)} ${formatShare(w.before)} to ${formatShare(w.after)}`).join(', ')}</li>` : ''}
      ${list('Options added', since.options_added)}
      ${list('Options removed', since.options_removed)}
      ${since.score_changes.length ? `<li>Scores: ${since.score_changes.map(s =>
        `${esc(s.option)} on ${esc(s.criterion)} ${fmt(s.before)} to ${fmt(s.after)}`).join(', ')}</li>` : ''}
      ${since.ranking_changed ? `<li>Ranking was ${since.ranking_before.map(esc).join(' > ')}; now ${since.ranking_after.map(esc).join(' > ')}</li>` : ''}
    </ul>
  </div>`;
}

// Update totals, shares, ranks, flags and the comparison in place, so typing
// a score never rebuilds (and never steals focus from) the inputs.
function _refreshComputed() {
  const editor = document.getElementById('decisions-editor');
  if (!editor || !_draft) return;
  const scoring = scoreOptions(_draft);
  editor.querySelectorAll('[data-share]').forEach((el) => {
    el.textContent = formatShare(scoring.weights[el.dataset.share] || 0);
  });
  editor.querySelectorAll('[data-total]').forEach((el) => {
    const r = scoring.options[el.dataset.total];
    el.textContent = r ? r.total.toFixed(2) : '';
  });
  editor.querySelectorAll('[data-rank]').forEach((el) => {
    const r = scoring.options[el.dataset.rank];
    el.textContent = r ? String(r.rank) : '';
  });
  editor.querySelectorAll('tr[data-option-id]').forEach((row) => {
    const r = scoring.options[row.dataset.optionId];
    row.classList.toggle('decisions-leader-row', !!r && scoring.leader_id === row.dataset.optionId);
    row.classList.toggle('decisions-ineligible', !!r && !r.eligible);
  });
  editor.querySelectorAll('[data-flag]').forEach((el) => {
    const r = scoring.options[el.dataset.flag];
    el.hidden = !r || r.eligible;
  });
  const leader = document.getElementById('decisions-leader');
  if (leader) {
    const lo = scoring.leader_id && _opt(scoring.leader_id);
    if (!_draft.criteria.length) leader.textContent = 'Add a criterion to score the options.';
    else if (!lo) leader.textContent = _draft.options.length ? 'No option meets every must-have.' : '';
    else {
      leader.textContent = `Leading: ${_optLabel(lo, _draft.options.indexOf(lo))} (${scoring.options[lo.id].total.toFixed(2)} of 10)`
        + (scoring.tied ? ', tied with the next option' : '');
    }
  }
  const status = _effectiveStatus();
  const statusEl = document.getElementById('decisions-editor-status');
  if (statusEl) {
    statusEl.className = `decisions-status decisions-status-${status}`;
    statusEl.textContent = STATUS_LABELS[status] || status;
  }
  const banner = document.getElementById('decisions-revisit-banner');
  if (banner) {
    banner.hidden = status !== 'revisit-due';
    banner.textContent = status === 'revisit-due'
      ? 'This decision is due for a second look. Check what changed below, then record it again or pick a new revisit date.'
      : '';
  }
  const history = _current?.history || [];
  const sinceEl = document.getElementById('decisions-since');
  if (sinceEl) sinceEl.innerHTML = history.length ? _sinceHtml(compareToSnapshot(_draft, history.at(-1))) : '';
}

// A key that survives a rebuild for any focusable in the editor: inputs carry
// data-key, buttons are identified by their action plus the row they act on.
function _focusKey(el) {
  if (!el || !el.closest || !el.closest('#decisions-editor')) return null;
  if (el.dataset.key) return `k=${el.dataset.key}`;
  if (el.id) return `#${el.id}`;
  const row = el.closest('[data-row]');
  const owner = el.closest('[data-option-id]');
  const where = `${row?.dataset.id || row?.dataset.index || ''}/${owner?.dataset.optionId || ''}`;
  if (el.dataset.act) return `a=${el.dataset.act}@${where}`;
  if (el.tagName === 'SUMMARY') return `s=${owner?.dataset.optionId || el.textContent}`;
  return null;
}

// Rebuild the editor (after adding/removing rows or renaming), keeping focus.
// ``focusDataKey`` moves focus to a new row's input instead.
function _rerender(focusDataKey = null) {
  const active = document.activeElement;
  const key = focusDataKey ? `k=${focusDataKey}` : _focusKey(active);
  const caret = (!focusDataKey && active && 'selectionStart' in active) ? active.selectionStart : null;
  const open = new Set([...document.querySelectorAll('#decisions-editor details[open]')]
    .map(el => el.dataset.optionId || el.querySelector('summary')?.textContent));
  _renderEditor();
  document.querySelectorAll('#decisions-editor details').forEach((el) => {
    if (open.has(el.dataset.optionId || el.querySelector('summary')?.textContent)) el.open = true;
  });
  if (!key) return;
  const target = [...document.querySelectorAll(
    '#decisions-editor input, #decisions-editor textarea, #decisions-editor select, #decisions-editor button, #decisions-editor summary',
  )].find(el => _focusKey(el) === key);
  if (!target) return;
  target.closest('details')?.setAttribute('open', '');
  target.focus();
  if (caret !== null && 'setSelectionRange' in target) {
    try { target.setSelectionRange(caret, caret); } catch (_) { /* number/date inputs */ }
  }
}

// ---------------------------------------------------------------------------
// Editor events
// ---------------------------------------------------------------------------

function _applyInput(el) {
  const d = _draft;
  if (el.dataset.field) {
    d[el.dataset.field] = el.value;
    return;
  }
  const row = el.closest('[data-row]');
  const optionRow = el.closest('[data-option-id]');
  if (el.dataset.score && optionRow) {
    const o = _opt(optionRow.dataset.optionId);
    if (!o) return;
    const raw = el.value.trim();
    const value = raw === '' ? null : Math.min(10, Math.max(0, Number(raw)));
    if (value === null || Number.isNaN(value)) delete o.scores[el.dataset.score];
    else o.scores[el.dataset.score] = value;
    return;
  }
  if (el.dataset.fail && optionRow) {
    const o = _opt(optionRow.dataset.optionId);
    if (!o) return;
    o.fails = o.fails.filter(id => id !== el.dataset.fail);
    if (el.checked) o.fails.push(el.dataset.fail);
    return;
  }
  if (!row && optionRow && el.dataset.prop) {
    const o = _opt(optionRow.dataset.optionId);
    if (o) o[el.dataset.prop] = el.value;
    return;
  }
  if (!row) return;
  const prop = el.dataset.prop;
  const kind = row.dataset.row;
  if (kind === 'constraint') {
    const c = d.constraints.find(x => x.id === row.dataset.id);
    if (c) c[prop] = prop === 'must_have' ? el.checked : el.value;
  } else if (kind === 'criterion') {
    const c = _crit(row.dataset.id);
    if (!c) return;
    if (prop === 'weight') {
      const w = Number(el.value);
      c.weight = el.value.trim() === '' || Number.isNaN(w) ? 0 : Math.min(100, Math.max(0, w));
    } else {
      c[prop] = el.value;
    }
  } else if (kind === 'source') {
    d.sources[Number(row.dataset.index)][prop] = el.value;
  } else if (kind === 'option-link') {
    const o = _opt(row.dataset.optionId);
    if (o) o.links[Number(row.dataset.index)][prop] = el.value;
  }
}

function _wireEditor(editor) {
  editor.addEventListener('input', (e) => {
    const el = e.target;
    if (!_draft) return;
    if (el.id === 'decision-rationale') { _draft.rationale = el.value; return; }
    if (!el.dataset.key) return;
    _applyInput(el);
    _refreshComputed();
    _scheduleSave();
  });
  editor.addEventListener('change', (e) => {
    const el = e.target;
    if (!_draft) return;
    if (el.id === 'decision-choice') { _draft.choice = el.value; return; }
    if (el.id === 'decision-calendar') {
      _draft.calendar = el.checked;
      _syncRevisitEvent(_calendarPlan());
      return;
    }
    if (!el.dataset.key) return;
    _applyInput(el);
    // Names and must-have flags appear in other sections (matrix headers,
    // tradeoff summaries, the fail checklists): rebuild once editing settles.
    const structural = el.dataset.prop === 'name' || el.dataset.prop === 'must_have'
      || (el.dataset.prop === 'text' && el.closest('[data-row="constraint"]'))
      || /^(source|option-link):/.test(el.dataset.key);
    // Defer the rebuild: `change` fires before Tab moves focus, so rebuilding
    // now would drop the focus the user is moving to the next field.
    if (structural) setTimeout(() => { if (_draft) _rerender(); }, 0);
    else _refreshComputed();
    _scheduleSave();
    if (el.id === 'decision-revisit') {
      // Queued now, in the order of the user's changes; runs once the new
      // date is saved, and not at all if the save fails.
      const saved = _saveNow();
      saved.catch(() => {});
      _syncRevisitEvent(_calendarPlan(), saved);
    }
  });
  editor.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-act]');
    if (!btn || !_draft) return;
    _handleAction(btn.dataset.act, btn);
  });
}

function _handleAction(act, btn) {
  const d = _draft;
  const row = btn.closest('[data-row]');
  const optionEl = btn.closest('[data-option-id]');
  switch (act) {
    case 'back':
      _setEditing(false);
      return;
    case 'add-constraint': {
      const id = newRowId();
      d.constraints.push({ id, text: '', must_have: false });
      _rerender(`k:${id}:text`);
      break;
    }
    case 'add-criterion': {
      const id = newRowId();
      d.criteria.push({ id, name: '', weight: 1 });
      _rerender(`c:${id}:name`);
      break;
    }
    case 'add-option': {
      const id = newRowId();
      d.options.push({ id, name: '', description: '', scores: idDict(), fails: [], links: [], prosText: '', consText: '' });
      _rerender(`o:${id}:name`);
      break;
    }
    case 'add-source':
      d.sources.push({ title: '', url: '' });
      _rerender(`source::${d.sources.length - 1}:url`);
      break;
    case 'add-option-link': {
      const o = _opt(optionEl.dataset.optionId);
      o.links.push({ title: '', url: '' });
      _rerender(`option-link:${o.id}:${o.links.length - 1}:url`);
      break;
    }
    case 'remove-constraint': {
      const id = row.dataset.id;
      d.constraints = d.constraints.filter(c => c.id !== id);
      d.options.forEach((o) => { o.fails = o.fails.filter(f => f !== id); });
      _rerender();
      break;
    }
    case 'remove-criterion': {
      const id = row.dataset.id;
      d.criteria = d.criteria.filter(c => c.id !== id);
      d.options.forEach((o) => { delete o.scores[id]; });
      _rerender();
      break;
    }
    case 'remove-option':
      d.options = d.options.filter(o => o.id !== optionEl.dataset.optionId);
      _rerender();
      break;
    case 'remove-source':
      d.sources.splice(Number(row.dataset.index), 1);
      _rerender();
      break;
    case 'remove-option-link': {
      const o = _opt(row.dataset.optionId);
      o.links.splice(Number(row.dataset.index), 1);
      _rerender();
      break;
    }
    case 'decide': _decide(); return;
    case 'discuss': _discussInChat(); return;
    case 'export': _export(); return;
    case 'archive': _toggleArchive(); return;
    case 'delete': _delete(); return;
    default: return;
  }
  _scheduleSave();
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

// The async actions below read everything they need from the open decision
// before their first await and write results back to that decision by id:
// the user can open another decision while a request is in flight, and only
// a decision that is still open may have its in-memory copy or view updated.
function _isShowing(id) { return !!_draft && _draft.id === id; }

async function _decide() {
  if (!_draft) return;
  const optionId = document.getElementById('decision-choice')?.value || _draft.choice;
  if (!optionId) { uiModule.showError('Add an option before recording a decision'); return; }
  const id = _draft.id;
  // No revisit_on: the save below stores the draft's date before Record is
  // sent, and the server keeps the stored date when it is omitted. Sending
  // it again would let a slow Record undo a date changed after the click
  // (date changes save on their own, independently of this request).
  const body = { option_id: optionId, rationale: _draft.rationale };
  const calendar = _calendarPlan();
  const saved = _saveNow();
  const recorded = saved.then(() => _api(`/${encodeURIComponent(id)}/decide`, { method: 'POST', body }));
  recorded.catch(() => {});   // reported below
  // The calendar sync is queued at the click, behind the record request: a
  // box cleared or a date changed while recording is queued after it and
  // wins. It is skipped if recording fails.
  const synced = _syncRevisitEvent(calendar, recorded);
  try { await saved; } catch (_) { return; }
  try {
    const decided = await recorded;
    _upsertSummary(decided);
    if (_isShowing(id)) {
      _current = decided;
      _rerender();
    }
    uiModule.showToast('Decision recorded');
  } catch (err) {
    uiModule.showError(`Could not record decision: ${err.message}`);
    return;
  }
  await synced;
}

function _nextDay(isoDate) {
  const [y, m, d] = isoDate.split('-').map(Number);
  const next = new Date(Date.UTC(y, m - 1, d + 1));
  return next.toISOString().slice(0, 10);
}

/** What the open decision wants on the calendar, read now (see _syncRevisitEvent). */
function _calendarPlan() {
  const d = _draft;
  if (!d) return null;
  const want = d.calendar && !!d.revisit_on;
  return {
    id: d.id,
    want,
    event: want ? {
      summary: `Revisit decision: ${d.title || 'Untitled decision'}`,
      description: `${d.question || ''}\n\nOpen Decisions in Odysseus to review the recorded choice.`.trim(),
      dtstart: d.revisit_on,
      dtend: _nextDay(d.revisit_on),
      all_day: true,
    } : null,
  };
}

// Keep one calendar event in step with a decision's revisit date when the
// user opted in: create it, move it, or delete it when the box is cleared.
// A decision's syncs run one at a time in the order the user acted, so the
// latest intent always lands last (a quick tick-and-clear deletes the event
// the tick created). `after` is the request the sync depends on (recording
// the decision, saving a new date): the sync waits for it, keeping its place
// in the queue, and is dropped if that request fails.
function _syncRevisitEvent(plan, after = null) {
  if (!plan) return Promise.resolve();
  const run = (_calendarQueues.get(plan.id) || Promise.resolve()).then(async () => {
    if (after) {
      try { await after; } catch (_) { return; }
    }
    await _applyCalendarPlan(plan);
  });
  const tail = run.catch(() => {});
  _calendarQueues.set(plan.id, tail);
  tail.then(() => { if (_calendarQueues.get(plan.id) === tail) _calendarQueues.delete(plan.id); });
  return run;
}

async function _applyCalendarPlan({ id, want, event }) {
  const path = `/${encodeURIComponent(id)}`;
  const call = (eventPath, method, body) => fetch(`/api/calendar/events${eventPath}`, {
    method,
    credentials: 'same-origin',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  try {
    // The event uid lives on the server copy of *this* decision; the open
    // one may be another decision, or a copy older than an earlier sync.
    const uid = (await _api(path)).revisit_event_uid || '';
    let next = uid;
    if (want) {
      if (uid) {
        const res = await call(`/${encodeURIComponent(uid)}`, 'PUT', event);
        if (res.status === 404) next = '';
        else if (!res.ok) throw new Error(`HTTP ${res.status}`);
      }
      if (!next) {
        const res = await call('', 'POST', event);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        next = (await res.json()).uid || '';
      }
    } else if (uid) {
      const res = await call(`/${encodeURIComponent(uid)}`, 'DELETE');
      if (!res.ok && res.status !== 404) throw new Error(`HTTP ${res.status}`);
      next = '';
    }
    if (next !== uid) {
      const saved = await _api(path, { method: 'PUT', body: { revisit_event_uid: next } });
      if (_isShowing(id)) _current = saved;
    }
  } catch (err) {
    uiModule.showError(`Could not update the calendar: ${err.message}`);
  }
}

async function _discussInChat() {
  if (!_draft) return;
  const id = _draft.id;
  try { await _saveNow(); } catch (_) { return; }
  let summary;
  try {
    const res = await fetch(`${API}/${encodeURIComponent(id)}/export?brief=true&today=${today()}`,
      { credentials: 'same-origin' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    summary = await res.text();
  } catch (err) {
    uiModule.showError(`Could not summarize the decision: ${err.message}`);
    return;
  }
  // Same path as the gallery's "Discuss in chat": a fresh chat on the
  // default model, composer prefilled and focused, nothing sent.
  try {
    const dc = await (await fetch('/api/default-chat', { credentials: 'same-origin' })).json();
    if (!dc?.endpoint_url || !dc?.model || !window.sessionModule?.createDirectChat) {
      uiModule.showError('Pick a chat model first');
      return;
    }
    window.sessionModule.createDirectChat(dc.endpoint_url, dc.model, dc.endpoint_id);
  } catch (err) {
    uiModule.showError(`Could not start a chat: ${err.message}`);
    return;
  }
  Modals.minimize(MODAL_ID);
  const input = document.getElementById('message');
  if (input) {
    input.value = summary.trim();
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.focus();
  }
}

function _export() {
  if (!_draft) return;
  const id = _draft.id;
  _saveNow().catch(() => {}).finally(() => {
    const a = document.createElement('a');
    a.href = `${API}/${encodeURIComponent(id)}/export?download=true&today=${today()}`;
    a.download = '';
    document.body.appendChild(a);
    a.click();
    a.remove();
  });
}

async function _toggleArchive() {
  if (!_draft) return;
  const id = _draft.id;
  const archived = _current?.status === 'archived';
  const status = archived ? (_current.chosen_option_id ? 'decided' : 'open') : 'archived';
  try {
    await _saveNow();
    const saved = await _api(`/${encodeURIComponent(id)}`, { method: 'PUT', body: { status } });
    _upsertSummary(saved);
    if (_isShowing(id)) {
      _current = saved;
      _rerender();
    }
  } catch (err) {
    uiModule.showError(`Could not update decision: ${err.message}`);
  }
}

async function _delete() {
  if (!_draft) return;
  const id = _draft.id;
  const ok = await uiModule.styledConfirm(
    `Delete "${_draft.title || 'Untitled decision'}" and its recorded history?`,
    { title: 'Delete decision', confirmText: 'Delete', danger: true });
  if (!ok) return;
  if (_isShowing(id)) {
    clearTimeout(_saveTimer);
    _dirty = false;
  }
  try {
    await _api(`/${encodeURIComponent(id)}`, { method: 'DELETE' });
  } catch (err) {
    uiModule.showError(`Could not delete decision: ${err.message}`);
    return;
  }
  _list = _list.filter(d => d.id !== id);
  _dueCount = _list.filter(d => d.revisit_due).length;
  if (_isShowing(id)) {
    _current = null;
    _draft = null;
    _renderEditor();
    _setEditing(false);
  }
  _renderBadges();
  _renderList();
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

// Delegated so the entry keeps working if the sidebar re-parents or rebuilds
// the tool list items (pinning / "more tools").
document.addEventListener('click', (e) => {
  if (!e.target.closest || !e.target.closest('#tool-decisions-btn, #rail-decisions')) return;
  toggleDecisions();
});

function _startBadge() {
  refreshDecisionBadge();
  setInterval(refreshDecisionBadge, BADGE_REFRESH_MS);
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', _startBadge, { once: true });
else _startBadge();

const decisionsModule = { openDecisions, closeDecisions, toggleDecisions, isDecisionsOpen, refreshDecisionBadge };
window.decisionsModule = decisionsModule;
export default decisionsModule;

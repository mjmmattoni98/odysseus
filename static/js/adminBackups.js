// static/js/adminBackups.js — Settings > System > Backups (admin only).
//
// Full server-side backups (src/backup_service.py): the schedule, the list of
// archives with Verify / Download / Restore / Delete, and the guided restore
// stepper. Not the JSON Export/Import card above it, which only downloads
// selected settings. Long operations run on the server; this module polls
// /api/backups/operation while one is running.

import uiModule from './ui.js?v=20260916largetoolscroll1';
import { formatBytes } from './cookbookOllamaFormat.js';

const API = '/api/backups';
const POLL_MS = 1500;
const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const KIND_LABELS = { manual: 'Manual', scheduled: 'Scheduled', safety: 'Safety' };
// .settings-row sets display:flex, which beats the [hidden] attribute.
const HIDE = ' style="display:none"';
const CHECK_SVG = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';

let root = null;
let data = null;           // last GET /api/backups
let restore = null;        // { name, preview, previewError, loadingPreview, confirm }
let scheduleDraft = null;  // unsaved schedule edits
let pollTimer = null;
let watchedOpId = null;    // operation started from this page; its result is shown once
let resultNote = null;     // { ok, text }

// ui.esc expects a string; counts and sizes are numbers.
function esc(s) { return uiModule.esc(s === null || s === undefined ? '' : String(s)); }

function fmtTime(value) {
  if (!value) return '';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function browserTimezone() {
  try { return Intl.DateTimeFormat().resolvedOptions().timeZone || ''; } catch (_) { return ''; }
}

async function api(path, options = {}) {
  const res = await fetch(API + path, {
    credentials: 'same-origin',
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || body.error || `Request failed (${res.status})`);
  return body;
}

function backupByName(name) {
  return (data?.backups || []).find(b => b.name === name) || null;
}

function runningOp() {
  return data?.operation?.current || null;
}

/* ── Loading and polling ── */

export async function loadBackups() {
  if (!root) return;
  try {
    data = await api('');
  } catch (e) {
    root.innerHTML = `<div class="admin-error">${esc(e.message)}</div>`;
    return;
  }
  if (restore && !backupByName(restore.name) && data.pending_restore?.backup !== restore.name) restore = null;
  render();
  syncPolling();
  if (restore && backupByName(restore.name)?.verification?.ok && !restore.preview && !restore.loadingPreview && !restore.previewError) {
    loadPreview();
  }
}

function syncPolling() {
  const running = !!runningOp();
  if (running && !pollTimer) pollTimer = setInterval(pollOperation, POLL_MS);
  if (!running && pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollOperation() {
  let status;
  try { status = await api('/operation'); } catch (_) { return; }
  if (!data) return;
  data.operation = status;
  if (status.current) {
    renderStatus();
    return;
  }
  clearInterval(pollTimer);
  pollTimer = null;
  const last = status.last;
  if (last && last.id === watchedOpId) {
    watchedOpId = null;
    resultNote = { ok: !!last.ok, text: last.message || (last.ok ? 'Done.' : 'Failed.') };
    if (restore && last.kind === 'restore' && !last.ok) restore.stageError = last.message;
  }
  await loadBackups();
}

function startWatching(op) {
  if (!data) return;
  watchedOpId = op?.id || null;
  resultNote = null;
  data.operation = { current: op, last: data.operation?.last || null };
  render();
  syncPolling();
}

async function loadPreview() {
  if (!restore) return;
  const name = restore.name;
  restore.loadingPreview = true;
  restore.previewError = null;
  render();
  try {
    const preview = await api(`/${encodeURIComponent(name)}/restore-preview`);
    if (restore?.name === name) restore.preview = preview;
  } catch (e) {
    if (restore?.name === name) restore.previewError = e.message;
  }
  if (restore?.name === name) restore.loadingPreview = false;
  render();
}

/* ── Rendering ── */

function render() {
  if (!root || !data) return;
  root.innerHTML = `
    <div data-bk-status>${statusHtml()}</div>
    ${scheduleHtml()}
    ${listHtml()}
    ${restore ? stepperHtml() : ''}
  `;
}

function renderStatus() {
  const slot = root?.querySelector('[data-bk-status]');
  if (slot) slot.innerHTML = statusHtml();
  const busy = !!runningOp();
  root?.querySelectorAll('[data-busy-disable]').forEach(btn => { btn.disabled = busy; });
}

function statusHtml() {
  const parts = [];
  const pending = data.pending_restore;
  if (pending) {
    parts.push(`
      <div class="admin-user-row" style="border-color:var(--accent, var(--red));">
        <div class="admin-toggle-label">Restore waiting for a restart</div>
        <div class="admin-toggle-sub">${esc(pending.backup)} is staged (safety backup ${esc(pending.safety_backup || 'none')}). Restart Odysseus to apply it: it is applied at startup, before the database opens.</div>
        <div style="margin-top:6px;"><button type="button" class="admin-btn-sm" data-action="cancel-pending" data-busy-disable>Cancel staged restore</button></div>
      </div>`);
  }
  const last = data.last_restore;
  if (last && !pending) {
    parts.push(`<div class="${last.ok ? 'admin-success' : 'admin-error'}" style="margin-bottom:6px;">Last restore (${esc(fmtTime(last.applied_at))}): ${esc(last.message || '')}</div>`);
  }
  const op = runningOp();
  if (op) {
    parts.push(`<div class="admin-toggle-sub" style="margin-bottom:6px;" role="status"><span class="admin-spinner"></span>${esc(op.label)}: ${esc(op.message || 'Working')}</div>`);
  } else if (resultNote) {
    parts.push(`<div class="${resultNote.ok ? 'admin-success' : 'admin-error'}" style="margin-bottom:6px;" role="status">${esc(resultNote.text)}</div>`);
  }
  return parts.join('');
}

function scheduleHtml() {
  const s = scheduleDraft || data.schedule;
  const sched = data.scheduler || {};
  const off = s.frequency === 'off';
  const option = (value, label, current) => `<option value="${esc(value)}"${String(current) === String(value) ? ' selected' : ''}>${esc(label)}</option>`;
  let line = '';
  if (!off) {
    line = sched.next_run ? `Next run ${esc(fmtTime(sched.next_run))}` : 'Next run is computed when the schedule is saved';
    if (!sched.active) {
      line += '. The in-process scheduler is not running on this server (ODYSSEUS_INPROCESS_TASKS is off), so scheduled backups will not fire.';
    }
  }
  if (sched.last_run) {
    const res = sched.last_result || {};
    line += `${line ? ' · ' : ''}Last scheduled run ${esc(fmtTime(sched.last_run))}: ${esc(res.message || '')}`;
  }
  return `
    <div class="settings-col" style="margin-bottom:10px;">
      <div class="settings-row">
        <label class="settings-label" for="adm-bk-frequency">Schedule</label>
        <select id="adm-bk-frequency" class="settings-select" data-field="frequency">
          ${option('off', 'Off', s.frequency)}${option('daily', 'Daily', s.frequency)}${option('weekly', 'Weekly', s.frequency)}
        </select>
      </div>
      <div class="settings-row"${s.frequency === 'weekly' ? '' : HIDE}>
        <label class="settings-label" for="adm-bk-weekday">Day</label>
        <select id="adm-bk-weekday" class="settings-select" data-field="weekday">
          ${WEEKDAYS.map((d, i) => option(i, d, s.weekday)).join('')}
        </select>
      </div>
      <div class="settings-row"${off ? HIDE : ''}>
        <label class="settings-label" for="adm-bk-time">Time</label>
        <input id="adm-bk-time" type="time" class="settings-select" data-field="time" value="${esc(s.time)}" style="width:120px;flex:0 0 auto;">
        <span class="admin-toggle-sub">${esc(s.timezone || browserTimezone() || 'UTC')}</span>
      </div>
      <div class="settings-row"${off ? HIDE : ''}>
        <label class="settings-label" for="adm-bk-retention">Keep</label>
        <input id="adm-bk-retention" type="text" inputmode="numeric" class="settings-select" data-field="retention" value="${esc(s.retention)}" style="width:120px;flex:0 0 auto;">
        <span class="admin-toggle-sub">newest scheduled backups; older ones are deleted after a verified run</span>
      </div>
      <div class="settings-row">
        <button type="button" class="admin-btn-add" data-action="save-schedule">Save schedule</button>
        <span id="adm-bk-scheduleMsg" class="admin-toggle-sub"></span>
      </div>
      ${line ? `<div class="admin-toggle-sub">${line}</div>` : ''}
    </div>`;
}

function verificationBadge(v) {
  if (!v) return '<span class="admin-toggle-sub">Not verified</span>';
  // Intact archive, but some paths could not be read when it was made: it is
  // not a restore point (retention and the safety backup ignore it).
  if (v.ok && v.complete === false) {
    const why = ['Some paths could not be read when this backup was made:', ...(v.unreadable || []).slice(0, 3)];
    return `<span class="admin-badge admin-badge-off" title="${esc(why.join('\n'))}">Incomplete</span>`;
  }
  if (v.ok) {
    return `<span class="admin-badge" style="background:color-mix(in srgb, var(--color-success) 20%, transparent);color:var(--color-success);" title="Checked ${esc(fmtTime(v.checked_at))}">Verified</span>`;
  }
  return `<span class="admin-badge admin-badge-off" title="${esc((v.problems || []).slice(0, 3).join('\n'))}">Failed check</span>`;
}

function listHtml() {
  const busy = !!runningOp();
  const rows = (data.backups || []).map(b => `
    <div class="admin-ep-item" style="flex-wrap:wrap;">
      <div class="admin-ep-info" style="min-width:180px;">
        <div class="admin-ep-name">${esc(fmtTime(b.created_at))}</div>
        <div class="admin-ep-detail" style="margin-top:0;" title="${esc(b.name)}">${esc(formatBytes(b.size))} · ${esc(KIND_LABELS[b.kind] || b.kind)} · ${verificationBadge(b.verification)}</div>
      </div>
      <div class="admin-ep-actions" style="margin-left:auto;">
        <button type="button" class="admin-btn-sm" data-action="verify" data-name="${esc(b.name)}" data-busy-disable${busy ? ' disabled' : ''}>Verify</button>
        <button type="button" class="admin-btn-sm" data-action="download" data-name="${esc(b.name)}">Download</button>
        <button type="button" class="admin-btn-sm" data-action="restore" data-name="${esc(b.name)}">Restore</button>
        <button type="button" class="admin-btn-delete" data-action="delete" data-name="${esc(b.name)}" data-busy-disable${busy ? ' disabled' : ''}>Delete</button>
      </div>
    </div>`).join('');
  return `
    <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin:4px 0 6px;">
      <div class="admin-toggle-label">Backups on this server</div>
      <button type="button" class="admin-btn-add" data-action="create" data-busy-disable${busy ? ' disabled' : ''}>Back up now</button>
    </div>
    ${rows || '<div class="admin-empty">No backups yet</div>'}`;
}

function stepHtml(n, title, status, body) {
  const color = status === 'done' ? 'var(--color-success)' : status === 'current' ? 'var(--accent, var(--red))' : 'var(--border)';
  const label = `Step ${n}, ${status === 'done' ? 'done' : status === 'current' ? 'current' : 'not started'}`;
  return `
    <li style="display:flex;gap:8px;"${status === 'current' ? ' aria-current="step"' : ''}>
      <span role="img" aria-label="${label}" style="flex:0 0 20px;height:20px;border-radius:50%;border:1px solid ${color};color:${color};font-size:11px;display:inline-flex;align-items:center;justify-content:center;">${status === 'done' ? CHECK_SVG : n}</span>
      <div style="flex:1;min-width:0;${status === 'pending' ? 'opacity:0.5;' : ''}">
        <div class="admin-toggle-label" style="font-size:12px;">${esc(title)}</div>
        ${body ? `<div style="margin-top:4px;font-size:12px;">${body}</div>` : ''}
      </div>
    </li>`;
}

function countCell(value, other) {
  if (value === null || value === undefined) return '<td style="padding:2px 6px;text-align:right;opacity:0.5;">none</td>';
  const differs = other !== value;
  return `<td style="padding:2px 6px;text-align:right;${differs ? 'color:var(--accent, var(--red));' : ''}">${esc(value)}</td>`;
}

function files(n) {
  const count = Number(n) || 0;
  return `${count} file${count === 1 ? '' : 's'}`;
}

function previewHtml(p) {
  const s = p.summary || {};
  const tableRows = (p.tables || []).map(t => `
    <tr><td style="padding:2px 6px;">${esc(t.name)}</td>${countCell(t.backup, t.current)}${countCell(t.current, t.backup)}</tr>`).join('');
  const areaRows = (p.areas || []).map(a => `
    <tr><td style="padding:2px 6px;">${esc(a.area)}</td>
      <td style="padding:2px 6px;text-align:right;">${esc(a.backup_files)} (${esc(formatBytes(a.backup_bytes))})</td>
      <td style="padding:2px 6px;text-align:right;">${esc(a.current_files)} (${esc(formatBytes(a.current_bytes))})</td></tr>`).join('');
  const head = (first) => `<tr style="border-bottom:1px solid var(--border);"><th style="padding:2px 6px;text-align:left;font-weight:600;">${first}</th><th style="padding:2px 6px;text-align:right;font-weight:600;">In backup</th><th style="padding:2px 6px;text-align:right;font-weight:600;">Now</th></tr>`;
  const tableStyle = 'width:100%;border-collapse:collapse;font-size:11px;margin-top:4px;';
  return `
    <div class="admin-toggle-sub">Backup from ${esc(fmtTime(p.created_at))}, made by Odysseus ${esc(p.app_version || 'unknown')} (this server runs ${esc(p.current_app_version)}).</div>
    <div style="margin-top:4px;">${esc(files(s.replace))} replaced, ${esc(s.add ?? 0)} added from the backup, ${esc(files(s.set_aside))} of the current state set aside (moved to .restore-previous, not deleted). ${p.replaces_database === false ? 'The current database is kept.' : 'The database is replaced as a whole.'}</div>
    ${(p.warnings || []).map(w => `<div class="admin-error">${esc(w)}</div>`).join('')}
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;">
      ${sectionToggle('tables', `Database tables (${(p.tables || []).length})`)}
      ${sectionToggle('areas', `Files by area (${(p.areas || []).length})`)}
      ${sectionToggle('untouched', 'Left untouched')}
    </div>
    ${restore.open?.tables ? `<table style="${tableStyle}">${head('Table')}${tableRows}</table>` : ''}
    ${restore.open?.areas ? `<table style="${tableStyle}">${head('Area')}${areaRows}</table>` : ''}
    ${restore.open?.untouched ? `<ul style="margin:4px 0 0 16px;padding:0;font-size:11px;">${(p.untouched || []).map(u => `<li><code>${esc(u.path)}</code>: ${esc(u.reason)}</li>`).join('')}</ul>` : ''}`;
}

// A plain disclosure button: <details> carries the chat's research-callout
// styling globally, which does not belong in Settings.
function sectionToggle(key, label) {
  const open = !!restore.open?.[key];
  return `<button type="button" class="admin-btn-sm" data-action="toggle-section" data-section="${key}" aria-expanded="${open}">${esc(label)}</button>`;
}

function stepperHtml() {
  const name = restore.name;
  const backup = backupByName(name);
  const op = runningOp();
  const pending = data.pending_restore?.backup === name ? data.pending_restore : null;
  const verification = backup?.verification || null;
  const verified = !!verification?.ok;
  const verifying = op && op.kind === 'verify' && op.label.endsWith(name);
  const staging = op && op.kind === 'restore';
  const previewReady = !!restore.preview;
  const confirmed = (restore.confirm || '').trim() === name;

  // Step 2: verification.
  let verifyBody;
  if (verifying) verifyBody = `<span class="admin-spinner"></span>${esc(op.message || 'Verifying')}`;
  else if (verified) {
    verifyBody = `<span class="admin-toggle-sub">Checked ${esc(fmtTime(verification.checked_at))}: ${esc(verification.files_checked)} files match the manifest; every database passed integrity_check and its row counts.</span>`;
    if (verification.complete === false) {
      verifyBody += `<div class="admin-error">This backup is incomplete: these paths could not be read when it was made, so the files there are not in it and restoring sets the current ones aside:</div><ul style="margin:4px 0 0 16px;padding:0;font-size:11px;">${(verification.unreadable || []).map(u => `<li>${esc(u)}</li>`).join('')}</ul>`;
    }
  }
  else if (verification) verifyBody = `<div class="admin-error">Verification failed, so this backup cannot be restored:</div><ul style="margin:4px 0 0 16px;padding:0;font-size:11px;">${(verification.problems || []).map(p => `<li>${esc(p)}</li>`).join('')}</ul>`;
  else verifyBody = `<div class="admin-toggle-sub">Every file is re-hashed against the manifest and each database copy gets an integrity check.</div><button type="button" class="admin-btn-sm" style="margin-top:4px;" data-action="verify" data-name="${esc(name)}" data-busy-disable${op ? ' disabled' : ''}>Verify now</button>`;

  // Step 3: preview.
  let previewBody = '';
  if (verified && !pending) {
    if (restore.loadingPreview) previewBody = '<span class="admin-spinner"></span>Comparing with the current state';
    else if (restore.previewError) previewBody = `<div class="admin-error">${esc(restore.previewError)}</div><button type="button" class="admin-btn-sm" data-action="preview">Try again</button>`;
    else if (previewReady) previewBody = previewHtml(restore.preview);
  }

  // Steps 4-5: confirmation, safety backup and staging.
  let confirmBody = '';
  if (previewReady && !pending) {
    confirmBody = `
      <label class="admin-toggle-sub" for="adm-bk-confirm">Type the backup name to confirm: <code>${esc(name)}</code></label>
      <input id="adm-bk-confirm" type="text" class="settings-select" autocomplete="off" spellcheck="false" data-field="confirm" value="${esc(restore.confirm || '')}" style="width:100%;margin-top:4px;">`;
  }
  let stageBody = '';
  if (previewReady && !pending) {
    stageBody = staging
      ? `<span class="admin-spinner"></span>${esc(op.message || 'Working')}`
      : `<div class="admin-toggle-sub">A safety backup of the current state is taken and verified first; then this backup is checked again and staged.</div>
         <button type="button" class="admin-btn-delete" style="margin-top:4px;" data-action="stage" data-name="${esc(name)}"${confirmed && !op ? '' : ' disabled'}>Back up current state and stage restore</button>
         ${restore.stageError ? `<div class="admin-error">${esc(restore.stageError)}</div>` : ''}`;
  }
  const restartBody = pending
    ? `<div>Staged at ${esc(fmtTime(pending.staged_at))}; safety backup <code>${esc(pending.safety_backup || '')}</code>. Restart Odysseus to apply it. The result is shown here after the restart.</div>
       <button type="button" class="admin-btn-sm" style="margin-top:4px;" data-action="cancel-pending" data-busy-disable${op ? ' disabled' : ''}>Cancel staged restore</button>`
    : '';

  const st = (done, current) => (done ? 'done' : current ? 'current' : 'pending');
  return `
    <div class="admin-user-row" style="margin-top:10px;">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
        <div class="admin-toggle-label">Restore a backup</div>
        <button type="button" class="admin-btn-sm" data-action="restore-close">Close</button>
      </div>
      <ol style="list-style:none;padding:0;margin:8px 0 0;display:flex;flex-direction:column;gap:10px;">
        ${stepHtml(1, 'Choose a backup', 'done', `${esc(fmtTime(backup?.created_at || pending?.staged_at))} <span class="admin-toggle-sub">${esc(name)}</span>`)}
        ${stepHtml(2, 'Verify it', st(verified || pending, !verified), verifyBody)}
        ${stepHtml(3, 'Preview what will be replaced', st(pending || (verified && previewReady && confirmed), verified && !pending), previewBody)}
        ${stepHtml(4, 'Confirm', st(pending || confirmed, previewReady && !pending), confirmBody)}
        ${stepHtml(5, 'Safety backup and stage', st(pending, confirmed && !pending), stageBody)}
        ${stepHtml(6, 'Restart to apply', st(false, !!pending), restartBody)}
      </ol>
    </div>`;
}

/* ── Actions ── */

function readScheduleForm() {
  const value = (field) => root.querySelector(`[data-field="${field}"]`)?.value;
  return {
    frequency: value('frequency') || 'off',
    weekday: Number(value('weekday') ?? 6),
    time: value('time') || '03:00',
    retention: value('retention') || '7',
    timezone: (scheduleDraft || data.schedule)?.timezone || browserTimezone(),
  };
}

async function saveSchedule() {
  const msg = root.querySelector('#adm-bk-scheduleMsg');
  const body = { ...readScheduleForm(), timezone: browserTimezone() };
  try {
    const saved = await api('/schedule', { method: 'PUT', body: JSON.stringify(body) });
    scheduleDraft = null;
    data.schedule = saved.schedule;
    data.scheduler = { ...(data.scheduler || {}), next_run: saved.next_run, last_run: saved.last_run, last_result: saved.last_result, active: saved.active };
    render();
    const fresh = root.querySelector('#adm-bk-scheduleMsg');
    if (fresh) { fresh.textContent = 'Saved'; fresh.className = 'admin-success'; }
  } catch (e) {
    if (msg) { msg.textContent = e.message; msg.className = 'admin-error'; }
  }
}

async function onClick(event) {
  const btn = event.target.closest('[data-action]');
  if (!btn || !root.contains(btn)) return;
  const action = btn.dataset.action;
  const name = btn.dataset.name || '';
  try {
    if (action === 'create') {
      const res = await api('', { method: 'POST' });
      startWatching(res.operation);
    } else if (action === 'verify') {
      const res = await api(`/${encodeURIComponent(name)}/verify`, { method: 'POST' });
      startWatching(res.operation);
    } else if (action === 'download') {
      const a = document.createElement('a');
      a.href = `${API}/${encodeURIComponent(name)}/download`;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } else if (action === 'delete') {
      if (!await uiModule.styledConfirm(`Delete backup ${name}? The archive is removed from the server.`, { confirmText: 'Delete', danger: true })) return;
      await api(`/${encodeURIComponent(name)}`, { method: 'DELETE' });
      if (restore?.name === name) restore = null;
      await loadBackups();
    } else if (action === 'restore') {
      restore = { name, preview: null, previewError: null, loadingPreview: false, confirm: '', stageError: null, open: {} };
      render();
      if (backupByName(name)?.verification?.ok) loadPreview();
      root.querySelector('[aria-current="step"]')?.scrollIntoView({ block: 'nearest' });
    } else if (action === 'restore-close') {
      restore = null;
      render();
    } else if (action === 'toggle-section') {
      if (!restore) return;
      restore.open = { ...restore.open, [btn.dataset.section]: !restore.open?.[btn.dataset.section] };
      render();
      root.querySelector(`[data-section="${btn.dataset.section}"]`)?.focus();
    } else if (action === 'preview') {
      loadPreview();
    } else if (action === 'stage') {
      if (!restore || (restore.confirm || '').trim() !== name) return;
      restore.stageError = null;
      const res = await api(`/${encodeURIComponent(name)}/restore`, { method: 'POST', body: JSON.stringify({ confirm: restore.confirm.trim() }) });
      startWatching(res.operation);
    } else if (action === 'cancel-pending') {
      if (!await uiModule.styledConfirm('Cancel the staged restore? Nothing will be restored at the next restart.', { confirmText: 'Cancel restore' })) return;
      await api('/restore/pending', { method: 'DELETE' });
      await loadBackups();
    } else if (action === 'save-schedule') {
      await saveSchedule();
    }
  } catch (e) {
    if (action === 'stage' && restore) { restore.stageError = e.message; render(); return; }
    resultNote = { ok: false, text: e.message };
    renderStatus();
  }
}

function onInput(event) {
  const field = event.target?.dataset?.field;
  if (field === 'confirm' && restore) {
    const was = (restore.confirm || '').trim() === restore.name;
    restore.confirm = event.target.value;
    if (((restore.confirm || '').trim() === restore.name) === was) return;
    // The match flips steps 3-5: re-render, then hand the caret back.
    const caret = event.target.selectionStart;
    render();
    const input = root.querySelector('#adm-bk-confirm');
    if (input) { input.focus(); input.setSelectionRange(caret, caret); }
  }
}

function onChange(event) {
  const field = event.target?.dataset?.field;
  if (!field || field === 'confirm') return;
  scheduleDraft = readScheduleForm();
  if (field === 'frequency') render();
}

export function initBackups() {
  root = document.getElementById('adm-backups');
  if (!root || root.dataset.bound) return;
  root.dataset.bound = '1';
  root.addEventListener('click', onClick);
  root.addEventListener('input', onInput);
  root.addEventListener('change', onChange);
}

export function stopBackupsPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

// ============================================
// Cookbook — Ollama tab
// Installed + loaded models, streamed pulls, delete, unload/keep-alive and
// parameter presets through /api/cookbook/ollama/* (admin-only). Servers
// are addressed by the backend's allowlisted ids, never by raw URL.
// ============================================

import uiModule from './ui.js';
import {
  formatBytes, formatContext, aggregatePullProgress, parseSseEvents, vramInfo,
  formatExpiresIn, capabilityBadges, allocatedContext, modelKvBytes, validPresetName,
  suggestPresetName, parseStopSequences, sameOllamaRoot,
} from './cookbookOllamaFormat.js';

const SERVER_KEY = 'cookbook_ollama_server_v1';
const SERVERS_TTL_MS = 30000;
const INSTALLED_TTL_MS = 60000;
const API = '/api/cookbook/ollama';

let _servers = null;
let _serversAt = 0;
let _installedCache = new Map();   // server id -> { at, data }
let _pull = null;                   // { controller, model, serverId, state }
let _pollTimer = null;
let _lastInstalled = [];
let _defaultContext = 0;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function _json(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data?.detail;
    const msg = typeof detail === 'string' ? detail : (detail?.message || data?.error || res.statusText || `HTTP ${res.status}`);
    const err = new Error(msg);
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  return data;
}

function _post(path, body) {
  return fetch(`${API}${path}`, {
    method: 'POST', credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(_json);
}

export function invalidateOllamaCache() {
  _installedCache = new Map();
  _servers = null;
}

export async function fetchOllamaServers(force = false) {
  if (!force && _servers && Date.now() - _serversAt < SERVERS_TTL_MS) return _servers;
  try {
    const data = await fetch(`${API}/servers`, { credentials: 'same-origin' }).then(_json);
    _servers = Array.isArray(data?.servers) ? data.servers : [];
  } catch {
    _servers = [];
  }
  _serversAt = Date.now();
  return _servers;
}

function _connectHost(host) {
  const h = String(host || '').trim();
  if (!h || h === 'local') return '';
  return (h.includes('@') ? h.split('@').pop() : h).toLowerCase();
}

// The reachable Ollama server that corresponds to a Cookbook server entry
// ('' = local). Local prefers the discovered local daemon, then endpoints on
// loopback / host.docker.internal.
export function pickServerForHost(servers, host) {
  const list = (servers || []).filter(s => s && s.reachable);
  const want = _connectHost(host);
  if (!want) {
    const localHosts = new Set(['127.0.0.1', 'localhost', '::1', 'host.docker.internal']);
    return list.find(s => s.source === 'local') || list.find(s => localHosts.has(String(s.host || '').toLowerCase())) || null;
  }
  return list.find(s => String(s.host || '').toLowerCase() === want) || null;
}

export function findServerByUrl(servers, url) {
  return (servers || []).find(s => s && s.reachable && sameOllamaRoot(s.url, url)) || null;
}

export async function fetchInstalled(serverId, force = false) {
  const hit = _installedCache.get(serverId);
  if (!force && hit && Date.now() - hit.at < INSTALLED_TTL_MS) return hit.data;
  const data = await fetch(`${API}/models?server=${encodeURIComponent(serverId)}`, { credentials: 'same-origin' }).then(_json);
  _installedCache.set(serverId, { at: Date.now(), data });
  return data;
}

// For HW Fit: installed models (with real sizes) on the Ollama server behind
// a Cookbook host, or null when none is reachable.
export async function fetchInstalledForHost(host) {
  try {
    const server = pickServerForHost(await fetchOllamaServers(), host);
    if (!server) return null;
    return await fetchInstalled(server.id);
  } catch {
    return null;
  }
}

// Unload a model on the Ollama server at `url` via the API. Returns false
// when no allowlisted, reachable server matches (caller may fall back).
export async function unloadOnServerUrl(url, model) {
  if (!url || !model) return false;
  const server = findServerByUrl(await fetchOllamaServers(), url);
  if (!server) return false;
  try {
    await _post('/unload', { server: server.id, model });
    return true;
  } catch {
    return false;
  }
}

// ── Panel ──

function _panel() { return document.getElementById('cookbook-ollama-panel'); }

function _selectedServerId() {
  return document.getElementById('cookbook-ollama-server')?.value || '';
}

function _panelVisible() {
  const group = document.querySelector('#cookbook-modal .cookbook-group[data-backend-group="Ollama"]');
  const modal = document.getElementById('cookbook-modal');
  return !!group && !group.classList.contains('hidden') && !!modal && !modal.classList.contains('hidden');
}

function _panelSkeleton() {
  return `
    <div class="cookbook-ollama-toolbar">
      <select class="memory-sort-select" id="cookbook-ollama-server" aria-label="Ollama server"></select>
      <button type="button" class="hwfit-gpu-btn" id="cookbook-ollama-refresh" title="Refresh" aria-label="Refresh Ollama server"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M1 4v6h6"/><path d="M23 20v-6h-6"/><path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10"/><path d="M3.51 15a9 9 0 0 0 14.85 3.36L23 14"/></svg></button>
      <span class="cookbook-ollama-version" id="cookbook-ollama-version"></span>
    </div>
    <div class="cookbook-ollama-notice" id="cookbook-ollama-notice" hidden></div>
    <div class="cookbook-ollama-section">
      <h3 class="cookbook-ollama-h">Loaded in memory</h3>
      <div class="cookbook-ollama-list" id="cookbook-ollama-running"></div>
    </div>
    <div class="cookbook-ollama-section">
      <h3 class="cookbook-ollama-h">Pull a model</h3>
      <div class="cookbook-ollama-pull-row">
        <input type="text" class="memory-search-input" id="cookbook-ollama-pull-input" placeholder="qwen3:8b or hf.co/org/repo:Q4_K_M" autocomplete="off" />
        <button type="button" class="cookbook-btn" id="cookbook-ollama-pull-btn">Pull</button>
        <button type="button" class="memory-toolbar-btn" id="cookbook-ollama-pull-cancel" hidden>Cancel</button>
      </div>
      <div class="cookbook-ollama-progress" id="cookbook-ollama-progress" hidden role="progressbar" aria-valuemin="0" aria-valuemax="100"><div class="cookbook-ollama-progress-fill"></div></div>
      <div class="cookbook-ollama-pull-status" id="cookbook-ollama-pull-status"></div>
    </div>
    <div class="cookbook-ollama-section">
      <h3 class="cookbook-ollama-h">Installed <span class="memory-count" id="cookbook-ollama-installed-count"></span></h3>
      <div class="cookbook-ollama-list" id="cookbook-ollama-installed"></div>
    </div>
    <details class="cookbook-ollama-section cookbook-ollama-preset" id="cookbook-ollama-preset">
      <summary class="cookbook-ollama-h">Create preset</summary>
      <p class="memory-desc doclib-desc">Save a copy of an installed model with its own defaults (context, sampling, system prompt). Weights are shared, so presets take almost no disk space.</p>
      <div class="cookbook-ollama-form">
        <label>Base model<select class="cookbook-field-input" data-p="from"></select></label>
        <label>Preset name<input type="text" class="cookbook-field-input" data-p="name" placeholder="qwen3:8b-64k" autocomplete="off" /></label>
        <label>Context (num_ctx)<input type="number" class="cookbook-field-input" data-p="num_ctx" min="256" max="1048576" step="256" /></label>
        <label>Temperature<input type="number" class="cookbook-field-input" data-p="temperature" min="0" max="2" step="0.05" /></label>
        <label>top_p<input type="number" class="cookbook-field-input" data-p="top_p" min="0" max="1" step="0.01" /></label>
        <label>top_k<input type="number" class="cookbook-field-input" data-p="top_k" min="0" max="1000" step="1" /></label>
        <label>min_p<input type="number" class="cookbook-field-input" data-p="min_p" min="0" max="1" step="0.01" /></label>
        <label>Repeat penalty<input type="number" class="cookbook-field-input" data-p="repeat_penalty" min="0" max="2" step="0.01" /></label>
        <label>Max tokens (num_predict)<input type="number" class="cookbook-field-input" data-p="num_predict" min="-2" step="1" placeholder="-1 = unlimited" /></label>
        <label class="cookbook-ollama-form-wide">Stop sequences (one per line, \\n for newline)<textarea class="cookbook-field-input" data-p="stop" rows="2"></textarea></label>
        <label class="cookbook-ollama-form-wide">System prompt (optional)<textarea class="cookbook-field-input" data-p="system" rows="3"></textarea></label>
      </div>
      <div class="cookbook-ollama-form-actions">
        <span class="cookbook-ollama-form-hint" id="cookbook-ollama-preset-hint"></span>
        <button type="button" class="cookbook-btn" id="cookbook-ollama-preset-create">Create preset</button>
      </div>
    </details>`;
}

function _notice(text, isError = false) {
  const el = document.getElementById('cookbook-ollama-notice');
  if (!el) return;
  el.hidden = !text;
  el.textContent = text || '';
  el.classList.toggle('is-error', !!isError);
}

function _badgesHtml(caps) {
  return capabilityBadges(caps)
    .map(b => `<span class="cookbook-ollama-badge cookbook-ollama-badge-${esc(b.key)}">${esc(b.label)}</span>`)
    .join('');
}

function _renderRunning(models) {
  const el = document.getElementById('cookbook-ollama-running');
  if (!el) return;
  if (!models.length) {
    el.innerHTML = '<div class="cookbook-ollama-empty">No model is loaded right now.</div>';
    return;
  }
  el.innerHTML = models.map(m => {
    const v = vramInfo(m);
    const meta = [
      formatBytes(m.size),
      v.label,
      m.context_length ? `ctx ${formatContext(m.context_length)}` : '',
      formatExpiresIn(m.expires_at),
    ].filter(Boolean).join(' · ');
    const kept = formatExpiresIn(m.expires_at) === 'kept loaded';
    return `<div class="cookbook-ollama-row" data-model="${esc(m.name)}">
      <div class="cookbook-ollama-row-main"><span class="cookbook-ollama-name">${esc(m.name)}</span></div>
      <div class="cookbook-ollama-vram${v.onGpu ? ' is-full' : ''}" title="${esc(`${formatBytes(m.size_vram)} of ${formatBytes(m.size)} in VRAM`)}"><div class="cookbook-ollama-vram-fill" style="width:${v.pct}%"></div></div>
      <div class="cookbook-ollama-meta">${esc(meta)}</div>
      <div class="cookbook-ollama-actions">
        ${kept ? '' : '<button type="button" class="memory-toolbar-btn" data-act="keep">Keep loaded</button>'}
        <button type="button" class="memory-toolbar-btn" data-act="unload">Unload</button>
      </div>
    </div>`;
  }).join('');
}

function _renderInstalled(models) {
  const el = document.getElementById('cookbook-ollama-installed');
  const count = document.getElementById('cookbook-ollama-installed-count');
  if (count) count.textContent = models.length ? String(models.length) : '';
  if (!el) return;
  if (!models.length) {
    el.innerHTML = '<div class="cookbook-ollama-empty">No models installed on this server yet.</div>';
    return;
  }
  el.innerHTML = models.map(m => {
    const ctx = allocatedContext(m, _defaultContext);
    const kv = modelKvBytes(m, ctx);
    const meta = [
      formatBytes(m.size),
      m.quantization,
      m.parameter_size,
      m.context_length ? `ctx max ${formatContext(m.context_length)}` : '',
      m.parameters?.num_ctx ? `preset ctx ${formatContext(m.parameters.num_ctx)}` : '',
      kv ? `KV@${formatContext(ctx)} ≤ ${formatBytes(kv)}` : '',
    ].filter(Boolean).join(' · ');
    const isEmbed = (m.capabilities || []).includes('embedding');
    return `<div class="cookbook-ollama-row" data-model="${esc(m.name)}">
      <div class="cookbook-ollama-row-main"><span class="cookbook-ollama-name">${esc(m.name)}</span>${_badgesHtml(m.capabilities)}</div>
      <div class="cookbook-ollama-meta" title="KV estimate is f16 (upper bound); q8_0 KV cache is about half.">${esc(meta)}</div>
      <div class="cookbook-ollama-actions">
        ${isEmbed ? '' : '<button type="button" class="memory-toolbar-btn" data-act="load">Load</button>'}
        ${isEmbed ? '' : '<button type="button" class="memory-toolbar-btn" data-act="preset">Preset…</button>'}
        <button type="button" class="memory-toolbar-btn danger" data-act="delete">Delete</button>
      </div>
    </div>`;
  }).join('');
}

function _fillPresetBase(models) {
  const sel = document.querySelector('#cookbook-ollama-preset [data-p="from"]');
  if (!sel) return;
  const prev = sel.value;
  const chat = models.filter(m => !(m.capabilities || []).includes('embedding'));
  sel.innerHTML = chat.map(m => `<option value="${esc(m.name)}">${esc(m.name)}</option>`).join('');
  if (prev && chat.some(m => m.name === prev)) sel.value = prev;
  _syncPresetPlaceholders();
}

function _presetField(name) {
  return document.querySelector(`#cookbook-ollama-preset [data-p="${name}"]`);
}

function _syncPresetPlaceholders() {
  const base = _presetField('from')?.value || '';
  const model = _lastInstalled.find(m => m.name === base);
  const params = model?.parameters || {};
  for (const key of ['num_ctx', 'temperature', 'top_p', 'top_k', 'min_p', 'repeat_penalty']) {
    const input = _presetField(key);
    if (input) input.placeholder = params[key] !== undefined ? String(params[key]) : 'model default';
  }
  const ctxInput = _presetField('num_ctx');
  if (ctxInput && model?.context_length) ctxInput.max = String(model.context_length);
  const hint = document.getElementById('cookbook-ollama-preset-hint');
  if (hint) hint.textContent = model?.context_length ? `Max context for ${base}: ${formatContext(model.context_length)}` : '';
  const nameInput = _presetField('name');
  if (nameInput && !nameInput.dataset.touched) {
    nameInput.value = base ? suggestPresetName(base, Number(ctxInput?.value) || 0) : '';
  }
}

async function _refreshServers(force = false) {
  const sel = document.getElementById('cookbook-ollama-server');
  const servers = await fetchOllamaServers(force);
  if (!sel) return servers;
  const saved = (() => { try { return localStorage.getItem(SERVER_KEY) || ''; } catch { return ''; } })();
  const prev = sel.value || saved;
  sel.innerHTML = servers.length
    ? servers.map(s => `<option value="${esc(s.id)}">${esc(s.label)}${s.reachable ? '' : ' (offline)'}</option>`).join('')
    : '<option value="">No Ollama server found</option>';
  const pick = servers.find(s => s.id === prev && s.reachable) || servers.find(s => s.reachable) || servers[0];
  if (pick) sel.value = pick.id;
  return servers;
}

async function _refreshData({ force = false } = {}) {
  const serverId = _selectedServerId();
  const server = (_servers || []).find(s => s.id === serverId);
  const ver = document.getElementById('cookbook-ollama-version');
  if (ver) ver.textContent = server ? (server.reachable ? `Ollama ${server.version} · ${server.url}` : `${server.url} — offline`) : '';
  if (!server) {
    _notice('No Ollama server is registered or reachable. Start Ollama on this machine, or add it in Settings → Models (native URL, e.g. http://host.docker.internal:11434).');
    _renderRunning([]);
    _renderInstalled([]);
    return;
  }
  if (!server.reachable) {
    _notice(`${server.label} (${server.url}) is not answering. Start Ollama there, then refresh.`, true);
    _renderRunning([]);
    _renderInstalled([]);
    return;
  }
  _notice('');
  try {
    const [running, installed] = await Promise.all([
      fetch(`${API}/running?server=${encodeURIComponent(serverId)}`, { credentials: 'same-origin' }).then(_json),
      fetchInstalled(serverId, force),
    ]);
    _defaultContext = Number(installed?.default_context) || 0;
    _lastInstalled = installed?.models || [];
    _renderRunning(running?.models || []);
    _renderInstalled(_lastInstalled);
    _fillPresetBase(_lastInstalled);
  } catch (e) {
    _notice(`Could not read models: ${e.message || e}`, true);
  }
}

async function _refreshRunningOnly() {
  const serverId = _selectedServerId();
  if (!serverId) return;
  try {
    const running = await fetch(`${API}/running?server=${encodeURIComponent(serverId)}`, { credentials: 'same-origin' }).then(_json);
    _renderRunning(running?.models || []);
  } catch { /* transient */ }
}

function _startPolling() {
  if (_pollTimer) return;
  _pollTimer = setInterval(() => {
    if (!_panel()?.isConnected || !_panelVisible()) { clearInterval(_pollTimer); _pollTimer = null; return; }
    _refreshRunningOnly();
  }, 10000);
}

function _renderPullState() {
  const bar = document.getElementById('cookbook-ollama-progress');
  const status = document.getElementById('cookbook-ollama-pull-status');
  const btn = document.getElementById('cookbook-ollama-pull-btn');
  const cancel = document.getElementById('cookbook-ollama-pull-cancel');
  const active = !!_pull && !_pull.finished;
  if (btn) btn.disabled = active;
  if (cancel) cancel.hidden = !active;
  if (!_pull) {
    if (bar) bar.hidden = true;
    if (status) status.textContent = '';
    return;
  }
  const st = _pull.state || {};
  if (bar) {
    bar.hidden = false;
    const pct = st.percent ?? 0;
    bar.setAttribute('aria-valuenow', String(pct));
    bar.classList.toggle('is-error', !!st.error);
    const fill = bar.firstElementChild;
    if (fill) fill.style.width = `${st.percent === null ? 3 : pct}%`;
  }
  if (status) {
    let text;
    if (st.error) text = `${_pull.model}: ${st.error}`;
    else if (st.done) text = `${_pull.model}: pulled.`;
    else if (_pull.cancelled) text = `${_pull.model}: cancelled — pulling again resumes where it stopped.`;
    else {
      const bytes = st.totalBytes ? ` · ${formatBytes(st.completedBytes)} / ${formatBytes(st.totalBytes)}` : '';
      text = `${_pull.model}: ${st.status || 'starting'}${st.percent !== null && st.percent !== undefined ? ` · ${st.percent}%` : ''}${bytes}`;
    }
    status.textContent = text;
    status.classList.toggle('is-error', !!st.error);
  }
}

export async function startOllamaPull(serverId, model) {
  const name = String(model || '').trim();
  if (!serverId || !name) return false;
  if (_pull && !_pull.finished) {
    uiModule.showToast(`Already pulling ${_pull.model}`);
    return false;
  }
  const controller = new AbortController();
  _pull = { controller, model: name, serverId, state: aggregatePullProgress(null, { event: 'progress', data: {} }), finished: false };
  _renderPullState();
  try {
    const res = await fetch(`${API}/pull`, {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server: serverId, model: name }),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) await _json(res);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parsed = parseSseEvents(buffer);
      buffer = parsed.rest;
      for (const ev of parsed.events) _pull.state = aggregatePullProgress(_pull.state, ev);
      _renderPullState();
    }
    if (!_pull.state.done && !_pull.state.error) _pull.state = { ..._pull.state, error: 'Connection closed before the pull finished.' };
  } catch (e) {
    if (e?.name === 'AbortError') _pull.cancelled = true;
    else _pull.state = { ..._pull.state, error: e.message || String(e) };
  }
  _pull.finished = true;
  _renderPullState();
  const ok = !!_pull.state.done;
  if (ok) {
    uiModule.showToast(`Pulled ${name}`);
    _installedCache.delete(serverId);
    if (_panel()?.isConnected && _selectedServerId() === serverId) _refreshData({ force: true });
  }
  return ok;
}

async function _deleteModel(serverId, model) {
  const ok = await uiModule.styledConfirm(`Delete ${model} from this Ollama server? Its weights are removed from disk (shared layers used by other models are kept).`, { title: 'Delete model', confirmText: 'Delete', danger: true });
  if (!ok) return;
  const url = (force) => `${API}/models?server=${encodeURIComponent(serverId)}&model=${encodeURIComponent(model)}${force ? '&force=true' : ''}`;
  try {
    await fetch(url(false), { method: 'DELETE', credentials: 'same-origin' }).then(_json);
  } catch (e) {
    if (e.status !== 409) { uiModule.showError(`Delete failed: ${e.message}`); return; }
    const uses = (e.detail?.in_use || []).map(u => u.setting).join(', ');
    const again = await uiModule.styledConfirm(`${model} is currently configured as: ${uses}. Those features will fail until you pick another model. Delete anyway?`, { title: 'Model in use', confirmText: 'Delete anyway', danger: true });
    if (!again) return;
    try {
      await fetch(url(true), { method: 'DELETE', credentials: 'same-origin' }).then(_json);
    } catch (e2) {
      uiModule.showError(`Delete failed: ${e2.message}`);
      return;
    }
  }
  uiModule.showToast(`Deleted ${model}`);
  _installedCache.delete(serverId);
  _refreshData({ force: true });
}

async function _keepAlive(serverId, model, keepAlive, confirmLoad) {
  if (confirmLoad) {
    const ok = await uiModule.styledConfirm(`Load ${model} and keep it in memory? With OLLAMA_MAX_LOADED_MODELS=1 this evicts the currently loaded model.`, { title: 'Load model', confirmText: 'Load' });
    if (!ok) return;
    uiModule.showToast(`Loading ${model}…`);
  }
  try {
    if (keepAlive === 0) await _post('/unload', { server: serverId, model });
    else await _post('/keep-alive', { server: serverId, model, keep_alive: keepAlive });
    uiModule.showToast(keepAlive === 0 ? `Unloaded ${model}` : `${model} will stay loaded`);
  } catch (e) {
    uiModule.showError(`${keepAlive === 0 ? 'Unload' : 'Load'} failed: ${e.message}`);
  }
  _refreshRunningOnly();
}

function _presetBody(serverId, overwrite) {
  const val = (k) => _presetField(k)?.value?.trim() ?? '';
  const parameters = {};
  for (const key of ['num_ctx', 'temperature', 'top_p', 'top_k', 'min_p', 'repeat_penalty', 'num_predict']) {
    const v = val(key);
    if (v !== '') parameters[key] = Number(v);
  }
  const stop = parseStopSequences(_presetField('stop')?.value || '');
  if (stop.length) parameters.stop = stop;
  const body = { server: serverId, name: val('name'), from: val('from'), parameters, overwrite: !!overwrite };
  const system = _presetField('system')?.value || '';
  if (system.trim()) body.system = system;
  return body;
}

async function _createPreset(serverId) {
  const body = _presetBody(serverId, false);
  if (!body.from) { uiModule.showError('Pick a base model.'); return; }
  if (!validPresetName(body.name)) {
    uiModule.showError('Preset name: lowercase letters, digits, ".", "_" or "-", with one optional ":tag" (e.g. qwen3:8b-64k).');
    return;
  }
  const btn = document.getElementById('cookbook-ollama-preset-create');
  if (btn) btn.disabled = true;
  try {
    try {
      await _post('/create', body);
    } catch (e) {
      if (e.status !== 409) throw e;
      const ok = await uiModule.styledConfirm(`${body.name} already exists. Replace it?`, { title: 'Replace preset', confirmText: 'Replace', danger: true });
      if (!ok) return;
      await _post('/create', { ...body, overwrite: true });
    }
    uiModule.showToast(`Created ${body.name}`);
    const nameInput = _presetField('name');
    if (nameInput) delete nameInput.dataset.touched;
    _installedCache.delete(serverId);
    _refreshData({ force: true });
  } catch (e) {
    uiModule.showError(`Create failed: ${e.message}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function _wirePanel(panel) {
  panel.querySelector('#cookbook-ollama-server')?.addEventListener('change', (e) => {
    try { localStorage.setItem(SERVER_KEY, e.target.value || ''); } catch {}
    _refreshData();
  });
  panel.querySelector('#cookbook-ollama-refresh')?.addEventListener('click', async () => {
    await _refreshServers(true);
    _refreshData({ force: true });
  });
  const pullInput = panel.querySelector('#cookbook-ollama-pull-input');
  const doPull = () => startOllamaPull(_selectedServerId(), pullInput?.value || '');
  panel.querySelector('#cookbook-ollama-pull-btn')?.addEventListener('click', doPull);
  pullInput?.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); doPull(); } });
  panel.querySelector('#cookbook-ollama-pull-cancel')?.addEventListener('click', () => {
    if (_pull && !_pull.finished) _pull.controller.abort();
  });
  const onRowAction = (e) => {
    const btn = e.target.closest('button[data-act]');
    const row = e.target.closest('.cookbook-ollama-row');
    if (!btn || !row) return;
    const model = row.dataset.model;
    const serverId = _selectedServerId();
    const act = btn.dataset.act;
    if (act === 'unload') _keepAlive(serverId, model, 0, false);
    else if (act === 'keep') _keepAlive(serverId, model, -1, false);
    else if (act === 'load') _keepAlive(serverId, model, -1, true);
    else if (act === 'delete') _deleteModel(serverId, model);
    else if (act === 'preset') {
      const details = document.getElementById('cookbook-ollama-preset');
      const sel = _presetField('from');
      if (sel) sel.value = model;
      const nameInput = _presetField('name');
      if (nameInput) delete nameInput.dataset.touched;
      _syncPresetPlaceholders();
      if (details) { details.open = true; details.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
    }
  };
  panel.querySelector('#cookbook-ollama-running')?.addEventListener('click', onRowAction);
  panel.querySelector('#cookbook-ollama-installed')?.addEventListener('click', onRowAction);
  _presetField('from')?.addEventListener('change', _syncPresetPlaceholders);
  _presetField('num_ctx')?.addEventListener('input', _syncPresetPlaceholders);
  _presetField('name')?.addEventListener('input', (e) => { e.target.dataset.touched = '1'; });
  panel.querySelector('#cookbook-ollama-preset-create')?.addEventListener('click', () => _createPreset(_selectedServerId()));
}

export async function renderOllamaPanel({ force = false } = {}) {
  const panel = _panel();
  if (!panel) return;
  if (!panel.dataset.ready) {
    panel.innerHTML = _panelSkeleton();
    panel.dataset.ready = '1';
    _wirePanel(panel);
  }
  _renderPullState();
  await _refreshServers(force);
  await _refreshData({ force });
  _startPolling();
}

// Download tab hook: when the target host has a reachable Ollama API, pull
// through it (real progress in the Ollama tab) instead of the CLI/tmux flow.
export async function pullViaApiForHost(host, model) {
  const server = pickServerForHost(await fetchOllamaServers(true), host);
  if (!server) return false;
  const tab = document.querySelector('#cookbook-modal .cookbook-tab[data-backend="Ollama"]');
  if (tab && !tab.classList.contains('active')) tab.click();
  await renderOllamaPanel();
  const sel = document.getElementById('cookbook-ollama-server');
  if (sel) sel.value = server.id;
  const input = document.getElementById('cookbook-ollama-pull-input');
  if (input) input.value = model;
  uiModule.showToast(`Pulling ${model} through the Ollama API on ${server.label}`);
  startOllamaPull(server.id, model);
  return true;
}

if (typeof window !== 'undefined') {
  window.cookbookOllama = { renderOllamaPanel, pullViaApiForHost, fetchInstalledForHost, unloadOnServerUrl, invalidateOllamaCache };
}

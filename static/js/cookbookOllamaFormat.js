// ============================================
// Cookbook Ollama tab — pure helpers (no DOM, no imports) so they can be
// unit-tested under node (tests/cookbook_ollama_format.test.mjs).
// ============================================

const _LOOPBACK = new Set(['localhost', '127.0.0.1', '0.0.0.0', '::1', '[::1]', '::', '[::]']);
const GIB = 1024 ** 3;

// Decimal units, matching `ollama list` (18157010252 → "18.2 GB").
export function formatBytes(n, digits = 1) {
  const v = Number(n);
  if (!Number.isFinite(v) || v <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let x = v;
  while (x >= 1000 && i < units.length - 1) { x /= 1000; i++; }
  return `${i === 0 ? Math.round(x) : x.toFixed(digits)} ${units[i]}`;
}

export function formatContext(n) {
  const v = Number(n);
  if (!Number.isFinite(v) || v <= 0) return '';
  if (v >= 1024 && v % 1024 === 0) return `${v / 1024}K`;
  return String(Math.round(v));
}

// Fold one `/api/cookbook/ollama/pull` SSE event into the aggregate state.
// Ollama reports progress per layer (digest); overall progress is the sum
// across layers seen so far. Returns a new state object.
export function aggregatePullProgress(state, event) {
  const prev = state || { layers: {}, status: '', done: false, error: '' };
  const next = { ...prev, layers: { ...prev.layers } };
  const type = event?.event || 'progress';
  const data = event?.data || {};
  if (type === 'error') {
    next.error = String(data.error || 'Pull failed');
  } else if (type === 'done') {
    next.done = true;
    next.status = 'success';
  } else if (type === 'progress') {
    if (data.status) next.status = String(data.status);
    const digest = data.digest || '';
    const total = Number(data.total) || 0;
    if (digest && total > 0) {
      const old = next.layers[digest] || { total: 0, completed: 0 };
      next.layers[digest] = {
        total,
        completed: Math.min(total, Math.max(old.completed, Number(data.completed) || 0)),
      };
    }
    if (data.status === 'success') next.done = true;
  }
  let totalBytes = 0;
  let completedBytes = 0;
  for (const layer of Object.values(next.layers)) {
    totalBytes += layer.total;
    completedBytes += layer.completed;
  }
  next.totalBytes = totalBytes;
  next.completedBytes = completedBytes;
  next.percent = next.done ? 100 : (totalBytes > 0 ? Math.floor((completedBytes / totalBytes) * 100) : null);
  return next;
}

// Split an SSE text buffer into complete events; returns the unparsed tail.
export function parseSseEvents(buffer) {
  const events = [];
  const text = String(buffer || '').replace(/\r\n/g, '\n');
  const blocks = text.split('\n\n');
  const rest = blocks.pop();
  for (const block of blocks) {
    let event = 'message';
    const dataLines = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) continue;
    let data;
    try { data = JSON.parse(dataLines.join('\n')); } catch { data = { raw: dataLines.join('\n') }; }
    events.push({ event, data });
  }
  return { events, rest };
}

// VRAM residency for a loaded model from /api/ps (size_vram vs size).
export function vramInfo(model) {
  const size = Number(model?.size) || 0;
  const vram = Number(model?.size_vram) || 0;
  if (!size) return { pct: 0, onGpu: false, label: 'unknown' };
  const pct = Math.max(0, Math.min(100, Math.round((vram / size) * 100)));
  if (pct >= 100) return { pct: 100, onGpu: true, label: '100% GPU' };
  if (pct <= 0) return { pct: 0, onGpu: false, label: '100% CPU' };
  return { pct, onGpu: false, label: `${pct}% GPU / ${100 - pct}% CPU` };
}

// "in 14m", "in 2h 5m", "kept loaded" (keep_alive < 0 → year ~2318), "unloading".
export function formatExpiresIn(expiresAt, nowMs = Date.now()) {
  if (!expiresAt) return '';
  const t = Date.parse(expiresAt);
  if (!Number.isFinite(t)) return '';
  const diff = t - nowMs;
  if (diff > 365 * 24 * 3600 * 1000) return 'kept loaded';
  if (diff <= 0) return 'unloading';
  const mins = Math.round(diff / 60000);
  if (mins < 1) return 'in <1m';
  if (mins < 60) return `in ${mins}m`;
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  if (h < 48) return m ? `in ${h}h ${m}m` : `in ${h}h`;
  return `in ${Math.round(h / 24)}d`;
}

const _CAP_LABELS = [
  ['tools', 'tools'],
  ['vision', 'vision'],
  ['thinking', 'thinking'],
  ['embedding', 'embedding'],
];

export function capabilityBadges(caps) {
  const have = new Set((Array.isArray(caps) ? caps : []).map(c => String(c).toLowerCase()));
  return _CAP_LABELS.filter(([key]) => have.has(key)).map(([key, label]) => ({ key, label }));
}

// Rough KV cache for `ctx` tokens from the backend's f16 estimate
// (per layer: kv_heads × (key_len + value_len) × 2 bytes per cached token).
// Full-attention layers cache `ctx` tokens; sliding-window layers cache at
// most `window` tokens. An upper bound: q8_0 KV is ~0.53×, q4_0 ~0.28×.
export function kvCacheBytes(bytesPerToken, ctx, swaBytesPerToken = 0, window = 0) {
  const b = Number(bytesPerToken) || 0;
  const c = Number(ctx) || 0;
  if (c <= 0) return 0;
  const s = Number(swaBytesPerToken) || 0;
  const w = Number(window) || 0;
  return (b > 0 ? b * c : 0) + (s > 0 && w > 0 ? s * Math.min(c, w) : 0);
}

export function modelKvBytes(model, ctx) {
  return kvCacheBytes(model?.kv_bytes_per_token, ctx, model?.kv_swa_bytes_per_token, model?.kv_sliding_window);
}

// Allocated context for an installed model: its preset num_ctx, else the
// server/app default.
export function allocatedContext(model, defaultContext = 0) {
  const numCtx = Number(model?.parameters?.num_ctx) || 0;
  const def = Number(defaultContext) || 0;
  let ctx = numCtx || def;
  const max = Number(model?.context_length) || 0;
  if (max && ctx > max) ctx = max;
  return ctx;
}

// Memory footprint (GiB) of an installed model: real on-disk weights plus the
// KV-cache estimate for the allocated context.
export function installedFootprintGb(model, ctx) {
  const size = Number(model?.size) || 0;
  if (!size) return 0;
  return (size + modelKvBytes(model, ctx)) / GIB;
}

export function normalizeOllamaTag(name) {
  const n = String(name || '').trim().toLowerCase();
  if (!n) return '';
  const last = n.split('/').pop();
  return last.includes(':') ? n : `${n}:latest`;
}

export function findInstalled(installed, tag) {
  const want = normalizeOllamaTag(tag);
  if (!want || !Array.isArray(installed)) return null;
  return installed.find(m => normalizeOllamaTag(m?.name) === want) || null;
}

function _hostPort(url) {
  try {
    const u = new URL(String(url || '').includes('://') ? url : `http://${url}`);
    let host = (u.hostname || '').toLowerCase();
    if (_LOOPBACK.has(host)) host = 'loopback';
    return `${host}:${u.port || '11434'}`;
  } catch {
    return '';
  }
}

// Same Ollama server? Ignores path (/v1, /api) and treats loopback aliases alike.
export function sameOllamaRoot(a, b) {
  const x = _hostPort(a);
  return !!x && x === _hostPort(b);
}

export function ollamaRootOf(url) {
  try {
    const u = new URL(String(url || ''));
    return `${u.protocol}//${u.host}`;
  } catch {
    return '';
  }
}

const _PRESET_NAME_RE = /^[a-z0-9][a-z0-9._-]{0,79}(?::[a-z0-9][a-z0-9._-]{0,47})?$/;

export function validPresetName(name) {
  return _PRESET_NAME_RE.test(String(name || ''));
}

// Suggest `<family>:<tag>-<ctx>k` for a preset, e.g. qwen3.8:27b + 65536 → qwen3.8:27b-64k.
export function suggestPresetName(base, numCtx) {
  const b = String(base || '').trim().toLowerCase().split('/').pop();
  if (!b) return '';
  const [fam, tag = 'latest'] = b.split(':');
  const ctx = Number(numCtx) || 0;
  const suffix = ctx >= 1024 ? `${Math.round(ctx / 1024)}k` : 'custom';
  return `${fam}:${tag}-${suffix}`.replace(/[^a-z0-9._:-]+/g, '-');
}

export function parseStopSequences(text) {
  return String(text || '')
    .split('\n')
    .map(s => s.replace(/\\n/g, '\n'))
    .filter(s => s.length > 0);
}

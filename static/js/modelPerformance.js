/**
 * Model Performance card (Settings → Added Models).
 *
 * Read-only summary of the signed-in user's recent reply metrics per
 * endpoint/model from GET /api/model-performance: time to first token,
 * generation and prompt tok/s, and model load time. Frequent reloads mean the
 * local server keeps swapping models (e.g. Ollama with one loaded model).
 * Loads lazily the first time the Added Models tab is opened.
 */

const WINDOW_DAYS = 30;

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
  ));
}

function fmt(value, suffix = '') {
  return typeof value === 'number' && Number.isFinite(value) ? `${value}${suffix}` : '—';
}

/** Render the summary rows as table HTML (exported for tests). */
export function renderPerformanceTable(data) {
  const rows = Array.isArray(data?.models) ? data.models : [];
  if (!rows.length) {
    return `<div class="admin-empty">No reply metrics in the last ${esc(data?.days ?? WINDOW_DAYS)} days yet.</div>`;
  }
  const head = ['Model', 'Endpoint', 'Replies', 'TTFT median / p90', 'Gen tok/s', 'Prompt tok/s', 'Load median', 'Reloads']
    .map((label) => `<th style="text-align:left;padding:4px 8px;font-weight:600;white-space:nowrap;">${label}</th>`)
    .join('');
  const body = rows.map((row) => {
    const gen = row.gen_tps_source === 'computed'
      ? `<span title="Estimated from wall-clock time (the backend reported no speed)">~${fmt(row.gen_tps_median)}</span>`
      : fmt(row.gen_tps_median);
    const ttft = row.ttft_median_s == null ? '—' : `${fmt(row.ttft_median_s, 's')} / ${fmt(row.ttft_p90_s, 's')}`;
    const cells = [
      esc(String(row.model || '').split('/').pop()),
      esc(row.endpoint_label || ''),
      fmt(row.messages),
      ttft,
      gen,
      fmt(row.prompt_tps_median),
      fmt(row.load_median_s, 's'),
      row.reloads ? `<span title="Replies whose model load took over ${esc(Math.round((data.reload_threshold_ms || 500) / 100) / 10)}s">${esc(row.reloads)}</span>` : '0',
    ];
    return `<tr>${cells.map((cell) => `<td style="padding:4px 8px;white-space:nowrap;">${cell}</td>`).join('')}</tr>`;
  }).join('');
  return `<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-size:12px;">
    <thead><tr style="border-bottom:1px solid var(--border);">${head}</tr></thead>
    <tbody>${body}</tbody>
  </table></div>`;
}

let loading = null;

async function load() {
  const target = document.getElementById('model-perf-table');
  if (!target || loading) return loading;
  target.innerHTML = '<div class="admin-empty">Loading…</div>';
  loading = fetch(`/api/model-performance?days=${WINDOW_DAYS}`)
    .then(async (response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      target.innerHTML = renderPerformanceTable(await response.json());
    })
    .catch((error) => {
      target.innerHTML = `<div class="admin-empty">Could not load model performance (${esc(error.message)}).</div>`;
    })
    .finally(() => { loading = null; });
  return loading;
}

let loadedOnce = false;

function init() {
  document.addEventListener('click', (event) => {
    if (event.target.closest?.('#model-perf-refresh')) {
      load();
      return;
    }
    if (!loadedOnce && event.target.closest?.('[data-settings-tab="added-models"]')) {
      loadedOnce = true;
      load();
    }
  });
}

if (typeof document !== 'undefined' && document.addEventListener) {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}

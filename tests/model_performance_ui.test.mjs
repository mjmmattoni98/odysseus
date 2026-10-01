import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';

// chatRenderer.js imports the whole UI graph, so evaluate just the helper.
async function speedSummary() {
  const source = await readFile(new URL('../static/js/chatRenderer.js', import.meta.url), 'utf8');
  const constant = source.match(/^const SIGNIFICANT_MODEL_LOAD_MS = .*$/m)[0];
  const fn = source.match(/^export function backendSpeedSummary\([\s\S]*?^\}/m)[0];
  return import(`data:text/javascript,${encodeURIComponent(`${constant}\n${fn}`)}`);
}

test('stats line shows prompt speed, significant model loads and length cut-offs', async () => {
  const { backendSpeedSummary } = await speedSummary();
  assert.deepEqual(
    backendSpeedSummary({ prefill_tps: 612.5, load_ms: 18240, finish_reason: 'length' }),
    { promptTps: 612.5, loadSeconds: 18.2, truncated: true },
  );
  // A warm model's load time is noise, not a model-swap signal.
  assert.equal(backendSpeedSummary({ load_ms: 120 }).loadSeconds, null);
  assert.deepEqual(backendSpeedSummary({}), { promptTps: null, loadSeconds: null, truncated: false });
});

test('performance table escapes names and marks wall-clock speed as estimated', async () => {
  const { renderPerformanceTable } = await import('../static/js/modelPerformance.js');
  const html = renderPerformanceTable({
    days: 30,
    reload_threshold_ms: 500,
    models: [{
      model: 'library/qwen<img src=x>', endpoint_label: 'Ollama', messages: 3,
      ttft_median_s: 2.1, ttft_p90_s: 9.5, gen_tps_median: 12, gen_tps_source: 'computed',
      prompt_tps_median: null, load_median_s: 0.1, reloads: 2,
    }],
  });
  assert.match(html, />qwen&lt;img src=x&gt;</);
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /~12/);
  assert.match(html, /2\.1s \/ 9\.5s/);
  assert.match(renderPerformanceTable({ days: 30, models: [] }), /No reply metrics in the last 30 days/);
});

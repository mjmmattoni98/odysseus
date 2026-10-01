import test from 'node:test';
import assert from 'node:assert/strict';
import {
  formatBytes, formatContext, aggregatePullProgress, parseSseEvents, vramInfo, formatExpiresIn,
  capabilityBadges, kvCacheBytes, modelKvBytes, allocatedContext, installedFootprintGb, findInstalled,
  sameOllamaRoot, validPresetName, suggestPresetName, parseStopSequences,
} from '../static/js/cookbookOllamaFormat.js';

test('bytes use decimal units like `ollama list`', () => {
  assert.equal(formatBytes(18157010252), '18.2 GB');
  assert.equal(formatBytes(45_000_000), '45.0 MB');
  assert.equal(formatBytes(512), '512 B');
  assert.equal(formatBytes(0), '0 B');
  assert.equal(formatBytes(undefined), '0 B');
  assert.equal(formatContext(65536), '64K');
  assert.equal(formatContext(5000), '5000');
});

test('pull progress sums bytes across layers and never goes backwards', () => {
  let s = aggregatePullProgress(null, { event: 'progress', data: { status: 'pulling manifest' } });
  assert.equal(s.percent, null);
  s = aggregatePullProgress(s, { event: 'progress', data: { status: 'pulling a', digest: 'a', total: 300, completed: 150 } });
  s = aggregatePullProgress(s, { event: 'progress', data: { status: 'pulling b', digest: 'b', total: 100, completed: 50 } });
  assert.equal(s.totalBytes, 400);
  assert.equal(s.completedBytes, 200);
  assert.equal(s.percent, 50);
  s = aggregatePullProgress(s, { event: 'progress', data: { status: 'pulling a', digest: 'a', total: 300, completed: 10 } });
  assert.equal(s.layers.a.completed, 150, 'a stale/retried frame must not shrink progress');
  s = aggregatePullProgress(s, { event: 'done', data: { status: 'success' } });
  assert.equal(s.done, true);
  assert.equal(s.percent, 100);
  const e = aggregatePullProgress(s, { event: 'error', data: { error: 'disk full' } });
  assert.equal(e.error, 'disk full');
});

test('SSE parser returns complete events and keeps the partial tail', () => {
  const { events, rest } = parseSseEvents('event: progress\ndata: {"status":"x","total":1}\n\nevent: done\ndata: {"sta');
  assert.deepEqual(events, [{ event: 'progress', data: { status: 'x', total: 1 } }]);
  assert.equal(rest, 'event: done\ndata: {"sta');
  const again = parseSseEvents(rest + 'tus":"success"}\n\n');
  assert.deepEqual(again.events, [{ event: 'done', data: { status: 'success' } }]);
  assert.equal(again.rest, '');
});

test('VRAM residency label', () => {
  assert.deepEqual(vramInfo({ size: 100, size_vram: 100 }), { pct: 100, onGpu: true, label: '100% GPU' });
  assert.deepEqual(vramInfo({ size: 100, size_vram: 62 }), { pct: 62, onGpu: false, label: '62% GPU / 38% CPU' });
  assert.equal(vramInfo({ size: 100, size_vram: 0 }).label, '100% CPU');
  assert.equal(vramInfo({}).label, 'unknown');
});

test('expires-in formatting', () => {
  const now = Date.parse('2026-09-30T10:00:00Z');
  assert.equal(formatExpiresIn('2026-09-30T10:14:00Z', now), 'in 14m');
  assert.equal(formatExpiresIn('2026-09-30T12:05:00Z', now), 'in 2h 5m');
  assert.equal(formatExpiresIn('2318-01-01T00:00:00Z', now), 'kept loaded');
  assert.equal(formatExpiresIn('2026-09-30T09:59:00Z', now), 'unloading');
  assert.equal(formatExpiresIn('', now), '');
});

test('capability badges keep a stable order and ignore completion', () => {
  assert.deepEqual(capabilityBadges(['completion', 'thinking', 'vision', 'tools']).map(b => b.key), ['tools', 'vision', 'thinking']);
  assert.deepEqual(capabilityBadges(['embedding']).map(b => b.key), ['embedding']);
  assert.deepEqual(capabilityBadges(null), []);
});

test('installed footprint = real size + KV estimate for the allocated context', () => {
  const model = { size: 18 * 1024 ** 3, kv_bytes_per_token: 69632, context_length: 262144, parameters: { num_ctx: 65536 } };
  assert.equal(allocatedContext(model, 32768), 65536);
  assert.equal(allocatedContext({ context_length: 8192 }, 32768), 8192, 'clamped to the model max');
  assert.equal(allocatedContext({}, 32768), 32768);
  assert.equal(kvCacheBytes(69632, 65536), 69632 * 65536);
  const gb = installedFootprintGb(model, 65536);
  assert.ok(Math.abs(gb - (18 + (69632 * 65536) / 1024 ** 3)) < 1e-9);
  assert.equal(installedFootprintGb({ size: 1024 ** 3 }, 65536), 1, 'unknown KV adds nothing');
  // Sliding-window layers are capped at the window.
  const gemma = { kv_bytes_per_token: 4096, kv_swa_bytes_per_token: 81920, kv_sliding_window: 1024 };
  assert.equal(modelKvBytes(gemma, 65536), 4096 * 65536 + 81920 * 1024);
  assert.equal(modelKvBytes(gemma, 512), 4096 * 512 + 81920 * 512);
});

test('installed lookup treats a bare name as :latest', () => {
  const installed = [{ name: 'llama3.2:latest' }, { name: 'qwen3:8b' }];
  assert.equal(findInstalled(installed, 'llama3.2')?.name, 'llama3.2:latest');
  assert.equal(findInstalled(installed, 'QWEN3:8B')?.name, 'qwen3:8b');
  assert.equal(findInstalled(installed, 'qwen3:14b'), null);
});

test('same Ollama server across /v1, native and loopback aliases', () => {
  assert.ok(sameOllamaRoot('http://localhost:11434/v1', 'http://127.0.0.1:11434'));
  assert.ok(sameOllamaRoot('http://host.docker.internal:11434/v1', 'http://host.docker.internal:11434'));
  assert.ok(!sameOllamaRoot('http://localhost:11435', 'http://localhost:11434'));
  assert.ok(!sameOllamaRoot('http://gpu-box:11434', 'http://localhost:11434'));
  assert.ok(!sameOllamaRoot('', 'http://localhost:11434'));
});

test('preset names and helpers', () => {
  assert.ok(validPresetName('qwen3.8:27b-64k'));
  assert.ok(!validPresetName('Qwen:27b'));
  assert.ok(!validPresetName('a:b:c'));
  assert.equal(suggestPresetName('qwen3.8:27b', 65536), 'qwen3.8:27b-64k');
  assert.equal(suggestPresetName('llama3.2', 0), 'llama3.2:latest-custom');
  assert.deepEqual(parseStopSequences('<|im_end|>\n\nUser:\\n'), ['<|im_end|>', 'User:\n']);
});

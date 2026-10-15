import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  memoryFitFacet, runtimeFacet, taskFacet, partitionChatFirst, installedOllamaFacets, withInstalledOllamaFacets,
} from '../static/js/cookbookFacets.js';

// Shared with tests/test_hwfit_recommendation_facets.py (services/hwfit/facets.py).
const NATIVE_CASES = JSON.parse(
  readFileSync(new URL('./fixtures/ollama_native_capability_cases.json', import.meta.url), 'utf8'),
).installed;

test('memory fit labels name memory, not suitability', () => {
  const perfect = memoryFitFacet('perfect');
  assert.notEqual(perfect.text.toLowerCase(), 'perfect');
  assert.match(perfect.title, /^Memory:/);
  assert.match(perfect.title, /Task and Engine/);
  assert.equal(memoryFitFacet('too_tight').text, 'no fit');
  assert.equal(memoryFitFacet(undefined).text, '?');
});

test('runtime facet reads the displayed engine and never invents "compatible"', () => {
  const row = { runtime: {
    llamacpp: { status: 'not_supported', reason: 'No GGUF build listed; llama.cpp needs GGUF files.' },
    vllm: { status: 'compatible', reason: 'Has an official vLLM recipe.' },
  } };
  assert.deepEqual(runtimeFacet(row, 'llamacpp'), {
    status: 'not_supported', label: 'not supported', reason: 'No GGUF build listed; llama.cpp needs GGUF files.',
  });
  assert.equal(runtimeFacet(row, 'vllm').status, 'compatible');
  // Missing engine, cached rows without facets, and unknown statuses fall back.
  assert.equal(runtimeFacet(row, 'sglang').status, 'unverified');
  assert.equal(runtimeFacet({}, 'vllm').status, 'unverified');
  assert.equal(runtimeFacet({ runtime: { vllm: { status: 'works!' } } }, 'vllm').status, 'unverified');
});

test('task facet keeps an explicit unknown fallback', () => {
  assert.deepEqual(
    { ...taskFacet({ task: 'time_series', task_label: 'Time series', chat_capable: false }), title: undefined },
    { task: 'time_series', label: 'Time series', chatCapable: false, title: undefined },
  );
  assert.match(taskFacet({ task: 'time_series', task_label: 'Time series', chat_capable: false }).title, /not a chat model/);
  const unknown = taskFacet({});
  assert.equal(unknown.task, 'unknown');
  assert.equal(unknown.label, 'Unknown');
  assert.equal(unknown.chatCapable, null);
});

test('chat-first partition is stable and reports the first non-chat row', () => {
  const rows = [
    { name: 'forecaster', chat_capable: false },
    { name: 'chat-a', chat_capable: true },
    { name: 'mystery', chat_capable: null },
    { name: 'embedder', chat_capable: false },
    { name: 'chat-b', chat_capable: true },
  ];
  const { rows: ordered, firstOtherIndex } = partitionChatFirst(rows);
  assert.deepEqual(ordered.map(r => r.name), ['chat-a', 'mystery', 'chat-b', 'forecaster', 'embedder']);
  assert.equal(firstOtherIndex, 3);
  assert.equal(partitionChatFirst([{ chat_capable: true }]).firstOtherIndex, -1);
  assert.equal(partitionChatFirst([{ chat_capable: false }]).firstOtherIndex, -1);
  assert.deepEqual(partitionChatFirst(null).rows, []);
});

test('installed-only Ollama tags take their task from Ollama capabilities', () => {
  assert.equal(installedOllamaFacets(['completion', 'vision']).task, 'vision');
  assert.equal(installedOllamaFacets(['completion', 'tools']).chat_capable, true);
  assert.equal(installedOllamaFacets(['embedding']).chat_capable, false);
  assert.equal(installedOllamaFacets(undefined).task, 'unknown');
  assert.equal(installedOllamaFacets([]).runtime.ollama.status, 'compatible');
});

for (const c of NATIVE_CASES) {
  test(`installed Ollama report: ${c.name}`, () => {
    const facets = installedOllamaFacets(c.capabilities);
    const picked = Object.fromEntries(Object.keys(c.expect).map(k => [k, facets[k]]));
    assert.deepEqual(picked, c.expect);
    assert.equal(facets.task_source, 'ollama');
    assert.equal(facets.runtime.ollama.status, 'compatible');
  });
}

test('an installed library tag takes its task from the native report, not the card hint', () => {
  const card = {
    name: 'seer:7b', task: 'vision', task_label: 'Vision chat', task_source: 'ollama', chat_capable: true,
    runtime: { ollama: { status: 'compatible', reason: 'Published in the Ollama library (may need a recent Ollama).' } },
  };
  const installed = withInstalledOllamaFacets(card, ['decision', 'vision']);
  assert.equal(installed.name, 'seer:7b');
  assert.equal(installed.task, 'classification');
  assert.equal(installed.chat_capable, false);
  assert.equal(installed.runtime.ollama.reason, 'Installed on your Ollama server.');
  // No report from the server: the library hint is all there is.
  assert.deepEqual(withInstalledOllamaFacets(card, []), card);
  assert.deepEqual(withInstalledOllamaFacets(card, undefined), card);
});

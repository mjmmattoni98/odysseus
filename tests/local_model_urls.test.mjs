import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';

// admin.js / slashCommands.js are browser modules with DOM imports; load just
// the named pure functions from their source.
function extractFunctions(source, names) {
  return names.map(name => {
    const start = source.search(new RegExp(`function ${name}\\(`));
    assert.ok(start >= 0, `${name} not found`);
    let depth = 0;
    for (let i = source.indexOf('{', start); i < source.length; i++) {
      if (source[i] === '{') depth++;
      else if (source[i] === '}' && --depth === 0) return source.slice(start, i + 1);
    }
    throw new Error(`unterminated ${name}`);
  }).join('\n');
}

async function load(file, names) {
  const source = await readFile(new URL(`../static/js/${file}`, import.meta.url), 'utf8');
  return new Function(`${extractFunctions(source, names)}; return { ${names.join(', ')} };`)();
}

test('admin URL normalization keeps Ollama native and /v1 for other servers', async () => {
  const { _normalizeBaseUrl } = await load('admin.js', ['_normalizeBaseUrl', '_looksLikeOllamaBase']);
  assert.equal(_normalizeBaseUrl('localhost:11434'), 'http://localhost:11434');
  assert.equal(_normalizeBaseUrl('http://host.docker.internal:11434/'), 'http://host.docker.internal:11434');
  assert.equal(_normalizeBaseUrl('http://localhost:11434/api/chat'), 'http://localhost:11434/api');
  assert.equal(_normalizeBaseUrl('http://ollama:11434'), 'http://ollama:11434');
  // Typed /v1 stays supported.
  assert.equal(_normalizeBaseUrl('http://localhost:11434/v1/chat/completions'), 'http://localhost:11434/v1');
  assert.equal(_normalizeBaseUrl('http://localhost:1234'), 'http://localhost:1234/v1');
  assert.equal(_normalizeBaseUrl('localhost:8080'), 'http://localhost:8080/v1');
});

test('setup guide registers Ollama natively and opens /api/chat', async () => {
  const fns = await load('slashCommands.js', [
    '_looksLikeOllamaBase', '_normalizeSetupBaseUrl', 'detectProvider', 'setupChatUrlForEndpoint',
  ]);
  assert.equal(fns._normalizeSetupBaseUrl('localhost:11434'), 'http://localhost:11434');
  assert.equal(fns._normalizeSetupBaseUrl('http://gpu:8000'), 'http://gpu:8000/v1');
  const ollama = fns.detectProvider('http://localhost:11434');
  assert.equal(ollama.base_url, 'http://localhost:11434');
  assert.equal(fns.setupChatUrlForEndpoint(ollama), 'http://localhost:11434/api/chat');
  const lmstudio = fns.detectProvider('localhost:1234');
  assert.equal(lmstudio.base_url, 'http://localhost:1234/v1');
  assert.equal(fns.setupChatUrlForEndpoint(lmstudio), 'http://localhost:1234/v1/chat/completions');
  assert.equal(fns.setupChatUrlForEndpoint({ base_url: 'http://localhost:11434/v1' }), 'http://localhost:11434/v1/chat/completions');
});

test('settings warns about roles that swap models on the chat server', async () => {
  const { localModelSwapWarnings, parseContextDefault } = await import('../static/js/settings/localModels.js');
  const endpoints = [
    { id: 'ol', name: 'Ollama (host)', category: 'local', models: ['qwen3.8:27b', 'gemma4:12b'] },
    { id: 'oa', name: 'OpenAI', category: 'api', models: ['gpt-4o'] },
  ];
  const settings = {
    default_endpoint_id: 'ol', default_model: 'qwen3.8:27b',
    utility_endpoint_id: 'ol', utility_model: 'gemma4:12b',
    research_endpoint_id: 'oa', research_model: 'gpt-4o',
    task_endpoint_id: 'ol', task_model: 'qwen3.8:27b',
    vision_model: 'gemma4:12b',
    teacher_enabled: true, teacher_model: 'gemma4:12b@ollama',
  };
  assert.deepEqual(localModelSwapWarnings(settings, endpoints), [
    'Utility uses gemma4:12b', 'Vision uses gemma4:12b', 'Teacher uses gemma4:12b',
  ]);
  assert.deepEqual(localModelSwapWarnings({ ...settings, default_endpoint_id: 'oa' }, endpoints), []);
  assert.equal(parseContextDefault('65536'), 65536);
  assert.equal(parseContextDefault('512'), null);
  assert.equal(parseContextDefault('32768.5'), null);
});

test('conversation context note explains reloads for non-default caps', async () => {
  globalThis.document = { readyState: 'loading', addEventListener() {}, getElementById: () => null };
  const source = (await readFile(new URL('../static/js/assistantControls.js', import.meta.url), 'utf8'))
    .replace("import ui from './ui.js';", 'const ui = { showError() {} };');
  const controls = await import(`data:text/javascript,${encodeURIComponent(source)}#${Math.random()}`);
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ preferences: { profile: 'everyday', context_limits: {} }, runtime: { native: true, model: 'm', default_context_limit: 32768 } }),
  });
  await controls.loadSession('s');
  assert.equal(controls.contextReloadNote(32768), '');
  assert.match(controls.contextReloadNote(65536), /Ollama reloads the model/);
});

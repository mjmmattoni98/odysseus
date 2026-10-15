import assert from 'node:assert/strict';
import test from 'node:test';

// jobs.js is a browser module; give it the few globals it touches.
const storage = new Map();
globalThis.localStorage = {
  getItem: key => (storage.has(key) ? storage.get(key) : null),
  setItem: (key, value) => storage.set(key, String(value)),
};
globalThis.window = globalThis;
const streams = [];
globalThis.EventSource = class {
  constructor(url) { this.url = url; streams.push(this); }
  close() { this.closed = true; }
};
// Job timers would keep the test process alive; they are not under test.
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};

let routes = {};
globalThis.fetch = async (url, options = {}) => {
  const key = `${options.method || 'GET'} ${String(url).split('?')[0]}`;
  const handler = routes[key];
  if (!handler) return { ok: false, status: 404, text: async () => '{"detail":"missing"}', json: async () => ({}) };
  const { status = 200, body } = handler();
  return { ok: status < 400, status, text: async () => JSON.stringify(body), json: async () => body };
};

const jobs = await import('../static/js/research/jobs.js');

const library = (items) => ({ status: 200, body: { research: items, total: items.length } });
const v1 = { id: 'rp-v1', query: 'Mars program', status: 'done', lineage_id: 'rp-v1', version: 1, source_count: 3 };

test('refresh starts a running version and the library then shows only the newest', async () => {
  routes = { 'GET /api/research/library': () => library([v1]) };
  await jobs.refreshLibrary({ force: true });
  assert.deepEqual(jobs.getJobs().map(j => j.id), ['rp-v1']);

  routes['POST /api/research/rp-v1/refresh'] = () => ({
    status: 200,
    body: {
      session_id: 'rp-v2', query: 'Mars program', version: 2, lineage_id: 'rp-v1',
      settings: { report_length: 'brief' }, source_rules: { allowed_domains: ['nasa'] },
      report_length: { name: 'brief' },
    },
  });
  const job = await jobs.refreshJob('rp-v1');
  assert.equal(job.status, 'running');
  assert.equal(job.id, 'rp-v2');
  assert.equal(job.version, 2);
  assert.equal(job.refreshOf, 'rp-v1');
  assert.deepEqual(job.settings.allowed_domains, ['nasa']);
  assert.equal(streams.at(-1).url, '/api/research/stream/rp-v2');

  // Once v2 is saved the library lists it in place of v1.
  job.status = 'done';
  routes['GET /api/research/library'] = () => library([{
    ...v1, id: 'rp-v2', version: 2, version_count: 2, previous_version_id: 'rp-v1',
    citation_warning_count: 1, change_counts: { changed: 1 },
  }]);
  await jobs.refreshLibrary({ force: true });
  const ids = jobs.getJobs().map(j => j.id);
  assert.deepEqual(ids, ['rp-v2']);
  const v2 = jobs.getJobs()[0];
  assert.equal(v2.versionCount, 2);
  assert.equal(v2.citationWarningCount, 1);
});

test('a rejected refresh reports the server reason and adds no job', async () => {
  routes['POST /api/research/rp-v2/refresh'] = () => ({ status: 400, body: { detail: 'Only completed research can be refreshed.' } });
  const before = jobs.getJobs().length;
  const job = await jobs.refreshJob('rp-v2');
  assert.equal(job.status, 'error');
  assert.equal(job.errorMsg, 'Only completed research can be refreshed.');
  assert.equal(jobs.getJobs().length, before);
});

test('comparison phase has a readable label', () => {
  assert.equal(jobs.formatPhase({ phase: 'comparing' }), 'Comparing with the previous version...');
});

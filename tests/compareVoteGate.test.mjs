// Compare voting in the DOM: static/js/compare/vote.js enables Vote and Tie
// only for panes whose latest answer finished with content
// (data-response-state="ok"), opens them once a second pane finishes, and
// records only those panes as the round's contestants.
//
// Runs the production vote.js in Chromium over a fake origin; ui.js,
// chatRenderer.js and theme.js are stubbed (vote.js only needs esc and a cost
// lookup from them). Stream-state stamping itself is covered by
// tests/compareStreamOutcome.test.mjs.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const ORIGIN = 'http://compare.test';
const STUBS = {
  '/static/js/ui.js': 'export default window.uiModule;',
  '/static/js/chatRenderer.js': 'export function getModelCost() { return null; }',
  '/static/js/theme.js': 'export default {};',
};
const MODELS = [
  { model: 'lab/alpha', name: 'Alpha' },
  { model: 'lab/beta', name: 'Beta' },
  { model: 'lab/gamma', name: 'Gamma' },
];

function pageHtml(n) {
  const panes = Array.from({ length: n }, (_, i) => `
    <div class="compare-pane" data-pane="${i}">
      <button class="pane-title" id="cmp-title-${i}" type="button">Model</button>
      <div class="chat-history" id="cmp-history-${i}"></div>
      <div class="pane-vote-footer"><button class="pane-vote-btn" data-pane="${i}" type="button" disabled>
        <span class="pane-vote-label">Vote</span></button></div>
    </div>`).join('');
  return `<!doctype html><html><body><div class="compare-grid">${panes}</div><div id="compare-vote-bar"></div></body></html>`;
}

let browser;
before(async () => { browser = await chromium.launch({ headless: true }); });
after(async () => { await browser?.close(); });

/** A compare round with `n` panes whose latest answers are in `states`. */
async function setup(states) {
  const page = await browser.newPage();
  const records = [];
  await page.route(`${ORIGIN}/**`, async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    if (path === '/') return route.fulfill({ contentType: 'text/html', body: pageHtml(states.length) });
    if (STUBS[path]) return route.fulfill({ contentType: 'text/javascript', body: STUBS[path] });
    if (path.startsWith('/static/js/')) {
      const source = await readFile(new URL('..' + path, import.meta.url), 'utf8');
      return route.fulfill({ contentType: 'text/javascript', body: source });
    }
    if (path === '/api/compare/record') {
      records.push(JSON.parse(req.postData()));
      return route.fulfill({ contentType: 'application/json', body: '{"ok":true}' });
    }
    return route.abort();
  });
  await page.goto(`${ORIGIN}/`);
  await page.evaluate(async ({ models, states }) => {
    const esc = (value) => { const node = document.createElement('div'); node.textContent = String(value ?? ''); return node.innerHTML; };
    window.uiModule = { esc };
    const vote = await import('/static/js/compare/vote.js');
    const state = (await import('/static/js/compare/state.js')).default;
    Object.assign(state, {
      _selectedModels: models.slice(0, states.length), _lastPrompt: 'Which is bigger, 9.11 or 9.9?',
      _voted: false, _blindMode: false, _streaming: true, _paneMetrics: [],
    });
    window.stops = 0;
    vote.registerCompareActions({ stopAll: () => { window.stops++; }, resetCompare: () => {} });
    // Same wiring as compare/index.js: a disabled pane button does nothing.
    document.querySelectorAll('.pane-vote-btn').forEach((btn) => btn.addEventListener('click', () => {
      if (!btn.disabled) vote.handleVote(Number(btn.dataset.pane));
    }));
    /** Stamp each pane's latest answer, as stream.js does when a stream settles. */
    window.setStates = (next) => {
      next.forEach((s, i) => {
        const hist = document.getElementById('cmp-history-' + i);
        hist.innerHTML = '<div class="msg msg-ai"><div class="role">AI</div><div class="body">answer</div></div>';
        if (s) hist.firstElementChild.dataset.responseState = s;
      });
      vote.syncVoteControls();
    };
    window.vote = vote;
    window.state = state;
    window.setStates(states);
    vote.buildVoteBar(states.length);
  }, { models: MODELS, states });
  return { page, records };
}

const controls = page => page.evaluate(() => ({
  panes: [...document.querySelectorAll('.pane-vote-btn')].map(b => !b.disabled),
  titles: [...document.querySelectorAll('.pane-vote-btn')].map(b => b.title),
  tie: !document.querySelector('.compare-vote-tie').disabled,
}));
const storedVotes = page => page.evaluate(() => JSON.parse(localStorage.getItem('odysseus-compare-votes') || '[]'));

test('Tie and Vote stay closed while both models are still answering', async () => {
  const { page } = await setup(['pending', 'pending']);
  try {
    assert.deepEqual(await controls(page), { panes: [false, false], titles: ['', ''], tie: false });
    // A stale or scripted Tie must not record a result or abort the streams.
    await page.evaluate(() => { window.vote.handleVote(-1); window.vote.handleVote(0); });
    assert.deepEqual(await storedVotes(page), []);
    assert.equal(await page.evaluate(() => window.state._voted), false);
    assert.equal(await page.evaluate(() => window.stops), 0, 'no request is aborted');
  } finally {
    await page.close();
  }
});

test('one finished pane is not enough; the second one finishing opens Vote and Tie', async () => {
  const { page, records } = await setup(['pending', 'pending']);
  try {
    await page.evaluate(() => window.setStates(['ok', 'pending']));
    assert.deepEqual(await controls(page), { panes: [false, false], titles: ['', ''], tie: false });
    await page.evaluate(() => window.setStates(['ok', 'ok']));
    assert.deepEqual(await controls(page), { panes: [true, true], titles: ['', ''], tie: true });

    const recorded = page.waitForRequest(`${ORIGIN}/api/compare/record`);
    await page.click('.compare-vote-tie');
    await (await recorded).response();
    assert.deepEqual(records, [{ prompt: 'Which is bigger, 9.11 or 9.9?', models: ['Alpha', 'Beta'], winner: 'tie', is_blind: false }]);
    assert.equal((await storedVotes(page))[0].winner, 'tie');
  } finally {
    await page.close();
  }
});

test('a stopped (partial) pane is no contestant: it cannot win, tie or lose', async () => {
  const { page, records } = await setup(['ok', 'ok', 'partial']);
  try {
    const before = await controls(page);
    assert.deepEqual(before.panes, [true, true, false]);
    assert.match(before.titles[2], /stopped before it finished/);
    const recorded = page.waitForRequest(`${ORIGIN}/api/compare/record`);
    await page.click('.compare-pane[data-pane="0"] .pane-vote-btn');
    await (await recorded).response();
    assert.deepEqual(records[0].models, ['Alpha', 'Beta']);
    assert.equal(records[0].winner, 'Alpha');
    const classes = await page.evaluate(() => [...document.querySelectorAll('.compare-pane')].map(p => p.className));
    assert.deepEqual(classes, ['compare-pane winner', 'compare-pane loser', 'compare-pane']);
  } finally {
    await page.close();
  }
});

test('voting early with a pane still streaming records only the finished panes', async () => {
  const { page, records } = await setup(['ok', 'pending', 'ok']);
  try {
    assert.deepEqual((await controls(page)).panes, [true, false, true]);
    const recorded = page.waitForRequest(`${ORIGIN}/api/compare/record`);
    await page.click('.compare-vote-tie');
    await (await recorded).response();
    assert.deepEqual(records[0].models, ['Alpha', 'Gamma']);
    assert.equal(await page.evaluate(() => window.stops), 1, 'the unfinished pane is stopped');
    const titles = await page.evaluate(() => [0, 1, 2].map(i => document.getElementById('cmp-title-' + i).textContent));
    assert.match(titles[0], /^=/);
    assert.doesNotMatch(titles[1], /^=/, 'the stopped pane takes no share of the tie');
    assert.match(titles[2], /^=/);
  } finally {
    await page.close();
  }
});

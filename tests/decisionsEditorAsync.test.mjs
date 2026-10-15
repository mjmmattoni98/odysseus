// Decisions editor: a slow request finishes on the decision it was started
// for, even when the user has opened another decision in the meantime.
//
// Drives the production static/js/decisions.js (and its real window helpers)
// in Chromium over a fake origin. /api/decisions and /api/calendar/events are
// an in-memory fake whose responses a test can hold, so a request can be
// overtaken by the user switching decisions, as on a slow network.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const ORIGIN = 'http://decisions.test';
const PAGE = '<!doctype html><html><body><button id="tool-decisions-btn" type="button">Decisions</button></body></html>';

let browser;
before(async () => { browser = await chromium.launch({ headless: true }); });
after(async () => { await browser?.close(); });

function decision(id, title, extra = {}) {
  return {
    id, title, question: `Why ${title}?`, status: 'open',
    constraints: [], criteria: [{ id: 'k', name: 'K', weight: 1 }],
    options: [{ id: `${id}1`, name: `${title} one`, scores: {} }, { id: `${id}2`, name: `${title} two`, scores: {} }],
    sources: [], revisit_on: '2027-01-15', revisit_event_uid: null,
    chosen_option_id: null, rationale: '', decided_at: null, history: [],
    ...extra,
  };
}

function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return { promise, resolve };
}

/** In-memory stand-in for the decision and calendar routes. */
function fakeServer(seed, events = {}) {
  const decisions = new Map(seed.map(d => [d.id, structuredClone(d)]));
  const calendar = new Map(Object.entries(events));
  const holds = [];
  const log = [];
  let inflight = 0;
  let nextUid = 1;

  const full = d => ({ ...d, effective_status: d.status, revisit_due: false, since_decision: null });
  const summary = d => ({ id: d.id, title: d.title, status: d.status, effective_status: d.status, revisit_due: false,
    revisit_on: d.revisit_on, chosen_option_name: null, leader_name: null, option_count: d.options.length });

  function handle({ method, path, body }) {
    const calendarMatch = path.match(/^\/api\/calendar\/events(?:\/([^/]+))?$/);
    if (calendarMatch) {
      const uid = calendarMatch[1] && decodeURIComponent(calendarMatch[1]);
      if (method === 'POST') {
        const created = `evt-${nextUid++}`;
        calendar.set(created, body);
        return [200, { uid: created }];
      }
      if (!calendar.has(uid)) return [404, { detail: 'not found' }];
      if (method === 'PUT') calendar.set(uid, body);
      if (method === 'DELETE') calendar.delete(uid);
      return [200, { ok: true }];
    }
    if (path === '/api/decisions' && method === 'GET') {
      return [200, { decisions: [...decisions.values()].map(summary), revisit_due_count: 0 }];
    }
    const m = path.match(/^\/api\/decisions\/([^/]+)(\/decide)?$/);
    const d = m && decisions.get(decodeURIComponent(m[1]));
    if (!d) return [404, { detail: 'Decision not found' }];
    if (m[2] && method === 'POST') {
      const chosen = d.options.find(o => o.id === body.option_id);
      if (!chosen) return [400, { detail: 'chosen option does not exist' }];
      Object.assign(d, { chosen_option_id: chosen.id, rationale: body.rationale, status: 'decided',
        decided_at: '2026-10-14T09:00:00Z' });
      // Like routes/decision: an omitted revisit_on keeps the stored date.
      if ('revisit_on' in body) d.revisit_on = body.revisit_on || null;
      d.history.push({ seq: d.history.length + 1, chosen_option_id: chosen.id, chosen_option_name: chosen.name,
        decided_at: d.decided_at, criteria: d.criteria, options: d.options, ranking: [], leader_id: null });
      return [200, full(d)];
    }
    if (method === 'PUT') {
      for (const key of ['title', 'question', 'constraints', 'criteria', 'options', 'sources', 'status', 'revisit_on']) {
        if (key in body) d[key] = body[key];
      }
      if ('revisit_event_uid' in body) d.revisit_event_uid = body.revisit_event_uid || null;
      return [200, full(d)];
    }
    return [200, full(d)];
  }

  return {
    decisions, calendar, log,
    /** Hold the next request `match` accepts until `release()` is called. */
    hold(match) {
      const h = { match, arrived: deferred(), released: deferred() };
      holds.push(h);
      return { arrived: h.arrived.promise, release: () => h.released.resolve() };
    },
    async respond(entry) {
      inflight++;
      try {
        log.push(entry);
        const i = holds.findIndex(h => h.match(entry));
        if (i >= 0) {
          const [h] = holds.splice(i, 1);
          h.arrived.resolve();
          await h.released.promise;
        }
        return handle(entry);
      } finally {
        inflight--;
      }
    },
    /** Resolve once no request has been in flight for a few event-loop turns. */
    async settled(page) {
      for (let quiet = 0; quiet < 3;) {
        await page.evaluate(() => new Promise(r => setTimeout(r, 40)));
        quiet = inflight === 0 ? quiet + 1 : 0;
      }
    },
  };
}

async function openDecisions(server, firstId) {
  const page = await browser.newPage();
  await page.route(`${ORIGIN}/**`, async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    if (path === '/') return route.fulfill({ contentType: 'text/html', body: PAGE });
    if (path === '/static/js/ui.js') {
      return route.fulfill({ contentType: 'text/javascript', body: 'export default window.uiModule;' });
    }
    if (path.startsWith('/static/js/')) {
      const source = await readFile(new URL('..' + path, import.meta.url), 'utf8');
      return route.fulfill({ contentType: 'text/javascript', body: source });
    }
    if (path.startsWith('/api/')) {
      const raw = req.postData();
      const [status, json] = await server.respond({ method: req.method(), path, body: raw ? JSON.parse(raw) : null });
      return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(json) });
    }
    return route.abort();
  });
  await page.goto(`${ORIGIN}/`);
  await page.evaluate(async (firstId) => {
    window.__errors = [];
    const esc = (value) => { const node = document.createElement('div'); node.textContent = String(value ?? ''); return node.innerHTML; };
    window.uiModule = {
      esc, isTouchInsideModal: () => false, showToast: () => {}, styledConfirm: async () => true,
      showError: (message) => window.__errors.push(String(message)),
    };
    await import('/static/js/decisions.js');
    await window.decisionsModule.openDecisions(firstId);
  }, firstId);
  return page;
}

async function switchTo(page, id, title) {
  await page.click(`[data-decision-id="${id}"]`);
  await page.waitForFunction(t => document.getElementById('decision-title')?.value === t, title);
}

const isCalendarPost = e => e.method === 'POST' && e.path === '/api/calendar/events';

test('a calendar event created for A is stored on A after the user opens B', async () => {
  const server = fakeServer([decision('A', 'Alpha'), decision('B', 'Beta')]);
  const page = await openDecisions(server, 'A');
  try {
    const post = server.hold(isCalendarPost);
    await page.click('#decision-calendar');
    await post.arrived;
    await switchTo(page, 'B', 'Beta');
    post.release();
    await server.settled(page);

    assert.equal(server.decisions.get('A').revisit_event_uid, 'evt-1', "A's event belongs to A");
    assert.equal(server.decisions.get('B').revisit_event_uid, null, 'B never asked for an event');
    assert.equal(server.calendar.get('evt-1').summary, 'Revisit decision: Alpha');
    assert.equal(await page.isChecked('#decision-calendar'), false, "B's box stays clear");
    assert.deepEqual(await page.evaluate(() => window.__errors), []);
  } finally {
    await page.close();
  }
});

test('recording A and opening B before the answer keeps A\'s event and B\'s view', async () => {
  const server = fakeServer(
    [decision('A', 'Alpha', { revisit_event_uid: 'evt-A' }), decision('B', 'Beta')],
    { 'evt-A': { summary: 'Revisit decision: Alpha' } },
  );
  const page = await openDecisions(server, 'A');
  try {
    assert.equal(await page.isChecked('#decision-calendar'), true);
    const decide = server.hold(e => e.method === 'POST' && e.path === '/api/decisions/A/decide');
    await page.click('[data-act="decide"]');
    await decide.arrived;
    await switchTo(page, 'B', 'Beta');
    decide.release();
    await server.settled(page);

    assert.equal(server.decisions.get('A').status, 'decided');
    assert.ok(server.calendar.has('evt-A'), "A's calendar event must survive");
    assert.equal(server.decisions.get('A').revisit_event_uid, 'evt-A');
    assert.equal(server.decisions.get('B').revisit_event_uid, null);
    assert.ok(!server.log.some(e => e.path === '/api/decisions/B' && e.method === 'PUT'), 'nothing is written to B');
    // B is still shown as B: undecided, not with A's recorded choice.
    assert.equal((await page.textContent('[data-act="decide"]')).trim(), 'Record decision');
    assert.equal(await page.isChecked('#decision-calendar'), false);
  } finally {
    await page.close();
  }
});

test('archiving A and opening B before the answer leaves B shown as open', async () => {
  const server = fakeServer([decision('A', 'Alpha'), decision('B', 'Beta')]);
  const page = await openDecisions(server, 'A');
  try {
    const archive = server.hold(e => e.method === 'PUT' && e.path === '/api/decisions/A' && e.body?.status);
    await page.click('[data-act="archive"]');
    await archive.arrived;
    await switchTo(page, 'B', 'Beta');
    archive.release();
    await server.settled(page);

    assert.equal(server.decisions.get('A').status, 'archived');
    assert.equal(server.decisions.get('B').status, 'open');
    assert.match(await page.textContent('[data-act="archive"]'), /^\s*Archive\s*$/);
    assert.equal(await page.textContent('#decisions-editor-status'), 'Open');
  } finally {
    await page.close();
  }
});

test('ticking and clearing the calendar box quickly leaves no event behind', async () => {
  const server = fakeServer([decision('A', 'Alpha'), decision('B', 'Beta')]);
  const page = await openDecisions(server, 'A');
  try {
    const post = server.hold(isCalendarPost);
    await page.click('#decision-calendar');
    await post.arrived;
    await page.click('#decision-calendar');
    post.release();
    await server.settled(page);

    assert.deepEqual([...server.calendar.keys()], [], 'the event created first is deleted again');
    assert.equal(server.decisions.get('A').revisit_event_uid, null);
  } finally {
    await page.close();
  }
});

test('clearing the calendar box while "Record" is pending leaves no event', async () => {
  const server = fakeServer(
    [decision('A', 'Alpha', { revisit_event_uid: 'evt-A' }), decision('B', 'Beta')],
    { 'evt-A': { summary: 'Revisit decision: Alpha' } },
  );
  const page = await openDecisions(server, 'A');
  try {
    const decide = server.hold(e => e.method === 'POST' && e.path === '/api/decisions/A/decide');
    await page.click('[data-act="decide"]');
    await decide.arrived;
    await page.click('#decision-calendar');   // the later intent: no calendar event
    decide.release();
    await server.settled(page);

    assert.equal(server.decisions.get('A').status, 'decided');
    assert.equal(await page.isChecked('#decision-calendar'), false);
    assert.deepEqual([...server.calendar.keys()], [], 'the cleared box wins over the earlier Record');
    assert.equal(server.decisions.get('A').revisit_event_uid, null);
  } finally {
    await page.close();
  }
});

test('a revisit date changed while "Record" is pending survives it, on the decision and the calendar', async () => {
  const server = fakeServer(
    [decision('A', 'Alpha', { revisit_on: '2027-01-15', revisit_event_uid: 'evt-A' }), decision('B', 'Beta')],
    { 'evt-A': { summary: 'Revisit decision: Alpha', dtstart: '2027-01-15' } },
  );
  const page = await openDecisions(server, 'A');
  try {
    const decide = server.hold(e => e.method === 'POST' && e.path === '/api/decisions/A/decide');
    await page.click('[data-act="decide"]');
    await decide.arrived;
    await page.fill('#decision-revisit', '2027-03-01');
    await page.dispatchEvent('#decision-revisit', 'change');   // saved at once, ahead of Record
    await page.waitForFunction(() => document.getElementById('decisions-save-state')?.textContent === 'Saved');
    assert.equal(server.decisions.get('A').revisit_on, '2027-03-01');
    decide.release();
    await server.settled(page);

    assert.equal(server.decisions.get('A').status, 'decided');
    assert.equal(server.decisions.get('A').revisit_on, '2027-03-01', 'the later date is kept');
    assert.equal(server.calendar.get('evt-A').dtstart, '2027-03-01', 'and the event matches it');
    assert.equal(await page.inputValue('#decision-revisit'), '2027-03-01');
  } finally {
    await page.close();
  }
});

test('a Record the server rejects leaves the calendar alone', async () => {
  // A successful Record would move A's existing event; a rejected one must not.
  const server = fakeServer(
    [decision('A', 'Alpha', { revisit_event_uid: 'evt-A' }), decision('B', 'Beta')],
    { 'evt-A': { summary: 'Revisit decision: Alpha', dtstart: '2027-01-15' } },
  );
  const page = await openDecisions(server, 'A');
  try {
    await page.evaluate(() => {
      const choice = document.getElementById('decision-choice');
      choice.add(new Option('Removed meanwhile', 'gone'));
      choice.value = 'gone';
    });
    const before = server.log.length;
    await page.click('[data-act="decide"]');
    await server.settled(page);
    assert.match((await page.evaluate(() => window.__errors)).join('\n'), /Could not record decision/);
    assert.equal(server.decisions.get('A').status, 'open');
    assert.ok(!server.log.slice(before).some(e => e.path.startsWith('/api/calendar/')), 'no calendar call');
  } finally {
    await page.close();
  }
});

test('edits typed into A while B is loading are saved to A', async () => {
  const server = fakeServer([decision('A', 'Alpha'), decision('B', 'Beta')]);
  const page = await openDecisions(server, 'A');
  try {
    const load = server.hold(e => e.method === 'GET' && e.path === '/api/decisions/B');
    await page.click('[data-decision-id="B"]');
    await load.arrived;
    await page.fill('#decision-title', 'Alpha, renamed');   // autosave timer is still pending
    load.release();
    await page.waitForFunction(() => document.getElementById('decision-title')?.value === 'Beta');
    await server.settled(page);
    assert.equal(server.decisions.get('A').title, 'Alpha, renamed');
    assert.equal(server.decisions.get('B').title, 'Beta');
  } finally {
    await page.close();
  }
});

test('going back to A while B is still loading keeps A open', async () => {
  const server = fakeServer([decision('A', 'Alpha'), decision('B', 'Beta')]);
  const page = await openDecisions(server, 'A');
  try {
    const load = server.hold(e => e.method === 'GET' && e.path === '/api/decisions/B');
    await page.click('[data-decision-id="B"]');
    await load.arrived;
    await page.click('[data-decision-id="A"]');
    load.release();
    await server.settled(page);
    assert.equal(await page.inputValue('#decision-title'), 'Alpha');
  } finally {
    await page.close();
  }
});

test('without a switch, recording creates the event and a new date moves it', async () => {
  const server = fakeServer([decision('A', 'Alpha'), decision('B', 'Beta')]);
  const page = await openDecisions(server, 'A');
  try {
    await page.click('#decision-calendar');
    await server.settled(page);
    await page.click('[data-act="decide"]');
    await server.settled(page);
    const uid = server.decisions.get('A').revisit_event_uid;
    assert.equal(uid, 'evt-1');
    assert.equal(server.calendar.size, 1, 'one event, not one per sync');
    assert.equal((await page.textContent('[data-act="decide"]')).trim(), 'Record again');

    await page.fill('#decision-revisit', '2027-03-01');
    await page.dispatchEvent('#decision-revisit', 'change');
    await server.settled(page);
    assert.equal(server.calendar.get(uid).dtstart, '2027-03-01');
    assert.equal(server.calendar.get(uid).dtend, '2027-03-02');
    assert.equal(server.calendar.size, 1);
    assert.equal(server.decisions.get('B').revisit_event_uid, null);
  } finally {
    await page.close();
  }
});

test('a score typed under a "__proto__" criterion counts and is saved', async () => {
  const server = fakeServer([decision('P', 'Proto', {
    criteria: [{ id: '__proto__', name: 'Proto', weight: 3 }, { id: 'constructor', name: 'Ctor', weight: 1 }],
    options: [{ id: 'o1', name: 'One', scores: {} }, { id: 'o2', name: 'Two', scores: {} }],
  })]);
  const page = await openDecisions(server, 'P');
  try {
    // Unscored "constructor" cells render blank, not as Object's source.
    assert.equal(await page.inputValue('input[data-key="s:o1:constructor"]'), '');
    await page.fill('input[data-key="s:o1:__proto__"]', '8');
    assert.equal(await page.textContent('[data-total="o1"]'), '6.00');
    assert.match(await page.textContent('#decisions-leader'), /Leading: One \(6\.00 of 10\)/);
    await page.waitForFunction(() => document.getElementById('decisions-save-state')?.textContent === 'Saved');
    await server.settled(page);
    const saved = server.log.filter(e => e.method === 'PUT' && e.body?.options).at(-1);
    assert.ok(saved, 'the edit was autosaved');
    const scores = saved.body.options.find(o => o.id === 'o1').scores;
    assert.ok(Object.prototype.hasOwnProperty.call(scores, '__proto__'), JSON.stringify(scores));
    assert.equal(scores['__proto__'], 8);
  } finally {
    await page.close();
  }
});

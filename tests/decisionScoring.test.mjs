import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  scoreOptions, compareToSnapshot, isRevisitDue, localToday, formatShare, newRowId, round4,
} from '../static/js/decisionScoring.js';

// Same cases tests/test_decision_notebook.py asserts against src/decisions.py.
const CASES = JSON.parse(readFileSync(new URL('./fixtures/decision_scoring_cases.json', import.meta.url), 'utf8'));

for (const c of CASES.scoring) {
  test(`scoring: ${c.name}`, () => {
    const result = scoreOptions(c.body);
    const totals = Object.fromEntries(Object.entries(result.options).map(([id, r]) => [id, r.total]));
    // Own entries only: the id-keyed maps carry no prototype (ids like "__proto__").
    assert.deepEqual({ ...result.weights }, c.expect.weights);
    assert.deepEqual(totals, c.expect.totals);
    assert.deepEqual(result.ranking, c.expect.ranking);
    assert.equal(result.leader_id, c.expect.leader_id);
    assert.equal(result.tied, c.expect.tied);
    for (const [id, violations] of Object.entries(c.expect.violations || {})) {
      assert.deepEqual(result.options[id].violations, violations);
      assert.equal(result.options[id].eligible, violations.length === 0);
    }
    for (const [id, unscored] of Object.entries(c.expect.unscored || {})) {
      assert.equal(result.options[id].unscored, unscored);
    }
  });
}

for (const c of CASES.comparison) {
  test(`comparison: ${c.name}`, () => {
    const result = compareToSnapshot(c.body, c.snapshot);
    for (const [key, value] of Object.entries(c.expect)) {
      assert.deepEqual(result[key], value, key);
    }
  });
}

test('an unchanged body compares as unchanged and the choice still wins', () => {
  const c = CASES.comparison[0];
  const result = compareToSnapshot(c.snapshot, c.snapshot);
  assert.equal(result.changed, false);
  assert.equal(result.chosen_still_wins, true);
});

test('a removed chosen option is reported, not counted as losing', () => {
  const c = CASES.comparison[0];
  const body = { ...c.body, options: c.body.options.filter(o => o.id !== 'a') };
  const result = compareToSnapshot(body, c.snapshot);
  assert.equal(result.chosen_removed, true);
  assert.equal(result.chosen_still_wins, null);
  assert.deepEqual(result.options_removed, ['A']);
});

test('blank score inputs count as unscored, not as zero-valued scores', () => {
  const body = {
    criteria: [{ id: 'k', name: 'K', weight: 1 }],
    options: [{ id: 'a', scores: { k: '' } }, { id: 'b', scores: { k: 0 } }],
  };
  const result = scoreOptions(body);
  assert.equal(result.options.a.unscored, 1);
  assert.equal(result.options.b.unscored, 0);
});

test('looking up an absent prototype-named id misses instead of finding Object.prototype', () => {
  const result = scoreOptions({
    criteria: [{ id: 'k', name: 'K', weight: 1 }],
    options: [{ id: 'a', scores: { k: 4 } }],
  });
  for (const id of ['constructor', '__proto__', 'toString', 'hasOwnProperty']) {
    assert.equal(result.weights[id], undefined, `weights.${id}`);
    assert.equal(result.options[id], undefined, `options.${id}`);
  }
  // An option without a "constructor" score is unscored there, not scored as Object.
  const ctor = scoreOptions({
    criteria: [{ id: 'constructor', name: 'C', weight: 1 }],
    options: [{ id: 'a', scores: {} }],
  });
  assert.equal(ctor.options.a.unscored, 1);
  assert.equal(ctor.options.a.total, 0);
});

test('revisit is due today or earlier, never when archived', () => {
  assert.equal(isRevisitDue('2026-10-12', 'decided', '2026-10-12'), true);
  assert.equal(isRevisitDue('2026-10-01', 'open', '2026-10-12'), true);
  assert.equal(isRevisitDue('2026-10-13', 'decided', '2026-10-12'), false);
  assert.equal(isRevisitDue('2026-10-01', 'archived', '2026-10-12'), false);
  assert.equal(isRevisitDue(null, 'decided', '2026-10-12'), false);
});

test('localToday uses the local calendar day, zero padded', () => {
  assert.equal(localToday(new Date(2026, 0, 5, 23, 59)), '2026-01-05');
});

test('shares format as percentages and rounding is half-up', () => {
  assert.equal(formatShare(0.375), '37.5%');
  assert.equal(formatShare(1), '100%');
  assert.equal(round4(0.00005), 0.0001);
});

test('new row ids match the server id pattern', () => {
  for (let i = 0; i < 20; i++) assert.match(newRowId(), /^[A-Za-z0-9_-]{1,40}$/);
});

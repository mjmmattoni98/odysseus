import test from 'node:test';
import assert from 'node:assert/strict';
import { sourceStatusLabel, searchStatusText } from '../static/js/searchEvidence.js';

test('page retrieval status never labels a snippet as a read page', () => {
  assert.equal(sourceStatusLabel({ read_status: 'snippet' }), 'Search snippet only');
  assert.match(sourceStatusLabel({ read_status: 'failed' }), /unavailable/);
  assert.equal(sourceStatusLabel({ read_status: 'read', partial: true }), 'Partial page text');
  assert.equal(sourceStatusLabel({}), '');
});

test('failures and fallbacks have visible explanations', () => {
  assert.match(searchStatusText({ state: 'failed' }), /could not be verified/);
  assert.match(searchStatusText({ state: 'empty' }), /no usable results/);
  assert.match(searchStatusText({ state: 'ok', results: 5, pages_read: 2, fallback: true, provider: 'duckduckgo' }), /5 sources; retrieved text from 2 pages.*fallback provider duckduckgo/);
});

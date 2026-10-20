import test from 'node:test';
import assert from 'node:assert/strict';
import { sourceStatusLabel, searchStatusText } from '../static/js/searchEvidence.js';

test('page retrieval status never labels a snippet as a read page', () => {
  assert.equal(sourceStatusLabel({ read_status: 'snippet' }), 'Search snippet only');
  assert.match(sourceStatusLabel({ read_status: 'failed' }), /unavailable/);
  assert.equal(sourceStatusLabel({ read_status: 'read', partial: true }), 'Page excerpt retrieved');
  assert.equal(sourceStatusLabel({}), '');
});

test('failures and fallbacks have visible explanations', () => {
  assert.match(searchStatusText({ state: 'failed' }), /could not be verified/);
  assert.match(searchStatusText({ state: 'empty' }), /no usable results/);
  assert.match(searchStatusText({ state: 'ok', results: 2, pages_read: 0, pages_failed: 2 }), /Only search snippets/);
  assert.match(searchStatusText({ state: 'ok', results: 5, pages_read: 2, fallback: true, provider: 'duckduckgo' }), /5 sources; retrieved text from 2 pages.*fallback provider duckduckgo/);
});

test('intelligent search distinguishes search, no search, and classifier fallback', () => {
  assert.equal(searchStatusText({ state: 'decision', mode: 'auto', needs_search: true }), 'Web: Auto — search needed.');
  assert.equal(searchStatusText({ state: 'decision', mode: 'off', needs_search: false }), 'Answered without search · Web: Off');
  assert.equal(searchStatusText({ state: 'decision', mode: 'auto', needs_search: false }), 'Answered without search');
  assert.match(searchStatusText({ state: 'decision', needs_search: null }), /Auto could not decide/);
});

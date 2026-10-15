// The login page imports theme.js for its background effects. theme.js's
// auto-init then pulled the user's theme and custom themes from
// /api/prefs/*, which need a session, so every logged-out visit made
// requests that could only answer 401 (QA audit 2026-10-12, item 7). The
// login page now marks itself as logged out and the init skips server sync.
// These tests run the shipped init against stubbed fetch/storage and pin that
// the logged-in app shell still syncs.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import test from 'node:test';

const themeSource = readFileSync(new URL('../static/js/theme.js', import.meta.url), 'utf8');

function slice(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.ok(start >= 0 && end > start, `could not locate ${startMarker}`);
  return source.slice(start, end);
}

const INIT_SOURCE = [
  slice(themeSource, 'async function _loadFromServer() {', '\nfunction syncPickers('),
  slice(themeSource, 'async function _initWithSync() {', "\nif (document.readyState === 'loading')"),
].join('\n');

function runInit({ loggedOutPage, savedTheme = null }) {
  const requests = [];
  let uiInits = 0;
  const ctx = {
    window: loggedOutPage ? { __odysseusLoggedOutPage: true } : {},
    fetch: async (url) => {
      requests.push(String(url));
      return { json: async () => ({ value: null }) };
    },
    getSaved: () => savedTheme,
    Storage: { setJSON() {} },
    applyColors() {},
    THEMES: {},
    LS_KEY: 'odysseus-theme',
    _loadCustomThemes: () => ({}),
    _saveCustomThemes() {},
    initThemeUI: () => { uiInits++; },
    console,
  };
  runInNewContext(INIT_SOURCE, ctx);
  return ctx._initWithSync().then(() => ({ requests, uiInits }));
}

test('logged-out login page makes no theme-preference requests', async () => {
  const { requests, uiInits } = await runInit({ loggedOutPage: true });
  assert.deepEqual(requests, []);
  // The local theme UI still initializes, as it did before.
  assert.equal(uiInits, 1);
});

test('the signed-in app still syncs theme and custom themes from the server', async () => {
  const { requests, uiInits } = await runInit({ loggedOutPage: false });
  assert.deepEqual(requests, ['/api/prefs/theme', '/api/prefs/custom-themes']);
  assert.equal(uiInits, 1);
});

test('the signed-in app with a local theme still merges server custom themes', async () => {
  const { requests } = await runInit({ loggedOutPage: false, savedTheme: { name: 'ume', colors: {} } });
  assert.deepEqual(requests, ['/api/prefs/custom-themes']);
});

test('login page marks itself logged out before it imports theme.js', () => {
  // Source-order check: proving this at runtime needs a real browser loading
  // login.html, which this Node suite does not have.
  const login = readFileSync(new URL('../static/login.html', import.meta.url), 'utf8');
  const flag = login.indexOf('window.__odysseusLoggedOutPage = true');
  const themeImport = login.indexOf("('/static/js/theme.js')");
  assert.ok(flag >= 0, 'login.html must set window.__odysseusLoggedOutPage');
  assert.ok(themeImport > flag, 'the flag must be set before theme.js is imported');
});

// Readability of the default workspace: muted/secondary text tokens meet
// WCAG AA against the theme surfaces, and the default (comfortable) density
// renders sidebar rows above the old 9.75px. Uses the shipped app cascade.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { appCss } from './helpers/stylesheets.mjs';
import { chromium } from 'playwright';

// The built-in palettes are plain data at the top of theme.js.
async function builtInThemes() {
  const src = await readFile(new URL('../static/js/theme.js', import.meta.url), 'utf8');
  const literal = /export const THEMES = (\{[\s\S]*?\n\});/.exec(src)[1];
  return Function(`return (${literal});`)();
}

const MUTED_TOKENS = ['--color-muted', '--color-muted-alt', '--color-subheader'];

function contrast(a, b) {
  const lum = ([r, g, b]) => [r, g, b]
    .map(v => { v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; })
    .reduce((s, v, i) => s + v * [0.2126, 0.7152, 0.0722][i], 0);
  const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}
const hex = (h) => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));

test('muted text tokens keep 4.5:1 on --bg and --panel in every built-in theme', async () => {
  const themes = await builtInThemes();
  const css = await appCss();
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.setContent(MUTED_TOKENS.map(t => `<span data-token="${t}" style="color:var(${t})">x</span>`).join(''));
    await page.addStyleTag({ content: css });
    const failures = [];
    for (const [name, c] of Object.entries(themes)) {
      // A theme whose own foreground misses AA cannot be fixed by a mix of it.
      if (contrast(hex(c.fg), hex(c.bg)) < 4.5) continue;
      const colors = await page.evaluate((c) => {
        const s = document.documentElement.style;
        for (const k of ['bg', 'fg', 'panel', 'border']) s.setProperty('--' + k, c[k]);
        const probe = document.createElement('canvas').getContext('2d');
        return Object.fromEntries([...document.querySelectorAll('[data-token]')].map(el => {
          probe.fillStyle = '#000';
          probe.fillStyle = getComputedStyle(el).color;
          probe.fillRect(0, 0, 1, 1);
          return [el.dataset.token, [...probe.getImageData(0, 0, 1, 1).data.slice(0, 3)]];
        }));
      }, c);
      for (const [token, rgb] of Object.entries(colors)) {
        for (const surface of ['bg', 'panel']) {
          const ratio = contrast(rgb, hex(c[surface]));
          if (ratio < 4.5) failures.push(`${name} ${token} on --${surface}: ${ratio.toFixed(2)}`);
        }
      }
    }
    assert.deepEqual(failures, []);
  } finally { await browser.close(); }
});

test('comfortable density (the default) renders sidebar rows at 11px; compact keeps 9.75px', async () => {
  const css = await appCss();
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    await page.setContent(`<nav class="sidebar"><div class="section" id="tools-section">
      <div class="section-header-flex"><span class="section-title">Tools</span></div>
      <div class="list-item" id="tool-notes-btn"><span class="grow">Notes</span></div></div></nav>`);
    await page.addStyleTag({ content: css });
    const sizes = () => page.evaluate(() => [
      getComputedStyle(document.querySelector('.list-item .grow')).fontSize,
      getComputedStyle(document.querySelector('.section-title')).fontSize,
    ]);
    assert.deepEqual(await sizes(), ['11px', '11px']);
    await page.evaluate(() => document.documentElement.classList.add('density-compact'));
    assert.deepEqual(await sizes(), ['9.75px', '10px']);
    await page.evaluate(() => document.documentElement.classList.replace('density-compact', 'density-spacious'));
    assert.deepEqual(await sizes(), ['12px', '12px']);
  } finally { await browser.close(); }
});

// Mobile tool sheets keep a visible, labelled Close in their header. Swipe
// down and Escape still dismiss, but neither is discoverable on its own.
// Applies the shipped app cascade to the sheet header markup each tool uses.
import test from 'node:test';
import assert from 'node:assert/strict';
import { appCss } from './helpers/stylesheets.mjs';
import { chromium } from 'playwright';

const SHEETS = `
  <div id="calendar-modal" class="modal"><div class="modal-content">
    <div class="modal-header"><h4>Calendar</h4>
      <button class="modal-minimize-btn" title="Minimize"></button>
      <button class="close-btn" id="cal-close" aria-label="Close Calendar">&#x2716;</button></div>
    <div class="modal-body"><button class="close-btn" id="nested-close">x</button></div>
  </div></div>
  <div id="memory-modal" class="modal"><div class="modal-content memory-modal-content">
    <div class="modal-header"><h4>Memory</h4>
      <button class="close-btn" id="close-memory-modal" aria-label="Close memory modal">&#x2716;</button></div>
  </div></div>
  <div id="gallery-modal" class="modal"><div class="modal-content">
    <div class="modal-header"><h4>Gallery</h4>
      <button class="modal-close" id="gallery-close" aria-label="Close Gallery">&times;</button></div>
  </div></div>
  <div id="theme-modal" class="modal"><div id="theme-popup" class="modal-content admin-modal-content">
    <div class="modal-header theme-popup-header"><h4>Theme</h4>
      <button class="close-btn" id="close-theme-popup" aria-label="Close theme">&#x2716;</button></div>
  </div></div>
  <div class="notes-pane"><div class="notes-pane-header">
    <h4 class="notes-pane-title">Notes</h4><span style="flex:1"></span>
    <button id="notes-minimize-btn" class="modal-minimize-btn" aria-label="Close notes"><svg width="14" height="14"></svg></button>
  </div></div>`;

const HEADER_CLOSE = ['cal-close', 'close-memory-modal', 'gallery-close', 'close-theme-popup', 'notes-minimize-btn'];

async function measure(page, width) {
  await page.setViewportSize({ width, height: 800 });
  // Sheets scale in on open; measure the settled geometry.
  await page.evaluate(() => Promise.all(document.getAnimations().map(a => a.finished)));
  return page.evaluate((ids) => Object.fromEntries(ids.map(id => {
    const el = document.getElementById(id);
    const r = el.getBoundingClientRect();
    return [id, { shown: getComputedStyle(el).display !== 'none' && r.width > 0,
      width: r.width, height: r.height, label: el.getAttribute('aria-label') }];
  })), [...HEADER_CLOSE, 'nested-close']);
}

test('every mobile sheet header shows a labelled Close with a 44px touch target', async () => {
  const css = await appCss();
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.setContent(SHEETS);
    await page.addStyleTag({ content: css });

    const phone = await measure(page, 375);
    for (const id of HEADER_CLOSE) {
      assert.ok(phone[id].shown, `${id} must be visible on a 375px sheet`);
      assert.ok(phone[id].width >= 44 && phone[id].height >= 44,
        `${id} touch target ${phone[id].width}x${phone[id].height} is below 44px`);
      assert.match(phone[id].label, /^Close\b/, `${id} needs a Close accessible name`);
    }
    assert.equal(phone['nested-close'].shown, false, 'close buttons inside a sheet body stay hidden on mobile');

    // Desktop keeps the compact round header control.
    const desktop = await measure(page, 1280);
    assert.equal(desktop['cal-close'].shown, true);
    assert.equal(desktop['cal-close'].width, 24);
  } finally { await browser.close(); }
});

// ============================================
// Section Management — collapse/expand + drag reorder + tool pins
// ============================================

import { bindMenuDismiss } from './escMenuStack.js';

/**
 * Initialize section collapse/expand with chevron buttons.
 * @param {Object} Storage - Storage module
 */
export function initSectionCollapse(Storage) {
  const _chevronHtml = '<button type="button" class="section-collapse-btn" title="Collapse section"><svg class="section-collapse-chevron" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg></button>';
  const savedState = Storage.getJSON('section-collapsed') || {};

  document.querySelectorAll('.section .section-header-flex').forEach(header => {
    const section = header.closest('.section');
    if (!section || !section.id) return;

    // Skip email section — it doesn't collapse (title opens popup instead)
    if (section.id === 'email-section') return;

    // Add chevron (always visible — rotates when collapsed)
    header.insertAdjacentHTML('beforeend', _chevronHtml);

    // Restore saved state
    if (savedState[section.id]) {
      section.classList.add('collapsed');
    }

    function toggleCollapse() {
      const wasCollapsed = section.classList.contains('collapsed');
      const willCollapse = !wasCollapsed;
      const state = Storage.getJSON('section-collapsed') || {};
      state[section.id] = willCollapse;
      Storage.setJSON('section-collapsed', state);

      // Always clear any in-flight animation classes from a previous toggle
      // so back-to-back clicks restart cleanly. Bump a generation token so
      // any callback still pending from a superseded toggle becomes a no-op.
      section.classList.remove('section-just-expanded', 'section-just-collapsing');
      const gen = (section._collapseGen = (section._collapseGen || 0) + 1);

      if (willCollapse) {
        // Domino-out: play the fade/slide-down on the row children BEFORE
        // actually adding .collapsed (which hides them via display:none),
        // then lock in collapse once the cascade finishes.
        //
        // We wait on the REAL animations (getAnimations) rather than a fixed
        // timeout. Different sections animate different rows — .list-item in
        // most, .models-row in #models-section — so any hard-coded duration
        // either stalls with a dead pause (when the selector matches nothing,
        // as it did for #models-section) or guesses the wrong length. Force a
        // reflow first so the keyframes restart from the top.
        // eslint-disable-next-line no-unused-expressions
        section.offsetHeight;
        section.classList.add('section-just-collapsing');

        const lockCollapsed = () => {
          if (section._collapseGen !== gen) return; // superseded by a newer toggle
          section.classList.remove('section-just-collapsing');
          section.classList.add('collapsed');
        };
        // Only the domino-out keyframes gate the collapse — ignore unrelated
        // (and possibly infinite, e.g. spinners) animations in the subtree.
        const dominoOut = section.getAnimations({ subtree: true })
          .filter(a => a.animationName === 'section-domino-out');
        if (dominoOut.length === 0) {
          lockCollapsed(); // nothing to animate — collapse now, no dead pause
        } else {
          Promise.allSettled(dominoOut.map(a => a.finished)).then(lockCollapsed);
          // Safety net: if an animation never settles (e.g. element removed),
          // still lock in the collapse so the section can't get stuck open.
          setTimeout(lockCollapsed, 600);
        }
      } else {
        // Expand path — remove .collapsed and replay the inbound domino.
        section.classList.remove('collapsed');
        // eslint-disable-next-line no-unused-expressions
        section.offsetHeight;
        section.classList.add('section-just-expanded');
        setTimeout(() => {
          if (section._collapseGen !== gen) return; // superseded by a newer toggle
          section.classList.remove('section-just-expanded');
        }, 700);
      }
    }

    // Click title to collapse/expand
    const title = header.querySelector('h4') || header.querySelector('.section-title');
    if (title) {
      title.style.cursor = 'pointer';
      title.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleCollapse();
      });
    }

    // Click chevron button
    const chevronBtn = header.querySelector('.section-collapse-btn');
    if (chevronBtn) {
      chevronBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleCollapse();
      });
    }

    // Chats can contain hundreds of rows, so provide a sticky collapse action
    // at the end of that list. Reuse the same handler/state as the header
    // chevron instead of creating a second persistence path.

    // Click anywhere on collapsed section to expand
    section.addEventListener('click', (e) => {
      if (!section.classList.contains('collapsed')) return;
      if (e.target.closest('button, select, .dropdown')) return;
      e.stopPropagation();
      toggleCollapse();
    });
  });
}

/**
 * Initialize section drag reorder (mouse-based, desktop only).
 * @param {Object} Storage - Storage module
 * @param {Function} loadUIVis - Function that returns UI visibility state
 */
export function initSectionDrag(Storage, loadUIVis) {
  const sidebar = document.getElementById('sidebar');
  const sidebarInner = sidebar ? sidebar.querySelector('.sidebar-inner') : null;
  if (!sidebarInner) return;

  // Disable draggable on mobile to prevent scroll interference
  if ('ontouchstart' in window) {
    document.querySelectorAll('.section[draggable]').forEach(s => {
      s.setAttribute('draggable', 'false');
    });
  }

  let draggedSection = null;
  let placeholder = null;
  let offsetY = 0;

  function getSections() {
    return Array.from(sidebar.querySelectorAll('.section[draggable="true"]'));
  }

  function onMouseDown(e) {
    if (!e.target.classList.contains('drag-handle')) return;

    // Check if drag reorder is enabled
    const uiState = loadUIVis();
    if (uiState['section-drag-reorder'] === false) return;

    const section = e.target.closest('.section[draggable="true"]');
    if (!section) return;

    e.preventDefault();

    const rect = section.getBoundingClientRect();
    offsetY = e.clientY - rect.top;
    draggedSection = section;

    // Create placeholder
    placeholder = document.createElement('div');
    placeholder.className = 'section-placeholder';
    placeholder.style.cssText = `
      height: ${rect.height}px;
      margin: 4px 0;
      border: 2px dashed rgba(0, 170, 255, 0.5);
      border-radius: 8px;
      background: rgba(0, 170, 255, 0.1);
    `;
    section.parentNode.insertBefore(placeholder, section);

    // Float the section
    Object.assign(section.style, {
      position: 'fixed',
      width: rect.width + 'px',
      left: rect.left + 'px',
      top: rect.top + 'px',
      zIndex: '9999',
      opacity: '0.95',
      boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
      pointerEvents: 'none',
      transition: 'none'
    });

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }

  function onMouseMove(e) {
    if (!draggedSection) return;

    // Only move vertically - horizontal stays locked
    draggedSection.style.top = (e.clientY - offsetY) + 'px';

    const sections = getSections().filter(s => s !== draggedSection);
    const dragRect = draggedSection.getBoundingClientRect();
    const dragTop = dragRect.top;

    let insertBefore = null;

    // Find which section we should go before
    for (let i = 0; i < sections.length; i++) {
      const section = sections[i];
      const rect = section.getBoundingClientRect();

      // If our top edge is above this section's bottom, we go before it
      if (dragTop < rect.bottom - 10) {
        insertBefore = section;
        break;
      }
    }

    // Move placeholder
    if (insertBefore) {
      if (placeholder.nextElementSibling !== insertBefore) {
        sidebarInner.insertBefore(placeholder, insertBefore);
      }
    } else if (sections.length > 0) {
      const lastSection = sections[sections.length - 1];
      if (placeholder !== lastSection.nextElementSibling) {
        sidebarInner.insertBefore(placeholder, lastSection.nextElementSibling);
      }
    }
  }

  function onMouseUp() {
    if (!draggedSection) return;

    document.removeEventListener('mousemove', onMouseMove);
    document.removeEventListener('mouseup', onMouseUp);

    // Snap to placeholder - fast!
    const phRect = placeholder.getBoundingClientRect();
    draggedSection.style.transition = 'top 0.08s ease-out';
    draggedSection.style.top = phRect.top + 'px';

    setTimeout(() => {
      placeholder.parentNode.insertBefore(draggedSection, placeholder);
      placeholder.remove();
      draggedSection.style.cssText = '';

      // Save order
      const ids = getSections().map(s => s.id).filter(Boolean);
      Storage.setJSON(Storage.KEYS.SECTION_ORDER, ids);

      draggedSection = null;
      placeholder = null;
    }, 80);
  }

  sidebar.addEventListener('mousedown', onMouseDown);

  // Restore saved order on load
  try {
    const saved = Storage.get(Storage.KEYS.SECTION_ORDER);
    if (saved) {
      const order = JSON.parse(saved);
      order.forEach(id => {
        const section = document.getElementById(id);
        if (section) sidebarInner.appendChild(section);
      });
    }
  } catch (e) {}
}

// ============================================
// Tool pins — everyday tools first, the rest behind "More tools"
// ============================================
//
// Every `.list-item` with an id directly inside #tools-section takes part;
// nothing is listed per tool here, so a tool added to the markup later (or
// injected at runtime) is picked up automatically. A tool's key comes from its
// id (`tool-notes-btn` → `notes`) and its icon-rail launcher is `#rail-<key>`
// unless the row names another one in `data-rail-id` (Library's launcher
// predates the convention and is `#rail-archive`).
//
// Saved state (localStorage `odysseus-tool-pins`; pins are mirrored to the
// `tool-pins` user preference so they follow the account):
//   { pin: [...], unpin: [...], moreExpanded: bool }
// `pin`/`unpin` are deltas against DEFAULT_PINNED_TOOLS, so a newly shipped
// everyday tool (e.g. Decisions) arrives pinned unless the user unpinned it,
// and any other new tool arrives under "More tools".

export const DEFAULT_PINNED_TOOLS = Object.freeze(
  ['notes', 'calendar', 'research', 'memory', 'library', 'decisions']);
const TOOL_PINS_STORAGE_KEY = 'odysseus-tool-pins';
const TOOL_PINS_PREF_URL = '/api/prefs/tool-pins';
const RAIL_ID_OVERRIDES = { library: 'rail-archive' };

/** `tool-notes-btn` → `notes`; ids outside the convention are used verbatim. */
export function toolKeyFromId(id) {
  const m = /^tool-(.+)-btn$/.exec(String(id || ''));
  return m ? m[1] : String(id || '');
}

/** Icon-rail launcher id for a tool key (an explicit data-rail-id wins). */
export function railIdForTool(key, railId) {
  return railId || RAIL_ID_OVERRIDES[key] || 'rail-' + key;
}

const _keyList = (v) => (Array.isArray(v)
  ? [...new Set(v.filter(k => typeof k === 'string' && k))] : []);

/** Coerce anything read from storage or the server into a valid state. */
export function normalizeToolPinState(raw) {
  const src = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
  const defaults = new Set(DEFAULT_PINNED_TOOLS);
  return {
    // Deltas only: a default can only be unpinned, anything else only pinned.
    pin: _keyList(src.pin).filter(k => !defaults.has(k)),
    unpin: _keyList(src.unpin).filter(k => defaults.has(k)),
    moreExpanded: src.moreExpanded === true,
  };
}

export function isToolPinned(state, key) {
  const s = normalizeToolPinState(state);
  if (s.pin.includes(key)) return true;
  if (s.unpin.includes(key)) return false;
  return DEFAULT_PINNED_TOOLS.includes(key);
}

/** New state with `key` pinned (true) or moved under "More tools" (false). */
export function setToolPinned(state, key, pinned) {
  const s = normalizeToolPinState(state);
  const isDefault = DEFAULT_PINNED_TOOLS.includes(key);
  s.pin = s.pin.filter(k => k !== key);
  s.unpin = s.unpin.filter(k => k !== key);
  if (pinned && !isDefault) s.pin.push(key);
  if (!pinned && isDefault) s.unpin.push(key);
  return s;
}

/** Split tool keys (already in display order) into pinned and unpinned. */
export function splitToolPins(keys, state) {
  const pinned = [];
  const unpinned = [];
  for (const key of keys) (isToolPinned(state, key) ? pinned : unpinned).push(key);
  return { pinned, unpinned };
}

const _SVG = 'width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"';
const _PIN_ICON = `<svg ${_SVG}><line x1="12" y1="17" x2="12" y2="22"/><path d="M5 17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1v4.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24Z"/></svg>`;
const _UNPIN_ICON = `<svg ${_SVG}><line x1="2" y1="2" x2="22" y2="22"/><line x1="12" y1="17" x2="12" y2="22"/><path d="M9 9v1.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V17h12"/><path d="M15 9.34V6h1a2 2 0 0 0 0-4H7.89"/></svg>`;
const _CHEVRON_ICON = `<svg class="tools-more-chevron" ${_SVG}><polyline points="6 9 12 15 18 9"/></svg>`;
const _MORE_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true"><circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/></svg>';

/**
 * Pin the everyday tools to the top of #tools-section (and the icon rail) and
 * fold the rest into a collapsible "More tools" group. Right-click, long-press
 * or Shift+F10 / the Menu key on a tool row or rail launcher opens Pin/Unpin.
 *
 * @param {Object} Storage - Storage module (getJSON/setJSON/KEYS)
 * @param {{doc?: Document, fetchImpl?: Function}} [opts] - injectable for tests
 * @returns {null|{apply, setPinned, setExpanded, openMenu, getState, ready}}
 */
export function initToolPins(Storage, {
  doc = typeof document !== 'undefined' ? document : null,
  fetchImpl = typeof fetch === 'function' ? fetch : null,
} = {}) {
  const section = doc && doc.getElementById('tools-section');
  if (!section) return null;
  const storageKey = (Storage.KEYS && Storage.KEYS.TOOL_PINS) || TOOL_PINS_STORAGE_KEY;
  let state = normalizeToolPinState(Storage.getJSON(storageKey, null));
  let pinsEditedLocally = false;
  let observer = null;
  let closeMenu = null;
  const rail = doc.getElementById('icon-rail');

  // The "More tools" row reuses the sidebar row styling (.list-item).
  const toggle = doc.createElement('div');
  toggle.id = 'tools-more-toggle';
  toggle.className = 'list-item tools-more-toggle';
  toggle.setAttribute('role', 'button');
  toggle.setAttribute('tabindex', '0');
  toggle.title = 'Right-click a tool (or press Shift+F10) to pin or unpin it';
  toggle.innerHTML = _CHEVRON_ICON
    + '<span class="grow">More tools</span><span class="tools-more-count"></span>';
  toggle.addEventListener('click', (e) => {
    e.stopPropagation();
    setExpanded(!state.moreExpanded);
  });

  let railToggle = null;
  if (rail) {
    railToggle = doc.createElement('button');
    railToggle.type = 'button';
    railToggle.id = 'rail-more-tools';
    railToggle.className = 'icon-rail-btn rail-more-tools';
    railToggle.innerHTML = _MORE_ICON;
    railToggle.addEventListener('click', () => setExpanded(!state.moreExpanded));
    // Sit at the end of the launchers, just above the spacer before Settings.
    const settings = doc.getElementById('rail-settings');
    const prev = settings && settings.previousElementSibling;
    const anchor = prev && !prev.id ? prev : settings;
    if (anchor && anchor.parentNode === rail) rail.insertBefore(railToggle, anchor);
    else rail.appendChild(railToggle);
  }

  const toolItems = () => Array.from(section.children).filter(el =>
    el !== toggle && el.id && el.classList.contains('list-item'));
  const labelOf = (el) => ((el.querySelector('.grow') || el).textContent || '').trim();
  const railFor = (item) => (rail
    ? doc.getElementById(railIdForTool(toolKeyFromId(item.id), item.dataset && item.dataset.railId))
    : null);
  const isShown = (el) => !!el && el.style.display !== 'none';
  // A collapsed tool with something live (a minimized window, a notification
  // dot) stays visible so the signal is never hidden behind "More tools".
  const hasAttention = (item) => item.classList.contains('rail-minimized')
    || Array.from(item.querySelectorAll('.sidebar-notif-dot, .cookbook-notif-dot, .tool-notes-dot'))
      .some(isShown);
  const hiddenByCollapse = (item) => item.classList.contains('tool-unpinned')
    && !state.moreExpanded && !item.classList.contains('rail-minimized')
    && !item.classList.contains('tool-attention');
  const sameOrder = (parent, desired) => {
    const current = Array.from(parent.children).filter(el => desired.includes(el));
    return current.length === desired.length && current.every((el, i) => el === desired[i]);
  };

  function apply() {
    const items = toolItems().sort((a, b) => labelOf(a).localeCompare(labelOf(b)));
    const byKey = new Map(items.map(el => [toolKeyFromId(el.id), el]));
    const { pinned, unpinned } = splitToolPins([...byKey.keys()], state);
    const pinnedEls = pinned.map(k => byKey.get(k));
    const unpinnedEls = unpinned.map(k => byKey.get(k));
    const collapsed = !state.moreExpanded;

    const desired = [...pinnedEls, toggle, ...unpinnedEls];
    if (!sameOrder(section, desired)) {
      const focused = doc.activeElement;
      for (const el of desired) section.appendChild(el);
      if (focused && focused !== doc.activeElement && desired.includes(focused)) {
        try { focused.focus(); } catch (e) {}
      }
    }
    for (const el of pinnedEls) el.classList.remove('tool-unpinned', 'tool-attention');
    for (const el of unpinnedEls) {
      el.classList.add('tool-unpinned');
      el.classList.toggle('tool-attention', hasAttention(el));
    }
    const moreCount = unpinnedEls.filter(isShown).length;
    section.classList.toggle('tools-more-collapsed', collapsed);
    toggle.classList.toggle('tools-more-empty', moreCount === 0);
    toggle.setAttribute('aria-expanded', String(!collapsed));
    const countEl = toggle.querySelector('.tools-more-count');
    if (countEl && countEl.textContent !== String(moreCount)) countEl.textContent = String(moreCount);

    if (railToggle) {
      const railPinned = pinnedEls.map(railFor).filter(Boolean);
      const railUnpinned = unpinnedEls.map(railFor).filter(Boolean);
      const email = doc.getElementById('rail-email');
      const lead = email && email.parentNode === rail ? [...railPinned, email] : railPinned;
      if (!sameOrder(rail, [...lead, railToggle, ...railUnpinned])) {
        for (const el of lead) rail.insertBefore(el, railToggle);
        let ref = railToggle;
        for (const el of railUnpinned) { rail.insertBefore(el, ref.nextSibling); ref = el; }
      }
      for (const el of railPinned) el.classList.remove('tool-unpinned');
      for (const el of railUnpinned) el.classList.add('tool-unpinned');
      rail.classList.toggle('rail-more-collapsed', collapsed);
      railToggle.classList.toggle('tools-more-empty', moreCount === 0);
      railToggle.setAttribute('aria-expanded', String(!collapsed));
      const label = collapsed ? `More tools (${moreCount})` : 'Fewer tools';
      railToggle.title = label;
      railToggle.setAttribute('aria-label', label);
    }
    // Drop the mutation records this pass produced so it cannot re-trigger.
    if (observer) observer.takeRecords();
  }

  function save({ sync = false } = {}) {
    Storage.setJSON(storageKey, state);
    if (!sync || !fetchImpl) return;
    try {
      Promise.resolve(fetchImpl(TOOL_PINS_PREF_URL, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ value: { pin: state.pin, unpin: state.unpin } }),
      })).catch(() => {});
    } catch (e) {}
  }

  function setExpanded(open) {
    state = { ...state, moreExpanded: !!open };
    save();
    apply();
  }

  function setPinned(key, pinned) {
    pinsEditedLocally = true;
    state = setToolPinned(state, key, pinned);
    save({ sync: true });
    apply();
  }

  function openMenu(item, at = null, returnTo = item) {
    if (closeMenu) closeMenu();
    const key = toolKeyFromId(item.id);
    const pinned = isToolPinned(state, key);
    const menu = doc.createElement('div');
    menu.className = 'dropdown session-dropdown-menu tool-pin-menu';
    menu.setAttribute('role', 'menu');
    menu.setAttribute('aria-label', `${labelOf(item) || key} options`);
    const row = doc.createElement('div');
    row.className = 'dropdown-item-compact';
    row.setAttribute('role', 'menuitem');
    row.setAttribute('tabindex', '-1');
    row.innerHTML = `<span class="dropdown-icon">${pinned ? _UNPIN_ICON : _PIN_ICON}</span><span></span>`;
    row.lastChild.textContent = pinned ? 'Unpin (move to More tools)' : 'Pin to top';
    menu.appendChild(row);

    let activated = false;
    const close = bindMenuDismiss(menu, () => {
      const hadFocus = menu.contains(doc.activeElement);
      menu.remove();
      if (closeMenu === close) closeMenu = null;
      if (activated || hadFocus) {
        const target = returnTo !== item || !hiddenByCollapse(item) ? returnTo : toggle;
        try { target.focus(); } catch (e) {}
      }
    });
    closeMenu = close;
    const activate = () => {
      activated = true;
      setPinned(key, !pinned);
      close();
    };
    row.addEventListener('click', (e) => { e.stopPropagation(); activate(); });
    menu.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate(); }
      else if (e.key === 'Tab') { e.preventDefault(); close(); }
    });

    doc.body.appendChild(menu);
    menu.style.display = 'block';
    const r = item.getBoundingClientRect();
    const box = menu.getBoundingClientRect();
    const vw = (doc.defaultView && doc.defaultView.innerWidth) || 1024;
    const vh = (doc.defaultView && doc.defaultView.innerHeight) || 768;
    const x = at ? at.x : r.left + 16;
    const y = at ? at.y : r.bottom + 2;
    menu.style.left = Math.max(4, Math.min(x, vw - box.width - 4)) + 'px';
    menu.style.top = Math.max(4, Math.min(y, vh - box.height - 4)) + 'px';
    try { row.focus(); } catch (e) {}
    return close;
  }

  const rowFrom = (target) => {
    const item = target && target.closest ? target.closest('.list-item') : null;
    return item && item !== toggle && item.id && item.parentNode === section ? item : null;
  };
  const pointAt = (e) => (e.clientX || e.clientY ? { x: e.clientX, y: e.clientY } : null);
  section.addEventListener('contextmenu', (e) => {
    const item = rowFrom(e.target);
    if (!item) return;
    e.preventDefault();
    openMenu(item, pointAt(e));
  });
  section.addEventListener('keydown', (e) => {
    if (e.key !== 'ContextMenu' && !(e.key === 'F10' && e.shiftKey)) return;
    const item = rowFrom(e.target);
    if (!item) return;
    e.preventDefault();
    openMenu(item, null);
  });
  if (rail) {
    rail.addEventListener('contextmenu', (e) => {
      const btn = e.target && e.target.closest ? e.target.closest('.icon-rail-btn') : null;
      const item = btn && toolItems().find(it => railFor(it) === btn);
      if (!item) return;
      e.preventDefault();
      openMenu(item, pointAt(e), btn);
    });
  }

  apply();
  // Re-apply when tools are added/removed, shown/hidden by Settings →
  // Appearance, or gain/lose a live indicator.
  if (typeof MutationObserver === 'function') {
    observer = new MutationObserver(() => apply());
    observer.observe(section, {
      childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'class'],
    });
    if (rail) observer.observe(rail, { childList: true });
  }

  const ready = (async () => {
    if (!fetchImpl) return;
    try {
      const res = await fetchImpl(TOOL_PINS_PREF_URL, { credentials: 'same-origin' });
      if (!res || !res.ok) return;
      const data = await res.json();
      const remote = data && data.value;
      if (pinsEditedLocally || !remote || typeof remote !== 'object') return;
      const next = normalizeToolPinState({ ...remote, moreExpanded: state.moreExpanded });
      if (JSON.stringify(next) === JSON.stringify(state)) return;
      state = next;
      save();
      apply();
    } catch (e) {}
  })();

  return { apply, setPinned, setExpanded, openMenu, getState: () => state, ready };
}

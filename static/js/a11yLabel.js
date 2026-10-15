// Keep a control's tooltip and accessible name in step.
//
// Some buttons change meaning with state: the composer button sends, queues,
// stops generation, records voice, or starts a new chat; Compare's copy of it
// sends to or stops all models. An `aria-label` outranks `title` when the
// browser computes the accessible name, so updating only `title` left screen
// readers announcing the stale action ("New chat" while the button stopped
// generation). Every state change goes through here so both stay identical.
//
// Deliberately side-effect free (unlike a11y.js, which enhances the DOM when
// it is evaluated) so any module can import it without changing load order.

export function setActionLabel(el, label) {
  if (!el) return;
  el.title = label;
  el.setAttribute('aria-label', label);
}

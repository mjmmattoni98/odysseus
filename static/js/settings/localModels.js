// Pure helpers for Settings → Local Model Context & Residency.

export const CONTEXT_MIN = 1024;
export const CONTEXT_MAX = 262144;

// Valid default context cap, or null when the input is out of range.
export function parseContextDefault(raw) {
  const value = Number(raw);
  if (!Number.isInteger(value) || value < CONTEXT_MIN || value > CONTEXT_MAX) return null;
  return value;
}

// Roles pinned to a different model on the same LOCAL endpoint as the
// default chat model. On a server that keeps one model loaded, every such
// call unloads the chat model and loads the other one.
export function localModelSwapWarnings(settings, endpoints) {
  const s = settings || {};
  const eps = Array.isArray(endpoints) ? endpoints : [];
  const chatEp = eps.find(ep => ep && ep.id === s.default_endpoint_id);
  const chatModel = s.default_model || '';
  if (!chatEp || !chatModel || chatEp.category !== 'local') return [];
  const out = [];
  const add = (label, model) => {
    if (model && model !== chatModel) out.push(`${label} uses ${model}`);
  };
  [['Utility', 'utility'], ['Research', 'research'], ['Background tasks', 'task']].forEach(([label, key]) => {
    if (s[`${key}_endpoint_id`] === chatEp.id) add(label, s[`${key}_model`]);
  });
  if (s.vision_model && (chatEp.models || []).includes(s.vision_model)) add('Vision', s.vision_model);
  const teacher = String(s.teacher_model || '');
  const at = teacher.lastIndexOf('@');
  const teacherEp = at > 0 ? teacher.slice(at + 1).toLowerCase() : '';
  if (s.teacher_enabled && teacherEp && String(chatEp.name || '').toLowerCase().includes(teacherEp)) {
    add('Teacher', teacher.slice(0, at));
  }
  return out;
}

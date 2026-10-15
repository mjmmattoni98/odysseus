/**
 * Decision notebook scoring — pure helpers, no DOM.
 *
 * Mirrors score_options / compare_to_snapshot in src/decisions.py so totals,
 * ranks and the "since decision" comparison update while the user types,
 * without a round trip. Both sides are pinned by the shared cases in
 * tests/fixtures/decision_scoring_cases.json; change them together.
 */

// Half-up to 4 places. Python's round() is banker's rounding, so the server
// uses the same floor(x + 0.5) form to keep ties identical on both sides.
export function round4(value) {
  return Math.floor(value * 10000 + 0.5) / 10000;
}

// Ids are user data and the server accepts any [A-Za-z0-9_-] id, including
// "__proto__" and "constructor". Every id-keyed dictionary therefore has no
// prototype (or is a Map), and every read is an own-property read, so an id
// can never hit Object.prototype the way it can't in Python.

/** A prototype-free object for id-keyed results (still JSON-serializable). */
export function idDict() {
  return Object.create(null);
}

/** `obj[id]` when `obj` has its own `id` key, else undefined. */
export function ownValue(obj, id) {
  return obj != null && Object.prototype.hasOwnProperty.call(obj, id) ? obj[id] : undefined;
}

/** Criterion id -> share of the total weight; all-zero weights count equally. */
export function normalizedWeights(criteria) {
  const list = criteria || [];
  const out = new Map();
  if (!list.length) return out;
  let total = 0;
  for (const c of list) total += Math.max(0, Number(c.weight) || 0);
  for (const c of list) {
    out.set(c.id, total <= 0 ? 1 / list.length : Math.max(0, Number(c.weight) || 0) / total);
  }
  return out;
}

/**
 * Weighted totals (0..10), must-have violations and ranking.
 * Options failing a must-have rank below every eligible option and never lead.
 * `weights` and `options` are keyed by id with no prototype (see idDict).
 */
export function scoreOptions(body) {
  const criteria = body.criteria || [];
  const options = body.options || [];
  const mustHave = new Set((body.constraints || []).filter(c => c.must_have).map(c => c.id));
  const weights = normalizedWeights(criteria);

  const results = options.map((option, index) => {
    let total = 0;
    let unscored = 0;
    for (const c of criteria) {
      const score = ownValue(option.scores, c.id);
      if (score === undefined || score === null || score === '') { unscored += 1; continue; }
      total += weights.get(c.id) * Number(score);
    }
    const violations = (option.fails || []).filter(id => mustHave.has(id));
    return {
      option_id: option.id,
      total: round4(total),
      unscored,
      violations,
      eligible: violations.length === 0,
      _index: index,
    };
  });

  const ranked = [...results].sort((a, b) =>
    (a.eligible === b.eligible ? 0 : (a.eligible ? -1 : 1))
    || (b.total - a.total)
    || (a._index - b._index));
  ranked.forEach((r, i) => { r.rank = i + 1; });
  const eligible = ranked.filter(r => r.eligible);
  const leaderId = eligible.length && criteria.length ? eligible[0].option_id : null;
  const tied = !!(leaderId && eligible.length > 1 && eligible[0].total === eligible[1].total);
  const byId = idDict();
  for (const r of results) { delete r._index; byId[r.option_id] = r; }
  const roundedWeights = idDict();
  for (const [id, share] of weights) roundedWeights[id] = round4(share);
  return {
    weights: roundedWeights,
    options: byId,
    ranking: ranked.map(r => r.option_id),
    leader_id: leaderId,
    tied,
  };
}

function _names(entries) {
  const out = new Map();
  for (const e of entries || []) out.set(e.id, e.name || 'Untitled');
  return out;
}

function _same(a, b) {
  const norm = v => (v === undefined || v === '' ? null : v);
  return norm(a) === norm(b);
}

/**
 * What changed since a recorded decision, matched by id (renames are not
 * removals), and whether the chosen option would still win — both under
 * today's weights and under the weights in force when it was chosen.
 */
export function compareToSnapshot(body, snapshot) {
  const now = scoreOptions(body);
  const thenCriteria = snapshot.criteria || [];
  const thenOptions = snapshot.options || [];
  const thenCrit = _names(thenCriteria);
  const nowCrit = _names(body.criteria);
  const thenOpt = _names(thenOptions);
  const nowOpt = _names(body.options);

  const weightChanges = [];
  for (const c of thenCriteria) {
    const after = ownValue(now.weights, c.id);
    if (after === undefined) continue;
    const before = round4(c.share || 0);
    if (round4(after) !== before) {
      weightChanges.push({ criterion: nowCrit.get(c.id), before, after });
    }
  }

  const nowOptions = new Map((body.options || []).map(o => [o.id, o]));
  const scoreChanges = [];
  for (const thenOption of thenOptions) {
    const current = nowOptions.get(thenOption.id);
    if (!current) continue;
    for (const cid of thenCrit.keys()) {
      if (!nowCrit.has(cid)) continue;
      const before = ownValue(thenOption.scores, cid);
      const after = ownValue(current.scores, cid);
      if (!_same(before, after)) {
        scoreChanges.push({
          option: nowOpt.get(current.id),
          criterion: nowCrit.get(cid),
          before: before ?? null,
          after: after ?? null,
        });
      }
    }
  }

  // Today's options judged by the decision-time criteria and weights; a
  // criterion deleted since then falls back to the recorded score.
  const thenScores = new Map(thenOptions.map(o => [o.id, o.scores]));
  const replay = {
    constraints: body.constraints || [],
    criteria: thenCriteria.map(c => ({ id: c.id, name: c.name, weight: c.weight ?? 0 })),
    options: (body.options || []).map(option => {
      const scores = idDict();
      for (const cid of thenCrit.keys()) {
        const value = nowCrit.has(cid)
          ? ownValue(option.scores, cid)
          : ownValue(thenScores.get(option.id), cid);
        if (value !== undefined && value !== null && value !== '') scores[cid] = value;
      }
      return { id: option.id, scores, fails: option.fails || [] };
    }),
  };
  const replayed = scoreOptions(replay);

  const chosenId = snapshot.chosen_option_id;
  const chosenPresent = nowOptions.has(chosenId);
  const thenRanking = snapshot.ranking || [];
  const result = {
    decided_at: snapshot.decided_at || null,
    chosen_option_id: chosenId,
    chosen_option_name: snapshot.chosen_option_name || null,
    chosen_removed: !chosenPresent,
    criteria_added: [...nowCrit.keys()].filter(id => !thenCrit.has(id)).map(id => nowCrit.get(id)),
    criteria_removed: [...thenCrit.keys()].filter(id => !nowCrit.has(id)).map(id => thenCrit.get(id)),
    weight_changes: weightChanges,
    options_added: [...nowOpt.keys()].filter(id => !thenOpt.has(id)).map(id => nowOpt.get(id)),
    options_removed: [...thenOpt.keys()].filter(id => !nowOpt.has(id)).map(id => thenOpt.get(id)),
    score_changes: scoreChanges,
    ranking_before: thenRanking.map(id => thenOpt.get(id) || 'Untitled'),
    ranking_after: now.ranking.map(id => nowOpt.get(id)),
    ranking_changed: thenRanking.length !== now.ranking.length
      || thenRanking.some((id, i) => id !== now.ranking[i]),
    leader_before: snapshot.leader_id ? (thenOpt.get(snapshot.leader_id) || null) : null,
    leader_after: now.leader_id ? nowOpt.get(now.leader_id) : null,
    chosen_still_wins: chosenPresent ? now.leader_id === chosenId : null,
    leader_under_original_weights: replayed.leader_id ? nowOpt.get(replayed.leader_id) : null,
    chosen_wins_under_original_weights: chosenPresent ? replayed.leader_id === chosenId : null,
  };
  result.changed = !!(result.criteria_added.length || result.criteria_removed.length
    || weightChanges.length || result.options_added.length || result.options_removed.length
    || scoreChanges.length || result.ranking_changed);
  return result;
}

/** The user's local calendar date as YYYY-MM-DD (what "revisit today" means). */
export function localToday(now = new Date()) {
  const pad = n => String(n).padStart(2, '0');
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
}

/** Revisit is due on or after the date unless the decision is archived. */
export function isRevisitDue(revisitOn, status, today) {
  return !!revisitOn && status !== 'archived' && revisitOn <= today;
}

/** Display string for a 0..1 weight share, e.g. 0.375 -> "37.5%". */
export function formatShare(share) {
  return `${Math.floor((share || 0) * 1000 + 0.5) / 10}%`;
}

/** Client-side id for a new row; matches the server's accepted id pattern. */
export function newRowId() {
  const bytes = new Uint8Array(6);
  (globalThis.crypto || {}).getRandomValues?.(bytes);
  let out = '';
  for (const b of bytes) out += b.toString(16).padStart(2, '0');
  return out === '000000000000' ? Math.random().toString(36).slice(2, 14) : out;
}

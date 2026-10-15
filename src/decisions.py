"""Decision notebook domain logic: validation, weighted scoring, snapshots.

Pure functions only (no DB, no FastAPI) so scoring, revisit and the
"since decision" comparison can be exercised directly. The browser mirrors
``score_options`` and ``compare_to_snapshot`` in
``static/js/decisionScoring.js`` so totals update while the user types; keep
the two in step (``tests/decisionScoring.test.mjs`` and
``tests/test_decision_notebook.py`` pin the same cases).

Shape of a decision body (the editable working copy)::

    {
      "question": str,
      "constraints": [{"id", "text", "must_have": bool}],
      "criteria":    [{"id", "name", "weight": 0..100}],
      "options":     [{"id", "name", "description",
                       "scores": {criterion_id: 0..10},
                       "pros": [str], "cons": [str],
                       "links": [{"title", "url"}],
                       "fails": [constraint_id]}],
      "sources":     [{"title", "url"}],
    }

Weights are normalized to shares that sum to 1, so a criterion weighted 3
next to one weighted 1 counts for 75%. An option's total is the weighted
mean of its 0..10 scores (unscored cells count as 0 and are reported). An
option marked as failing a must-have constraint is still scored but ranks
below every option that satisfies all must-haves and can never be the leader.
"""

from __future__ import annotations

import copy
import math
import re
import uuid
from datetime import date
from typing import Any, Dict, Iterable, List, Optional

from src.url_safety import check_link_url

STATUSES = ("open", "decided", "archived")
REVISIT_DUE = "revisit-due"

SCORE_MIN, SCORE_MAX = 0.0, 10.0
WEIGHT_MIN, WEIGHT_MAX = 0.0, 100.0
DEFAULT_WEIGHT = 1.0

MAX_TITLE = 200
MAX_QUESTION = 4000
MAX_ITEM_TEXT = 500
MAX_NAME = 200
MAX_DESCRIPTION = 4000
MAX_RATIONALE = 8000
MAX_LINK_TITLE = 300
MAX_URL = 2048
MAX_CONSTRAINTS = 30
MAX_CRITERIA = 20
MAX_OPTIONS = 20
MAX_LIST_ITEMS = 20
MAX_LINKS = 20
MAX_SOURCES = 50

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class DecisionValidationError(ValueError):
    """Raised for user input that cannot be stored; the message is user-facing."""


# ---------------------------------------------------------------------------
# Validation / normalization
# ---------------------------------------------------------------------------

def _text(value: Any, field: str, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DecisionValidationError(f"{field} must be text")
    value = value.strip()
    if len(value) > limit:
        raise DecisionValidationError(f"{field} is longer than {limit} characters")
    return value


def _list(value: Any, field: str, limit: int) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise DecisionValidationError(f"{field} must be a list")
    if len(value) > limit:
        raise DecisionValidationError(f"{field} has more than {limit} entries")
    return value


def _obj(value: Any, field: str) -> dict:
    if not isinstance(value, dict):
        raise DecisionValidationError(f"{field} entries must be objects")
    return value


def _number(value: Any, field: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionValidationError(f"{field} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < low or value > high:
        raise DecisionValidationError(f"{field} must be between {low:g} and {high:g}")
    return value


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _ids(entries: List[dict], field: str) -> None:
    """Assign ids to entries without one; reject malformed or duplicate ids."""
    seen = set()
    for entry in entries:
        raw = entry.get("id")
        if raw in (None, ""):
            raw = _new_id()
        if not isinstance(raw, str) or not _ID_RE.match(raw):
            raise DecisionValidationError(f"{field} id must be 1-40 letters, digits, '-' or '_'")
        if raw in seen:
            raise DecisionValidationError(f"duplicate {field} id: {raw}")
        seen.add(raw)
        entry["id"] = raw


def normalize_link(value: Any, field: str = "link") -> Dict[str, str]:
    value = _obj(value, field)
    url = value.get("url")
    if not isinstance(url, str):
        raise DecisionValidationError(f"{field} URL is required")
    ok, reason = check_link_url(url, max_length=MAX_URL)
    if not ok:
        raise DecisionValidationError(f"{field} URL rejected: {reason}")
    return {"title": _text(value.get("title"), f"{field} title", MAX_LINK_TITLE), "url": url.strip()}


def _text_list(value: Any, field: str) -> List[str]:
    out = []
    for item in _list(value, field, MAX_LIST_ITEMS):
        text = _text(item, field, MAX_ITEM_TEXT)
        if text:
            out.append(text)
    return out


def normalize_revisit_on(value: Any) -> Optional[str]:
    """Accept ``YYYY-MM-DD`` (a calendar date, no time) or empty to clear."""
    if value in (None, ""):
        return None
    if not isinstance(value, str) or not _DATE_RE.match(value.strip()):
        raise DecisionValidationError("revisit date must be YYYY-MM-DD")
    value = value.strip()
    try:
        date.fromisoformat(value)
    except ValueError:
        raise DecisionValidationError("revisit date is not a real date")
    return value


def normalize_title(value: Any) -> str:
    return _text(value, "title", MAX_TITLE)


def normalize_rationale(value: Any) -> str:
    return _text(value, "rationale", MAX_RATIONALE)


def normalize_status(value: Any) -> str:
    if value not in STATUSES:
        raise DecisionValidationError(f"status must be one of: {', '.join(STATUSES)}")
    return value


def normalize_body(raw: Optional[dict]) -> Dict[str, Any]:
    """Validate a decision body and return a clean copy.

    Ids are kept stable (the client generates them so scores can reference
    criteria before the first save); missing ids are assigned. Scores and
    must-have markers that reference a criterion or constraint which no
    longer exists are dropped, because deleting a criterion deletes its
    column.
    """
    raw = raw or {}
    if not isinstance(raw, dict):
        raise DecisionValidationError("decision body must be an object")

    constraints = []
    for entry in _list(raw.get("constraints"), "constraints", MAX_CONSTRAINTS):
        entry = _obj(entry, "constraint")
        constraints.append({
            "id": entry.get("id"),
            "text": _text(entry.get("text"), "constraint", MAX_ITEM_TEXT),
            "must_have": bool(entry.get("must_have")),
        })
    _ids(constraints, "constraint")

    criteria = []
    for entry in _list(raw.get("criteria"), "criteria", MAX_CRITERIA):
        entry = _obj(entry, "criterion")
        weight = entry.get("weight", DEFAULT_WEIGHT)
        criteria.append({
            "id": entry.get("id"),
            "name": _text(entry.get("name"), "criterion name", MAX_NAME),
            "weight": DEFAULT_WEIGHT if weight is None else _number(weight, "weight", WEIGHT_MIN, WEIGHT_MAX),
        })
    _ids(criteria, "criterion")
    criterion_ids = {c["id"] for c in criteria}
    constraint_ids = {c["id"] for c in constraints}

    options = []
    for entry in _list(raw.get("options"), "options", MAX_OPTIONS):
        entry = _obj(entry, "option")
        raw_scores = entry.get("scores") or {}
        if not isinstance(raw_scores, dict):
            raise DecisionValidationError("option scores must be an object")
        scores = {}
        for cid, score in raw_scores.items():
            if cid not in criterion_ids or score is None or score == "":
                continue
            scores[cid] = _number(score, "score", SCORE_MIN, SCORE_MAX)
        fails = []
        for cid in _list(entry.get("fails"), "option must-have markers", MAX_CONSTRAINTS):
            if isinstance(cid, str) and cid in constraint_ids and cid not in fails:
                fails.append(cid)
        options.append({
            "id": entry.get("id"),
            "name": _text(entry.get("name"), "option name", MAX_NAME),
            "description": _text(entry.get("description"), "option description", MAX_DESCRIPTION),
            "scores": scores,
            "pros": _text_list(entry.get("pros"), "pros"),
            "cons": _text_list(entry.get("cons"), "cons"),
            "links": [normalize_link(link, "option link")
                      for link in _list(entry.get("links"), "option links", MAX_LINKS)],
            "fails": fails,
        })
    _ids(options, "option")

    sources = [normalize_link(link, "source")
               for link in _list(raw.get("sources"), "sources", MAX_SOURCES)]

    return {
        "question": _text(raw.get("question"), "question", MAX_QUESTION),
        "constraints": constraints,
        "criteria": criteria,
        "options": options,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def round4(value: float) -> float:
    """Round half-up to 4 places, identically to the JS mirror.

    Python's round() is banker's rounding and JS Math.round is half-up, so
    using either directly would let the two disagree on a tie.
    """
    return math.floor(value * 10000 + 0.5) / 10000


def normalized_weights(criteria: Iterable[dict]) -> Dict[str, float]:
    """Criterion id -> share of the total weight (shares sum to 1).

    When every weight is zero (nothing prioritized yet) the criteria count
    equally rather than the whole matrix scoring zero.
    """
    criteria = list(criteria)
    if not criteria:
        return {}
    raw = {c["id"]: max(0.0, float(c.get("weight") or 0)) for c in criteria}
    total = sum(raw[c["id"]] for c in criteria)
    if total <= 0:
        return {c["id"]: 1.0 / len(criteria) for c in criteria}
    return {c["id"]: raw[c["id"]] / total for c in criteria}


def score_options(body: dict) -> Dict[str, Any]:
    """Weighted totals, must-have violations and ranking for a body.

    Returns ``{"weights": {cid: share}, "options": {oid: result},
    "ranking": [oid...], "leader_id": oid|None, "tied": bool}`` where each
    result has ``total`` (0..10), ``rank``, ``unscored`` (criteria without a
    score), ``violations`` (must-have constraint ids it fails) and
    ``eligible`` (no violations).
    """
    criteria = body.get("criteria") or []
    options = body.get("options") or []
    must_have = {c["id"] for c in (body.get("constraints") or []) if c.get("must_have")}
    weights = normalized_weights(criteria)

    results = []
    for index, option in enumerate(options):
        scores = option.get("scores") or {}
        total = 0.0
        unscored = 0
        for criterion in criteria:
            score = scores.get(criterion["id"])
            if score is None:
                unscored += 1
                continue
            total += weights[criterion["id"]] * float(score)
        violations = [cid for cid in (option.get("fails") or []) if cid in must_have]
        results.append({
            "option_id": option["id"],
            "total": round4(total),
            "unscored": unscored,
            "violations": violations,
            "eligible": not violations,
            "_index": index,
        })

    ranked = sorted(results, key=lambda r: (not r["eligible"], -r["total"], r["_index"]))
    for rank, result in enumerate(ranked, 1):
        result["rank"] = rank
    eligible = [r for r in ranked if r["eligible"]]
    leader_id = eligible[0]["option_id"] if eligible and criteria else None
    tied = bool(leader_id and len(eligible) > 1 and eligible[0]["total"] == eligible[1]["total"])
    for result in results:
        result.pop("_index")
    return {
        "weights": {cid: round4(share) for cid, share in weights.items()},
        "options": {r["option_id"]: r for r in results},
        "ranking": [r["option_id"] for r in ranked],
        "leader_id": leader_id,
        "tied": tied,
    }


# ---------------------------------------------------------------------------
# Revisit
# ---------------------------------------------------------------------------

def is_revisit_due(revisit_on: Optional[str], status: str, today: str) -> bool:
    """True when a non-archived decision's revisit date is today or past.

    Dates are compared as ``YYYY-MM-DD`` strings in the caller's local day,
    which the client supplies, so "today" means the user's today rather than
    the server's.
    """
    return bool(revisit_on) and status != "archived" and revisit_on <= today


def effective_status(status: str, revisit_on: Optional[str], today: str) -> str:
    return REVISIT_DUE if is_revisit_due(revisit_on, status, today) else status


# ---------------------------------------------------------------------------
# Snapshots ("why it won") and the since-decision comparison
# ---------------------------------------------------------------------------

def build_snapshot(
    body: dict,
    *,
    title: str,
    chosen_option_id: str,
    rationale: str,
    decided_at: str,
    revisit_on: Optional[str],
) -> Dict[str, Any]:
    """Freeze everything needed to explain the choice later.

    The snapshot is a deep copy, so later edits to the working body cannot
    reach it; it is stored once and never updated.
    """
    body = copy.deepcopy(body)
    scoring = score_options(body)
    chosen = next((o for o in body["options"] if o["id"] == chosen_option_id), None)
    if chosen is None:
        raise DecisionValidationError("chosen option does not exist")
    weights = scoring["weights"]
    return {
        "version": 1,
        "title": title,
        "question": body["question"],
        "decided_at": decided_at,
        "chosen_option_id": chosen_option_id,
        "chosen_option_name": chosen["name"],
        "rationale": rationale,
        "revisit_on": revisit_on,
        "constraints": body["constraints"],
        "criteria": [dict(c, share=weights.get(c["id"], 0.0)) for c in body["criteria"]],
        "options": [
            dict(o, **{k: scoring["options"][o["id"]][k]
                       for k in ("total", "rank", "violations", "eligible", "unscored")})
            for o in body["options"]
        ],
        "sources": body["sources"],
        "ranking": scoring["ranking"],
        "leader_id": scoring["leader_id"],
    }


def _names(entries: Iterable[dict]) -> Dict[str, str]:
    return {e["id"]: (e.get("name") or "Untitled") for e in entries}


def compare_to_snapshot(body: dict, snapshot: dict) -> Dict[str, Any]:
    """What changed since a decision was recorded, and would it still win?

    Criteria, options and scores are matched by id, so renaming something is
    not reported as remove + add. ``leader_under_original_weights`` re-scores
    today's options with the criteria and weights in force at decision time
    (using the recorded score for a criterion that has since been deleted),
    which answers "judged by the same criteria, would we still pick it?".
    """
    now = score_options(body)
    then_criteria = snapshot.get("criteria") or []
    then_options = snapshot.get("options") or []
    then_crit_names = _names(then_criteria)
    now_crit_names = _names(body.get("criteria") or [])
    then_opt_names = _names(then_options)
    now_opt_names = _names(body.get("options") or [])
    then_weights = {c["id"]: c.get("share", 0.0) for c in then_criteria}

    weight_changes = []
    for cid, share in then_weights.items():
        if cid in now["weights"] and round4(now["weights"][cid]) != round4(share):
            weight_changes.append({
                "criterion": now_crit_names[cid],
                "before": round4(share),
                "after": now["weights"][cid],
            })

    score_changes = []
    now_options = {o["id"]: o for o in body.get("options") or []}
    for then_option in then_options:
        current = now_options.get(then_option["id"])
        if current is None:
            continue
        for cid in then_crit_names:
            if cid not in now_crit_names:
                continue
            before = (then_option.get("scores") or {}).get(cid)
            after = (current.get("scores") or {}).get(cid)
            if before != after:
                score_changes.append({
                    "option": now_opt_names[current["id"]],
                    "criterion": now_crit_names[cid],
                    "before": before,
                    "after": after,
                })

    # Today's options judged by the decision-time criteria and weights.
    then_scores = {o["id"]: o.get("scores") or {} for o in then_options}
    replay = {
        "constraints": body.get("constraints") or [],
        "criteria": [{"id": c["id"], "name": c.get("name"), "weight": c.get("weight", 0)}
                     for c in then_criteria],
        "options": [],
    }
    for option in body.get("options") or []:
        scores = {}
        for cid in then_crit_names:
            if cid in now_crit_names:
                value = (option.get("scores") or {}).get(cid)
            else:
                value = then_scores.get(option["id"], {}).get(cid)
            if value is not None:
                scores[cid] = value
        replay["options"].append({"id": option["id"], "scores": scores, "fails": option.get("fails") or []})
    replayed = score_options(replay)

    chosen_id = snapshot.get("chosen_option_id")
    chosen_present = chosen_id in now_options
    ranking_before = [then_opt_names.get(oid, "Untitled") for oid in snapshot.get("ranking") or []]
    ranking_after = [now_opt_names[oid] for oid in now["ranking"]]
    leader_before = snapshot.get("leader_id")
    result = {
        "decided_at": snapshot.get("decided_at"),
        "chosen_option_id": chosen_id,
        "chosen_option_name": snapshot.get("chosen_option_name"),
        "chosen_removed": not chosen_present,
        "criteria_added": [now_crit_names[c] for c in now_crit_names if c not in then_crit_names],
        "criteria_removed": [then_crit_names[c] for c in then_crit_names if c not in now_crit_names],
        "weight_changes": weight_changes,
        "options_added": [now_opt_names[o] for o in now_opt_names if o not in then_opt_names],
        "options_removed": [then_opt_names[o] for o in then_opt_names if o not in now_opt_names],
        "score_changes": score_changes,
        "ranking_before": ranking_before,
        "ranking_after": ranking_after,
        "ranking_changed": list(snapshot.get("ranking") or []) != now["ranking"],
        "leader_before": then_opt_names.get(leader_before) if leader_before else None,
        "leader_after": now_opt_names.get(now["leader_id"]) if now["leader_id"] else None,
        "chosen_still_wins": (now["leader_id"] == chosen_id) if chosen_present else None,
        "leader_under_original_weights": (
            now_opt_names.get(replayed["leader_id"]) if replayed["leader_id"] else None
        ),
        "chosen_wins_under_original_weights": (
            (replayed["leader_id"] == chosen_id) if chosen_present else None
        ),
    }
    result["changed"] = bool(
        result["criteria_added"] or result["criteria_removed"] or weight_changes
        or result["options_added"] or result["options_removed"] or score_changes
        or result["ranking_changed"]
    )
    return result


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def _cell(value: Any) -> str:
    return str(value if value is not None else "").replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _inline(value: Any, limit: Optional[int] = None) -> str:
    text = " ".join(str(value or "").split())
    if limit and len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def _md_link(link: dict) -> str:
    title = (link.get("title") or link["url"]).replace("[", "\\[").replace("]", "\\]")
    return f"[{_inline(title)}](<{link['url']}>)"


def _pct(share: float) -> str:
    return f"{math.floor(share * 1000 + 0.5) / 10:g}%"


def _score(value: float) -> str:
    return f"{value:.2f}"


def to_markdown(decision: dict, *, brief: bool = False) -> str:
    """Render a serialized decision (route output shape) as Markdown.

    ``brief`` is the short summary dropped into a chat composer; the full
    form is the export, including every snapshot in the history.
    """
    body = decision
    scoring = decision.get("scoring") or score_options(body)
    criteria = body.get("criteria") or []
    options = body.get("options") or []
    constraints = {c["id"]: c for c in body.get("constraints") or []}
    names = _names(options)
    ranked = [o for oid in scoring["ranking"] for o in options if o["id"] == oid]
    title = decision.get("title") or "Untitled decision"
    chosen_id = decision.get("chosen_option_id")

    if brief:
        lines = [f"I'm working on a decision: **{_inline(title)}**", ""]
        if body.get("question"):
            lines += [f"Question: {_inline(body['question'], 600)}", ""]
        must = [c["text"] for c in constraints.values() if c.get("must_have") and c.get("text")]
        other = [c["text"] for c in constraints.values() if not c.get("must_have") and c.get("text")]
        if must:
            lines.append("Must-haves: " + "; ".join(_inline(t, 160) for t in must))
        if other:
            lines.append("Other constraints: " + "; ".join(_inline(t, 160) for t in other))
        if criteria:
            lines.append("Criteria (share of weight): " + ", ".join(
                f"{_inline(c['name'] or 'Untitled', 60)} {_pct(scoring['weights'].get(c['id'], 0))}" for c in criteria))
        if must or other or criteria:
            lines.append("")
        if ranked:
            lines.append("Options (weighted score out of 10):")
            for index, option in enumerate(ranked, 1):
                result = scoring["options"][option["id"]]
                parts = [f"{index}. {_inline(option['name'] or 'Untitled', 80)} - {_score(result['total'])}"]
                if result["violations"]:
                    parts.append("fails must-have: " + "; ".join(
                        _inline(constraints[c]["text"], 80) for c in result["violations"] if c in constraints))
                if option.get("pros"):
                    parts.append("pros: " + "; ".join(_inline(p, 100) for p in option["pros"][:3]))
                if option.get("cons"):
                    parts.append("cons: " + "; ".join(_inline(p, 100) for p in option["cons"][:3]))
                lines.append(" | ".join(parts))
            lines.append("")
        if chosen_id and chosen_id in names:
            choice = f"Current choice: {_inline(names[chosen_id], 80)}"
            if decision.get("rationale"):
                choice += f" - {_inline(decision['rationale'], 400)}"
            lines.append(choice)
        if decision.get("revisit_on"):
            lines.append(f"Revisit on: {decision['revisit_on']}")
        lines += ["", "Help me pressure-test this decision: what am I missing, "
                      "and does the weighting reflect what matters?"]
        return "\n".join(lines).strip() + "\n"

    status = decision.get("effective_status") or decision.get("status") or "open"
    meta = [f"**Status:** {status}"]
    if decision.get("decided_at"):
        meta.append(f"**Decided:** {decision['decided_at'][:10]}")
    if decision.get("revisit_on"):
        meta.append(f"**Revisit on:** {decision['revisit_on']}")
    lines = [f"# {_inline(title)}", "", " | ".join(meta), ""]
    if body.get("question"):
        lines += ["## Question", "", body["question"], ""]
    if constraints:
        lines += ["## Constraints", ""]
        for c in constraints.values():
            lines.append(f"- {'**Must-have:** ' if c.get('must_have') else ''}{_inline(c['text'])}")
        lines.append("")
    if criteria:
        lines += ["## Criteria", "", "| Criterion | Weight | Share |", "|---|---|---|"]
        for c in criteria:
            lines.append(f"| {_cell(c['name'] or 'Untitled')} | {c['weight']:g} | "
                         f"{_pct(scoring['weights'].get(c['id'], 0))} |")
        lines.append("")
    if ranked:
        header = ["Rank", "Option", "Total"] + [_cell(c["name"] or "Untitled") for c in criteria] + ["Must-have issues"]
        lines += ["## Options", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for option in ranked:
            result = scoring["options"][option["id"]]
            row = [str(result["rank"]), _cell(option["name"] or "Untitled"), _score(result["total"])]
            for c in criteria:
                value = option["scores"].get(c["id"])
                row.append("" if value is None else f"{value:g}")
            row.append(_cell("; ".join(constraints[v]["text"] for v in result["violations"] if v in constraints)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines += ["## Tradeoffs", ""]
        for option in ranked:
            lines += [f"### {_inline(option['name'] or 'Untitled')}", ""]
            if option.get("description"):
                lines += [option["description"], ""]
            for label, key in (("Pros", "pros"), ("Cons", "cons")):
                if option.get(key):
                    lines += [f"{label}:", ""] + [f"- {_inline(p)}" for p in option[key]] + [""]
            if option.get("links"):
                lines += ["Links:", ""] + [f"- {_md_link(link)}" for link in option["links"]] + [""]
    if body.get("sources"):
        lines += ["## Sources", ""] + [f"- {_md_link(link)}" for link in body["sources"]] + [""]
    if chosen_id:
        lines += ["## Decision", "", f"Chosen: **{_inline(names.get(chosen_id, 'removed option'))}**", ""]
        if decision.get("rationale"):
            lines += [decision["rationale"], ""]
    history = decision.get("history") or []
    if history:
        lines += ["## History", ""]
        for snap in history:
            leader_total = next((o["total"] for o in snap.get("options") or []
                                 if o["id"] == snap.get("chosen_option_id")), None)
            weights = ", ".join(f"{_inline(c.get('name') or 'Untitled')} {_pct(c.get('share', 0))}"
                                for c in snap.get("criteria") or [])
            entry = f"- {str(snap.get('decided_at') or '')[:10]}: chose **{_inline(snap.get('chosen_option_name'))}**"
            if leader_total is not None:
                entry += f" ({_score(leader_total)}/10)"
            if weights:
                entry += f"; weights: {weights}"
            if snap.get("rationale"):
                entry += f". Rationale: {_inline(snap['rationale'], 300)}"
            lines.append(entry)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

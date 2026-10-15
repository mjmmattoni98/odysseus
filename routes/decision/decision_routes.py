# routes/decision/decision_routes.py
"""Decision notebook API: options, weighted criteria, recorded choices.

Owner scoping matches the notes routes: ``require_user`` fails closed when
auth is configured but the request carries no identity, and resolves to the
single-user mode (no owner filter) only in the documented anonymous modes.
Another account's decision is always a 404, never a 403, so ids do not leak.

``POST /{id}/decide`` appends a DecisionSnapshot; nothing ever updates one.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from core.database import Decision, DecisionSnapshot, SessionLocal, utcnow_naive
from src import decisions as dm
from src.auth_helpers import require_user

logger = logging.getLogger(__name__)

_MAX_EVENT_UID = 200


# ---------------------------------------------------------------------------
# Request models — shape only; src/decisions.py owns the validation rules.
# ---------------------------------------------------------------------------

class DecisionCreate(BaseModel):
    title: str = ""
    question: str = ""
    constraints: List[Any] = []
    criteria: List[Any] = []
    options: List[Any] = []
    sources: List[Any] = []
    revisit_on: Optional[str] = None


class DecisionUpdate(BaseModel):
    title: Optional[str] = None
    question: Optional[str] = None
    constraints: Optional[List[Any]] = None
    criteria: Optional[List[Any]] = None
    options: Optional[List[Any]] = None
    sources: Optional[List[Any]] = None
    status: Optional[str] = None
    revisit_on: Optional[str] = None
    revisit_event_uid: Optional[str] = None


class DecideRequest(BaseModel):
    option_id: str
    rationale: str = ""
    revisit_on: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(value) -> Optional[str]:
    # Timestamp columns hold naive UTC; say so, or browsers read local time.
    return value.isoformat() + "Z" if value else None


def _today(raw: Optional[str]) -> str:
    """The caller's local date (the client sends it); server date otherwise."""
    if raw:
        try:
            return dm.normalize_revisit_on(raw)
        except dm.DecisionValidationError:
            raise HTTPException(400, "today must be YYYY-MM-DD")
    from src.user_time import now_user_local
    return now_user_local().date().isoformat()


def _load_body(row: Decision) -> Dict[str, Any]:
    try:
        body = json.loads(row.body) if row.body else {}
    except (json.JSONDecodeError, TypeError):
        body = {}
    # Stored bodies were normalized on write; re-normalizing would turn a
    # later-tightened limit into an unreadable decision.
    return {
        "question": body.get("question") or "",
        "constraints": body.get("constraints") or [],
        "criteria": body.get("criteria") or [],
        "options": body.get("options") or [],
        "sources": body.get("sources") or [],
    }


def _snapshots(db, decision_id: str) -> List[DecisionSnapshot]:
    return (db.query(DecisionSnapshot)
            .filter(DecisionSnapshot.decision_id == decision_id)
            .order_by(DecisionSnapshot.seq.asc())
            .all())


def _snapshot_dict(snap: DecisionSnapshot) -> Dict[str, Any]:
    try:
        payload = json.loads(snap.payload)
    except (json.JSONDecodeError, TypeError):
        payload = {}
    return {**payload, "id": snap.id, "seq": snap.seq, "recorded_at": _iso(snap.created_at)}


def _chosen_name(body: dict, chosen_id: Optional[str]) -> Optional[str]:
    if not chosen_id:
        return None
    return next((o.get("name") or "Untitled" for o in body["options"] if o["id"] == chosen_id), None)


def _summary(row: Decision, today: str) -> Dict[str, Any]:
    body = _load_body(row)
    scoring = dm.score_options(body)
    leader = scoring["leader_id"]
    return {
        "id": row.id,
        "title": row.title or "",
        "status": row.status or "open",
        "effective_status": dm.effective_status(row.status or "open", row.revisit_on, today),
        "revisit_due": dm.is_revisit_due(row.revisit_on, row.status or "open", today),
        "revisit_on": row.revisit_on,
        "chosen_option_name": _chosen_name(body, row.chosen_option_id),
        "leader_name": _chosen_name(body, leader),
        "option_count": len(body["options"]),
        "decided_at": _iso(row.decided_at),
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


def _full(db, row: Decision, today: str) -> Dict[str, Any]:
    body = _load_body(row)
    history = [_snapshot_dict(s) for s in _snapshots(db, row.id)]
    since = dm.compare_to_snapshot(body, history[-1]) if history else None
    return {
        **_summary(row, today),
        **body,
        "chosen_option_id": row.chosen_option_id,
        "rationale": row.rationale or "",
        "revisit_event_uid": row.revisit_event_uid,
        "scoring": dm.score_options(body),
        "history": history,
        "since_decision": since,
    }


def _bad_request(exc: dm.DecisionValidationError) -> HTTPException:
    return HTTPException(400, str(exc))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def setup_decision_routes() -> APIRouter:
    router = APIRouter(prefix="/api/decisions", tags=["decisions"])

    def _owner(request: Request) -> Optional[str]:
        # Same contract as the notes routes: 401 with no identity when auth
        # is configured; None (single-user, unfiltered) only in the
        # documented anonymous modes.
        return require_user(request) or None

    def _get_owned(db, decision_id: str, user: Optional[str]) -> Decision:
        row = db.query(Decision).filter(Decision.id == decision_id).first()
        # Strict ownership: a null-owner row is not shared with every account.
        if not row or (user is not None and row.owner != user):
            raise HTTPException(404, "Decision not found")
        return row

    @router.get("")
    def list_decisions(request: Request, today: Optional[str] = None):
        user = _owner(request)
        day = _today(today)
        db = SessionLocal()
        try:
            q = db.query(Decision)
            if user is not None:
                q = q.filter(Decision.owner == user)
            rows = [_summary(r, day) for r in q.order_by(Decision.updated_at.desc()).all()]
            # Revisit-due first, archived last, otherwise most recently edited.
            rows.sort(key=lambda d: (not d["revisit_due"], d["status"] == "archived"))
            return {
                "decisions": rows,
                "revisit_due_count": sum(1 for d in rows if d["revisit_due"]),
                "today": day,
            }
        finally:
            db.close()

    @router.post("")
    def create_decision(request: Request, body: DecisionCreate, today: Optional[str] = None):
        user = _owner(request)
        day = _today(today)
        try:
            title = dm.normalize_title(body.title)
            clean = dm.normalize_body(body.model_dump(exclude={"title", "revisit_on"}))
            revisit_on = dm.normalize_revisit_on(body.revisit_on)
        except dm.DecisionValidationError as exc:
            raise _bad_request(exc)
        db = SessionLocal()
        try:
            row = Decision(
                id=str(uuid.uuid4()),
                owner=user,
                title=title,
                status="open",
                body=json.dumps(clean),
                revisit_on=revisit_on,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return _full(db, row, day)
        finally:
            db.close()

    @router.get("/{decision_id}")
    def get_decision(request: Request, decision_id: str, today: Optional[str] = None):
        user = _owner(request)
        day = _today(today)
        db = SessionLocal()
        try:
            return _full(db, _get_owned(db, decision_id, user), day)
        finally:
            db.close()

    @router.put("/{decision_id}")
    def update_decision(request: Request, decision_id: str, body: DecisionUpdate,
                        today: Optional[str] = None):
        user = _owner(request)
        day = _today(today)
        db = SessionLocal()
        try:
            row = _get_owned(db, decision_id, user)
            fields = body.model_fields_set
            try:
                if body.title is not None:
                    row.title = dm.normalize_title(body.title)
                body_keys = {"question", "constraints", "criteria", "options", "sources"} & fields
                if body_keys:
                    merged = _load_body(row)
                    for key in body_keys:
                        value = getattr(body, key)
                        if value is not None:
                            merged[key] = value
                    row.body = json.dumps(dm.normalize_body(merged))
                if "revisit_on" in fields:
                    row.revisit_on = dm.normalize_revisit_on(body.revisit_on)
                if body.status is not None:
                    status = dm.normalize_status(body.status)
                    if status == "decided" and not row.chosen_option_id:
                        raise dm.DecisionValidationError("record a choice before marking this decided")
                    row.status = status
            except dm.DecisionValidationError as exc:
                raise _bad_request(exc)
            if "revisit_event_uid" in fields:
                uid = (body.revisit_event_uid or "").strip()
                if len(uid) > _MAX_EVENT_UID:
                    raise HTTPException(400, "calendar event id is too long")
                row.revisit_event_uid = uid or None
            row.updated_at = utcnow_naive()
            db.commit()
            db.refresh(row)
            return _full(db, row, day)
        finally:
            db.close()

    @router.delete("/{decision_id}")
    def delete_decision(request: Request, decision_id: str):
        user = _owner(request)
        db = SessionLocal()
        try:
            row = _get_owned(db, decision_id, user)
            db.query(DecisionSnapshot).filter(DecisionSnapshot.decision_id == row.id).delete(
                synchronize_session=False)
            db.delete(row)
            db.commit()
            return {"ok": True}
        finally:
            db.close()

    @router.post("/{decision_id}/decide")
    def decide(request: Request, decision_id: str, body: DecideRequest, today: Optional[str] = None):
        """Record a choice. Each call appends a new snapshot (re-deciding keeps history)."""
        user = _owner(request)
        day = _today(today)
        db = SessionLocal()
        try:
            row = _get_owned(db, decision_id, user)
            current = _load_body(row)
            fields = body.model_fields_set
            try:
                rationale = dm.normalize_rationale(body.rationale)
                revisit_on = (dm.normalize_revisit_on(body.revisit_on)
                              if "revisit_on" in fields else row.revisit_on)
                now = utcnow_naive()
                payload = dm.build_snapshot(
                    current,
                    title=row.title or "",
                    chosen_option_id=body.option_id,
                    rationale=rationale,
                    decided_at=_iso(now),
                    revisit_on=revisit_on,
                )
            except dm.DecisionValidationError as exc:
                raise _bad_request(exc)
            last = _snapshots(db, row.id)
            db.add(DecisionSnapshot(
                id=str(uuid.uuid4()),
                decision_id=row.id,
                owner=row.owner,
                seq=(last[-1].seq + 1) if last else 1,
                payload=json.dumps(payload),
                created_at=now,
            ))
            row.chosen_option_id = body.option_id
            row.rationale = rationale
            row.decided_at = now
            # Only an explicit revisit_on moves the date: a Record that omits
            # it must not touch a date saved concurrently by another request.
            if "revisit_on" in fields:
                row.revisit_on = revisit_on
            row.status = "decided"
            row.updated_at = now
            db.commit()
            db.refresh(row)
            return _full(db, row, day)
        finally:
            db.close()

    @router.get("/{decision_id}/export")
    def export_decision(request: Request, decision_id: str, brief: bool = False,
                        download: bool = False, today: Optional[str] = None):
        user = _owner(request)
        day = _today(today)
        db = SessionLocal()
        try:
            full = _full(db, _get_owned(db, decision_id, user), day)
        finally:
            db.close()
        text = dm.to_markdown(full, brief=brief)
        headers = {}
        if download:
            slug = "".join(ch if ch.isalnum() else "-" for ch in (full["title"] or "decision").lower())
            slug = "-".join(part for part in slug.split("-") if part)[:60] or "decision"
            headers["Content-Disposition"] = f'attachment; filename="{slug}.md"'
        return Response(text, media_type="text/markdown; charset=utf-8", headers=headers)

    return router

"""Backup routes.

Two features share this router and must not be confused:

* ``/api/export`` and ``/api/import`` download and merge a JSON export of
  selected user data (memories, presets, settings, skills, preferences);
* ``/api/backups/...`` manage full server-side backups (src.backup_service):
  database plus file-backed workspace state, a schedule, verification, and a
  guided restore applied at the next startup.
"""

import json
import logging
from datetime import datetime

from fastapi import APIRouter, Body, HTTPException, Request, Response
from fastapi.responses import FileResponse
from core.middleware import INTERNAL_TOOL_HEADER, INTERNAL_TOOL_USER, require_admin
from services.memory import MemoryStoreUnreadable
from src import backup_service
from src.auth_helpers import get_current_user
from src.settings import load_settings, save_settings, load_features, save_features

logger = logging.getLogger(__name__)


def _require_backup_admin(request: Request) -> None:
    """Admin only, and never the agent's loopback identity.

    Backups hold every user's data plus the app key, and a restore replaces
    all of it, so the internal-tool bypass that require_admin grants agent
    tools does not extend to this surface.
    """
    require_admin(request)
    if request.headers.get(INTERNAL_TOOL_HEADER) or getattr(request.state, "current_user", None) == INTERNAL_TOOL_USER:
        raise HTTPException(403, "Backups are managed from Settings only")


def _backup_errors(exc: Exception) -> HTTPException:
    if isinstance(exc, backup_service.BackupBusy):
        return HTTPException(409, str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, "Backup not found")
    return HTTPException(400, str(exc))


def setup_backup_routes(memory_manager, preset_manager, skills_manager) -> APIRouter:
    router = APIRouter(tags=["backup"])

    # ── Server-side backups ──
    # Plain `def` handlers: they touch the filesystem, so FastAPI runs them in
    # its thread pool. Creating, verifying and staging can outlast the request
    # timeout, so those start a background operation the UI polls.

    @router.get("/api/backups")
    def backups_overview(request: Request):
        _require_backup_admin(request)
        return backup_service.overview()

    @router.put("/api/backups/schedule")
    def backups_schedule(request: Request, body: dict = Body(...)):
        _require_backup_admin(request)
        try:
            return {"schedule": backup_service.save_schedule(body), **backup_service.overview()["scheduler"]}
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.get("/api/backups/operation")
    def backups_operation(request: Request):
        _require_backup_admin(request)
        return backup_service.operation_status()

    @router.post("/api/backups")
    def backups_create(request: Request):
        _require_backup_admin(request)
        actor = get_current_user(request)
        try:
            op = backup_service.start_operation(
                "manual", "Backup",
                lambda progress: backup_service.run_backup("manual", actor=actor, progress=progress),
            )
        except Exception as exc:
            raise _backup_errors(exc)
        return {"operation": op}

    @router.post("/api/backups/{name}/verify")
    def backups_verify(name: str, request: Request):
        _require_backup_admin(request)
        try:
            backup_service.archive_path(name)
            op = backup_service.start_operation(
                "verify", "Verify " + name,
                lambda progress: backup_service.verify_backup(name, progress=progress),
            )
        except Exception as exc:
            raise _backup_errors(exc)
        return {"operation": op}

    @router.get("/api/backups/{name}/download")
    def backups_download(name: str, request: Request):
        _require_backup_admin(request)
        try:
            path = backup_service.archive_path(name)
        except Exception as exc:
            raise _backup_errors(exc)
        return FileResponse(path, media_type="application/gzip", filename=name,
                            headers={"Cache-Control": "no-store"})

    @router.delete("/api/backups/{name}")
    def backups_delete(name: str, request: Request):
        _require_backup_admin(request)
        try:
            backup_service.delete_backup(name)
        except Exception as exc:
            raise _backup_errors(exc)
        return {"ok": True}

    @router.get("/api/backups/{name}/restore-preview")
    def backups_restore_preview(name: str, request: Request):
        _require_backup_admin(request)
        try:
            return backup_service.restore_preview(name)
        except Exception as exc:
            raise _backup_errors(exc)

    @router.post("/api/backups/{name}/restore")
    def backups_restore(name: str, request: Request, body: dict = Body(default={})):
        _require_backup_admin(request)
        confirm = str((body or {}).get("confirm") or "")
        actor = get_current_user(request)
        try:
            # Cheap checks up front so a mistyped name is a 400, not a failed job.
            backup_service.check_restore_allowed(name, confirm)
            op = backup_service.start_operation(
                "restore", "Restore " + name,
                lambda progress: backup_service.stage_restore(name, confirm, actor=actor, progress=progress),
            )
        except Exception as exc:
            raise _backup_errors(exc)
        return {"operation": op}

    @router.delete("/api/backups/restore/pending")
    def backups_cancel_restore(request: Request):
        _require_backup_admin(request)
        return {"ok": True, "cancelled": backup_service.cancel_pending_restore()}

    # ── JSON export / import ──

    @router.get("/api/export")
    async def export_data(request: Request):
        """Export all user data as a downloadable JSON file."""
        require_admin(request)
        user = get_current_user(request)

        # Memories (filtered by owner when auth is enabled)
        memories = memory_manager.load(owner=user)

        # Presets (shared across users — export all)
        presets = preset_manager.get_all()

        # Skills (filtered by owner when auth is enabled)
        skills = skills_manager.load(owner=user)

        # Settings
        settings = load_settings()

        # Feature flags
        features = load_features()

        # User preferences
        from routes.prefs_routes import _load_for_user
        preferences = _load_for_user(user)

        export_data = {
            "version": 1,
            "exported_at": datetime.now().isoformat(),
            "exported_by": user,
            "memories": memories,
            "presets": presets,
            "skills": skills,
            "settings": settings,
            "features": features,
            "preferences": preferences,
        }

        filename = f"odysseus_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return Response(
            content=json.dumps(export_data, indent=2, ensure_ascii=False),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @router.post("/api/import")
    async def import_data(request: Request):
        """Import user data from a previously exported JSON file. Merges with existing data."""
        require_admin(request)
        user = get_current_user(request)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid JSON")

        if not isinstance(body, dict):
            raise HTTPException(400, "Expected a JSON object")

        imported = []

        # ── Memories ──
        if "memories" in body and isinstance(body["memories"], list):
            # Strict load: importing on top of an unreadable store would write
            # only the incoming rows back and drop everything already saved.
            try:
                existing = memory_manager.load_all_for_update()
            except MemoryStoreUnreadable as e:
                logger.error("Refusing to import memories: %s", e)
                raise HTTPException(
                    503, "Memory store is temporarily unreadable — nothing was imported."
                )
            # Dedup against THIS user's own memories only. Using every tenant's
            # rows (load_all) meant a memory whose text matched any other
            # user's was silently skipped, so the importing user lost their own
            # data. The full store is still saved back below.
            existing_texts = {e.get("text", "").strip().lower()
                              for e in existing if e.get("owner") == user}
            added = 0
            for mem in body["memories"]:
                if not isinstance(mem, dict) or not mem.get("text"):
                    continue
                if mem["text"].strip().lower() in existing_texts:
                    continue  # skip duplicates
                # Assign owner when auth is enabled
                if user and not mem.get("owner"):
                    mem["owner"] = user
                existing.append(mem)
                existing_texts.add(mem["text"].strip().lower())
                added += 1
            memory_manager.save(existing)
            imported.append(f"{added} memories")

        # ── Skills ──
        if "skills" in body and isinstance(body["skills"], list):
            existing = skills_manager.load_all()
            # Dedup against THIS user's own skills only. Using every tenant's
            # rows (load_all) meant a skill whose id/name/title matched any
            # other user's was silently skipped, so the importing user lost
            # their own data — same cross-tenant bug fixed for memories above.
            # The full store is still saved back below.
            own = [s for s in existing if s.get("owner") == user]
            existing_names = {s.get("name") for s in own if s.get("name")}
            existing_ids = {s.get("id") for s in own if s.get("id")}
            existing_titles = {
                (s.get("title") or s.get("description") or "").strip().lower()
                for s in own
            }
            added = 0
            for skill in body["skills"]:
                if not isinstance(skill, dict):
                    continue
                title = (
                    skill.get("title") or skill.get("description")
                    or skill.get("name") or ""
                ).strip()
                if not title:
                    continue
                sid = skill.get("id") or skill.get("name")
                if sid and sid in existing_ids:
                    continue
                nm = skill.get("name")
                if nm and nm in existing_names:
                    continue
                if title.lower() in existing_titles:
                    continue
                owner = skill.get("owner")
                if user and not owner:
                    owner = user
                # Skills live on disk as SKILL.md files; the old JSON-era
                # skills_manager.save() no longer exists. Write each new skill
                # via add_skill (source="user" skips auto-dedup — this is an
                # explicit backup restore).
                result = skills_manager.add_skill(
                    title=title,
                    name=skill.get("name"),
                    description=skill.get("description"),
                    problem=skill.get("problem", ""),
                    solution=skill.get("solution", ""),
                    steps=skill.get("steps"),
                    tags=skill.get("tags"),
                    source="user",
                    teacher_model=skill.get("teacher_model"),
                    confidence=skill.get("confidence", 0.8),
                    owner=owner,
                    category=skill.get("category", "general"),
                    when_to_use=skill.get("when_to_use"),
                    procedure=skill.get("procedure"),
                    pitfalls=skill.get("pitfalls"),
                    verification=skill.get("verification"),
                    platforms=skill.get("platforms"),
                    requires_toolsets=skill.get("requires_toolsets"),
                    fallback_for_toolsets=skill.get("fallback_for_toolsets"),
                    status=skill.get("status", "draft"),
                    version=skill.get("version", "1.0.0"),
                )
                if result.get("_deduped"):
                    continue
                if result.get("name"):
                    existing_names.add(result["name"])
                if result.get("id"):
                    existing_ids.add(result["id"])
                existing_titles.add(title.lower())
                added += 1
            imported.append(f"{added} skills")

        # ── Presets ──
        if "presets" in body and isinstance(body["presets"], dict):
            current = preset_manager.get_all()
            for key, value in body["presets"].items():
                if isinstance(value, dict):
                    current[key] = value
                elif isinstance(value, list):
                    current[key] = value
            preset_manager.save(current)
            imported.append("presets")

        # ── Settings ──
        if "settings" in body and isinstance(body["settings"], dict):
            current = load_settings()
            current.update(body["settings"])
            save_settings(current)
            imported.append("settings")

        # ── Features ──
        if "features" in body and isinstance(body["features"], dict):
            current = load_features()
            current.update(body["features"])
            save_features(current)
            imported.append("features")

        # ── Preferences ──
        if "preferences" in body and isinstance(body["preferences"], dict):
            from routes.prefs_routes import _load_for_user, _save_for_user
            current = _load_for_user(user)
            current.update(body["preferences"])
            _save_for_user(user, current)
            imported.append("preferences")

        if not imported:
            return {"ok": False, "message": "No recognized data found in the file"}

        return {"ok": True, "imported": imported, "message": f"Imported: {', '.join(imported)}"}

    return router

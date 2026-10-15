"""Scheduled backups, verification records and the guided restore flow.

The archive format and the restore mechanics live in ``src.backup_archive``;
this module adds the parts that need the running app:

* the admin's schedule, kept in settings.json under ``backup_schedule``
  through ``src.settings`` (off / daily / weekly, time of day in an IANA
  timezone, how many scheduled archives to keep);
* ``BACKUP_STATE_FILE`` with the last and next scheduled run, the last result,
  each archive's verification record and the outcome of the last restore;
* a single-flight runner: creating, verifying and staging are long, so the
  routes start them in a worker thread and the UI polls their progress, and
  the scheduler skips a tick rather than run two at once;
* :func:`backup_scheduler_loop`, which ``app.py`` starts next to the task
  scheduler and only when in-process tasks are enabled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from src import backup_archive as archive
from src import constants

logger = logging.getLogger(__name__)

SCHEDULE_SETTING_KEY = "backup_schedule"
FREQUENCIES = ("off", "daily", "weekly")
RETENTION_MIN, RETENTION_MAX = 1, 365
DEFAULT_SCHEDULE = {
    "frequency": "off",
    "time": "03:00",
    "weekday": 6,  # 0 = Monday, matching src.task_scheduler.compute_next_run
    "timezone": "",  # "" = UTC
    "retention": 7,
}
_SCHEDULER_POLL_SECONDS = 60.0


@dataclass(frozen=True)
class BackupPaths:
    data_dir: str
    backups_dir: str
    staging_dir: str
    previous_dir: str
    state_file: str
    db_path: str | None


def default_paths() -> BackupPaths:
    return BackupPaths(
        data_dir=constants.DATA_DIR,
        backups_dir=constants.BACKUPS_DIR,
        staging_dir=constants.RESTORE_STAGING_DIR,
        previous_dir=constants.RESTORE_PREVIOUS_DIR,
        state_file=constants.BACKUP_STATE_FILE,
        db_path=archive.resolve_app_db_path(),
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ── Schedule ────────────────────────────────────────────────────────────────


def normalize_schedule(raw, *, strict: bool = False) -> dict:
    """A complete, valid schedule. ``strict`` raises ValueError on bad input
    (the settings route); otherwise bad fields fall back to their defaults."""
    raw = raw if isinstance(raw, dict) else {}
    out = dict(DEFAULT_SCHEDULE)

    def bad(message):
        if strict:
            raise ValueError(message)

    frequency = str(raw.get("frequency", out["frequency"])).strip().lower()
    if frequency in FREQUENCIES:
        out["frequency"] = frequency
    else:
        bad(f"frequency must be one of {', '.join(FREQUENCIES)}")

    time_value = str(raw.get("time", out["time"])).strip()
    try:
        hour, minute = (int(p) for p in time_value.split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
        out["time"] = f"{hour:02d}:{minute:02d}"
    except ValueError:
        bad("time must be HH:MM")

    try:
        weekday = int(raw.get("weekday", out["weekday"]))
        if not 0 <= weekday <= 6:
            raise ValueError
        out["weekday"] = weekday
    except (TypeError, ValueError):
        bad("weekday must be 0 (Monday) to 6 (Sunday)")

    tz_name = str(raw.get("timezone") or "").strip()
    if tz_name and tz_name.upper() != "UTC":
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(tz_name)
            out["timezone"] = tz_name
        except Exception:
            bad(f"unknown timezone {tz_name!r}")

    try:
        retention = int(raw.get("retention", out["retention"]))
        if not RETENTION_MIN <= retention <= RETENTION_MAX:
            raise ValueError
        out["retention"] = retention
    except (TypeError, ValueError):
        bad(f"retention must be {RETENTION_MIN} to {RETENTION_MAX}")
    return out


def load_schedule() -> dict:
    from src.settings import load_settings
    return normalize_schedule(load_settings().get(SCHEDULE_SETTING_KEY))


def save_schedule(raw) -> dict:
    """Validate and persist the schedule, then recompute the next run."""
    schedule = normalize_schedule(raw, strict=True)
    from src.settings import load_settings, save_settings
    settings = dict(load_settings())
    settings[SCHEDULE_SETTING_KEY] = schedule
    save_settings(settings)
    _reschedule(schedule)
    return schedule


def next_run_after(schedule: dict, after: datetime) -> datetime | None:
    """Next scheduled run strictly after ``after`` (aware UTC), or None when off."""
    if schedule.get("frequency") not in ("daily", "weekly"):
        return None
    from src.task_scheduler import compute_next_run
    naive = compute_next_run(
        schedule["frequency"],
        schedule["time"],
        scheduled_day=schedule.get("weekday"),
        after=after.astimezone(timezone.utc).replace(tzinfo=None),
        tz_name=schedule.get("timezone") or None,
    )
    return naive.replace(tzinfo=timezone.utc) if naive else None


def _signature(schedule: dict) -> str:
    return json.dumps(schedule, sort_keys=True)


def _reschedule(schedule: dict, *, paths: BackupPaths | None = None, now: datetime | None = None) -> None:
    nxt = next_run_after(schedule, now or _now())

    def mutate(state):
        state["next_run"] = archive.iso(nxt) if nxt else None
        state["schedule_signature"] = _signature(schedule)

    update_state(mutate, paths=paths)


# ── State file ──────────────────────────────────────────────────────────────

_STATE_LOCK = threading.RLock()


def load_state(paths: BackupPaths | None = None) -> dict:
    paths = paths or default_paths()
    try:
        state = archive.read_json(paths.state_file)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def update_state(mutate, *, paths: BackupPaths | None = None) -> dict:
    paths = paths or default_paths()
    with _STATE_LOCK:
        state = load_state(paths)
        mutate(state)
        archive.ensure_private_dir(os.path.dirname(paths.state_file))
        archive.write_json_atomic(paths.state_file, state)
        return state


# ── Single-flight operations ────────────────────────────────────────────────


class BackupBusy(RuntimeError):
    """Another backup operation is running."""


_OP_LOCK = threading.Lock()
_OPS_GUARD = threading.Lock()
_ops: dict = {"current": None, "last": None}


def operation_status() -> dict:
    with _OPS_GUARD:
        return {
            "current": dict(_ops["current"]) if _ops["current"] else None,
            "last": dict(_ops["last"]) if _ops["last"] else None,
        }


def _run_operation(kind: str, label: str, work, *, background: bool):
    """Run ``work(progress)`` holding the operation lock.

    Raises BackupBusy when another operation holds it. In the background form
    the op record is returned at once and the result lands in ``last``.
    """
    if not _OP_LOCK.acquire(blocking=False):
        raise BackupBusy("Another backup operation is still running.")
    op = {
        "id": uuid.uuid4().hex[:12], "kind": kind, "label": label, "state": "running",
        "started_at": archive.iso(_now()), "message": "Starting",
    }
    with _OPS_GUARD:
        _ops["current"] = op

    def progress(message: str) -> None:
        with _OPS_GUARD:
            op["message"] = message

    def run():
        try:
            result = work(progress)
            ok = bool(result.get("ok", True)) if isinstance(result, dict) else True
            with _OPS_GUARD:
                op.update(state="done", ok=ok, result=result,
                          message=(result or {}).get("message") or ("Done" if ok else "Failed"))
            return result
        except Exception as exc:
            logger.exception("Backup operation %s failed", kind)
            with _OPS_GUARD:
                op.update(state="failed", ok=False, result=None, message=str(exc) or type(exc).__name__)
            return None
        finally:
            with _OPS_GUARD:
                op["finished_at"] = archive.iso(_now())
                _ops["last"] = op
                _ops["current"] = None
                # Inside the guard: once a poller sees no current operation,
                # the next one can start. (acquire is non-blocking, so no deadlock.)
                _OP_LOCK.release()

    if background:
        threading.Thread(target=run, name=f"backup-{kind}", daemon=True).start()
        with _OPS_GUARD:
            return dict(op)
    return run()


def start_operation(kind: str, label: str, work) -> dict:
    return _run_operation(kind, label, work, background=True)


# ── Archives ────────────────────────────────────────────────────────────────


def archive_path(name: str, paths: BackupPaths | None = None) -> str:
    """Path of an existing archive; ``name`` must be a backup name, nothing else."""
    paths = paths or default_paths()
    if not archive.parse_name(name):
        raise ValueError("Not a backup name")
    path = os.path.join(paths.backups_dir, name)
    if not os.path.isfile(path) or os.path.islink(path):
        raise FileNotFoundError(name)
    return path


def _verification_is_current(record, size, mtime) -> bool:
    return bool(record) and record.get("size") == size and record.get("mtime") == mtime


def _is_restore_point(item: dict, verifications: dict) -> bool:
    """A complete archive whose current bytes passed verification."""
    record = verifications.get(item["name"])
    return (_verification_is_current(record, item["size"], item["mtime"])
            and record.get("ok") is True and record.get("complete") is not False)


def _first_unreadable(unreadable: list, limit: int = 3) -> str:
    shown = ", ".join(unreadable[:limit])
    return shown + (f" and {len(unreadable) - limit} more" if len(unreadable) > limit else "")


def list_backups(paths: BackupPaths | None = None) -> list[dict]:
    paths = paths or default_paths()
    verifications = load_state(paths).get("verifications") or {}
    out = []
    for item in archive.list_archives(paths.backups_dir):
        record = verifications.get(item["name"])
        current = _verification_is_current(record, item["size"], item["mtime"])
        out.append({
            "name": item["name"],
            "kind": item["kind"],
            "created_at": item["created_at"],
            "size": item["size"],
            "verification": _public_verification(record) if current else None,
        })
    return out


def _public_verification(record) -> dict | None:
    if not record:
        return None
    return {k: record.get(k) for k in (
        "ok", "checked_at", "files_checked", "bytes_checked", "problem_count", "problems", "databases",
        "complete", "unreadable",
    )}


def verify_backup(name: str, *, paths: BackupPaths | None = None, progress=None) -> dict:
    """Re-hash every member, integrity-check every database, record the result."""
    paths = paths or default_paths()
    path = archive_path(name, paths)
    (progress or (lambda _m: None))("Verifying " + name)
    st = os.stat(path)
    scratch = tempfile.mkdtemp(prefix=".verify-", dir=paths.backups_dir)
    try:
        report = archive.scan_archive(path, scratch_dir=scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    report.pop("manifest", None)
    record = {**report, "size": st.st_size, "mtime": st.st_mtime}

    def mutate(state):
        state.setdefault("verifications", {})[name] = record

    update_state(mutate, paths=paths)
    summary = _public_verification(record)
    if not report["ok"]:
        summary["message"] = f"Verification of {name} failed: {report['problem_count']} problem(s)."
    else:
        summary["message"] = (
            f"Verified {name}: {report['files_checked']} files match the manifest and every database passed integrity_check."
        )
        if report.get("complete") is False:
            summary["message"] += " The backup is incomplete: some paths could not be read when it was made."
    return summary


def delete_backup(name: str, *, paths: BackupPaths | None = None) -> None:
    paths = paths or default_paths()
    path = archive_path(name, paths)
    plan = archive.pending_restore(paths.staging_dir)
    if plan and name == plan.get("safety_backup"):
        raise ValueError("This is the safety backup of the restore waiting for a restart.")
    os.unlink(path)
    try:
        os.unlink(archive.sidecar_manifest_path(path))
    except OSError:
        pass

    def mutate(state):
        (state.get("verifications") or {}).pop(name, None)

    update_state(mutate, paths=paths)


def prune_scheduled(retention: int, *, paths: BackupPaths | None = None) -> list[str]:
    """Delete the scheduled archives that ``retention`` no longer covers.

    Only restore points count toward ``retention``: complete archives whose
    current bytes passed verification. The newest ``retention`` of them are
    kept. Any other scheduled archive (incomplete, failed or never verified)
    stays until ``retention`` restore points are newer than it, and at most
    ``retention`` of those are kept, so a persistent fault cannot fill the
    disk. Manual and safety backups stay until an admin deletes them.
    """
    paths = paths or default_paths()
    keep = max(retention, RETENTION_MIN)
    verifications = load_state(paths).get("verifications") or {}
    newer_points = newer_others = 0
    doomed = []
    for item in archive.list_archives(paths.backups_dir):  # newest first
        if item["kind"] != "scheduled":
            continue
        if _is_restore_point(item, verifications):
            drop = newer_points >= keep
            newer_points += 1
        else:
            drop = newer_points >= keep or newer_others >= keep
            newer_others += 1
        if drop:
            doomed.append(item)
    removed = []
    for item in doomed:
        try:
            delete_backup(item["name"], paths=paths)
            removed.append(item["name"])
        except (OSError, ValueError) as exc:
            logger.warning("Could not prune backup %s: %s", item["name"], exc)
    return removed


def run_backup(kind: str = "manual", *, actor: str | None = None, paths: BackupPaths | None = None,
               progress=None, retention: int | None = None) -> dict:
    """Create an archive, verify it, and (scheduled only) prune old ones.

    ``ok`` needs a passing verification and a complete archive. A scheduled
    run prunes whether or not it was ok: :func:`prune_scheduled` only drops a
    restore point once ``retention`` newer ones exist, so a broken or
    incomplete run never costs a good backup, and a persistent fault cannot
    pile up incomplete archives.
    """
    paths = paths or default_paths()
    say = progress or (lambda _m: None)
    from src.constants import APP_BUILD_VERSION, APP_VERSION
    archive.ensure_private_dir(paths.backups_dir)
    archive.cleanup_scratch(paths.backups_dir)
    manifest = archive.create_backup(
        kind=kind,
        backups_dir=paths.backups_dir,
        data_dir=paths.data_dir,
        db_path=paths.db_path,
        app_version=APP_VERSION,
        build=APP_BUILD_VERSION,
        created_by=actor,
        skip=(paths.staging_dir, paths.previous_dir),
        progress=say,
    )
    name = manifest["name"]
    verification = verify_backup(name, paths=paths, progress=say)
    complete = archive.manifest_is_complete(manifest)
    unreadable = archive.declared_unreadable(manifest)
    result = {
        "ok": bool(verification["ok"]) and complete,
        "name": name,
        "kind": kind,
        "files": manifest["totals"]["files"],
        "bytes": manifest["totals"]["bytes"],
        "size": os.path.getsize(os.path.join(paths.backups_dir, name)),
        "complete": complete,
        "unreadable": unreadable[:archive.MAX_REPORTED_PROBLEMS],
        "notes": manifest.get("notes") or [],
        "verification": verification,
        "pruned": [],
    }
    if kind == "scheduled" and retention:
        result["pruned"] = prune_scheduled(retention, paths=paths)
    if not verification["ok"]:
        result["message"] = f"Backup {name} was created but failed verification."
    elif not complete:
        result["message"] = (
            f"Backup {name} is incomplete: {len(unreadable)} path(s) could not be read "
            f"({_first_unreadable(unreadable)}). It is kept, but it does not count as a restore point. "
            "Fix the permissions and back up again."
        )
    else:
        result["message"] = f"Backup {name} created and verified."

    def mutate(state):
        state["last_backup"] = {k: result[k] for k in ("ok", "name", "kind", "message")} | {"at": archive.iso(_now())}

    update_state(mutate, paths=paths)
    return result


# ── Guided restore ──────────────────────────────────────────────────────────


def _area(rel: str) -> str:
    return rel.split("/", 1)[0] if "/" in rel else "(top-level files)"


def restore_preview(name: str, *, paths: BackupPaths | None = None) -> dict:
    """What a restore of ``name`` would replace, compared with the current state."""
    paths = paths or default_paths()
    path = archive_path(name, paths)
    manifest = archive.read_manifest(path)
    from src.constants import APP_VERSION

    main = (manifest.get("database") or {}).get("member")
    db_real = os.path.realpath(paths.db_path) if paths.db_path else None
    backup_tables = ((manifest.get("sqlite_files") or {}).get(main) or {}).get("tables") or {}
    current_tables = {}
    if paths.db_path and os.path.isfile(paths.db_path):
        current_tables = archive.describe_sqlite(paths.db_path, integrity=False)["tables"]
    tables = [
        {"name": t, "backup": backup_tables.get(t), "current": current_tables.get(t)}
        for t in sorted(set(backup_tables) | set(current_tables))
    ]

    backup_files = {}
    for entry in manifest.get("files") or []:
        member = entry.get("path") or ""
        if member == main or not member.startswith(archive.DATA_PREFIX + "/"):
            continue
        rel = member[len(archive.DATA_PREFIX) + 1:]
        if not main and db_real and os.path.realpath(os.path.join(paths.data_dir, rel)) == db_real:
            continue  # a stray copy at the live database's path: a files-only restore keeps the database
        backup_files[rel] = int(entry.get("size") or 0)
    current_files = {}
    inv = archive.inventory(paths.data_dir, skip=(paths.backups_dir, paths.staging_dir, paths.previous_dir))
    for abs_path, rel in inv.files:
        if db_real and os.path.realpath(abs_path) == db_real:
            continue
        try:
            current_files[rel] = os.path.getsize(abs_path)
        except OSError:
            continue

    areas: dict[str, dict] = {}
    for label, files in (("backup", backup_files), ("current", current_files)):
        for rel, size in files.items():
            row = areas.setdefault(_area(rel), {"area": _area(rel), "backup_files": 0, "backup_bytes": 0,
                                               "current_files": 0, "current_bytes": 0})
            row[f"{label}_files"] += 1
            row[f"{label}_bytes"] += size

    warnings = []
    if manifest.get("app_version") and manifest.get("app_version") != APP_VERSION:
        warnings.append(
            f"This backup was made by Odysseus {manifest.get('app_version')}; this server runs {APP_VERSION}. "
            "An older database is migrated at startup; a newer one may not load."
        )
    if not archive.manifest_is_complete(manifest):
        unreadable = archive.declared_unreadable(manifest)
        warnings.append(
            f"This backup is incomplete: {len(unreadable)} path(s) could not be read when it was made "
            f"({_first_unreadable(unreadable)}). Files there are not in it, so restoring sets the current ones aside."
        )
    if not main:
        warnings.append("This backup has no app database: only files will be restored, and the current database is kept.")
    elif not paths.db_path:
        warnings.append("DATABASE_URL does not point to a SQLite file here, so this backup cannot be restored.")
    if archive.pending_restore(paths.staging_dir):
        warnings.append("Another restore is already waiting for a restart; confirming replaces it.")

    return {
        "name": name,
        "created_at": manifest.get("created_at"),
        "app_version": manifest.get("app_version"),
        "current_app_version": APP_VERSION,
        "replaces_database": bool(main),
        "tables": tables,
        "areas": sorted(areas.values(), key=lambda r: r["area"]),
        "summary": {
            "replace": len(set(backup_files) & set(current_files)),
            "add": len(set(backup_files) - set(current_files)),
            "set_aside": len(set(current_files) - set(backup_files)),
        },
        "untouched": [{"path": p, "reason": r} for p, r in sorted(archive.excluded_paths().items())],
        "warnings": warnings,
    }


def check_restore_allowed(name: str, confirm: str, *, paths: BackupPaths | None = None) -> str:
    """Raise ValueError unless ``confirm`` is the exact name and the archive
    has a passing verification of its current bytes. Returns its path."""
    paths = paths or default_paths()
    path = archive_path(name, paths)
    if (confirm or "").strip() != name:
        raise ValueError("Type the backup name exactly to confirm the restore.")
    st = os.stat(path)
    record = (load_state(paths).get("verifications") or {}).get(name)
    if not (_verification_is_current(record, st.st_size, st.st_mtime) and record.get("ok")):
        raise ValueError("Verify this backup first: a restore needs a passing verification.")
    if not paths.db_path and (archive.read_manifest(path).get("database") or {}).get("member"):
        raise ValueError("DATABASE_URL does not point to a SQLite file here, so this backup cannot be restored.")
    return path


def stage_restore(name: str, confirm: str, *, actor: str | None = None,
                  paths: BackupPaths | None = None, progress=None) -> dict:
    """Steps 4-6 of the guided restore: confirm, safety backup, stage.

    Refuses unless ``confirm`` is the exact backup name and the archive has a
    passing verification for its current bytes. The staged files are applied
    by :func:`src.backup_archive.apply_pending_restore` at the next startup.
    """
    paths = paths or default_paths()
    say = progress or (lambda _m: None)
    path = check_restore_allowed(name, confirm, paths=paths)

    say("Taking a safety backup of the current state")
    safety = run_backup("safety", actor=actor, paths=paths, progress=say)
    if not safety["ok"]:
        # An incomplete safety backup could not undo the restore it guards.
        if safety["verification"]["ok"]:
            message = (f"The safety backup is incomplete: {_first_unreadable(safety['unreadable'])} could not be "
                       "read, so nothing was staged. Fix the permissions and try again.")
        else:
            message = "The safety backup failed verification, so nothing was staged."
        return {"ok": False, "safety_backup": safety["name"], "unreadable": safety["unreadable"],
                "message": message}

    plan = {
        "backup": name,
        "staged_at": archive.iso(_now()),
        "requested_by": actor,
        "safety_backup": safety["name"],
        # The schedule is configuration of this server, not part of the
        # restored state: the startup hook carries the value it finds then
        # over the restored settings.json.
        "preserve_settings_keys": [SCHEDULE_SETTING_KEY],
    }
    say("Extracting and checking the backup")
    scratch = tempfile.mkdtemp(prefix=".verify-", dir=paths.backups_dir)
    try:
        report = archive.stage_restore(path, staging_dir=paths.staging_dir, scratch_dir=scratch, plan=plan,
                                       db_path=paths.db_path)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if not report["ok"]:
        return {"ok": False, "safety_backup": safety["name"], "problems": report["problems"],
                "message": "The backup failed its checks while being staged, so nothing will be restored."}
    return {
        "ok": True,
        "backup": name,
        "safety_backup": safety["name"],
        "message": f"Restore of {name} is staged. Restart Odysseus to apply it.",
    }


def pending_restore_info(paths: BackupPaths | None = None) -> dict | None:
    paths = paths or default_paths()
    plan = archive.pending_restore(paths.staging_dir)
    if not plan:
        return None
    return {k: plan.get(k) for k in ("backup", "staged_at", "requested_by", "safety_backup")}


def cancel_pending_restore(paths: BackupPaths | None = None) -> bool:
    paths = paths or default_paths()
    return archive.discard_staged_restore(paths.staging_dir, paths.db_path)


# ── Scheduler ───────────────────────────────────────────────────────────────

_scheduler_started = False


def scheduler_active() -> bool:
    """True once this process runs :func:`backup_scheduler_loop`."""
    return _scheduler_started


def run_due_backup(*, now: datetime | None = None, paths: BackupPaths | None = None) -> dict | None:
    """One scheduler tick: run the scheduled backup when it is due.

    A run missed while the server was down fires once at the next tick, then
    the schedule continues from now. A failed run still advances ``next_run``
    so a persistent fault is retried at the next slot, not every minute.
    """
    paths = paths or default_paths()
    now = now or _now()
    schedule = load_schedule()
    state = load_state(paths)
    if schedule["frequency"] == "off":
        if state.get("next_run"):
            _reschedule(schedule, paths=paths, now=now)
        return None
    due = _parse_time(state.get("next_run"))
    if due is None or state.get("schedule_signature") != _signature(schedule):
        _reschedule(schedule, paths=paths, now=now)
        return None
    if now < due:
        return None
    try:
        result = _run_operation(
            "scheduled", "Scheduled backup",
            lambda progress: run_backup("scheduled", actor="scheduler", paths=paths,
                                        progress=progress, retention=schedule["retention"]),
            background=False,
        )
    except BackupBusy:
        return None  # a manual operation is running; try at the next tick
    summary = {
        "ok": bool(result and result.get("ok")),
        "name": (result or {}).get("name"),
        "message": (result or {}).get("message") or "Scheduled backup failed; see the server log.",
        "pruned": (result or {}).get("pruned") or [],
    }
    nxt = next_run_after(schedule, _now())

    def mutate(s):
        s["last_run"] = archive.iso(now)
        s["last_result"] = summary
        s["next_run"] = archive.iso(nxt) if nxt else None
        s["schedule_signature"] = _signature(schedule)

    update_state(mutate, paths=paths)
    return summary


async def backup_scheduler_loop(*, poll_seconds: float = _SCHEDULER_POLL_SECONDS, initial_delay: float = 45.0):
    """In-process scheduled backups. app.py starts this only when
    ODYSSEUS_INPROCESS_TASKS allows the in-process task runner."""
    global _scheduler_started
    _scheduler_started = True
    try:
        await asyncio.sleep(initial_delay)
        while True:
            try:
                await asyncio.to_thread(run_due_backup)
            except Exception:
                logger.exception("Scheduled backup tick failed")
            await asyncio.sleep(poll_seconds)
    finally:
        _scheduler_started = False


def overview(paths: BackupPaths | None = None) -> dict:
    """Everything the Settings > Backups card renders."""
    paths = paths or default_paths()
    state = load_state(paths)
    return {
        "schedule": load_schedule(),
        "scheduler": {
            "active": scheduler_active(),
            "next_run": state.get("next_run"),
            "last_run": state.get("last_run"),
            "last_result": state.get("last_result"),
        },
        "backups": list_backups(paths),
        "operation": operation_status(),
        "pending_restore": pending_restore_info(paths),
        "last_restore": state.get("last_restore"),
        "excluded": [{"path": p, "reason": r} for p, r in sorted(archive.excluded_paths().items())],
    }

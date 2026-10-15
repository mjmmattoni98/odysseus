"""Scheduled backups: archive format, verification, retention, staged restore.

Everything runs against a throwaway data directory under ``tmp_path`` with a
real file-backed SQLite database; no test touches the configured DATA_DIR.
"""
import dataclasses
import errno
import io
import json
import os
import shutil
import sqlite3
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import backup_archive as archive
from src import backup_service as service
from tests.helpers.cli_loader import load_script

PHOTO = bytes(range(256)) * 64


def _make_instance(tmp_path):
    data = tmp_path / "data"
    (data / "uploads").mkdir(parents=True)
    (data / "logs").mkdir()
    db = data / "app.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
    conn.executemany("INSERT INTO notes (body) VALUES (?)", [(f"note {i} " * 20,) for i in range(400)])
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO sessions VALUES ('s1')")
    conn.commit()
    conn.close()
    (data / "uploads" / "photo.bin").write_bytes(PHOTO)
    (data / "settings.json").write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
    (data / "logs" / "app.log").write_text("log line\n", encoding="utf-8")
    paths = service.BackupPaths(
        data_dir=str(data),
        backups_dir=str(data / "backups"),
        staging_dir=str(data / ".restore-staging"),
        previous_dir=str(data / ".restore-previous"),
        state_file=str(data / "backups" / "backup-state.json"),
        db_path=str(db),
    )
    return data, db, paths


def _count(db, table):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _rewrite(archive_path, mutate):
    """Rewrite an archive member by member: mutate(name, bytes) -> bytes | None (drop)."""
    members = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for m in tar.getmembers():
            members.append((m.name, tar.extractfile(m).read()))
    with tarfile.open(archive_path, "w:gz") as tar:
        for name, payload in members:
            payload = mutate(name, payload)
            if payload is None:
                continue
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def _manifest_with(manifest, member, payload):
    """The manifest re-hashed for a replaced member, so only content checks can fail."""
    import hashlib
    for entry in manifest["files"]:
        if entry["path"] == member:
            entry["size"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
    return json.dumps(manifest).encode()


def _backup(paths, kind="manual"):
    return service.run_backup(kind, actor="admin", paths=paths)


def _unreadable(monkeypatch, *targets):
    """Make ``targets`` fail with EACCES the way a permission problem does.

    Patched instead of chmod'ed: permission bits are ignored when the suite
    runs as root (as in the Docker image), and the test must not depend on it.
    """
    blocked = {os.path.abspath(t) for t in targets}
    real_scandir, real_spool = os.scandir, archive._spool

    def scandir(path="."):
        if not isinstance(path, int) and os.path.abspath(path) in blocked:  # shutil.rmtree passes fds
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_scandir(path)

    def spool(path, work_dir):
        if os.path.abspath(path) in blocked:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_spool(path, work_dir)

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(archive, "_spool", spool)


def _apply_restore(paths):
    return archive.apply_pending_restore(
        data_dir=paths.data_dir, db_path=paths.db_path, staging_dir=paths.staging_dir,
        previous_dir=paths.previous_dir, backups_dir=paths.backups_dir, state_file=paths.state_file,
        settings_file=os.path.join(paths.data_dir, "settings.json"),
    )


# ── Archive contents and verification ──


def test_backup_captures_db_and_workspace_files_and_verifies(tmp_path):
    data, db, paths = _make_instance(tmp_path)
    (data / "x.json.tmp.0123456789abcdef0123456789abcdef").write_text("partial", encoding="utf-8")

    result = _backup(paths)

    assert result["ok"], result["verification"]["problems"]
    manifest = archive.read_manifest(os.path.join(paths.backups_dir, result["name"]))
    members = {f["path"] for f in manifest["files"]}
    assert members == {"data/app.db", "data/settings.json", "data/uploads/photo.bin"}
    assert manifest["database"]["member"] == "data/app.db"
    assert manifest["database"]["tables"] == {"notes": 400, "sessions": 1}
    assert manifest["created_at"].endswith("+00:00")
    assert all(len(f["sha256"]) == 64 and f["size"] >= 0 for f in manifest["files"])
    assert oct(os.stat(paths.backups_dir).st_mode & 0o777) == oct(0o700)
    assert oct(os.stat(os.path.join(paths.backups_dir, result["name"])).st_mode & 0o777) == oct(0o600)
    listed = service.list_backups(paths)
    assert listed[0]["name"] == result["name"] and listed[0]["verification"]["ok"] is True


def test_exclusions_cover_caches_logs_runtime_state_and_the_backups_dir():
    excluded = archive.excluded_paths()
    for rel in ("backups", "logs", "tts_cache", "chroma", "bg_jobs", ".restore-staging", ".restore-previous"):
        assert rel in excluded, rel
    assert all(reason for reason in excluded.values())


def test_tampered_member_fails_verification(tmp_path):
    _data, _db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    path = os.path.join(paths.backups_dir, name)

    _rewrite(path, lambda n, b: b"swapped" if n == "data/uploads/photo.bin" else b)
    report = service.verify_backup(name, paths=paths)

    assert report["ok"] is False
    assert any("data/uploads/photo.bin" in p and "does not match" in p for p in report["problems"])
    assert service.list_backups(paths)[0]["verification"]["ok"] is False


def test_corrupted_database_copy_fails_integrity_check(tmp_path):
    _data, _db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    path = os.path.join(paths.backups_dir, name)
    manifest = archive.read_manifest(path)

    def corrupt(member, payload):
        if member == "data/app.db":
            broken = bytearray(payload)
            for page in range(1, len(broken) // 4096):  # every b-tree page header after the first
                broken[page * 4096: page * 4096 + 64] = b"\xff" * 64
            corrupt.payload = bytes(broken)
            return corrupt.payload
        if member == archive.MANIFEST_MEMBER:
            return _manifest_with(manifest, "data/app.db", corrupt.payload)
        return payload

    _rewrite(path, corrupt)
    report = service.verify_backup(name, paths=paths)

    assert report["ok"] is False
    assert not any("does not match" in p for p in report["problems"])  # hashes agree
    assert report["databases"]["data/app.db"]["integrity"] != "ok"
    assert any("integrity_check failed" in p for p in report["problems"])


def test_row_count_drift_fails_verification(tmp_path):
    _data, _db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    path = os.path.join(paths.backups_dir, name)

    def lie(member, payload):
        if member == archive.MANIFEST_MEMBER:
            manifest = json.loads(payload)
            manifest["sqlite_files"]["data/app.db"]["tables"]["notes"] = 999
            return json.dumps(manifest).encode()
        return payload

    _rewrite(path, lie)
    report = service.verify_backup(name, paths=paths)

    assert report["ok"] is False
    assert any("table notes has 400 rows, manifest says 999" in p for p in report["problems"])


@pytest.mark.parametrize("hostile", ["data/../../escape.txt", "/etc/escape.txt", "data/./x", "elsewhere/x"])
def test_path_traversal_member_is_rejected_by_verify_and_stage(tmp_path, hostile):
    _data, _db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    path = os.path.join(paths.backups_dir, name)
    with tarfile.open(path, "r:gz") as tar:
        members = [(m.name, tar.extractfile(m).read()) for m in tar.getmembers()]
    with tarfile.open(path, "w:gz") as tar:
        evil = tarfile.TarInfo(hostile)
        evil.size = 3
        tar.addfile(evil, io.BytesIO(b"bad"))
        for member, payload in members:
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    report = service.verify_backup(name, paths=paths)
    assert report["ok"] is False
    staged = archive.stage_restore(path, staging_dir=paths.staging_dir, scratch_dir=str(tmp_path), plan={})
    assert staged["ok"] is False
    assert not os.path.exists(paths.staging_dir)
    assert not (tmp_path / "escape.txt").exists()


def test_symlink_member_is_rejected(tmp_path):
    _data, _db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    path = os.path.join(paths.backups_dir, name)
    with tarfile.open(path, "w:gz") as tar:
        link = tarfile.TarInfo("data/uploads")
        link.type = tarfile.SYMTYPE
        link.linkname = str(tmp_path)
        tar.addfile(link)

    report = service.verify_backup(name, paths=paths)

    assert report["ok"] is False
    assert any("not a regular file" in p for p in report["problems"])


def test_backup_does_not_follow_symlinks_out_of_the_data_dir(tmp_path):
    data, _db, paths = _make_instance(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not ours", encoding="utf-8")
    os.symlink(outside, data / "linked")

    result = _backup(paths)

    manifest = archive.read_manifest(os.path.join(paths.backups_dir, result["name"]))
    assert not any(f["path"].startswith("data/linked") for f in manifest["files"])
    assert any("symbolic link" in note for note in manifest["notes"])


# ── Retention and schedule ──


def test_retention_prunes_only_the_oldest_scheduled_backups(tmp_path):
    data, db, paths = _make_instance(tmp_path)
    for day in (1, 2, 3):
        archive.create_backup(kind="scheduled", backups_dir=paths.backups_dir, data_dir=str(data),
                              db_path=str(db), app_version="t",
                              created_at=datetime(2026, 10, day, 3, 0, tzinfo=timezone.utc))
    archive.create_backup(kind="manual", backups_dir=paths.backups_dir, data_dir=str(data), db_path=str(db),
                          app_version="t", created_at=datetime(2026, 9, 1, tzinfo=timezone.utc))

    removed = service.prune_scheduled(2, paths=paths)

    assert removed == ["odysseus-backup-20261001T030000Z-scheduled.tar.gz"]
    names = [b["name"] for b in service.list_backups(paths)]
    assert names == [
        "odysseus-backup-20261003T030000Z-scheduled.tar.gz",
        "odysseus-backup-20261002T030000Z-scheduled.tar.gz",
        "odysseus-backup-20260901T000000Z-manual.tar.gz",
    ]
    assert not os.path.exists(os.path.join(paths.backups_dir, "odysseus-backup-20261001T030000Z-scheduled.manifest.json"))


@pytest.mark.parametrize("blocked", ["uploads", "uploads/photo.bin"])
def test_incomplete_scheduled_backup_never_costs_the_last_complete_one(tmp_path, monkeypatch, blocked):
    data, _db, paths = _make_instance(tmp_path)
    first = service.run_backup("scheduled", paths=paths, retention=1)
    assert first["ok"] is True and first["complete"] is True
    _unreadable(monkeypatch, data / blocked)

    second = service.run_backup("scheduled", paths=paths, retention=1)

    assert second["ok"] is False and second["complete"] is False
    assert second["pruned"] == []
    assert [u.split(":")[0] for u in second["unreadable"]] == [f"data/{blocked}"]
    assert "incomplete" in second["message"]
    manifest = archive.read_manifest(os.path.join(paths.backups_dir, second["name"]))
    assert manifest["complete"] is False and manifest["unreadable"] == second["unreadable"]
    listed = {b["name"]: b["verification"] for b in service.list_backups(paths)}
    assert set(listed) == {first["name"], second["name"]}
    assert listed[first["name"]]["complete"] is True
    assert listed[second["name"]]["ok"] is True and listed[second["name"]]["complete"] is False


def test_retention_counts_only_complete_verified_scheduled_backups(tmp_path, monkeypatch):
    data, db, paths = _make_instance(tmp_path)

    def scheduled(day, *, blocked=None):
        with monkeypatch.context() as m:
            if blocked:
                _unreadable(m, data / blocked)
            name = archive.create_backup(kind="scheduled", backups_dir=paths.backups_dir, data_dir=str(data),
                                         db_path=str(db), app_version="t",
                                         created_at=datetime(2026, 10, day, 3, 0, tzinfo=timezone.utc))["name"]
        service.verify_backup(name, paths=paths)
        return name

    day1, day2 = scheduled(1), scheduled(2)
    day3 = scheduled(3, blocked="uploads")

    assert service.prune_scheduled(2, paths=paths) == []  # day3 does not count toward the two kept

    day4 = scheduled(4)

    assert service.prune_scheduled(2, paths=paths) == [day1]
    day5 = scheduled(5)

    # day3 goes once two complete backups are newer than it.
    assert sorted(service.prune_scheduled(2, paths=paths)) == sorted([day2, day3])
    assert [b["name"] for b in service.list_backups(paths)] == [day5, day4]


def test_persistent_read_failure_keeps_at_most_retention_incomplete_backups(tmp_path, monkeypatch):
    data, db, paths = _make_instance(tmp_path)
    _unreadable(monkeypatch, data / "uploads")
    names = []
    for day in (1, 2, 3):
        names.append(archive.create_backup(kind="scheduled", backups_dir=paths.backups_dir, data_dir=str(data),
                                           db_path=str(db), app_version="t",
                                           created_at=datetime(2026, 10, day, 3, 0, tzinfo=timezone.utc))["name"])
        service.verify_backup(names[-1], paths=paths)

    assert service.prune_scheduled(2, paths=paths) == [names[0]]


def test_persistent_incomplete_scheduled_runs_are_pruned_to_retention(tmp_path, monkeypatch):
    data, _db, paths = _make_instance(tmp_path)
    good = service.run_backup("scheduled", paths=paths, retention=1)
    assert good["ok"] is True
    _unreadable(monkeypatch, data / "uploads")

    runs = [service.run_backup("scheduled", paths=paths, retention=1) for _ in range(4)]

    assert all(run["ok"] is False for run in runs)
    assert [b["name"] for b in service.list_backups(paths)] == [runs[-1]["name"], good["name"]]


def test_same_second_backups_list_in_creation_order(tmp_path):
    data, db, paths = _make_instance(tmp_path)
    at = datetime(2026, 10, 1, 3, 0, tzinfo=timezone.utc)
    made = [archive.create_backup(kind="scheduled", backups_dir=paths.backups_dir, data_dir=str(data),
                                  db_path=str(db), app_version="t", created_at=at)["name"] for _ in range(11)]

    assert [a["name"] for a in archive.list_archives(paths.backups_dir)] == made[::-1]
    os.unlink(os.path.join(paths.backups_dir, made[1]))  # pruning frees a sequence number
    made.remove(made[1])
    made.append(archive.create_backup(kind="scheduled", backups_dir=paths.backups_dir, data_dir=str(data),
                                      db_path=str(db), app_version="t", created_at=at)["name"])

    assert archive.list_archives(paths.backups_dir)[0]["name"] == made[-1]


def test_next_run_daily_honours_the_timezone():
    schedule = service.normalize_schedule({"frequency": "daily", "time": "03:00", "timezone": "Europe/Madrid"})
    after = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)

    assert service.next_run_after(schedule, after) == datetime(2026, 10, 13, 1, 0, tzinfo=timezone.utc)


def test_next_run_weekly_and_off():
    weekly = service.normalize_schedule({"frequency": "weekly", "time": "03:00", "weekday": 6})
    monday_noon = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)

    assert service.next_run_after(weekly, monday_noon) == datetime(2026, 10, 18, 3, 0, tzinfo=timezone.utc)
    assert service.next_run_after(service.normalize_schedule({"frequency": "off"}), monday_noon) is None


@pytest.mark.parametrize("bad", [
    {"frequency": "hourly"}, {"time": "25:00"}, {"weekday": 7}, {"retention": 0}, {"timezone": "Mars/Base"},
])
def test_schedule_validation_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        service.normalize_schedule(bad, strict=True)


def test_operation_lock_is_free_once_status_shows_no_operation(monkeypatch):
    """Pollers (the UI, the route tests) start the next operation as soon as
    ``current`` is empty, so the lock must be released in the same critical
    section that empties it. The thread race cannot be forced reliably, so
    this pins the ordering at the release call instead."""
    import threading

    guard_held_at_release = []

    class ProbeLock:
        def __init__(self):
            self._lock = threading.Lock()

        def acquire(self, blocking=True):
            return self._lock.acquire(blocking)

        def release(self):
            guard_held_at_release.append(service._OPS_GUARD.locked())
            self._lock.release()

    monkeypatch.setattr(service, "_OP_LOCK", ProbeLock())

    service._run_operation("verify", "probe", lambda progress: {"ok": True}, background=False)

    assert guard_held_at_release == [True]


def test_due_scheduled_backup_runs_once_and_reschedules(tmp_path, monkeypatch):
    _data, _db, paths = _make_instance(tmp_path)
    schedule = service.normalize_schedule({"frequency": "daily", "time": "03:00", "retention": 2})
    monkeypatch.setattr(service, "load_schedule", lambda: schedule)
    before = datetime(2026, 10, 12, 2, 0, tzinfo=timezone.utc)

    assert service.run_due_backup(now=before, paths=paths) is None  # first tick only schedules
    assert service.load_state(paths)["next_run"] == "2026-10-12T03:00:00+00:00"
    assert service.run_due_backup(now=before, paths=paths) is None  # not due yet

    result = service.run_due_backup(now=datetime(2026, 10, 12, 3, 0, 30, tzinfo=timezone.utc), paths=paths)

    assert result["ok"] is True
    state = service.load_state(paths)
    assert state["last_result"]["name"] == result["name"]
    assert state["last_run"] == "2026-10-12T03:00:30+00:00"
    assert service._parse_time(state["next_run"]) > datetime(2026, 10, 12, 3, 0, 30, tzinfo=timezone.utc)
    assert [b["kind"] for b in service.list_backups(paths)] == ["scheduled"]


# ── Guided restore ──


def test_restore_requires_exact_confirmation_and_a_passing_verification(tmp_path):
    _data, _db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]

    with pytest.raises(ValueError, match="exactly"):
        service.check_restore_allowed(name, "yes", paths=paths)
    _rewrite(os.path.join(paths.backups_dir, name), lambda n, b: b)  # new bytes: verification is stale
    with pytest.raises(ValueError, match="Verify this backup first"):
        service.check_restore_allowed(name, name, paths=paths)
    with pytest.raises(ValueError):
        service.check_restore_allowed("../app.db", "../app.db", paths=paths)


def test_staged_restore_is_applied_by_the_startup_hook(tmp_path, monkeypatch):
    data, db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    # Change everything after the backup.
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM notes WHERE id > 10")
    conn.execute("INSERT INTO sessions VALUES ('later')")
    conn.commit()
    conn.close()
    (data / "uploads" / "photo.bin").unlink()
    (data / "uploads" / "new.txt").write_text("made after the backup", encoding="utf-8")
    (data / "settings.json").write_text(json.dumps({"theme": "light", "backup_schedule": {"frequency": "daily"}}),
                                        encoding="utf-8")

    staged = service.stage_restore(name, name, actor="admin", paths=paths)
    # A schedule saved after staging but before the restart is the one kept.
    (data / "settings.json").write_text(json.dumps({"theme": "light", "backup_schedule": {"frequency": "weekly"}}),
                                        encoding="utf-8")

    assert staged["ok"] is True, staged
    safety = [b for b in service.list_backups(paths) if b["kind"] == "safety"]
    assert [b["name"] for b in safety] == [staged["safety_backup"]]
    assert safety[0]["verification"]["ok"] is True
    assert service.pending_restore_info(paths)["backup"] == name
    assert _count(db, "notes") == 10  # nothing applied before the restart

    result = archive.apply_pending_restore(
        data_dir=str(data), db_path=str(db), staging_dir=paths.staging_dir, previous_dir=paths.previous_dir,
        backups_dir=paths.backups_dir, state_file=paths.state_file, settings_file=str(data / "settings.json"),
    )

    assert result["ok"] is True, result
    assert _count(db, "notes") == 400 and _count(db, "sessions") == 1
    assert (data / "uploads" / "photo.bin").read_bytes() == PHOTO
    assert not (data / "uploads" / "new.txt").exists()
    assert (tmp_path / "data" / ".restore-previous" / "data" / "uploads" / "new.txt").read_text(encoding="utf-8") == "made after the backup"
    settings = json.loads((data / "settings.json").read_text(encoding="utf-8"))
    assert settings == {"theme": "dark", "backup_schedule": {"frequency": "weekly"}}
    assert (data / "logs" / "app.log").read_text(encoding="utf-8") == "log line\n"  # excluded: untouched
    assert not os.path.exists(paths.staging_dir)
    assert not os.path.exists(str(db) + "-wal")
    assert service.load_state(paths)["last_restore"]["ok"] is True
    assert os.path.isfile(os.path.join(paths.backups_dir, staged["safety_backup"]))
    # A second start finds nothing to do.
    assert archive.apply_pending_restore(data_dir=str(data), db_path=str(db), staging_dir=paths.staging_dir,
                                         state_file=paths.state_file) is None


def test_incomplete_safety_backup_stops_the_restore_before_staging(tmp_path, monkeypatch):
    data, _db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    _unreadable(monkeypatch, data / "uploads")

    staged = service.stage_restore(name, name, actor="admin", paths=paths)

    assert staged["ok"] is False and "incomplete" in staged["message"]
    assert staged["unreadable"] and staged["unreadable"][0].startswith("data/uploads:")
    assert service.pending_restore_info(paths) is None
    assert not os.path.exists(paths.staging_dir)


def test_restore_preview_warns_that_a_backup_is_incomplete(tmp_path, monkeypatch):
    data, _db, paths = _make_instance(tmp_path)
    with monkeypatch.context() as m:
        _unreadable(m, data / "uploads")
        name = _backup(paths)["name"]

    preview = service.restore_preview(name, paths=paths)

    assert any("incomplete" in w and "data/uploads" in w for w in preview["warnings"])


def test_hook_refuses_when_a_directory_cannot_be_listed_at_startup(tmp_path, monkeypatch):
    data, db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    assert service.stage_restore(name, name, paths=paths)["ok"] is True
    (data / "uploads" / "photo.bin").write_bytes(b"current photo")
    _unreadable(monkeypatch, data / "uploads")

    result = _apply_restore(paths)

    assert result["ok"] is False and result["rolled_back"] is True, result
    assert "data/uploads" in result["message"]
    assert (data / "uploads" / "photo.bin").read_bytes() == b"current photo"
    assert _count(db, "notes") == 400


def test_hook_refuses_a_target_behind_a_symlink_out_of_the_data_dir(tmp_path, monkeypatch):
    data, db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    assert service.stage_restore(name, name, paths=paths)["ok"] is True
    outside = tmp_path / "outside"
    outside.mkdir()
    for item in (data / "uploads").iterdir():
        item.unlink()
    (data / "uploads").rmdir()
    os.symlink(outside, data / "uploads")

    result = archive.apply_pending_restore(
        data_dir=str(data), db_path=str(db), staging_dir=paths.staging_dir, previous_dir=paths.previous_dir,
        backups_dir=paths.backups_dir, state_file=paths.state_file, settings_file=str(data / "settings.json"),
    )

    assert result["ok"] is False
    assert list(outside.iterdir()) == []
    assert _count(db, "notes") == 400  # data left as it was


def test_failed_post_restore_check_rolls_everything_back(tmp_path, monkeypatch):
    data, db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    assert service.stage_restore(name, name, paths=paths)["ok"] is True
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM notes WHERE id > 5")
    conn.commit()
    conn.close()
    (data / "uploads" / "new.txt").write_text("current", encoding="utf-8")
    monkeypatch.setattr(archive, "describe_sqlite", lambda path, integrity=True: {"integrity": "broken", "tables": {}})

    result = archive.apply_pending_restore(
        data_dir=str(data), db_path=str(db), staging_dir=paths.staging_dir, previous_dir=paths.previous_dir,
        backups_dir=paths.backups_dir, state_file=paths.state_file, settings_file=str(data / "settings.json"),
    )

    assert result["ok"] is False and result["rolled_back"] is True
    assert _count(db, "notes") == 5
    assert (data / "uploads" / "new.txt").read_text(encoding="utf-8") == "current"
    assert (data / "uploads" / "photo.bin").read_bytes() == PHOTO
    assert service.load_state(paths)["last_restore"]["ok"] is False


@pytest.mark.parametrize("layout", ["db_in_data_dir", "stray_db_copy_in_backup", "db_outside_data_dir"])
def test_files_only_restore_keeps_the_live_database(tmp_path, layout):
    data, db, paths = _make_instance(tmp_path)
    if layout == "db_outside_data_dir":
        outside = tmp_path / "db" / "app.db"
        outside.parent.mkdir()
        os.replace(db, outside)
        db, paths = outside, dataclasses.replace(paths, db_path=str(outside))
    # A backup made where DATABASE_URL is not a SQLite file has no app database.
    skip = [paths.staging_dir, paths.previous_dir]
    if layout == "db_in_data_dir":
        skip.append(str(db))
    manifest = archive.create_backup(kind="manual", backups_dir=paths.backups_dir, data_dir=str(data),
                                     db_path=None, app_version="t", skip=skip)
    name = manifest["name"]
    assert manifest["database"]["included"] is False
    assert service.verify_backup(name, paths=paths)["ok"] is True
    preview = service.restore_preview(name, paths=paths)
    assert preview["replaces_database"] is False
    assert preview["summary"] == {"replace": 2, "add": 0, "set_aside": 0}
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO sessions VALUES ('later')")
    conn.commit()
    conn.close()
    (data / "uploads" / "photo.bin").write_bytes(b"changed")
    assert service.stage_restore(name, name, paths=paths)["ok"] is True
    live = sqlite3.connect(db)  # held open across the restore, its last write still in the WAL
    try:
        live.execute("PRAGMA wal_autocheckpoint=0")
        live.execute("INSERT INTO sessions VALUES ('after-staging')")
        live.commit()
        assert os.path.exists(str(db) + "-wal")

        result = _apply_restore(paths)

        assert result["ok"] is True, result
        assert "database" in result["message"]
        assert os.path.exists(str(db) + "-wal")
        assert _count(db, "sessions") == 3 and _count(db, "notes") == 400
    finally:
        live.close()
    assert (data / "uploads" / "photo.bin").read_bytes() == PHOTO
    assert not any(p.name.startswith("app.db") for p in (tmp_path / "data" / ".restore-previous").rglob("*"))


def _symlinked_database(data, db):
    """app.db -> databases/current.db, the real file holding the data."""
    (data / "databases").mkdir()
    real = data / "databases" / "current.db"
    os.replace(db, real)
    os.symlink(os.path.join("databases", "current.db"), db)
    return real


def test_files_only_restore_keeps_a_symlinked_live_database(tmp_path):
    data, db, paths = _make_instance(tmp_path)
    real = _symlinked_database(data, db)
    name = archive.create_backup(kind="manual", backups_dir=paths.backups_dir, data_dir=str(data), db_path=None,
                                 app_version="t", skip=[paths.staging_dir, paths.previous_dir, str(db), str(real)])["name"]
    assert service.verify_backup(name, paths=paths)["ok"] is True
    assert service.stage_restore(name, name, paths=paths)["ok"] is True
    live = sqlite3.connect(db)  # SQLite keeps the WAL next to the real file
    try:
        live.execute("PRAGMA wal_autocheckpoint=0")
        live.execute("INSERT INTO sessions VALUES ('after-staging')")
        live.commit()
        assert os.path.exists(str(real) + "-wal")

        result = _apply_restore(paths)

        assert result["ok"] is True, result
        assert os.path.islink(db) and os.path.exists(str(real) + "-wal")
        assert _count(db, "sessions") == 2
    finally:
        live.close()
    assert not any(p.name.startswith("current.db") for p in (tmp_path / "data" / ".restore-previous").rglob("*"))


def test_database_restore_writes_through_a_symlinked_database_path(tmp_path):
    data, db, paths = _make_instance(tmp_path)
    real = _symlinked_database(data, db)
    name = _backup(paths)["name"]
    manifest = archive.read_manifest(os.path.join(paths.backups_dir, name))
    # Captured once, under its real path: a second copy via the link would land on the same file.
    assert [f["path"] for f in manifest["files"] if f["path"].endswith(".db")] == ["data/databases/current.db"]
    assert manifest["database"]["member"] == "data/databases/current.db"
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM notes WHERE id > 10")
    conn.commit()
    conn.close()
    assert service.stage_restore(name, name, paths=paths)["ok"] is True

    result = _apply_restore(paths)

    assert result["ok"] is True, result
    assert os.path.islink(db) and os.path.realpath(db) == str(real)
    assert _count(db, "notes") == 400


@pytest.mark.parametrize("outcome", ["applied", "rolled_back"])
def test_restore_sets_aside_a_database_that_lives_in_an_excluded_area(tmp_path, monkeypatch, outcome):
    data, db, paths = _make_instance(tmp_path)
    real = data / "logs" / "live.db"  # logs/ is excluded: inventory() never lists it
    os.replace(db, real)
    os.symlink(os.path.join("logs", "live.db"), db)
    name = _backup(paths)["name"]
    assert archive.read_manifest(os.path.join(paths.backups_dir, name))["database"]["member"] == "data/logs/live.db"
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM notes WHERE id > 10")
    conn.commit()
    conn.close()
    assert service.stage_restore(name, name, paths=paths)["ok"] is True
    if outcome == "rolled_back":
        monkeypatch.setattr(archive, "describe_sqlite", lambda path, integrity=True: {"integrity": "broken", "tables": {}})

    result = _apply_restore(paths)

    if outcome == "applied":
        assert result["ok"] is True, result
        assert _count(db, "notes") == 400
        assert _count(tmp_path / "data" / ".restore-previous" / "database" / "live.db", "notes") == 10
    else:
        assert result["ok"] is False and result["rolled_back"] is True, result
        assert _count(db, "notes") == 10
    assert os.path.islink(db)


@pytest.mark.parametrize("protected", [".restore-previous", "backups"])
def test_restore_refuses_a_database_inside_a_protected_directory(tmp_path, protected):
    data, db, paths = _make_instance(tmp_path)
    (data / protected).mkdir(exist_ok=True)
    os.replace(db, data / protected / "live.db")
    os.symlink(os.path.join(protected, "live.db"), db)
    name = _backup(paths)["name"]
    assert service.stage_restore(name, name, paths=paths)["ok"] is True

    result = _apply_restore(paths)

    assert result["ok"] is False and "Nothing was changed" in result["message"], result
    assert protected in result["message"]
    assert _count(db, "notes") == 400 and (data / "uploads" / "photo.bin").read_bytes() == PHOTO


@pytest.mark.parametrize("folder", ["notes ", " notes", "~root"])
def test_restore_keeps_directory_names_exactly_as_archived(tmp_path, folder):
    data, _db, paths = _make_instance(tmp_path)
    (data / folder).mkdir()
    (data / folder / "report.md").write_text("from the backup", encoding="utf-8")
    name = _backup(paths)["name"]
    (data / folder / "report.md").write_text("changed since", encoding="utf-8")
    assert service.stage_restore(name, name, paths=paths)["ok"] is True

    result = _apply_restore(paths)

    assert result["ok"] is True, result
    assert (data / folder / "report.md").read_text(encoding="utf-8") == "from the backup"
    assert sorted(p.name for p in data.iterdir() if "notes" in p.name or "~" in p.name) == [folder]


def test_restore_keeps_file_names_that_differ_only_by_trailing_space(tmp_path):
    data, _db, paths = _make_instance(tmp_path)
    (data / "uploads" / "a.txt").write_text("plain", encoding="utf-8")
    (data / "uploads" / "a.txt ").write_text("spaced", encoding="utf-8")
    name = _backup(paths)["name"]
    assert service.stage_restore(name, name, paths=paths)["ok"] is True

    result = _apply_restore(paths)

    assert result["ok"] is True, result
    assert (data / "uploads" / "a.txt").read_text(encoding="utf-8") == "plain"
    assert (data / "uploads" / "a.txt ").read_text(encoding="utf-8") == "spaced"


def test_restore_writes_each_file_at_its_own_path_despite_a_file_symlink(tmp_path):
    data, _db, paths = _make_instance(tmp_path)
    (data / "a.txt").write_text("A", encoding="utf-8")
    (data / "b.txt").write_text("B", encoding="utf-8")
    name = _backup(paths)["name"]
    (data / "b.txt").unlink()
    os.symlink("a.txt", data / "b.txt")
    assert service.stage_restore(name, name, paths=paths)["ok"] is True

    result = _apply_restore(paths)

    assert result["ok"] is True, result
    assert (data / "a.txt").read_text(encoding="utf-8") == "A"
    assert not os.path.islink(data / "b.txt") and (data / "b.txt").read_text(encoding="utf-8") == "B"
    assert os.path.islink(tmp_path / "data" / ".restore-previous" / "data" / "b.txt")


@pytest.mark.parametrize("protected", ["backups", ".restore-previous"])
def test_restore_refuses_to_write_into_a_protected_area_through_a_symlink(tmp_path, protected):
    data, db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    assert service.stage_restore(name, name, paths=paths)["ok"] is True
    shutil.rmtree(data / "uploads")
    (data / protected).mkdir(exist_ok=True)
    os.symlink(protected, data / "uploads")

    result = _apply_restore(paths)

    assert result["ok"] is False and "Nothing was changed" in result["message"], result
    assert protected in result["message"]
    assert not (data / protected / "photo.bin").exists()
    assert (data / "settings.json").exists() and _count(db, "notes") == 400


def test_restore_refuses_two_members_joined_by_a_directory_symlink(tmp_path):
    data, _db, paths = _make_instance(tmp_path)
    (data / "media").mkdir()
    (data / "media" / "photo.bin").write_bytes(b"media copy")
    name = _backup(paths)["name"]
    shutil.rmtree(data / "uploads")
    os.symlink("media", data / "uploads")
    assert service.stage_restore(name, name, paths=paths)["ok"] is True

    result = _apply_restore(paths)

    assert result["ok"] is False and "Nothing was changed" in result["message"], result
    assert "data/media/photo.bin" in result["message"] and "data/uploads/photo.bin" in result["message"]
    assert (data / "media" / "photo.bin").read_bytes() == b"media copy"


def test_restore_follows_a_relocated_subdirectory_symlink_inside_the_data_dir(tmp_path):
    data, _db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    os.replace(data / "uploads", data / "media")  # relocated, linked back
    os.symlink("media", data / "uploads")
    (data / "media" / "photo.bin").write_bytes(b"changed since")
    assert service.stage_restore(name, name, paths=paths)["ok"] is True

    result = _apply_restore(paths)

    assert result["ok"] is True, result
    assert os.path.islink(data / "uploads")
    assert (data / "media" / "photo.bin").read_bytes() == PHOTO


def _file_where_a_directory_now_stands(tmp_path):
    data, db, paths = _make_instance(tmp_path)
    (data / "notes.md").write_text("from the backup", encoding="utf-8")
    name = _backup(paths)["name"]
    (data / "notes.md").unlink()
    (data / "notes.md" / "sub").mkdir(parents=True)
    (data / "notes.md" / "sub" / "inner.txt").write_text("current", encoding="utf-8")
    os.symlink(data / "uploads", data / "notes.md" / "link")  # never captured, so never set aside on its own
    assert service.stage_restore(name, name, paths=paths)["ok"] is True
    return data, paths


def test_restoring_a_file_where_a_directory_now_stands_sets_the_directory_aside(tmp_path):
    data, paths = _file_where_a_directory_now_stands(tmp_path)

    result = _apply_restore(paths)

    assert result["ok"] is True, result
    assert (data / "notes.md").read_text(encoding="utf-8") == "from the backup"
    aside = tmp_path / "data" / ".restore-previous" / "data" / "notes.md"
    assert (aside / "sub" / "inner.txt").read_text(encoding="utf-8") == "current"
    assert os.path.islink(aside / "link")


def test_failed_restore_puts_back_a_directory_that_stood_where_a_file_goes(tmp_path, monkeypatch):
    data, paths = _file_where_a_directory_now_stands(tmp_path)
    monkeypatch.setattr(archive, "describe_sqlite", lambda path, integrity=True: {"integrity": "broken", "tables": {}})

    result = _apply_restore(paths)

    assert result["ok"] is False and result["rolled_back"] is True, result
    assert (data / "notes.md" / "sub" / "inner.txt").read_text(encoding="utf-8") == "current"
    assert os.path.islink(data / "notes.md" / "link")


def test_restore_never_moves_a_protected_area_out_of_the_way(tmp_path, monkeypatch):
    real = archive.excluded_paths
    monkeypatch.setattr(archive, "excluded_paths", lambda: {**real(), "notes.md/cache": "test: never touched"})
    data, db, paths = _make_instance(tmp_path)
    (data / "notes.md").write_text("from the backup", encoding="utf-8")
    name = _backup(paths)["name"]
    (data / "notes.md").unlink()
    (data / "notes.md" / "cache").mkdir(parents=True)
    (data / "notes.md" / "cache" / "kept.bin").write_bytes(b"cache")
    assert service.stage_restore(name, name, paths=paths)["ok"] is True

    result = _apply_restore(paths)

    assert result["ok"] is False and result["rolled_back"] is True, result
    assert "notes.md/cache" in result["message"]
    assert (data / "notes.md" / "cache" / "kept.bin").read_bytes() == b"cache"


def test_restoring_a_directory_where_a_file_now_stands(tmp_path):
    data, _db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    shutil.rmtree(data / "uploads")
    (data / "uploads").write_text("now a file", encoding="utf-8")
    assert service.stage_restore(name, name, paths=paths)["ok"] is True

    result = _apply_restore(paths)

    assert result["ok"] is True, result
    assert (data / "uploads" / "photo.bin").read_bytes() == PHOTO
    assert (tmp_path / "data" / ".restore-previous" / "data" / "uploads").read_text(encoding="utf-8") == "now a file"


def _rename_fails(exc):
    def replace(src, dst):
        raise exc
    return replace


def _another_filesystem(real):
    """A rename that fails with EXDEV unless it stays in one directory, the
    way it does when the data dir holds a sub-mount."""
    def rename(src, dst):
        if os.path.dirname(os.path.abspath(src)) != os.path.dirname(os.path.abspath(dst)):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real(src, dst)
    return rename


def test_move_copies_only_across_filesystems(tmp_path, monkeypatch):
    """Drives the private mover: a second filesystem (EXDEV) cannot be mounted in a test."""
    monkeypatch.setattr(os, "replace", _another_filesystem(os.replace))
    monkeypatch.setattr(os, "rename", _another_filesystem(os.rename))
    src = tmp_path / "a.txt"
    src.write_text("payload", encoding="utf-8")

    archive._move(str(src), str(tmp_path / "out" / "a.txt"))

    assert (tmp_path / "out" / "a.txt").read_text(encoding="utf-8") == "payload" and not src.exists()
    src.write_text("payload", encoding="utf-8")
    busy = tmp_path / "out" / "busy"  # another directory, so the rename crosses filesystems
    busy.mkdir()
    with pytest.raises(IsADirectoryError):
        archive._move(str(src), str(busy))
    assert src.exists() and list(busy.iterdir()) == []


def _out_of_space_copying(monkeypatch, name):
    """shutil.copyfile writes part of ``name`` and then runs out of space."""
    real_copyfile = shutil.copyfile

    def copyfile(src, dst, *, follow_symlinks=True):
        if os.path.basename(src) == name:
            with open(dst, "wb") as fh:
                fh.write(b"partial")
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_copyfile(src, dst, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(shutil, "copyfile", copyfile)


def test_failed_cross_filesystem_copy_leaves_neither_a_partial_file_nor_a_lost_source(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "replace", _another_filesystem(os.replace))
    monkeypatch.setattr(os, "rename", _another_filesystem(os.rename))
    _out_of_space_copying(monkeypatch, "a.txt")
    src = tmp_path / "a.txt"
    src.write_text("payload", encoding="utf-8")

    with pytest.raises(OSError):
        archive._move(str(src), str(tmp_path / "out" / "a.txt"))

    assert src.read_text(encoding="utf-8") == "payload"
    assert list((tmp_path / "out").iterdir()) == []


def test_rollback_after_a_failed_cross_filesystem_copy_leaves_no_partial_file(tmp_path, monkeypatch):
    data, _db, paths = _make_instance(tmp_path)
    (data / "uploads" / "new.bin").write_bytes(PHOTO)
    name = _backup(paths)["name"]
    (data / "uploads" / "new.bin").unlink()  # only in the backup: nothing comes back over it
    assert service.stage_restore(name, name, paths=paths)["ok"] is True
    payload = os.path.join(paths.staging_dir, archive.PAYLOAD_DIRNAME)
    real_replace, real_rename = os.replace, os.rename

    def across(real):
        def move(src, dst):
            if str(src).startswith(payload):  # the staged files sit on another filesystem
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real(src, dst)
        return move

    monkeypatch.setattr(os, "replace", across(real_replace))
    monkeypatch.setattr(os, "rename", across(real_rename))
    _out_of_space_copying(monkeypatch, "new.bin")

    result = _apply_restore(paths)

    assert result["ok"] is False and result["rolled_back"] is True, result
    assert sorted(p.name for p in (data / "uploads").iterdir()) == ["photo.bin"]
    assert (data / "uploads" / "photo.bin").read_bytes() == PHOTO


@pytest.mark.parametrize("undo", ["works", "fails_once"])
def test_rollback_removes_a_copied_file_whose_staged_source_could_not_be_deleted(tmp_path, monkeypatch, undo):
    data, _db, paths = _make_instance(tmp_path)
    (data / "uploads" / "new.bin").write_bytes(PHOTO)
    name = _backup(paths)["name"]
    (data / "uploads" / "new.bin").unlink()
    assert service.stage_restore(name, name, paths=paths)["ok"] is True
    payload = os.path.join(paths.staging_dir, archive.PAYLOAD_DIRNAME)
    staged_new = os.path.join(payload, "data", "uploads", "new.bin")
    real_replace, real_rename, real_unlink = os.replace, os.rename, os.unlink

    def across(real):
        def move(src, dst):
            if str(src).startswith(payload):  # the staged files sit on another filesystem
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real(src, dst)
        return move

    placed_new = str(data / "uploads" / "new.bin")
    undo_failures = [1] if undo == "fails_once" else []

    def unlink(path, *, dir_fd=None):
        if str(path) == staged_new:
            raise OSError(errno.EIO, "Input/output error", str(path))
        if str(path) == placed_new and undo_failures:  # the mover's own undo fails too
            undo_failures.pop()
            raise OSError(errno.EIO, "Input/output error", str(path))
        return real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "replace", across(real_replace))
    monkeypatch.setattr(os, "rename", across(real_rename))
    monkeypatch.setattr(os, "unlink", unlink)

    result = _apply_restore(paths)

    assert result["ok"] is False and result["rolled_back"] is True, result
    assert sorted(p.name for p in (data / "uploads").iterdir()) == ["photo.bin"]


def test_move_does_not_copy_around_other_rename_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "replace", _rename_fails(PermissionError(errno.EACCES, "Permission denied")))
    src = tmp_path / "a.txt"
    src.write_text("payload", encoding="utf-8")

    with pytest.raises(PermissionError):
        archive._move(str(src), str(tmp_path / "b.txt"))

    assert src.exists() and not (tmp_path / "b.txt").exists()


def test_hook_rejects_staged_files_changed_after_staging(tmp_path, monkeypatch):
    data, db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    assert service.stage_restore(name, name, paths=paths)["ok"] is True
    staged_photo = os.path.join(paths.staging_dir, archive.PAYLOAD_DIRNAME, "data", "uploads", "photo.bin")
    with open(staged_photo, "wb") as fh:
        fh.write(b"tampered")
    (data / "uploads" / "keep.txt").write_text("current", encoding="utf-8")

    result = archive.apply_pending_restore(
        data_dir=str(data), db_path=str(db), staging_dir=paths.staging_dir, previous_dir=paths.previous_dir,
        backups_dir=paths.backups_dir, state_file=paths.state_file, settings_file=str(data / "settings.json"),
    )

    assert result["ok"] is False and "Nothing was changed" in result["message"]
    assert (data / "uploads" / "keep.txt").exists()
    assert not os.path.exists(paths.staging_dir)


# ── Startup hook placement ──


def test_restore_hook_module_does_not_open_the_database():
    """Importing the hook must not import core.database (which opens the DB)."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; import src.backup_archive as a; "
        "a.apply_pending_restore(staging_dir=sys.argv[1]); "
        "bad = [m for m in sys.modules if m == 'core' or m.startswith(('core.', 'sqlalchemy'))]; "
        "assert not bad, bad"
    )
    env = {**os.environ, "PYTHONPATH": str(root)}
    done = subprocess.run([sys.executable, "-c", code, str(root / "does-not-exist")],
                          cwd=root, env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_app_applies_a_staged_restore_before_any_core_import():
    """Source-order check, a deliberate exception to behavior-first testing:
    importing app.py starts the whole server, so the ordering that keeps the
    database closed while files are swapped is pinned on the module AST."""
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8"))
    hook_line = next(
        node.lineno for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", "") == "apply_pending_restore"
    )
    first_core = min(
        node.lineno for node in tree.body
        if (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "core")
        or (isinstance(node, ast.Import) and any(a.name.split(".")[0] == "core" for a in node.names))
    )
    assert hook_line < first_core


# ── Compatibility ──


@pytest.mark.parametrize("url", [
    "sqlite:////srv/odysseus/app.db",
    "sqlite:///relative/app.db",
    "sqlite:///:memory:",
    "sqlite://",
    "sqlite:///file:/srv/odysseus/app.db?uri=true",
    "sqlite:///file:/srv/odysseus/app.db?mode=memory&uri=true",
    "sqlite:///file:shared?mode=memory&cache=shared&uri=true",
    "postgresql://user@db.example/odysseus",
])
def test_db_path_resolver_matches_core_database(url):
    from sqlalchemy.engine import make_url

    import core.database as db

    expected = db._sqlite_db_path(make_url(db._normalize_sqlite_url(url)))
    assert archive.resolve_app_db_path(url) == expected


def test_cli_verifies_and_restores_a_scheduled_backup_archive(tmp_path, monkeypatch):
    data, _db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    cli = load_script("odysseus-backup")
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    monkeypatch.setattr(cli, "_REPO_ROOT", repo)
    monkeypatch.setattr(cli, "_DATA_DIR", repo / "data")
    monkeypatch.setattr(cli, "emit", lambda payload, args: None)
    from types import SimpleNamespace

    cli.cmd_verify(SimpleNamespace(path=os.path.join(paths.backups_dir, name), pretty=False))
    cli.cmd_restore(SimpleNamespace(path=os.path.join(paths.backups_dir, name), yes=True, pretty=False))

    assert (repo / "data" / "uploads" / "photo.bin").read_bytes() == PHOTO
    assert not (repo / "manifest.json").exists()


def test_refused_restore_keeps_a_database_that_moved_into_staging(tmp_path):
    # A restore is staged, then the database ends up inside the staging
    # directory before the restart that applies it. Refusing must not delete
    # the staging directory, which now holds the live database.
    data, db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    assert service.stage_restore(name, name, paths=paths)["ok"] is True
    staging = Path(paths.staging_dir)
    os.replace(db, staging / "live.db")
    os.symlink(os.path.relpath(staging / "live.db", data), db)

    result = _apply_restore(paths)

    assert result["ok"] is False and "Nothing was changed" in result["message"], result
    assert _count(db, "notes") == 400
    assert not (staging / archive.PLAN_FILE).exists()  # disarmed, never retried
    assert _apply_restore(paths) is None


def test_staging_a_restore_refuses_while_the_database_is_in_staging(tmp_path):
    data, db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    staging = Path(paths.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    os.replace(db, staging / "live.db")
    os.symlink(os.path.relpath(staging / "live.db", data), db)

    with pytest.raises(archive.BackupError, match="staging"):
        archive.stage_restore(os.path.join(paths.backups_dir, name), staging_dir=paths.staging_dir,
                              scratch_dir=str(tmp_path), plan={}, db_path=paths.db_path)
    assert _count(db, "notes") == 400
    assert service.cancel_pending_restore(paths) is True and _count(db, "notes") == 400


def test_cancel_keeps_a_live_database_stored_at_the_plan_path(tmp_path):
    data, db, paths = _make_instance(tmp_path)
    staging = Path(paths.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    plan_db = staging / archive.PLAN_FILE
    os.replace(db, plan_db)
    paths = dataclasses.replace(paths, db_path=str(plan_db))

    assert service.cancel_pending_restore(paths) is False
    assert _count(plan_db, "notes") == 400


def test_discard_keeps_a_database_link_that_lives_in_staging(tmp_path):
    # The configured database path is a link inside staging pointing at the
    # real file elsewhere: removing staging would remove the configured path.
    data, db, paths = _make_instance(tmp_path)
    name = _backup(paths)["name"]
    assert service.stage_restore(name, name, paths=paths)["ok"] is True
    link = Path(paths.staging_dir) / "app-link.db"
    os.symlink(db, link)
    paths = dataclasses.replace(paths, db_path=str(link))

    assert service.cancel_pending_restore(paths) is True
    assert os.path.islink(link) and _count(link, "notes") == 400
    assert not (Path(paths.staging_dir) / archive.PLAN_FILE).exists()

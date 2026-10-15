"""Backup archives: snapshot, verify, stage a restore, apply it at startup.

One archive (``<name>.tar.gz``) holds the persisted state of the instance:

    data/<path>       every file under DATA_DIR the backup policy keeps
    database/<file>   the app database, only when DATABASE_URL points outside
                      DATA_DIR (inside it, it is an ordinary ``data/`` member)
    manifest.json     written last, once every hash is known

SQLite files are captured with the online backup API into a scratch copy, so
the archive holds a consistent database while the app keeps writing. Every
other file is read to EOF once, and that exact byte stream is both hashed and
archived. The manifest records each member's size and SHA-256, every SQLite
member's table list with row counts, the creation time and the app version. A
copy sits next to the archive (``<name>.manifest.json``) so listing and the
restore preview do not decompress anything; verification and restore trust
only the copy inside the archive.

A path in scope that cannot be read (a permission problem, say) does not stop
the backup, but the manifest marks the archive incomplete and lists what was
missed. ``src.backup_service`` never counts an incomplete archive as a restore
point: it does not satisfy retention or the restore flow's safety backup.

Stdlib-only apart from ``src.constants`` and ``src.path_confinement`` on
purpose: ``app.py`` calls :func:`apply_pending_restore` before
``core.database`` is imported, because importing that module opens the
database, and replacing a live SQLite file under an open engine corrupts it.
Scheduling, state and the HTTP surface live in ``src.backup_service``.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sqlite3
import tarfile
import tempfile
import uuid
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src import constants
from src.path_confinement import PathEscape

logger = logging.getLogger(__name__)

FORMAT = "odysseus-backup"
FORMAT_VERSION = 1
MANIFEST_MEMBER = "manifest.json"
DATA_PREFIX = "data"
DATABASE_PREFIX = "database"
ARCHIVE_SUFFIX = ".tar.gz"
MANIFEST_SIDECAR_SUFFIX = ".manifest.json"
BACKUP_KINDS = ("manual", "scheduled", "safety")
SQLITE_MAGIC = b"SQLite format 3\x00"
SQLITE_SIDECARS = ("-wal", "-shm", "-journal")

# Archive names are the only handle the HTTP layer accepts, so they double as
# the path-traversal guard: anything that does not match is not a backup.
NAME_RE = re.compile(
    r"^odysseus-backup-(?P<stamp>\d{8}T\d{6}Z)-(?P<kind>manual|scheduled|safety)"
    r"(?:-(?P<seq>\d{1,3}))?\.tar\.gz$"
)

# Files inside a staged restore.
PLAN_FILE = "plan.json"
STAGED_MANIFEST_FILE = "manifest.json"
PAYLOAD_DIRNAME = "payload"
_APPLY_LOCK_FILE = "apply.lock"

_CHUNK = 1024 * 1024
_SPOOL_MEMORY_BYTES = 8 * 1024 * 1024
_MANIFEST_MAX_BYTES = 256 * 1024 * 1024
MAX_REPORTED_PROBLEMS = 50

# Never captured at any depth: OS litter, scratch and lock files, and the
# SQLite sidecars of a database the backup snapshots (the copy is
# self-contained). Sidecars only count when their database sits next to them,
# so an upload that merely ends in "-journal" is still captured.
_TRANSIENT_NAMES = frozenset({"__pycache__", ".DS_Store", "Thumbs.db"})
_TRANSIENT_SUFFIXES = (".tmp", ".temp", ".part", ".partial", ".crdownload", ".swp", ".lock", ".pid")
_ATOMIC_TEMP_RE = re.compile(r"\.tmp\.[0-9a-f]{32}$")  # core.atomic_io sibling temp files


class BackupError(ValueError):
    """An archive, a staged restore or a request about one is not usable."""


# ── Backup scope ────────────────────────────────────────────────────────────


def excluded_paths() -> dict[str, str]:
    """DATA_DIR-relative paths a backup never captures, with the reason.

    Everything else under DATA_DIR is captured, so data added by a new feature
    is covered without being listed anywhere. What is left out: the backups
    and restore working directories themselves, logs, caches and indexes that
    are rebuilt from captured data or a remote server, and runtime records of
    processes that will not exist after a restore. A path overridden by env to
    live outside DATA_DIR is never walked, so it needs no entry.

    Restores leave these paths alone in both directions: they are neither
    replaced from the archive nor set aside.
    """
    entries = (
        (constants.BACKUPS_DIR, "backup archives (the backups directory itself)"),
        (constants.RESTORE_STAGING_DIR, "restore waiting for the next restart"),
        (constants.RESTORE_PREVIOUS_DIR, "files set aside by the last restore"),
        (constants.LOGS_DIR, "application logs"),
        (constants.WEB_CACHE_DIR, "cache: web search results and fetched pages"),
        (constants.TTS_CACHE_DIR, "cache: synthesized speech"),
        (constants.EMOJI_CACHE_DIR, "cache: emoji images"),
        (constants.EMAIL_URGENCY_CACHE_DIR, "cache: email urgency scores"),
        (constants.EMAIL_CACHE_DB, "cache: IMAP message index, re-synced from the mail server"),
        (constants.MAIL_ATTACHMENTS_DIR, "cache: extracted mail attachments"),
        (constants.FASTEMBED_CACHE_DIR, "cache: downloaded embedding models"),
        (constants.CHROMA_DIR, "vector index, rebuilt by re-indexing (ChromaDB runs as a separate service)"),
        (constants.RAG_DIR, "vector index, rebuilt by re-indexing"),
        (constants.MEMORY_VECTORS_DIR, "vector index, rebuilt from memories"),
        (constants.BG_JOBS_DIR, "runtime: background job output of the running process"),
        (constants.BG_JOBS_FILE, "runtime: background job registry"),
        (constants.PROCESS_RESOURCES_DIR, "runtime: process launch records"),
        (constants.BROWSER_RESOURCES_DIR, "runtime: browser session records"),
        (constants.CONTAINMENT_STATE_FILE, "runtime: sandbox grants of running processes"),
    )
    root = os.path.abspath(constants.DATA_DIR)
    out: dict[str, str] = {}
    for path, reason in entries:
        rel = _relative_to(root, path)
        if rel:
            out[rel] = reason
    return out


def _relative_to(root: str, path: str) -> str | None:
    """POSIX path of ``path`` under ``root``, or None when it is not under it."""
    try:
        rel = os.path.relpath(os.path.abspath(path), root)
    except ValueError:  # another Windows drive
        return None
    if rel == "." or rel == ".." or rel.startswith(".." + os.sep):
        return None
    return Path(rel).as_posix()


def _rel_is_excluded(rel: str, excluded) -> bool:
    return any(rel == ex or rel.startswith(ex + "/") for ex in excluded)


def _is_transient(name: str, siblings) -> bool:
    if name in _TRANSIENT_NAMES or name.endswith(_TRANSIENT_SUFFIXES):
        return True
    if name.endswith("~") or name.startswith((".#", ".~")) or _ATOMIC_TEMP_RE.search(name):
        return True
    return any(name.endswith(s) and name[: -len(s)] in siblings for s in SQLITE_SIDECARS)


@dataclass
class Inventory:
    files: list = field(default_factory=list)  # (absolute path, POSIX path under data dir)
    skipped_links: int = 0
    skipped_special: int = 0
    unreadable: list = field(default_factory=list)  # (POSIX path under data dir, "" for itself; reason)


def _data_member(rel: str) -> str:
    """Archive-style name of a path under the data dir, for messages."""
    return f"{DATA_PREFIX}/{rel}" if rel else DATA_PREFIX


def inventory(data_dir: str, *, skip=()) -> Inventory:
    """Files a backup of ``data_dir`` captures (and a restore replaces).

    Symlinks are never followed or captured: one pointing outside the data
    directory would otherwise pull foreign files into the archive, and on
    restore it could redirect a write. ``skip`` adds absolute paths to leave
    out on top of :func:`excluded_paths`. Directories and entries that cannot
    be read land in ``unreadable``; ones removed while listing are ignored.
    """
    root = os.path.abspath(data_dir)
    excluded = set(excluded_paths())
    for path in skip:
        rel = _relative_to(root, path) if path else None
        if rel:
            excluded.add(rel)
    workspace = _relative_to(os.path.abspath(constants.DATA_DIR), constants.AGENT_WORKSPACE_DIR)
    inv = Inventory()
    stack = [("", root)]
    while stack:
        rel_dir, abs_dir = stack.pop()
        try:
            with os.scandir(abs_dir) as it:
                entries = sorted(it, key=lambda e: e.name)
        except FileNotFoundError:
            continue  # removed while listing (or no data directory at all)
        except OSError as exc:
            inv.unreadable.append((rel_dir, exc.strerror or str(exc)))
            continue
        names = {e.name for e in entries}
        for entry in entries:
            rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            if rel in excluded or _is_transient(entry.name, names):
                continue
            try:
                if entry.is_symlink():
                    inv.skipped_links += 1
                    continue
                if entry.is_dir(follow_symlinks=False):
                    # The agent's per-turn scratch directory (its /tmp).
                    in_workspace = workspace and (rel_dir == workspace or rel_dir.startswith(workspace + "/"))
                    if in_workspace and entry.name == constants.AGENT_ISOLATED_TMP_DIRNAME:
                        continue
                    stack.append((rel, entry.path))
                elif entry.is_file(follow_symlinks=False):
                    inv.files.append((entry.path, rel))
                else:
                    inv.skipped_special += 1
            except FileNotFoundError:
                continue
            except OSError as exc:
                inv.unreadable.append((rel, exc.strerror or str(exc)))
    inv.files.sort(key=lambda item: item[1])
    return inv


# ── Database location ───────────────────────────────────────────────────────


def resolve_app_db_path(database_url: str | None = None) -> str | None:
    """Filesystem path of the app's SQLite database, or None when it has none.

    Mirrors ``core.database`` (``_default_database_url``,
    ``_normalize_sqlite_url`` and ``_sqlite_db_path``). It cannot import that
    module: the import opens the database, and the restore hook must run before
    anything does. ``tests/test_backup_archive.py`` pins the two together.
    """
    from sqlalchemy.engine import make_url

    if database_url is None:
        database_url = os.getenv("DATABASE_URL", f"sqlite:///{Path(constants.DATA_DIR) / 'app.db'}")
    try:
        url = make_url(database_url)
    except Exception:
        return None
    if url.get_backend_name() != "sqlite":
        return None
    db_path = url.database
    if not db_path or db_path == ":memory:":
        return None
    db_path = str(db_path)
    is_file_uri = db_path.lower().startswith("file:")
    if not is_file_uri and not os.path.isabs(db_path):
        from src.runtime_paths import get_app_root
        return (Path(get_app_root()) / db_path).resolve().as_posix()
    query = {
        str(key).lower(): str(value).strip().lower()
        for key, value in dict(getattr(url, "query", {}) or {}).items()
    }
    if query.get("uri") not in {"1", "true", "yes", "on"} or not is_file_uri:
        return db_path
    if db_path.lower().startswith("file::memory:") or query.get("mode") == "memory":
        return None
    from urllib.parse import unquote, urlparse
    parsed = urlparse(db_path)
    fs_path = parsed.path or ""
    if not fs_path or fs_path == ":memory:":
        return None
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        fs_path = f"//{parsed.netloc}{fs_path}"
    return unquote(fs_path)


# ── SQLite ──────────────────────────────────────────────────────────────────


def is_sqlite_file(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC
    except OSError:
        return False


def snapshot_sqlite(source_path: str, target_path: str) -> None:
    """Consistent copy of a live SQLite database via the online backup API.

    The copy is switched to rollback-journal mode so the archived file is
    self-contained (no -wal/-shm needed); the app turns WAL back on per
    connection after a restore.
    """
    source = sqlite3.connect(source_path, timeout=30)
    try:
        target = sqlite3.connect(target_path)
        try:
            source.backup(target)
            target.execute("PRAGMA journal_mode=DELETE")
        finally:
            target.close()
    finally:
        source.close()


def describe_sqlite(path: str, *, integrity: bool = True) -> dict:
    """``PRAGMA integrity_check`` verdict and row count per table, read-only."""
    result: dict = {"integrity": None, "tables": {}}
    try:
        conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    except sqlite3.Error as exc:
        result["integrity"] = f"unreadable: {exc}"
        return result
    try:
        if integrity:
            rows = conn.execute("PRAGMA integrity_check").fetchall()
            ok = [tuple(r) for r in rows] == [("ok",)]
            result["integrity"] = "ok" if ok else "; ".join(str(r[0]) for r in rows[:5])
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        for name in names:
            quoted = '"' + name.replace('"', '""') + '"'
            try:
                result["tables"][name] = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            except sqlite3.Error:
                result["tables"][name] = None  # e.g. a virtual table whose module is missing
    except sqlite3.Error as exc:
        result["integrity"] = f"unreadable: {exc}"
    finally:
        conn.close()
    return result


# ── Small file helpers ──────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _chmod(path: str, mode: int) -> None:
    if os.name == "nt":
        return
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def ensure_private_dir(path: str) -> str:
    """Create ``path`` (0o700) if needed. Raises OSError when it cannot."""
    os.makedirs(path, mode=0o700, exist_ok=True)
    _chmod(path, 0o700)
    return path


def write_json_atomic(path: str, data, *, mode: int = 0o600) -> None:
    """tmp + fsync + replace. A local copy of core.atomic_io's write because
    importing anything under ``core`` opens the database (see module doc)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        _chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def read_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sidecar_manifest_path(archive_path: str) -> str:
    return archive_path[: -len(ARCHIVE_SUFFIX)] + MANIFEST_SIDECAR_SUFFIX


def parse_name(name: str) -> dict | None:
    match = NAME_RE.match(name or "")
    if not match:
        return None
    created = datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return {"name": name, "kind": match.group("kind"), "created_at": iso(created),
            "seq": int(match.group("seq") or 0)}


def list_archives(backups_dir: str) -> list[dict]:
    """Archives in ``backups_dir``, newest first."""
    out = []
    try:
        entries = list(os.scandir(backups_dir))
    except OSError:
        return out
    for entry in entries:
        info = parse_name(entry.name)
        if not info:
            continue
        try:
            if not entry.is_file(follow_symlinks=False):
                continue
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        info.update(size=st.st_size, mtime=st.st_mtime, path=entry.path)
        out.append(info)
    # The sequence number orders archives made within one second; comparing
    # names would put "-1" after the unnumbered first one and "-10" before "-9".
    out.sort(key=lambda a: (a["created_at"], a["seq"]), reverse=True)
    return out


def _unique_name(backups_dir: str, created_at: datetime, kind: str) -> str:
    stamp = created_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"odysseus-backup-{stamp}-{kind}"
    # One past the highest sequence in use, not the first free one:
    # list_archives orders same-second archives by sequence, so a number freed
    # by pruning must not go to a newer archive.
    try:
        names = os.listdir(backups_dir)
    except OSError:
        names = []
    taken = [info["seq"] for info in map(parse_name, names)
             if info and (info["name"] == base + ARCHIVE_SUFFIX or info["name"].startswith(base + "-"))]
    seq = max(taken) + 1 if taken else 0
    name = f"{base}{'' if seq == 0 else f'-{seq}'}{ARCHIVE_SUFFIX}"
    if seq > 999 or os.path.lexists(os.path.join(backups_dir, name)):
        raise BackupError("too many backups with the same timestamp")
    return name


def cleanup_scratch(backups_dir: str) -> None:
    """Drop work directories and partial archives a killed run left behind."""
    try:
        entries = list(os.scandir(backups_dir))
    except OSError:
        return
    for entry in entries:
        if not entry.name.startswith((".work-", ".verify-", ".partial-")):
            continue
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path, ignore_errors=True)
            else:
                os.unlink(entry.path)
        except OSError:
            pass


# ── Creating a backup ───────────────────────────────────────────────────────


def _member_info(arcname: str, size: int, mtime: float) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arcname)
    info.size = size
    info.mtime = int(mtime)
    info.mode = 0o600
    info.type = tarfile.REGTYPE
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _spool(path: str, work_dir: str):
    """Read ``path`` to EOF once; return (spool, size, sha256) of those bytes."""
    digest = hashlib.sha256()
    spool = tempfile.SpooledTemporaryFile(max_size=_SPOOL_MEMORY_BYTES, dir=work_dir)
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK), b""):
                digest.update(chunk)
                spool.write(chunk)
        size = spool.tell()
        spool.seek(0)
    except BaseException:
        spool.close()
        raise
    return spool, size, digest.hexdigest()


def create_backup(
    *,
    kind: str,
    backups_dir: str,
    data_dir: str,
    db_path: str | None,
    app_version: str,
    build: str = "",
    created_by: str | None = None,
    created_at: datetime | None = None,
    skip=(),
    progress=None,
) -> dict:
    """Write one archive into ``backups_dir`` and return its manifest.

    Raises OSError / BackupError; on failure nothing is left in
    ``backups_dir`` except what :func:`cleanup_scratch` removes. A file or
    directory in scope that cannot be read is listed under ``unreadable`` and
    sets ``complete`` to False (the app database itself must be readable).
    """
    if kind not in BACKUP_KINDS:
        raise BackupError(f"unknown backup kind {kind!r}")
    say = progress or (lambda _msg: None)
    created_at = created_at or _utcnow()
    ensure_private_dir(backups_dir)
    name = _unique_name(backups_dir, created_at, kind)
    final_path = os.path.join(backups_dir, name)
    work_dir = tempfile.mkdtemp(prefix=".work-", dir=backups_dir)
    partial = os.path.join(backups_dir, f".partial-{uuid.uuid4().hex}")
    data_root = os.path.abspath(data_dir)
    skip = tuple(skip) + (backups_dir,)

    say("Listing files")
    inv = inventory(data_root, skip=skip)
    # The real file, not a symlink to it: inventory() lists the target, so a
    # link path would capture the database twice (and restore it twice).
    db_abs = os.path.realpath(db_path) if db_path else None
    main_member = None
    database: dict
    if not db_abs:
        database = {"included": False, "reason": "DATABASE_URL is not a file-backed SQLite database"}
    elif not os.path.isfile(db_abs):
        database = {"included": False, "reason": "the database file does not exist yet"}
    else:
        rel = _relative_to(os.path.realpath(data_root), db_abs)
        main_member = f"{DATA_PREFIX}/{rel}" if rel else f"{DATABASE_PREFIX}/{os.path.basename(db_abs)}"
        database = {"included": True, "member": main_member}

    files: list[dict] = []
    sqlite_files: dict[str, dict] = {}
    notes: list[str] = []
    unreadable = [f"{_data_member(rel)}: {reason}" for rel, reason in inv.unreadable]
    total = 0
    try:
        with tarfile.open(partial, "w:gz", compresslevel=6, format=tarfile.PAX_FORMAT) as tar:
            sources = []
            if main_member:
                sources.append((db_abs, main_member))
            for abs_path, rel in inv.files:
                arcname = f"{DATA_PREFIX}/{rel}"
                if arcname != main_member:
                    sources.append((abs_path, arcname))
            for index, (abs_path, arcname) in enumerate(sources, 1):
                if index == 1 or index % 200 == 0:
                    say(f"Archiving file {index} of {len(sources)}")
                try:
                    st = os.stat(abs_path)
                    if is_sqlite_file(abs_path):
                        copy = os.path.join(work_dir, f"db-{index}.sqlite")
                        try:
                            snapshot_sqlite(abs_path, copy)
                        except sqlite3.Error as exc:
                            # Not snapshottable (corrupt, locked past the timeout):
                            # keep the raw bytes; verification reports the result.
                            notes.append(f"{arcname}: copied without the SQLite backup API ({exc})")
                            shutil.copyfile(abs_path, copy)
                        sqlite_files[arcname] = describe_sqlite(copy)
                        spool, size, sha = _spool(copy, work_dir)
                        os.unlink(copy)
                    else:
                        spool, size, sha = _spool(abs_path, work_dir)
                except FileNotFoundError:
                    continue  # deleted between listing and reading
                except OSError as exc:
                    if arcname == main_member:
                        raise
                    unreadable.append(f"{arcname}: {exc.strerror or exc}")
                    continue
                with spool:
                    tar.addfile(_member_info(arcname, size, st.st_mtime), spool)
                files.append({
                    "path": arcname, "size": size, "sha256": sha,
                    "mode": st.st_mode & 0o777, "mtime": int(st.st_mtime),
                })
                total += size
            if main_member and main_member in sqlite_files:
                database.update(sqlite_files[main_member])

            if inv.skipped_links:
                notes.append(f"{inv.skipped_links} symbolic link(s) were not followed")
            if inv.skipped_special:
                notes.append(f"{inv.skipped_special} special file(s) (sockets, pipes) were skipped")

            manifest = {
                "format": FORMAT,
                "format_version": FORMAT_VERSION,
                "name": name,
                "kind": kind,
                "created_at": iso(created_at),
                "created_by": created_by,
                "app_version": app_version,
                "build": build,
                "database": database,
                "sqlite_files": sqlite_files,
                "files": files,
                "totals": {"files": len(files), "bytes": total},
                "excluded": excluded_paths(),
                "complete": not unreadable,
                "unreadable": unreadable,
                "notes": notes,
            }
            payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            tar.addfile(_member_info(MANIFEST_MEMBER, len(payload), created_at.timestamp()),
                        io.BytesIO(payload))
        _chmod(partial, 0o600)
        write_json_atomic(sidecar_manifest_path(final_path), manifest)
        os.replace(partial, final_path)
    except BaseException:
        for leftover in (partial, sidecar_manifest_path(final_path)):
            try:
                os.unlink(leftover)
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return manifest


# ── Reading an archive ──────────────────────────────────────────────────────


def manifest_is_complete(manifest) -> bool:
    """False when the backup could not read part of its scope."""
    return isinstance(manifest, dict) and manifest.get("complete") is not False


def declared_unreadable(manifest) -> list[str]:
    """The ``path: reason`` entries a backup recorded as unreadable."""
    listed = manifest.get("unreadable") if isinstance(manifest, dict) else None
    return [str(item) for item in listed] if isinstance(listed, list) else []


def validate_member_name(name) -> str:
    """Accept only the three member shapes this format writes.

    Absolute paths, ``..``/``.``/empty components, backslashes and NUL are
    refused outright; extraction additionally confines every target with
    :func:`_confine_member`.
    """
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise BackupError(f"unsafe archive entry name {name!r}")
    if name.startswith("/") or (os.name == "nt" and ":" in name):
        raise BackupError(f"archive entry {name!r} is an absolute path")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise BackupError(f"archive entry {name!r} has an empty, '.' or '..' component")
    if name == MANIFEST_MEMBER:
        return name
    if parts[0] == DATA_PREFIX and len(parts) >= 2:
        return name
    if parts[0] == DATABASE_PREFIX and len(parts) == 2:
        return name
    raise BackupError(f"unexpected archive entry {name!r}")


def _confine_member(root: str, rel: str) -> str:
    """``rel`` (a validated, ``/``-separated archive path; "" for the root)
    resolved under ``root``, symlinks followed, or PathEscape.

    Unlike :func:`src.path_confinement.confine`, the name is taken exactly as
    archived: no whitespace stripping and no ``~`` expansion, so a directory
    called ``"notes "`` or ``"~root"`` restores to itself and not to
    ``notes`` or ``/root``.
    """
    base = os.path.realpath(root)
    resolved = os.path.realpath(os.path.join(base, *rel.split("/"))) if rel else base
    try:
        inside = os.path.commonpath([os.path.normcase(resolved), os.path.normcase(base)]) == os.path.normcase(base)
    except ValueError:  # another Windows drive
        inside = False
    if not inside:
        raise PathEscape(base, rel)
    return resolved


def validate_member(member: tarfile.TarInfo) -> None:
    validate_member_name(member.name)
    if not member.isreg():
        raise BackupError(
            f"archive entry {member.name!r} is not a regular file "
            "(links, directories and devices are refused)"
        )


def read_manifest(archive_path: str) -> dict:
    """The sidecar manifest, or the archive's own when the sidecar is missing."""
    sidecar = sidecar_manifest_path(archive_path)
    try:
        manifest = read_json(sidecar)
        if isinstance(manifest, dict) and manifest.get("format") == FORMAT:
            return manifest
    except (OSError, ValueError):
        pass
    with tarfile.open(archive_path, mode="r|gz") as tar:
        for member in tar:
            validate_member(member)
            if member.name == MANIFEST_MEMBER:
                if member.size > _MANIFEST_MAX_BYTES:
                    raise BackupError("manifest is too large")
                return json.loads(tar.extractfile(member).read())
    raise BackupError("archive has no manifest")


def scan_archive(archive_path: str, *, scratch_dir: str, extract_root: str | None = None) -> dict:
    """Re-hash every member against the manifest and check every database.

    Each SQLite member is written out (to ``scratch_dir``, or into
    ``extract_root`` when extracting) and gets ``PRAGMA integrity_check`` plus
    a table-by-table row-count comparison against the manifest. With
    ``extract_root`` every member is extracted there, confined to it.

    Returns a report whose ``ok`` is True only when nothing was wrong; the
    parsed manifest rides along under ``manifest`` for callers that need it.
    ``complete`` and ``unreadable`` repeat what the archive's own manifest
    declares: ``ok`` speaks for the archive's integrity, not its coverage.
    """
    problems: list[str] = []
    seen: dict[str, tuple[int, str]] = {}
    sqlite_copies: dict[str, str] = {}
    manifest = None

    def finish(databases=None) -> dict:
        return {
            "ok": not problems,
            "checked_at": iso(_utcnow()),
            "files_checked": len(seen),
            "bytes_checked": sum(size for size, _ in seen.values()),
            "problem_count": len(problems),
            "problems": problems[:MAX_REPORTED_PROBLEMS],
            "databases": databases or {},
            "complete": manifest_is_complete(manifest) if isinstance(manifest, dict) else None,
            "unreadable": declared_unreadable(manifest)[:MAX_REPORTED_PROBLEMS],
            "manifest": manifest,
        }

    try:
        with tarfile.open(archive_path, mode="r|gz") as tar:
            for member in tar:
                try:
                    validate_member(member)
                except BackupError as exc:
                    problems.append(str(exc))
                    return finish()
                name = member.name
                if name in seen or (name == MANIFEST_MEMBER and manifest is not None):
                    problems.append(f"{name}: appears twice in the archive")
                    return finish()
                handle = tar.extractfile(member)
                if name == MANIFEST_MEMBER:
                    if member.size > _MANIFEST_MAX_BYTES:
                        problems.append("manifest.json is implausibly large")
                        return finish()
                    try:
                        manifest = json.loads(handle.read())
                    except ValueError as exc:
                        problems.append(f"manifest.json is not valid JSON: {exc}")
                        return finish()
                    continue
                digest = hashlib.sha256()
                size = 0
                first = handle.read(_CHUNK)
                target = None
                if extract_root is not None:
                    try:
                        target = _confine_member(extract_root, name)
                    except PathEscape:
                        problems.append(f"{name}: resolves outside the restore directory")
                        return finish()
                    if os.path.lexists(target):
                        problems.append(f"{name}: extraction target already exists")
                        return finish()
                elif first.startswith(SQLITE_MAGIC):
                    target = os.path.join(scratch_dir, f"db-{len(sqlite_copies)}.sqlite")
                out = None
                if target is not None:
                    os.makedirs(os.path.dirname(target), mode=0o700, exist_ok=True)
                    out = open(target, "xb")
                try:
                    chunk = first
                    while chunk:
                        digest.update(chunk)
                        size += len(chunk)
                        if out is not None:
                            out.write(chunk)
                        chunk = handle.read(_CHUNK)
                finally:
                    if out is not None:
                        out.close()
                if first.startswith(SQLITE_MAGIC):
                    sqlite_copies[name] = target
                seen[name] = (size, digest.hexdigest())
    except (tarfile.TarError, EOFError, zlib.error, OSError, ValueError) as exc:
        problems.append(f"archive is unreadable: {exc}")
        return finish()

    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        problems.append("manifest.json is missing or not an Odysseus backup manifest")
        return finish()
    entries = manifest.get("files")
    if not isinstance(entries, list):
        problems.append("manifest.json has no file list")
        return finish()

    expected: dict[str, tuple] = {}
    for entry in entries:
        try:
            name = validate_member_name(entry.get("path") if isinstance(entry, dict) else None)
            expected[name] = (int(entry["size"]), str(entry["sha256"]))
        except (BackupError, KeyError, TypeError, ValueError) as exc:
            problems.append(f"manifest entry is malformed: {exc}")
    for name in sorted(set(expected) - set(seen)):
        problems.append(f"{name}: listed in the manifest but missing from the archive")
    for name in sorted(set(seen) - set(expected)):
        problems.append(f"{name}: in the archive but not listed in the manifest")
    for name in sorted(set(seen) & set(expected)):
        if seen[name] != expected[name]:
            problems.append(
                f"{name}: content does not match the manifest "
                f"({seen[name][0]} bytes, manifest says {expected[name][0]})"
            )

    databases = {}
    declared = manifest.get("sqlite_files") or {}
    main = (manifest.get("database") or {}).get("member")
    if main and main not in declared:
        problems.append(f"{main}: the app database has no recorded table list")
    for name, info in sorted(declared.items()):
        copy = sqlite_copies.get(name)
        if copy is None:
            problems.append(f"{name}: database listed in the manifest but not found in the archive")
            continue
        actual = describe_sqlite(copy)
        db_problems = []
        if actual["integrity"] != "ok":
            db_problems.append(f"integrity_check failed: {actual['integrity']}")
        want = (info or {}).get("tables") or {}
        got = actual["tables"]
        for table in sorted(set(want) | set(got)):
            if table not in got:
                db_problems.append(f"table {table} is missing")
            elif table not in want:
                db_problems.append(f"table {table} is not in the manifest")
            elif want[table] is not None and want[table] != got[table]:
                db_problems.append(f"table {table} has {got[table]} rows, manifest says {want[table]}")
        databases[name] = {
            "integrity": actual["integrity"],
            "tables": len(got),
            "rows": sum(v for v in got.values() if isinstance(v, int)),
            "problems": db_problems,
        }
        problems.extend(f"{name}: {p}" for p in db_problems)
    return finish(databases=databases)


# ── Staging a restore ───────────────────────────────────────────────────────


def pending_restore(staging_dir: str) -> dict | None:
    try:
        plan = read_json(os.path.join(staging_dir, PLAN_FILE))
    except (OSError, ValueError):
        return None
    return plan if isinstance(plan, dict) else None


_UNSET = object()


def _db_files(db_path) -> set:
    """Lexical and resolved paths of the live database and its sidecars."""
    if db_path is _UNSET:
        try:
            db_path = resolve_app_db_path()
        except Exception:
            db_path = None
    if not db_path:
        return set()
    out = set()
    for base in {os.path.abspath(db_path), os.path.realpath(db_path)}:
        for suffix in ("", "-wal", "-shm", "-journal"):
            out.add(os.path.abspath(base + suffix))
            out.add(os.path.realpath(base + suffix))
    return out


def _db_inside(area: str, db_path) -> bool:
    """Whether the live database (or a sidecar), or a link to it, is inside ``area``."""
    areas = {os.path.abspath(area), os.path.realpath(area)}
    return any(_under(f, a) for f in _db_files(db_path) for a in areas)


def discard_staged_restore(staging_dir: str, db_path=_UNSET) -> bool:
    if not os.path.lexists(staging_dir):
        return False
    if _db_inside(staging_dir, db_path):
        # Removing the directory would take the live database with it: only
        # disarm the restore so it is never applied, and never by deleting a
        # file that is (or links to) the database itself.
        plan_path = os.path.join(staging_dir, PLAN_FILE)
        db_files = _db_files(db_path)
        if os.path.lexists(plan_path):
            if {os.path.abspath(plan_path), os.path.realpath(plan_path)} & db_files:
                logger.error("The app database is %s; the staged restore was left in place.", plan_path)
                return False
            os.unlink(plan_path)
        logger.error("The app database is inside %s; the staged restore was disarmed, not removed.", staging_dir)
        return True
    if os.path.islink(staging_dir):
        os.unlink(staging_dir)
    else:
        shutil.rmtree(staging_dir)
    return True


def stage_restore(archive_path: str, *, staging_dir: str, scratch_dir: str, plan: dict, db_path=_UNSET) -> dict:
    """Extract and check ``archive_path`` into ``staging_dir``; arm the restore.

    The plan file is written last, so a staging directory without one is
    never applied. A failed check removes everything that was staged.
    """
    if _db_inside(staging_dir, db_path):
        raise BackupError("the app database is inside the restore staging directory; move it out first")
    discard_staged_restore(staging_dir, db_path)
    ensure_private_dir(staging_dir)
    payload = os.path.join(staging_dir, PAYLOAD_DIRNAME)
    ensure_private_dir(payload)
    try:
        report = scan_archive(archive_path, scratch_dir=scratch_dir, extract_root=payload)
    except BaseException:
        discard_staged_restore(staging_dir, db_path)
        raise
    if not report["ok"]:
        discard_staged_restore(staging_dir, db_path)
        return report
    write_json_atomic(os.path.join(staging_dir, STAGED_MANIFEST_FILE), report["manifest"])
    write_json_atomic(os.path.join(staging_dir, PLAN_FILE), {**plan, "phase": "staged"})
    return report


# ── Applying a staged restore (startup hook) ────────────────────────────────


def _move(src: str, dst: str) -> None:
    """Rename ``src`` to exactly ``dst``, replacing a file already there.

    Copying stands in for the rename only across filesystems (EXDEV: a
    sub-mount inside the data dir), and never into a directory standing at
    ``dst``, which would nest ``src`` inside it. Any other failure propagates,
    so a restore rolls back instead of placing a file somewhere else.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.replace(src, dst)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        if os.path.isdir(dst):
            raise IsADirectoryError(errno.EISDIR, "a directory is in the way", dst) from exc
    _copy_then_remove(src, dst)


def _copy_then_remove(src: str, dst: str) -> None:
    """Cross-filesystem move: copy to a temp sibling of ``dst``, rename it
    into place, then remove ``src``. A failed copy (a full disk, say) leaves
    no partial ``dst`` for a rollback to miss, and ``src`` untouched; a file
    whose source cannot be removed has its new copy removed again, so the
    move is undone rather than half done. The temp name matches the
    atomic-write pattern, so a crash leftover is never captured or set aside."""
    tmp = os.path.join(os.path.dirname(dst), f"{os.path.basename(dst)}.tmp.{uuid.uuid4().hex}")
    try:
        if os.path.islink(src):
            os.symlink(os.readlink(src), tmp)
        elif os.path.isdir(src):
            shutil.copytree(src, tmp, symlinks=True)
        else:
            shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.isdir(tmp) and not os.path.islink(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise
    if os.path.isdir(src) and not os.path.islink(src):
        # On failure the complete copy stays at dst and what is left of src
        # is a subset of it, which a rollback merges back without loss.
        shutil.rmtree(src)
        return
    try:
        os.unlink(src)
    except BaseException:
        try:
            os.unlink(dst)
        except OSError:
            pass
        raise


def _under(path: str, area: str) -> bool:
    return path == area or path.startswith(area.rstrip(os.sep) + os.sep)


def _planned_moves(manifest: dict, payload: str, data_dir: str, db_path: str | None, protected=()) -> list:
    """(staged source, destination, manifest entry) for every member to place.

    ``db_path`` is the database's real path. Directory symlinks inside the
    data dir are followed (a relocated subdirectory keeps working), but each
    file lands at its own name: a symlink standing there is set aside by
    _set_aside_current, never written through. Refused before anything moves:
    a destination that, links resolved, falls in an excluded area or one of
    the ``protected`` directories (backups, staging, set-aside files), and
    two members sharing a destination or one landing where another needs a
    directory.
    """
    excluded = set(excluded_paths())
    real_root = os.path.realpath(data_dir)
    protected_real = [os.path.realpath(p) for p in protected if p]
    main = (manifest.get("database") or {}).get("member")
    # A live database inside the backups, staging or set-aside directories
    # would be overwritten there or deleted with them (the previous set-aside
    # generation is removed before the swap).
    if db_path:
        area = next((p for p in protected_real if _under(os.path.realpath(db_path), p)), None)
        if area:
            where = _relative_to(real_root, area)
            raise BackupError(f"the app database is inside {_data_member(where) if where else area}, "
                              "which a restore replaces or clears; move it out first")
    # A backup without the app database restores files only, so a stray copy
    # it holds at the live database's path is left out: the database and its
    # sidecars stay as they are (see _set_aside_current's keep_db).
    live_db = set()
    if not main and db_path:
        real_db = os.path.realpath(db_path)
        live_db = {real_db + suffix for suffix in ("",) + SQLITE_SIDECARS}
    moves = []
    claimed: dict[str, str] = {}
    for entry in manifest.get("files") or []:
        name = validate_member_name(entry.get("path"))
        source = _confine_member(payload, name)
        if name == main:
            if not db_path:
                raise BackupError(
                    "the backup contains the app database, but DATABASE_URL does not "
                    "point to a SQLite file here"
                )
            target = os.path.abspath(db_path)
        elif name.startswith(DATA_PREFIX + "/"):
            rel = name[len(DATA_PREFIX) + 1:]
            if _rel_is_excluded(rel, excluded):
                raise BackupError(f"{name}: targets an area restores never write")
            # The parent's symlinks are resolved and confined, so a link inside
            # the data directory cannot redirect the write outside it.
            parent_rel, _sep, leaf = rel.rpartition("/")
            target = os.path.join(_confine_member(data_dir, parent_rel), leaf)
            if os.path.realpath(target) in live_db:
                continue
            real_rel = _relative_to(real_root, target)
            if real_rel != rel and real_rel and _rel_is_excluded(real_rel, excluded):
                raise BackupError(f"{name}: a symbolic link leads it into {_data_member(real_rel)}, "
                                  "an area restores never write")
            area = next((p for p in protected_real if _under(target, p)), None)
            if area:
                where = _relative_to(real_root, area)
                raise BackupError(f"{name}: a symbolic link leads it into {_data_member(where) if where else area}, "
                                  "which restores never write")
        else:
            raise BackupError(f"{name}: unexpected member")
        key = os.path.normcase(target)
        if key in claimed:
            raise BackupError(f"{claimed[key]} and {name} would both be written to "
                              f"{_data_member(_relative_to(real_root, target) or target)}: a symbolic link joins their paths")
        claimed[key] = name
        moves.append((source, target, entry))
    # A file where another member needs a directory (possible only through links).
    parents = {}
    for _source, target, entry in moves:
        parent = os.path.dirname(target)
        while _under(parent, real_root) and parent != real_root and os.path.normcase(parent) not in parents:
            parents[os.path.normcase(parent)] = entry.get("path")
            parent = os.path.dirname(parent)
    for _source, target, entry in moves:
        if os.path.normcase(target) in parents:
            raise BackupError(f"{entry.get('path')} is a file, but {parents[os.path.normcase(target)]} "
                              "needs a directory there: a symbolic link joins their paths")
    return moves


def _staged_payload_problems(moves) -> list[str]:
    problems = []
    for source, _target, entry in moves:
        try:
            digest = hashlib.sha256()
            size = 0
            with open(source, "rb") as fh:
                for chunk in iter(lambda: fh.read(_CHUNK), b""):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as exc:
            problems.append(f"{entry.get('path')}: {exc.strerror or exc}")
            continue
        if size != entry.get("size") or digest.hexdigest() != entry.get("sha256"):
            problems.append(f"{entry.get('path')}: changed since it was staged")
    return problems


def _set_aside_current(data_dir: str, db_path: str | None, previous_dir: str, skip, *,
                       keep_db: bool = False, file_targets=()) -> int:
    """Move the current files in the backup's scope into ``previous_dir``.

    ``db_path`` is the database's real path (a symlink to it is never moved).
    ``keep_db`` leaves the database and its sidecars where they are: the
    backup has no database to put in their place. A directory or a symlink
    standing where the backup has a file (one of ``file_targets``) is set
    aside whole, so the file lands at its exact path. Raises BackupError
    before moving anything when part of the scope cannot be listed (its files
    could be neither set aside nor safely replaced), or when a directory in
    the way holds a path restores never move.
    """
    root = os.path.abspath(data_dir)
    real_root = os.path.realpath(root)
    db_rel = _relative_to(real_root, db_path) if db_path else None
    keep = tuple(skip)
    if keep_db and db_rel:
        # inventory() walks the lexical root; name the file the way it sees it.
        keep += (os.path.join(root, *db_rel.split("/")),)
    inv = inventory(root, skip=keep)
    if inv.unreadable:
        listed = ", ".join(f"{_data_member(rel)} ({reason})" for rel, reason in inv.unreadable[:5])
        raise BackupError(f"cannot read {listed}, so the files there cannot be set aside; "
                          "fix the permissions and stage the restore again")

    untouchable = set(excluded_paths())
    for path in keep:
        rel = _relative_to(root, path) if path else None
        if rel:
            untouchable.add(rel)
    in_the_way = []
    for target in file_targets:
        if os.path.islink(target):
            rel = _relative_to(real_root, target)
            if rel:
                in_the_way.append(rel)  # the link itself goes aside, never written through
            continue
        if not os.path.isdir(target):
            continue
        rel = _relative_to(real_root, target)
        if not rel:
            continue  # outside the data dir (the database path): placing it fails and rolls back
        held = sorted(p for p in untouchable
                      if (p == rel or p.startswith(rel + "/")) and os.path.lexists(os.path.join(root, *p.split("/"))))
        if held:
            raise BackupError(f"{_data_member(rel)} is a file in the backup, but the directory there now "
                              f"holds {_data_member(held[0])}, which restores never move")
        in_the_way.append(rel)

    moved = 0
    for rel in in_the_way:
        moved += sum(1 for _abs, r in inv.files if r.startswith(rel + "/"))
        _move(os.path.join(root, *rel.split("/")), os.path.join(previous_dir, DATA_PREFIX, *rel.split("/")))
    for abs_path, rel in inv.files:
        if in_the_way and any(rel.startswith(d + "/") for d in in_the_way):
            continue  # went aside with its directory
        target = os.path.join(previous_dir, DATA_PREFIX, *rel.split("/"))
        _move(abs_path, target)
        moved += 1
        for suffix in SQLITE_SIDECARS:
            if os.path.lexists(abs_path + suffix):
                _move(abs_path + suffix, target + suffix)
    # Whatever inventory() did not take (a database outside the data dir, or
    # inside an excluded area such as logs/) goes aside explicitly: placement
    # overwrites it, and a rollback must be able to put it back.
    if db_path and not keep_db and os.path.lexists(db_path):
        moved += 1
        for suffix in ("",) + SQLITE_SIDECARS:
            if os.path.lexists(db_path + suffix):
                _move(db_path + suffix, os.path.join(previous_dir, DATABASE_PREFIX, os.path.basename(db_path) + suffix))
    return moved


def _rollback(moves, data_dir: str, db_path: str | None, previous_dir: str, attempted=frozenset()) -> bool:
    """Undo a partly applied restore: drop what was placed, put back what was set aside.

    Placed files are the ones whose staged source is gone (this start or an
    interrupted earlier one), plus every target this start tried to place
    (``attempted``): a failed cross-filesystem move can leave its copy behind
    with the source still staged. Whatever stood at a target was set aside
    first, so what is there now came from the backup. Sidecars need no
    special case: the set-aside step moved the old ones with their database,
    and the restored copy is in rollback-journal mode.
    """
    clean = True
    for source, target, _entry in moves:
        if (not os.path.lexists(source) or target in attempted) and os.path.lexists(target):
            try:
                os.unlink(target)
            except OSError:
                clean = False
    for sub, dest_root in ((DATA_PREFIX, data_dir), (DATABASE_PREFIX, os.path.dirname(db_path or "") or None)):
        base = os.path.join(previous_dir, sub)
        if not dest_root or not os.path.isdir(base):
            continue
        for dirpath, dirs, names in os.walk(base):
            # os.walk lists a symlink to a directory among the directories
            # without descending: it moves back like a file.
            links = [d for d in dirs if os.path.islink(os.path.join(dirpath, d))]
            for name in names + links:
                src = os.path.join(dirpath, name)
                rel = os.path.relpath(src, base)
                try:
                    _move(src, os.path.join(dest_root, rel))
                except OSError:
                    clean = False
    return clean


def _read_settings(settings_file: str) -> dict:
    try:
        current = read_json(settings_file)
        return current if isinstance(current, dict) else {}
    except (OSError, ValueError):
        return {}


def _preserve_settings(settings_file: str, preserved: dict) -> None:
    current = _read_settings(settings_file)
    for key, value in preserved.items():
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value
    write_json_atomic(settings_file, current)


def _record_restore_result(state_file: str, result: dict) -> None:
    try:
        try:
            state = read_json(state_file)
            if not isinstance(state, dict):
                state = {}
        except (OSError, ValueError):
            state = {}
        state["last_restore"] = result
        ensure_private_dir(os.path.dirname(state_file))
        write_json_atomic(state_file, state)
    except OSError as exc:
        logger.warning("Could not record the restore result in %s: %s", state_file, exc)


class _ApplyLock:
    """Serialize appliers (uvicorn workers) and make later ones wait it out."""

    def __init__(self, path: str):
        self.path = path
        self.handle = None

    def __enter__(self):
        if os.name == "nt":
            return self
        import fcntl
        self.handle = open(self.path, "a+b")
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            import fcntl
            try:
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            finally:
                self.handle.close()
        return False




def apply_pending_restore(
    *,
    data_dir: str | None = None,
    db_path=_UNSET,
    staging_dir: str | None = None,
    previous_dir: str | None = None,
    backups_dir: str | None = None,
    state_file: str | None = None,
    settings_file: str | None = None,
) -> dict | None:
    """Apply a staged restore, if there is one. Never raises.

    Called by ``app.py`` before ``core.database`` is imported. The current
    files in the backup's scope (and the app database) are moved into
    ``RESTORE_PREVIOUS_DIR``, then the staged files are moved into place, then
    the database is checked. Any failure rolls the set-aside files back. The
    phase is persisted in the plan, so a crash mid-way resumes on the next
    start instead of mixing old and new files. The outcome is recorded in the
    backup state file for the Settings panel to show.
    """
    staging_dir = staging_dir or constants.RESTORE_STAGING_DIR
    plan_path = os.path.join(staging_dir, PLAN_FILE)
    if not os.path.isfile(plan_path):
        return None
    data_dir = os.path.abspath(data_dir or constants.DATA_DIR)
    previous_dir = previous_dir or constants.RESTORE_PREVIOUS_DIR
    backups_dir = backups_dir or constants.BACKUPS_DIR
    state_file = state_file or constants.BACKUP_STATE_FILE
    if settings_file is None:
        rel = _relative_to(os.path.abspath(constants.DATA_DIR), constants.SETTINGS_FILE)
        settings_file = os.path.join(data_dir, *rel.split("/")) if rel else constants.SETTINGS_FILE
    try:
        with _ApplyLock(os.path.join(staging_dir, _APPLY_LOCK_FILE)):
            if not os.path.isfile(plan_path):  # another worker finished it
                return None
            if db_path is _UNSET:
                db_path = resolve_app_db_path()
            result = _apply(plan_path, staging_dir, data_dir, db_path, previous_dir,
                            (backups_dir, staging_dir, previous_dir), settings_file)
    except Exception as exc:  # the hook must never stop the app from starting
        logger.exception("Applying the staged restore failed")
        result = {
            "ok": False,
            "applied_at": iso(_utcnow()),
            "message": f"Restore failed unexpectedly: {exc}. Check the data directory and the safety backup.",
        }
    _record_restore_result(state_file, result)
    log = logger.warning if result.get("ok") else logger.error
    log("Staged restore: %s", result.get("message"))
    return result


def _apply(plan_path, staging_dir, data_dir, db_path, previous_dir, skip, settings_file) -> dict:
    plan = read_json(plan_path)
    base = {
        "backup": plan.get("backup"),
        "staged_at": plan.get("staged_at"),
        "requested_by": plan.get("requested_by"),
        "safety_backup": plan.get("safety_backup"),
        "applied_at": iso(_utcnow()),
    }
    # Work on the database's real file: a symlink to it stays a symlink, and
    # the file's sidecars (SQLite keeps them next to the real file) go with it.
    db_path = os.path.realpath(db_path) if db_path else None
    try:
        manifest = read_json(os.path.join(staging_dir, STAGED_MANIFEST_FILE))
        moves = _planned_moves(manifest, os.path.join(staging_dir, PAYLOAD_DIRNAME), data_dir, db_path,
                               protected=skip)
    except (OSError, ValueError) as exc:
        discard_staged_restore(staging_dir, db_path)
        return {**base, "ok": False, "message": f"Restore not applied: {exc}. Nothing was changed."}
    main = (manifest.get("database") or {}).get("member")
    keep_db = bool(db_path) and not main  # files-only backup: the live database stays

    phase = plan.get("phase")
    if phase == "staged":
        problems = _staged_payload_problems(moves)
        if problems:
            discard_staged_restore(staging_dir, db_path)
            return {**base, "ok": False, "problems": problems[:MAX_REPORTED_PROBLEMS],
                    "message": "Restore not applied: staged files no longer match the backup. Nothing was changed."}
        if os.path.lexists(previous_dir):
            shutil.rmtree(previous_dir)  # one generation of set-aside files
        # Settings that describe this server rather than the restored state
        # (the backup schedule) keep their value from right before the swap.
        current = _read_settings(settings_file)
        plan["preserve_settings"] = {k: current.get(k) for k in plan.get("preserve_settings_keys") or []}
        phase = plan["phase"] = "setting_aside"
        write_json_atomic(plan_path, plan)
    set_aside = 0
    attempted: set[str] = set()
    try:
        if phase == "setting_aside":
            set_aside = _set_aside_current(data_dir, db_path, previous_dir, skip, keep_db=keep_db,
                                           file_targets=[target for _source, target, _entry in moves])
            phase = plan["phase"] = "placing"
            write_json_atomic(plan_path, plan)
        if phase != "placing":
            raise BackupError(f"unknown restore phase {phase!r}")
        placed = 0
        for source, target, entry in moves:
            if not os.path.lexists(source):
                continue  # placed before an interrupted start
            attempted.add(target)
            _move(source, target)
            mode = 0o600 if db_path and target == os.path.abspath(db_path) else int(entry.get("mode") or 0o600) & 0o755
            _chmod(target, mode)
            placed += 1
        check = None
        if main and db_path:
            check = describe_sqlite(db_path)
            want = ((manifest.get("sqlite_files") or {}).get(main) or {}).get("tables") or {}
            if check["integrity"] != "ok":
                raise BackupError(f"restored database failed integrity_check: {check['integrity']}")
            wrong = [t for t, n in want.items() if n is not None and check["tables"].get(t) != n]
            if wrong:
                raise BackupError(f"restored database row counts differ in {', '.join(sorted(wrong)[:5])}")
    except Exception as exc:
        logger.exception("Restore failed while moving files; rolling back")
        clean = _rollback(moves, data_dir, db_path, previous_dir, attempted)
        discard_staged_restore(staging_dir, db_path)
        if clean:
            message = f"Restore failed ({exc}); rolled back, your data is as it was before."
        else:
            message = (f"Restore failed ({exc}) and could not be fully rolled back. "
                       f"Restore the safety backup {plan.get('safety_backup')} from Settings or with odysseus-backup.")
        return {**base, "ok": False, "rolled_back": clean, "message": message}

    preserved = plan.get("preserve_settings")
    if isinstance(preserved, dict) and preserved:
        try:
            _preserve_settings(settings_file, preserved)
        except OSError as exc:
            logger.warning("Could not carry the backup schedule over the restore: %s", exc)
    discard_staged_restore(staging_dir, db_path)
    result = {
        **base,
        "ok": True,
        "files_restored": placed,
        "files_set_aside": set_aside,
        "previous_dir": previous_dir,
        "message": (f"Restored {plan.get('backup')}. The files it replaced are kept in "
                    f"{os.path.basename(previous_dir)} until the next restore."
                    + (" The backup has no app database, so the current one was kept." if keep_db else "")),
    }
    if check is not None:
        result["database"] = {"integrity": check["integrity"], "tables": len(check["tables"])}
    return result

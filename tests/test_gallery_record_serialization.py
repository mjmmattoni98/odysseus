"""Gallery and editor-draft records: UTC timestamps and in-place edit refresh.

QA 2026-10-12: a photo uploaded at 23:59:55 UTC showed as "2 hours ago" in
Madrid 33 s later, because created_at/updated_at were serialized without an
offset and the browser read them as local time. EXIF ``taken_at`` is camera
wall-clock time with no zone and must stay offset-less.

Rotating a 320x180 image persisted 180x320, but the inspector kept showing the
old size because the rotate response did not carry the updated record.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import core.database as cdb
from core.database import GalleryAlbum, GalleryImage
from src.user_time import utc_isoformat

CREATED = datetime(2026, 10, 11, 23, 59, 55, 397856)
TAKEN = datetime(2024, 7, 1, 12, 0, 0)


def _png(width, height):
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (width, height), "blue").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def gallery(monkeypatch, tmp_path):
    import routes.gallery_routes as gallery_routes

    engine = create_engine(
        f"sqlite:///{tmp_path / 'gallery.db'}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    cdb.Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    image_dir = tmp_path / "generated_images"
    image_dir.mkdir()
    monkeypatch.setattr(gallery_routes, "SessionLocal", session_factory)
    monkeypatch.setattr(gallery_routes, "GALLERY_IMAGE_DIR", image_dir)
    monkeypatch.setattr(gallery_routes, "get_current_user", lambda request: "alice")

    app = FastAPI()
    app.include_router(gallery_routes.setup_gallery_routes())
    return SimpleNamespace(
        client=TestClient(app), db=session_factory, image_dir=image_dir,
    )


def _seed_image(gallery, *, filename="photo.png", content=None, **fields):
    if content is not None:
        (gallery.image_dir / filename).write_bytes(content)
    db = gallery.db()
    try:
        db.add(GalleryImage(
            id="img-1", filename=filename, prompt="photo", model="imported",
            owner="alice", is_active=True, created_at=CREATED, updated_at=CREATED,
            file_size=len(content) if content is not None else 10, **fields,
        ))
        db.commit()
    finally:
        db.close()


def _as_utc(stamp):
    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None, f"{stamp!r} has no UTC offset"
    return parsed


# ---- utc_isoformat ----

def test_utc_isoformat_marks_naive_values_as_utc():
    assert utc_isoformat(CREATED) == "2026-10-11T23:59:55.397856Z"
    assert utc_isoformat(None) is None


def test_utc_isoformat_converts_aware_values_to_utc():
    madrid_summer = timezone(timedelta(hours=2))
    stamp = utc_isoformat(datetime(2026, 10, 12, 1, 59, 55, tzinfo=madrid_summer))
    assert stamp == "2026-10-11T23:59:55Z"


# ---- gallery image / album serialization ----

def test_gallery_image_timestamps_carry_utc_offset(gallery):
    _seed_image(gallery, width=320, height=180)

    photo = gallery.client.get("/api/gallery/img-1").json()

    assert photo["created_at"] == "2026-10-11T23:59:55.397856Z"
    assert _as_utc(photo["created_at"]) == CREATED.replace(tzinfo=timezone.utc)
    assert _as_utc(photo["updated_at"]) == CREATED.replace(tzinfo=timezone.utc)
    assert [photo["width"], photo["height"]] == [320, 180]
    # Never edited in place (no content hash): the file's bytes are fixed.
    assert photo["url"] == "/api/generated-image/photo.png"


def test_gallery_exif_taken_at_stays_zone_less(gallery):
    _seed_image(gallery, taken_at=TAKEN)

    photo = gallery.client.get("/api/gallery/img-1").json()
    listed = gallery.client.get("/api/gallery/library").json()["items"][0]

    assert photo["taken_at"] == "2024-07-01T12:00:00"
    assert listed["taken_at"] == "2024-07-01T12:00:00"
    assert listed["created_at"] == "2026-10-11T23:59:55.397856Z"


def test_gallery_album_created_at_and_cover_url(gallery):
    db = gallery.db()
    try:
        db.add(GalleryAlbum(id="album-1", name="Trip", owner="alice", created_at=CREATED))
        db.commit()
    finally:
        db.close()
    _seed_image(gallery, album_id="album-1", file_hash="ab" * 32)

    albums = gallery.client.get("/api/gallery/albums").json()["albums"]

    assert albums[0]["created_at"] == "2026-10-11T23:59:55.397856Z"
    assert albums[0]["cover_url"] == "/api/generated-image/photo.png?v=abababababab"


# ---- in-place edits return the refreshed record ----

def test_rotate_response_carries_updated_record(gallery):
    pytest.importorskip("PIL")
    _seed_image(gallery, content=_png(320, 180), width=320, height=180)

    response = gallery.client.post("/api/gallery/img-1/rotate", json={"angle": 90})

    assert response.status_code == 200
    body = response.json()
    on_disk = (gallery.image_dir / "photo.png").read_bytes()
    assert [body["width"], body["height"]] == [180, 320]
    record = body["image"]
    assert record["id"] == "img-1"
    assert [record["width"], record["height"]] == [180, 320]
    assert record["file_size"] == len(on_disk)
    assert _as_utc(record["updated_at"]) > CREATED.replace(tzinfo=timezone.utc)
    # The detail GET agrees with what the rotate response told the client.
    fetched = gallery.client.get("/api/gallery/img-1").json()
    fields = ("width", "height", "file_size", "updated_at", "url")
    assert {k: fetched[k] for k in fields} == {k: record[k] for k in fields}


def test_rotate_changes_versioned_image_url(gallery):
    """Images are served with immutable cache headers, so the URL must change
    with the bytes or the browser keeps showing the pre-rotation file."""
    pytest.importorskip("PIL")
    original = _png(320, 180)
    _seed_image(gallery, content=original, file_hash=hashlib.sha256(original).hexdigest())
    before = gallery.client.get("/api/gallery/img-1").json()["url"]

    record = gallery.client.post("/api/gallery/img-1/rotate", json={"angle": 90}).json()["image"]

    rotated_hash = hashlib.sha256((gallery.image_dir / "photo.png").read_bytes()).hexdigest()
    assert before == f"/api/generated-image/photo.png?v={hashlib.sha256(original).hexdigest()[:12]}"
    assert record["url"] == f"/api/generated-image/photo.png?v={rotated_hash[:12]}"
    assert record["url"] != before
    listed = gallery.client.get("/api/gallery/library").json()["items"][0]
    assert listed["url"] == record["url"]


def test_replace_refreshes_size_hash_and_returns_record(gallery):
    pytest.importorskip("PIL")
    pytest.importorskip("python_multipart")
    _seed_image(gallery, content=_png(320, 180), width=320, height=180)
    edited = _png(64, 48)

    response = gallery.client.post(
        "/api/gallery/img-1/replace",
        files={"image": ("edited.png", edited, "image/png")},
    )

    assert response.status_code == 200
    record = response.json()["image"]
    assert [record["width"], record["height"]] == [64, 48]
    assert record["file_size"] == len(edited)
    assert record["url"] == f"/api/generated-image/photo.png?v={hashlib.sha256(edited).hexdigest()[:12]}"
    db = gallery.db()
    try:
        assert db.get(GalleryImage, "img-1").file_hash == hashlib.sha256(edited).hexdigest()
    finally:
        db.close()


# ---- editor drafts ----

def test_editor_draft_summary_timestamps_carry_utc_offset():
    import routes.editor_draft_routes as drafts

    summary = drafts._summary(SimpleNamespace(
        id="d1", name="Poster", source_image_id=None, width=10, height=10,
        thumbnail=None, created_at=CREATED, updated_at=CREATED + timedelta(seconds=30),
    ))

    assert summary["created_at"] == "2026-10-11T23:59:55.397856Z"
    assert summary["updated_at"] == "2026-10-12T00:00:25.397856Z"

"""gallery_helpers.py — extracted helpers, models, and small utilities.

Imported by gallery_routes.py."""

"""Gallery routes — browsable library for photos and AI-generated images."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from pydantic import BaseModel

from core.database import GalleryImage
from src.auth_helpers import _auth_disabled
from src.user_time import utc_isoformat

logger = logging.getLogger(__name__)


# ---- Request schemas ----

class GalleryPatch(BaseModel):
    tags: Optional[str] = None
    favorite: Optional[bool] = None
    album_id: Optional[str] = None


# ---- EXIF extraction ----

def _extract_exif(content: bytes) -> dict:
    """Extract EXIF metadata from image bytes. Returns dict of fields."""
    result = {"width": None, "height": None}
    try:
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(content))
        # Read the raw EXIF before any transpose: exif_transpose strips the
        # orientation tag and with it the parsed EXIF view.
        exif = img._getexif() if hasattr(img, '_getexif') else None

        # Record DISPLAY dimensions (EXIF-rotated), matching upload_handler.
        # A phone photo with Orientation 6/8 is stored landscape but shown
        # portrait, so the raw width/height swap the aspect ratio.
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img) or img
        except Exception:
            pass
        result["width"] = img.width
        result["height"] = img.height

        if not exif:
            return result

        # EXIF tag IDs
        # 271=Make, 272=Model, 306=DateTime, 36867=DateTimeOriginal
        # 34853=GPSInfo
        result["camera_make"] = str(exif.get(271, "")).strip() or None
        result["camera_model"] = str(exif.get(272, "")).strip() or None

        # Date taken
        for tag_id in (36867, 36868, 306):  # DateTimeOriginal, DateTimeDigitized, DateTime
            raw = exif.get(tag_id)
            if raw:
                try:
                    result["taken_at"] = datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
                    break
                except (ValueError, TypeError):
                    pass

        # GPS
        gps_info = exif.get(34853)
        if gps_info and isinstance(gps_info, dict):
            try:
                def _to_deg(vals):
                    d, m, s = [float(v) for v in vals]
                    return d + m / 60 + s / 3600
                if 2 in gps_info and 4 in gps_info:
                    lat = _to_deg(gps_info[2])
                    lng = _to_deg(gps_info[4])
                    if gps_info.get(1) == 'S': lat = -lat
                    if gps_info.get(3) == 'W': lng = -lng
                    result["gps_lat"] = f"{lat:.6f}"
                    result["gps_lng"] = f"{lng:.6f}"
            except Exception:
                pass
    except Exception as e:
        # User-visible failure (photo loses metadata): surface at WARNING
        # and record on the result so the upload endpoint can pass it back.
        logger.warning(f"EXIF extraction failed: {e}")
        result["exif_error"] = str(e)
    return result


# ---- Helpers ----

def _image_url(filename: str, file_hash: Optional[str] = None) -> str:
    """URL for a gallery file, versioned by its content hash.

    Rotate and replace overwrite the file under the same name, but the serve
    route sends ``immutable`` cache headers. The ``v`` query, which the route
    ignores, gives each content version its own browser cache entry. Rows
    without a hash have never been edited in place, so their bytes are fixed.
    """
    url = f"/api/generated-image/{filename}"
    return f"{url}?v={file_hash[:12]}" if file_hash else url


def _image_to_dict(img: GalleryImage, session_name: str = None) -> Dict[str, Any]:
    return {
        "id": img.id,
        "filename": img.filename,
        "url": _image_url(img.filename, img.file_hash),
        "prompt": img.prompt,
        "caption": img.caption or "",
        "model": img.model,
        "size": img.size,
        "quality": img.quality,
        "tags": img.tags or "",
        "ai_tags": img.ai_tags or "",
        "user_tags": img.tags or "",
        "session_id": img.session_id,
        "session_name": session_name,
        "album_id": img.album_id,
        "is_active": img.is_active,
        "favorite": img.favorite or False,
        # EXIF capture time is camera wall-clock time with no zone, so it
        # stays offset-less; created_at/updated_at are stored UTC.
        "taken_at": img.taken_at.isoformat() if img.taken_at else None,
        "camera": f"{img.camera_make or ''} {img.camera_model or ''}".strip() or None,
        "gps": {"lat": img.gps_lat, "lng": img.gps_lng} if img.gps_lat else None,
        "width": img.width,
        "height": img.height,
        "file_size": img.file_size,
        "created_at": utc_isoformat(img.created_at),
        "updated_at": utc_isoformat(img.updated_at),
    }


def _owner_filter(q, user, model_cls=GalleryImage):
    """Apply owner filtering to a gallery query.

    ``get_current_user`` returns None both in auth-disabled single-user mode
    and when auth is enabled but no current user was resolved. Preserve the
    single-user behavior, but fail closed for auth-enabled null-user states.
    """
    if user is not None:
        return q.filter(model_cls.owner == user)
    if _auth_disabled():
        return q
    return q.filter(False)



def _human_size(nbytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"

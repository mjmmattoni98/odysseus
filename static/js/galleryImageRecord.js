// Pure (browser-free) helpers for gallery image records from /api/gallery/*.
// Extracted so they can be unit-tested.
import { parseUtcTimestamp } from './util/utcTimestamp.js';

// The date a photo is labelled with. EXIF taken_at is the camera's wall-clock
// time with no zone, so it keeps the browser's local-time parse; created_at is
// a UTC column.
export function photoDisplayDate(img) {
  if (img?.taken_at) {
    const taken = new Date(img.taken_at);
    if (!Number.isNaN(taken.getTime())) return taken;
  }
  return parseUtcTimestamp(img?.created_at);
}

// When the photo was last edited, or null if updated_at is not meaningfully
// later than created_at. Every row gets updated_at on insert via the ORM
// timestamp mixin, so a gap under 10 s is not an edit.
export function photoEditedDate(img) {
  const updated = parseUtcTimestamp(img?.updated_at);
  const created = parseUtcTimestamp(img?.created_at);
  if (!updated || !created) return null;
  return updated.getTime() - created.getTime() > 10000 ? updated : null;
}

// Fields an in-place pixel edit (rotate, replace) changes on the server. The
// url carries a content version (?v=<hash>), so it changes with the pixels.
const EDITED_FIELDS = ['width', 'height', 'file_size', 'updated_at', 'url'];

// Fold a rotate/replace response into a cached record so the inspector stops
// showing pre-edit dimensions and size, and the image element picks up the
// new file. Accepts the full record under `image` or a bare {width, height}
// body. Other fields are left alone: the response record has no session name.
export function applyImageEdit(img, response) {
  if (!img || !response || typeof response !== 'object') return img;
  const fresh = response.image && typeof response.image === 'object' ? response.image : response;
  for (const key of EDITED_FIELDS) {
    if (fresh[key] !== undefined && fresh[key] !== null) img[key] = fresh[key];
  }
  return img;
}

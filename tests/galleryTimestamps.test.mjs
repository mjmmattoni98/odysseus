// Gallery record timestamps and in-place edit reconciliation.
// QA 2026-10-12: in Europe/Madrid a 33-second-old upload showed "2 hours ago"
// because naive UTC timestamps were parsed as local time, and the inspector
// kept "320 x 180" after a rotation persisted 180 x 320.
process.env.TZ = 'Europe/Madrid';

import test from 'node:test';
import assert from 'node:assert/strict';
import { parseUtcTimestamp } from '../static/js/util/utcTimestamp.js';
import { applyImageEdit, photoDisplayDate, photoEditedDate } from '../static/js/galleryImageRecord.js';

const UPLOADED_UTC = Date.UTC(2026, 9, 11, 23, 59, 55, 397);

test('naive UTC-column values are read as UTC, not local time', () => {
  assert.equal(parseUtcTimestamp('2026-10-11T23:59:55.397856').getTime(), UPLOADED_UTC);
  assert.equal(parseUtcTimestamp('2026-10-11 23:59:55.397856').getTime(), UPLOADED_UTC);
});

test('offset-bearing values keep their offset', () => {
  assert.equal(parseUtcTimestamp('2026-10-11T23:59:55.397856Z').getTime(), UPLOADED_UTC);
  assert.equal(parseUtcTimestamp('2026-10-11T23:59:55.397856+00:00').getTime(), UPLOADED_UTC);
  assert.equal(parseUtcTimestamp('2026-10-12T01:59:55.397+02:00').getTime(), UPLOADED_UTC);
  assert.equal(parseUtcTimestamp('2026-10-11T18:59:55.397-0500').getTime(), UPLOADED_UTC);
});

test('missing or unparseable values return null', () => {
  for (const value of [null, undefined, '', '   ', 'not a date', {}]) {
    assert.equal(parseUtcTimestamp(value), null, `value ${JSON.stringify(value)}`);
  }
});

test('a just-uploaded photo is seconds old, not hours, in Madrid', () => {
  const inspectedAt = Date.UTC(2026, 9, 12, 0, 0, 28); // 02:00:28 Madrid
  for (const createdAt of ['2026-10-11T23:59:55.397856', '2026-10-11T23:59:55.397856Z']) {
    const shown = photoDisplayDate({ created_at: createdAt });
    const ageSeconds = (inspectedAt - shown.getTime()) / 1000;
    assert.ok(ageSeconds > 30 && ageSeconds < 35, `${createdAt}: ${ageSeconds}s`);
  }
});

test('EXIF taken_at keeps camera wall-clock time', () => {
  const shown = photoDisplayDate({ taken_at: '2024-07-01T12:00:00', created_at: '2026-10-11T23:59:55Z' });
  assert.equal(shown.getTime(), new Date(2024, 6, 1, 12, 0, 0).getTime());
  assert.equal(shown.getHours(), 12);
  assert.notEqual(shown.getTime(), Date.UTC(2024, 6, 1, 12, 0, 0), 'EXIF time must not be read as UTC');
});

test('created_at is the fallback when EXIF taken_at is absent or invalid', () => {
  assert.equal(photoDisplayDate({ taken_at: null, created_at: '2026-10-11T23:59:55.397856' }).getTime(), UPLOADED_UTC);
  assert.equal(photoDisplayDate({ taken_at: 'garbage', created_at: '2026-10-11T23:59:55.397Z' }).getTime(), UPLOADED_UTC);
  assert.equal(photoDisplayDate({}), null);
});

test('edited date requires updated_at meaningfully after created_at', () => {
  assert.equal(photoEditedDate({ created_at: '2026-10-11T23:59:55Z', updated_at: '2026-10-11T23:59:58Z' }), null);
  // Mixed legacy/new formats compare on the same (UTC) clock.
  const edited = photoEditedDate({ created_at: '2026-10-11T23:59:55', updated_at: '2026-10-12T00:00:30Z' });
  assert.equal(edited.getTime(), Date.UTC(2026, 9, 12, 0, 0, 30));
  assert.equal(photoEditedDate({ created_at: '2026-10-11T23:59:55Z' }), null);
});

test('rotation response refreshes dimensions, size, updated_at, and versioned url', () => {
  const img = {
    id: 'img-1', width: 320, height: 180, file_size: 4403,
    created_at: '2026-10-11T23:59:55Z', updated_at: '2026-10-11T23:59:55Z',
    session_name: 'Source chat', url: '/api/generated-image/a.png?v=111111111111',
  };
  const response = {
    ok: true, width: 180, height: 320,
    image: {
      id: 'img-1', width: 180, height: 320, file_size: 4512,
      updated_at: '2026-10-12T00:01:00Z', session_name: null, url: '/api/generated-image/a.png?v=222222222222',
    },
  };

  assert.equal(applyImageEdit(img, response), img);
  assert.deepEqual(
    [img.width, img.height, img.file_size, img.updated_at],
    [180, 320, 4512, '2026-10-12T00:01:00Z'],
  );
  assert.equal(img.session_name, 'Source chat');
  assert.equal(img.url, '/api/generated-image/a.png?v=222222222222');
  assert.ok(photoEditedDate(img));
});

test('legacy {width, height} rotate body still updates dimensions', () => {
  const img = { width: 320, height: 180, file_size: 4403, url: '/api/generated-image/a.png' };
  applyImageEdit(img, { ok: true, width: 180, height: 320 });
  assert.deepEqual([img.width, img.height, img.file_size], [180, 320, 4403]);
  assert.equal(img.url, '/api/generated-image/a.png');
  assert.deepEqual(applyImageEdit(img, null), img);
});

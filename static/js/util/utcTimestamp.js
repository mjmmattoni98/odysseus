// Pure (browser-free) parser for timestamps from naive-UTC database columns
// (created_at / updated_at). Extracted so it can be unit-tested.
//
// The server serializes these with an explicit "Z", but older payloads carry a
// bare "2026-10-11T23:59:55.397856". Browsers parse an offset-less date-time as
// LOCAL time, which shifts it by the user's UTC offset, so a missing offset is
// read as UTC here. Do not use this for zone-less local wall times such as EXIF
// capture dates; those must keep the browser's local-time parse.
const ISO_RE = /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?)?\s*(Z|[+-]\d{2}:?\d{2})?$/i;

function _validDate(date) {
  return Number.isNaN(date.getTime()) ? null : date;
}

// Returns a Date, or null when the value is missing or unparseable.
export function parseUtcTimestamp(value) {
  if (value instanceof Date) return _validDate(value);
  if (typeof value === 'number') return _validDate(new Date(value));
  if (typeof value !== 'string' || !value.trim()) return null;
  const match = ISO_RE.exec(value.trim());
  if (!match) return _validDate(new Date(value));
  const [, year, month, day, hour = '0', minute = '0', second = '0', fraction = '', zone] = match;
  // Date.UTC keeps millisecond precision; Python emits microseconds.
  const millis = Number((fraction + '000').slice(0, 3));
  let time = Date.UTC(+year, +month - 1, +day, +hour, +minute, +second, millis);
  if (zone && zone.toUpperCase() !== 'Z') {
    const sign = zone[0] === '-' ? -1 : 1;
    const digits = zone.slice(1).replace(':', '');
    const offsetMinutes = Number(digits.slice(0, 2)) * 60 + Number(digits.slice(2, 4));
    time -= sign * offsetMinutes * 60000;
  }
  return _validDate(new Date(time));
}

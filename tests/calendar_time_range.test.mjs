// Shared calendar time-range formatter (static/js/calendar/utils.js).
// Pinned to a DST-observing zone so tz-aware instants and DST days are
// deterministic regardless of the machine running the suite.
process.env.TZ = 'America/New_York';

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  _fmtEventTimeRange, _eventTimeRangeTitle, _eventEndDayOffset, _calendarDayDiff, _endDayHint, _fmtShortDate,
} from '../static/js/calendar/utils.js';

const shortDate = (ds) => new Date(`${ds}T00:00:00`).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });

test('same-day ranges have no marker', () => {
  const ev = { dtstart: '2026-10-12T12:00:00', dtend: '2026-10-12T13:00:00' };
  assert.equal(_fmtEventTimeRange(ev), '12:00 – 13:00');
  assert.equal(_fmtEventTimeRange(ev, { sep: '–' }), '12:00–13:00');
  assert.equal(_eventTimeRangeTitle(ev), '12:00 – 13:00');
});

test('an end on the next day is marked and the title names the end date', () => {
  // The QA case: 12:00 to 11:00 the following day.
  const ev = { dtstart: '2026-10-12T12:00:00-04:00', dtend: '2026-10-13T11:00:00-04:00' };
  assert.equal(_fmtEventTimeRange(ev), '12:00 – 11:00 (+1 day)');
  assert.equal(_eventTimeRangeTitle(ev), `12:00 – 11:00 (+1 day) · ends ${shortDate('2026-10-13')}`);
  assert.equal(_fmtShortDate('2026-10-13'), shortDate('2026-10-13'));
});

test('an end at exactly midnight is literally on the next day', () => {
  const ev = { dtstart: '2026-10-12T22:00:00', dtend: '2026-10-13T00:00:00' };
  assert.equal(_fmtEventTimeRange(ev), '22:00 – 00:00 (+1 day)');
});

test('multi-day timed events count the days', () => {
  const ev = { dtstart: '2026-10-12T09:00:00', dtend: '2026-10-15T17:00:00' };
  assert.equal(_fmtEventTimeRange(ev), '09:00 – 17:00 (+3 days)');
  assert.equal(_eventEndDayOffset(ev), 3);
});

test('all-day events keep their own semantics', () => {
  const single = { all_day: true, dtstart: '2026-10-12', dtend: '2026-10-13' };
  const multi = { all_day: true, dtstart: '2026-10-12', dtend: '2026-10-16' };
  assert.equal(_fmtEventTimeRange(single), 'All day');
  assert.equal(_fmtEventTimeRange(multi), 'All day');
  assert.equal(_eventEndDayOffset(multi), 0);
  assert.equal(_eventTimeRangeTitle(single), 'All day');
});

test('tz-aware instants are bucketed by the local day, not the UTC day', () => {
  // 22:00Z–02:00Z crosses the UTC date line but is 18:00–22:00 in New York.
  assert.equal(_fmtEventTimeRange({ dtstart: '2026-10-12T22:00:00Z', dtend: '2026-10-13T02:00:00Z' }), '18:00 – 22:00');
  // Same UTC date, but 23:00 Oct 12 – 01:00 Oct 13 locally.
  assert.equal(_fmtEventTimeRange({ dtstart: '2026-10-13T03:00:00Z', dtend: '2026-10-13T05:00:00Z' }), '23:00 – 01:00 (+1 day)');
});

test('DST days are compared as calendar days, not 24-hour blocks', () => {
  // Spring forward (Mar 8 2026): a 3-hour overnight event is still +1 day.
  assert.equal(
    _fmtEventTimeRange({ dtstart: '2026-03-07T23:00:00-05:00', dtend: '2026-03-08T03:00:00-04:00' }),
    '23:00 – 03:00 (+1 day)',
  );
  // Fall back (Nov 1 2026): 25 hours elapse but the event stays on one day.
  assert.equal(
    _fmtEventTimeRange({ dtstart: '2026-11-01T00:30:00-04:00', dtend: '2026-11-01T23:30:00-05:00' }),
    '00:30 – 23:30',
  );
  assert.equal(_calendarDayDiff('2026-03-08', '2026-03-09'), 1);
  assert.equal(_calendarDayDiff('2026-11-01', '2026-11-02'), 1);
  assert.equal(_calendarDayDiff('bad', '2026-11-02'), 0);
});

test('continuation days prefix the start date so the marker stays anchored', () => {
  const ev = { dtstart: '2026-10-12T12:00:00', dtend: '2026-10-13T11:00:00' };
  assert.equal(_fmtEventTimeRange(ev, { onDay: '2026-10-12' }), '12:00 – 11:00 (+1 day)');
  assert.equal(_fmtEventTimeRange(ev, { onDay: '2026-10-13' }), `${shortDate('2026-10-12')} 12:00 – 11:00 (+1 day)`);
});

test('missing or partial events degrade gracefully', () => {
  assert.equal(_fmtEventTimeRange(null), '');
  assert.equal(_fmtEventTimeRange({ dtstart: '2026-10-12T12:00:00' }), '12:00');
  assert.equal(_fmtEventTimeRange({ dtstart: '2026-10-12' }), '');
});

test('event form hint says when the end rolls over to a later day', () => {
  const base = { startDate: '2026-10-12', startTime: '12:00' };
  assert.equal(_endDayHint({ ...base, endDate: '2026-10-12', endTime: '11:00' }), `Ends the next day (${shortDate('2026-10-13')})`);
  assert.equal(_endDayHint({ ...base, endDate: '2026-10-12', endTime: '12:00' }), `Ends the next day (${shortDate('2026-10-13')})`);
  assert.equal(_endDayHint({ ...base, endDate: '2026-10-13', endTime: '11:00' }), `Ends the next day (${shortDate('2026-10-13')})`);
  assert.equal(_endDayHint({ ...base, endDate: '2026-10-14', endTime: '13:00' }), `Ends 2 days later (${shortDate('2026-10-14')})`);
  assert.equal(_endDayHint({ ...base, endDate: '2026-10-12', endTime: '13:00' }), '');
  assert.equal(_endDayHint({ ...base, endDate: '2026-10-12', endTime: '11:00', allDay: true }), '');
  assert.equal(_endDayHint({}), '');
});

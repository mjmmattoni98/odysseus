// Tests for the hands-free conversation logic (static/js/voiceConversationCore.js):
// energy VAD (onset, trailing-silence end, max length, noise rejection), the
// listening -> transcribing -> thinking -> speaking state machine with
// barge-in, the end-of-turn tracker, and transcript gating.
//
// Pure functions with injected timestamps: no DOM, no audio, no real clock.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  VAD_DEFAULTS,
  computeRms,
  conversationStep,
  createTurnTracker,
  createVad,
  shouldSendTranscript,
} from '../static/js/voiceConversationCore.js';

const FRAME = 50; // ms between analysis frames

/** Feed `ms` of frames at `rms` starting at t; return [events, nextT]. */
function feed(vad, t, ms, rms) {
  const events = [];
  const end = t + ms;
  for (; t < end; t += FRAME) {
    const ev = vad.update(rms, t);
    if (ev) events.push({ ...ev, t });
  }
  return [events, t];
}

test('computeRms measures signal level', () => {
  assert.equal(computeRms(new Float32Array(0)), 0);
  assert.equal(computeRms(new Float32Array([0, 0, 0])), 0);
  assert.ok(Math.abs(computeRms(new Float32Array([0.5, -0.5, 0.5, -0.5])) - 0.5) < 1e-9);
});

test('speech then silence yields speech_start and a speech_end after the silence timeout', () => {
  const vad = createVad({ threshold: 0.02, silenceMs: 1200 });
  let t = 0;
  let ev;
  [ev, t] = feed(vad, t, 500, 0.001);       // room noise
  assert.deepEqual(ev, []);
  [ev, t] = feed(vad, t, 1000, 0.1);        // talking
  assert.equal(ev.length, 1);
  assert.equal(ev[0].type, 'speech_start');
  assert.equal(ev[0].at, 500);
  [ev, t] = feed(vad, t, 1100, 0.001);      // pause shorter than the timeout
  assert.deepEqual(ev, []);
  [ev, t] = feed(vad, t, 200, 0.001);
  assert.equal(ev.length, 1);
  assert.equal(ev[0].type, 'speech_end');
  assert.equal(ev[0].durationMs, 1450 - 500); // last voiced frame - onset
  assert.equal(vad.inSpeech, false);
});

test('a pause inside an utterance does not end it', () => {
  const vad = createVad({ threshold: 0.02, silenceMs: 1200 });
  let t = 0;
  let ev;
  [ev, t] = feed(vad, t, 400, 0.1);
  [ev, t] = feed(vad, t, 800, 0.0);         // thinking pause
  [ev, t] = feed(vad, t, 400, 0.1);         // keeps talking
  assert.deepEqual(ev, []);
  assert.equal(vad.inSpeech, true);
});

test('short clicks below minSpeechMs never start an utterance', () => {
  const vad = createVad({ threshold: 0.02, minSpeechMs: 200, gapToleranceMs: 250 });
  let t = 0;
  let events = [];
  for (let i = 0; i < 10; i++) {
    let ev;
    [ev, t] = feed(vad, t, 50, 0.3);        // 1 loud frame
    events = events.concat(ev);
    [ev, t] = feed(vad, t, 600, 0.0);       // then quiet
    events = events.concat(ev);
  }
  assert.deepEqual(events, []);
});

test('max utterance length forces an end', () => {
  const vad = createVad({ threshold: 0.02, maxUtteranceMs: 2000 });
  const [ev] = feed(vad, 0, 2200, 0.2);
  assert.deepEqual(ev.map((e) => e.type), ['speech_start', 'max_length']);
  assert.ok(ev[1].durationMs >= 2000);
});

test('barge-in threshold ignores quieter speaker echo', () => {
  const vad = createVad({ threshold: 0.015 });
  vad.setThreshold(VAD_DEFAULTS.bargeInThreshold);
  let [ev, t] = feed(vad, 0, 1000, 0.03);   // echo of TTS through the mic
  assert.deepEqual(ev, []);
  [ev, t] = feed(vad, t, 500, 0.2);         // the user talks over it
  assert.equal(ev[0].type, 'speech_start');
});

test('transcript gating drops empty, tiny and silence-hallucination transcripts', () => {
  for (const text of ['', '   ', '.', 'a', '¿?', 'Thank you.', ' you ', 'Subtítulos realizados por la comunidad de Amara.org']) {
    assert.equal(shouldSendTranscript(text), false, JSON.stringify(text));
  }
  for (const text of ['sí', 'Hola, ¿qué tiempo hace hoy?', 'Thank you for the summary', 'ok']) {
    assert.equal(shouldSendTranscript(text), true, JSON.stringify(text));
  }
});

function run(events, start = 'off') {
  let state = start;
  const log = [];
  for (const event of events) {
    const step = conversationStep(state, event);
    state = step.state;
    log.push([state, step.actions]);
  }
  return log;
}

test('full turn: listen, hear, transcribe, send, speak, re-arm', () => {
  const log = run([
    { type: 'enable' },
    { type: 'speech_start' },
    { type: 'speech_end', durationMs: 1500 },
    { type: 'transcript', text: '¿Qué hora es?' },
    { type: 'playback_start' },
    { type: 'turn_done' },
  ]);
  assert.deepEqual(log, [
    ['listening', ['arm']],
    ['recording', []],
    ['transcribing', ['end_capture']],
    ['thinking', ['send']],
    ['speaking', []],
    ['listening', ['arm']],
  ]);
});

test('reply without audio (TTS off) re-arms after the turn ends', () => {
  const log = run([{ type: 'turn_done' }], 'thinking');
  assert.deepEqual(log, [['listening', ['arm']]]);
});

test('very short utterances are discarded without transcription', () => {
  const log = run([{ type: 'speech_end', durationMs: 120 }], 'recording');
  assert.deepEqual(log, [['listening', ['discard', 'arm']]]);
});

test('empty or too-short transcripts are not sent', () => {
  assert.deepEqual(run([{ type: 'transcript', text: '' }], 'transcribing'),
    [['listening', ['notify_ignored', 'arm']]]);
  assert.deepEqual(run([{ type: 'transcript', text: 'Thank you.' }], 'transcribing'),
    [['listening', ['notify_ignored', 'arm']]]);
  assert.deepEqual(run([{ type: 'error' }], 'transcribing'), [['listening', ['arm']]]);
});

test('barge-in: speech during playback stops it and captures the user', () => {
  const log = run([
    { type: 'speech_start' },
    { type: 'speech_end', durationMs: 900 },
  ], 'speaking');
  assert.deepEqual(log, [
    ['recording', ['stop_playback', 'begin_capture']],
    ['transcribing', ['end_capture']],
  ]);
});

test('playback that starts while listening pauses the mic (no echo capture)', () => {
  assert.deepEqual(run([{ type: 'playback_start' }], 'listening'), [['speaking', ['pause_capture']]]);
});

test('max length sends what was heard even if long', () => {
  assert.deepEqual(run([{ type: 'max_length', durationMs: 30000 }], 'recording'),
    [['transcribing', ['end_capture']]]);
});

test('disable exits from any state and releases the mic; unrelated events are ignored', () => {
  for (const state of ['listening', 'recording', 'transcribing', 'thinking', 'speaking']) {
    assert.deepEqual(conversationStep(state, { type: 'disable' }), { state: 'off', actions: ['stop_playback', 'teardown'] });
  }
  assert.deepEqual(conversationStep('off', { type: 'disable' }), { state: 'off', actions: [] });
  assert.deepEqual(conversationStep('off', { type: 'speech_start' }), { state: 'off', actions: [] });
  assert.deepEqual(conversationStep('thinking', { type: 'speech_start' }), { state: 'thinking', actions: [] });
  assert.deepEqual(conversationStep('transcribing', { type: 'speech_end', durationMs: 900 }), { state: 'transcribing', actions: [] });
});

test('turn tracker waits for stream end and playback drain plus grace', () => {
  const tracker = createTurnTracker({ graceMs: 800, startTimeoutMs: 5000 });
  tracker.start(0);
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: false }, 100), null); // not started yet
  assert.equal(tracker.update({ chatBusy: true, ttsBusy: false }, 500), null);
  assert.equal(tracker.update({ chatBusy: true, ttsBusy: true }, 900), 'playback_start');
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: true }, 3000), null);
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: false }, 3100), null);
  // A sentence enqueued just after the stream ended resets the grace window.
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: true }, 3300), null);
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: false }, 3400), null);
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: false }, 4100), null);
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: false }, 4200), 'turn_done');
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: false }, 9000), null); // once
  assert.equal(tracker.active, false);
});

test('turn tracker ends a turn that never started after the timeout', () => {
  const tracker = createTurnTracker({ graceMs: 800, startTimeoutMs: 5000 });
  tracker.start(0);
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: false }, 4900), null);
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: false }, 5000), null);
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: false }, 5800), 'turn_done');
});

test('turn tracker can follow playback that began without a send', () => {
  const tracker = createTurnTracker({ graceMs: 500 });
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: true }, 0), null); // inactive
  tracker.watchPlayback(0);
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: true }, 50), 'playback_start');
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: false }, 1000), null);
  assert.equal(tracker.update({ chatBusy: false, ttsBusy: false }, 1500), 'turn_done');
});

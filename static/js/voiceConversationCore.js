// static/js/voiceConversationCore.js
//
// Pure logic for hands-free conversation mode (no DOM, no Web Audio, no
// timers): energy-based voice-activity detection, the conversation state
// machine, the "is the assistant turn over?" tracker, and transcript gating.
// static/js/voiceConversation.js wires these to the microphone, STT, chat
// submit and TTS playback; tests/voice_conversation_core.test.mjs covers them.

export const VAD_DEFAULTS = Object.freeze({
  threshold: 0.015,        // frame RMS (0..1) that counts as voice
  bargeInThreshold: 0.05,  // stricter while the assistant is speaking (speaker echo)
  minSpeechMs: 200,        // voiced time needed before an utterance starts
  gapToleranceMs: 250,     // silence allowed inside the onset window
  silenceMs: 1200,         // trailing silence that ends an utterance
  maxUtteranceMs: 30000,   // hard stop for one utterance
  minUtteranceMs: 350,     // shorter utterances are dropped, not transcribed
});

/** Root-mean-square level of time-domain samples in [-1, 1]. */
export function computeRms(samples) {
  if (!samples || !samples.length) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

/**
 * Energy VAD. Feed it one RMS value per analysis frame with a monotonic
 * timestamp; it returns an event object or null:
 *   { type: 'speech_start', at }
 *   { type: 'speech_end', durationMs }   after `silenceMs` of trailing silence
 *   { type: 'max_length', durationMs }   utterance reached `maxUtteranceMs`
 * `setThreshold()` switches levels (e.g. to the barge-in threshold).
 */
export function createVad(options = {}) {
  const cfg = { ...VAD_DEFAULTS, ...options };
  let threshold = cfg.threshold;
  let inSpeech = false;
  let onsetStart = null;    // first voiced frame of a candidate onset
  let voicedMs = 0;         // voiced time accumulated in the onset window
  let lastVoiceAt = null;
  let speechStartAt = null;
  let lastAt = null;

  function reset() {
    inSpeech = false;
    onsetStart = null;
    voicedMs = 0;
    lastVoiceAt = null;
    speechStartAt = null;
    lastAt = null;
  }

  function update(rms, now) {
    const dt = lastAt === null ? 0 : Math.max(0, now - lastAt);
    lastAt = now;
    const voiced = rms >= threshold;

    if (!inSpeech) {
      if (voiced) {
        if (onsetStart === null) { onsetStart = now; voicedMs = 0; }
        else voicedMs += dt;
        lastVoiceAt = now;
        if (voicedMs >= cfg.minSpeechMs) {
          inSpeech = true;
          speechStartAt = onsetStart;
          onsetStart = null;
          return { type: 'speech_start', at: speechStartAt };
        }
      } else if (onsetStart !== null && now - lastVoiceAt > cfg.gapToleranceMs) {
        onsetStart = null;  // a click or a breath, not speech
        voicedMs = 0;
      }
      return null;
    }

    if (voiced) lastVoiceAt = now;
    if (now - speechStartAt >= cfg.maxUtteranceMs) {
      const durationMs = now - speechStartAt;
      reset();
      return { type: 'max_length', durationMs };
    }
    if (!voiced && now - lastVoiceAt >= cfg.silenceMs) {
      const durationMs = lastVoiceAt - speechStartAt;
      reset();
      return { type: 'speech_end', durationMs };
    }
    return null;
  }

  return {
    update,
    reset,
    setThreshold(value) { threshold = value; },
    get threshold() { return threshold; },
    get inSpeech() { return inSpeech; },
  };
}

// Phrases Whisper-style models emit for silence or noise. Only an exact
// (normalized) match is dropped, so real sentences containing them still send.
const SILENCE_HALLUCINATIONS = new Set([
  'you',
  'thank you',
  'thanks for watching',
  'thank you for watching',
  'subtitles by the amara org community',
  'subtitulos realizados por la comunidad de amara org',
  'gracias por ver el video',
  'gracias por ver',
]);

function normalizeTranscript(text) {
  return String(text || '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim();
}

/** True when a transcript is worth auto-sending (not empty/too short/noise). */
export function shouldSendTranscript(text, { minChars = 2 } = {}) {
  const normalized = normalizeTranscript(text);
  if (normalized.replace(/\s/g, '').length < minChars) return false;
  return !SILENCE_HALLUCINATIONS.has(normalized);
}

/**
 * Conversation state machine.
 *   off -> listening -> recording -> transcribing -> thinking -> speaking -> listening ...
 * Returns { state, actions } where actions are commands for the browser glue:
 *   'arm'            (re)start listening for the next utterance
 *   'begin_capture'  start capturing now (barge-in: the recorder was paused)
 *   'end_capture'    stop capturing and transcribe what was heard
 *   'discard'        stop capturing and throw the audio away
 *   'pause_capture'  stop listening while the assistant speaks
 *   'send'           send event.text as the next chat message
 *   'stop_playback'  cut TTS playback (barge-in)
 *   'teardown'       release the microphone
 *   'notify_ignored' transcript was empty/too short and not sent
 */
export function conversationStep(state, event, options = {}) {
  const minUtteranceMs = options.minUtteranceMs ?? VAD_DEFAULTS.minUtteranceMs;
  const type = event && event.type;

  if (type === 'disable') {
    return state === 'off' ? { state, actions: [] } : { state: 'off', actions: ['stop_playback', 'teardown'] };
  }
  if (state === 'off') {
    return type === 'enable' ? { state: 'listening', actions: ['arm'] } : { state, actions: [] };
  }

  switch (state) {
    case 'listening':
      if (type === 'speech_start') return { state: 'recording', actions: [] };
      if (type === 'playback_start') return { state: 'speaking', actions: ['pause_capture'] };
      break;
    case 'recording':
      if (type === 'speech_end' || type === 'max_length') {
        if (type === 'speech_end' && (event.durationMs || 0) < minUtteranceMs) {
          return { state: 'listening', actions: ['discard', 'arm'] };
        }
        return { state: 'transcribing', actions: ['end_capture'] };
      }
      break;
    case 'transcribing':
      if (type === 'transcript') {
        if (shouldSendTranscript(event.text)) return { state: 'thinking', actions: ['send'] };
        return { state: 'listening', actions: ['notify_ignored', 'arm'] };
      }
      if (type === 'error') return { state: 'listening', actions: ['arm'] };
      break;
    case 'thinking':
      if (type === 'playback_start') return { state: 'speaking', actions: [] };
      if (type === 'turn_done') return { state: 'listening', actions: ['arm'] };
      break;
    case 'speaking':
      if (type === 'speech_start') return { state: 'recording', actions: ['stop_playback', 'begin_capture'] };
      if (type === 'turn_done') return { state: 'listening', actions: ['arm'] };
      break;
    default:
      break;
  }
  return { state, actions: [] };
}

/**
 * Decides when the assistant's turn is over after a message was sent: the
 * reply stream finished AND TTS playback drained, held for `graceMs` (TTS
 * sentences are enqueued slightly after the stream flag flips). If the chat
 * never reports busy (send rejected, empty reply), the turn ends after
 * `startTimeoutMs`. Feed update({ chatBusy, ttsBusy }, now); it returns
 * 'playback_start' once when audio starts, 'turn_done' once at the end.
 */
export function createTurnTracker({ graceMs = 800, startTimeoutMs = 5000 } = {}) {
  let startedAt = null;
  let sawBusy = false;
  let announcedPlayback = false;
  let idleSince = null;
  let done = true;

  return {
    start(now) {
      startedAt = now;
      sawBusy = false;
      announcedPlayback = false;
      idleSince = null;
      done = false;
    },
    /** Track playback that begins without a send (e.g. a queued reply). */
    watchPlayback(now) {
      if (done) this.start(now);
      sawBusy = true;
    },
    update({ chatBusy, ttsBusy }, now) {
      if (done) return null;
      if (chatBusy || ttsBusy) sawBusy = true;
      if (ttsBusy && !announcedPlayback) {
        announcedPlayback = true;
        idleSince = null;
        return 'playback_start';
      }
      if (chatBusy || ttsBusy) { idleSince = null; return null; }
      if (!sawBusy && now - startedAt < startTimeoutMs) return null;
      if (idleSince === null) idleSince = now;
      if (now - idleSince >= graceMs) { done = true; return 'turn_done'; }
      return null;
    },
    get active() { return !done; },
  };
}

/** Status label for each state (English UI strings). */
export const CONVERSATION_LABELS = Object.freeze({
  off: 'Conversation mode',
  listening: 'Listening…',
  recording: 'Hearing you…',
  transcribing: 'Transcribing…',
  thinking: 'Thinking…',
  speaking: 'Speaking…',
});

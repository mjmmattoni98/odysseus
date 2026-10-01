// static/js/voiceConversation.js
//
// Hands-free conversation mode: the mic listens, a browser-side energy VAD
// ends the utterance after a short silence, the configured STT (server
// faster-whisper / OpenAI-compatible endpoint, or the browser Web Speech API)
// transcribes it, the transcript is sent as a chat message, the reply is read
// aloud with TTS (sentence-by-sentence while streaming), and the mic re-arms
// when playback ends. Talking over the reply stops playback (barge-in).
// Esc or the toggle exits. Decisions live in voiceConversationCore.js.

import voiceRecorderModule from './voiceRecorder.js';
import {
  CONVERSATION_LABELS,
  VAD_DEFAULTS,
  computeRms,
  conversationStep,
  createTurnTracker,
  createVad,
} from './voiceConversationCore.js';

const FRAME_MS = 50;
// Listening with no speech for this long restarts the recorder so a silent
// wait never uploads minutes of audio.
const IDLE_RESTART_MS = 20000;
const RECOGNITION_STOP_TIMEOUT_MS = 2000;

let state = 'off';
let session = 0;                 // bumps on every enable; stale async results are dropped
let stream = null;
let audioCtx = null;
let analyser = null;
let samples = null;
let frameTimer = null;
let vad = null;
const tracker = createTurnTracker();
let prevAutoPlay = null;

let recorder = null;
let recorderChunks = [];
let captureStartedAt = 0;

let recognition = null;
let recognitionText = '';
let recognitionWanted = false;

let btn = null;
let labelEl = null;

function toast(msg, ms) {
  if (window.uiModule && window.uiModule.showToast) window.uiModule.showToast(msg, ms);
}
function showError(msg) {
  if (window.uiModule && window.uiModule.showError) window.uiModule.showError(msg);
  else toast(msg);
}

function sttProvider() { return voiceRecorderModule._sttProvider || 'disabled'; }
function usesBrowserStt() { return sttProvider() === 'browser'; }
function ttsManager() { return window.aiTTSManager || null; }
function ttsBusy() {
  const mgr = ttsManager();
  return !!(mgr && typeof mgr.isBusy === 'function' && mgr.isBusy());
}
function chatBusy() {
  return !!window.__odysseusChatBusy
    || Date.now() < (window.__odysseusChatBusyUntil || 0)
    || !!document.querySelector('.send-btn[data-mode="streaming"], .send-btn.send-pending');
}

// ── State machine plumbing ──

function dispatch(event) {
  const prev = state;
  const step = conversationStep(state, event, { minUtteranceMs: VAD_DEFAULTS.minUtteranceMs });
  state = step.state;
  if (state !== prev) onEnter(state);
  for (const action of step.actions) runAction(action, event);
  if (state !== prev) render();
}

function onEnter(next) {
  if (!vad) return;
  // While the assistant speaks, only clearly louder input (the user, not the
  // speaker echo) counts as a barge-in.
  vad.setThreshold(next === 'speaking' ? VAD_DEFAULTS.bargeInThreshold : VAD_DEFAULTS.threshold);
  if (next === 'speaking' || next === 'listening') vad.reset();
}

function runAction(action, event) {
  switch (action) {
    case 'arm':
      startCapture();
      break;
    case 'begin_capture':
      startCapture();
      break;
    case 'end_capture':
      finishCapture();
      break;
    case 'discard':
    case 'pause_capture':
      stopCapture();
      break;
    case 'send':
      sendTranscript(event.text);
      break;
    case 'stop_playback': {
      const mgr = ttsManager();
      if (mgr && mgr.isBusy && mgr.isBusy()) mgr.stop();
      break;
    }
    case 'teardown':
      teardown();
      break;
    case 'notify_ignored':
      toast("Didn't catch that. Listening again.", 1500);
      break;
    default:
      break;
  }
}

// ── Audio frames ──

function onFrame() {
  if (!analyser || state === 'off') return;
  analyser.getFloatTimeDomainData(samples);
  const now = performance.now();
  const ev = vad.update(computeRms(samples), now);
  if (ev) dispatch(ev);
  if (state === 'off') return;

  const playing = ttsBusy();
  // Playback that starts outside a tracked turn (e.g. a reply that was queued
  // before a barge-in) must still pause the mic and re-arm when it ends.
  if (playing && !tracker.active && (state === 'listening' || state === 'speaking')) {
    tracker.watchPlayback(now);
  }
  const turn = tracker.update({ chatBusy: chatBusy(), ttsBusy: playing }, now);
  if (turn) dispatch({ type: turn });

  if (state === 'listening' && recorder && !vad.inSpeech && now - captureStartedAt > IDLE_RESTART_MS) {
    startCapture();
  }
}

// ── Capture: MediaRecorder (server STT) or Web Speech (browser STT) ──

function startCapture() {
  stopCapture();
  captureStartedAt = performance.now();
  if (usesBrowserStt()) {
    startRecognition();
    return;
  }
  if (!stream) return;
  recorderChunks = [];
  const options = (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported('audio/webm'))
    ? { mimeType: 'audio/webm' } : undefined;
  try {
    recorder = new MediaRecorder(stream, options);
  } catch (e) {
    recorder = null;
    showError('Microphone recorder unavailable: ' + e.message);
    return;
  }
  recorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) recorderChunks.push(e.data); };
  recorder.start();
}

function stopCapture() {
  if (recorder) {
    const rec = recorder;
    recorder = null;
    rec.ondataavailable = null;
    rec.onstop = null;
    try { if (rec.state !== 'inactive') rec.stop(); } catch (_) { /* ignore */ }
  }
  stopRecognition();
}

function stopRecorderForBlob() {
  return new Promise((resolve) => {
    const rec = recorder;
    recorder = null;
    if (!rec || rec.state === 'inactive') { resolve(null); return; }
    rec.onstop = () => resolve(new Blob(recorderChunks, { type: rec.mimeType || 'audio/webm' }));
    try { rec.stop(); } catch (_) { resolve(null); }
  });
}

async function finishCapture() {
  const token = session;
  let text = '';
  try {
    if (usesBrowserStt()) {
      text = await stopRecognitionForText();
    } else {
      const blob = await stopRecorderForBlob();
      if (token !== session) return;
      text = blob && blob.size ? await voiceRecorderModule.transcribeOnServer(blob) : '';
    }
  } catch (e) {
    if (token !== session) return;
    showError('Transcription failed: ' + e.message);
    dispatch({ type: 'error' });
    return;
  }
  if (token !== session) return;
  dispatch({ type: 'transcript', text: (text || '').trim() });
}

function startRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return;
  recognitionText = '';
  recognitionWanted = true;
  const rec = new SR();
  rec.continuous = true;
  rec.interimResults = false;
  rec.lang = voiceRecorderModule.browserSttLang() || navigator.language || '';
  rec.onresult = (event) => {
    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) recognitionText += event.results[i][0].transcript + ' ';
    }
  };
  rec.onerror = (e) => {
    if (e.error !== 'no-speech' && e.error !== 'aborted') console.warn('Conversation STT error:', e.error);
  };
  rec.onend = () => {
    if (rec._onStopped) { rec._onStopped(); return; }
    // Chrome ends continuous recognition on its own after a while; keep
    // listening as long as we still want it.
    if (recognition === rec && recognitionWanted && (state === 'listening' || state === 'recording')) {
      setTimeout(() => { if (recognition === rec && recognitionWanted) { try { rec.start(); } catch (_) {} } }, 250);
    }
  };
  recognition = rec;
  try { rec.start(); } catch (e) { console.warn('Conversation STT start failed:', e); }
}

function stopRecognition() {
  recognitionWanted = false;
  if (!recognition) return;
  const rec = recognition;
  recognition = null;
  try { rec.abort(); } catch (_) { /* ignore */ }
}

function stopRecognitionForText() {
  return new Promise((resolve) => {
    const rec = recognition;
    recognitionWanted = false;
    recognition = null;
    if (!rec) { resolve(recognitionText); return; }
    // Final results can arrive after stop(); wait for onend (bounded).
    const done = () => { clearTimeout(timer); resolve(recognitionText); };
    const timer = setTimeout(done, RECOGNITION_STOP_TIMEOUT_MS);
    rec._onStopped = done;
    try { rec.stop(); } catch (_) { done(); }
  });
}

// ── Chat ──

function sendTranscript(text) {
  const input = document.getElementById('message');
  const form = document.getElementById('chat-form');
  tracker.start(performance.now());
  if (!input || !form) return;
  const existing = input.value.trim();
  input.value = existing ? existing + ' ' + text : text;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  // Same path as pressing Enter; a reply still streaming (after a barge-in)
  // queues this message instead of dropping it.
  if (chatBusy()) window.__odysseusQueueStreamingSubmit = Date.now();
  if (form.requestSubmit) form.requestSubmit();
  else form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
}

// ── Enable / disable ──

async function enable() {
  if (state !== 'off') return;
  if (sttProvider() === 'disabled') {
    showError('Set up speech-to-text (Settings) to use conversation mode.');
    return;
  }
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError('Microphone requires HTTPS. Use a reverse proxy with SSL or access via localhost.');
    return;
  }
  if (usesBrowserStt() && !(window.SpeechRecognition || window.webkitSpeechRecognition)) {
    showError('This browser has no speech recognition. Choose a server STT provider in Settings.');
    return;
  }
  if (voiceRecorderModule.getIsRecording()) voiceRecorderModule.stopRecording();

  const token = ++session;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (error) {
    if (error.name === 'NotAllowedError') showError('Microphone access denied. Check browser permissions.');
    else if (error.name === 'NotFoundError') showError('No microphone found.');
    else showError('Microphone error: ' + error.message);
    return;
  }
  if (token !== session) { stream.getTracks().forEach((t) => t.stop()); stream = null; return; }

  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctx();
    if (audioCtx.resume) await audioCtx.resume();
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    samples = new Float32Array(analyser.fftSize);
  } catch (e) {
    showError('Audio analysis unavailable: ' + e.message);
    teardown();
    return;
  }

  vad = createVad();
  const mgr = ttsManager();
  if (mgr) {
    prevAutoPlay = mgr.autoPlay;
    mgr.autoPlay = true;
    if (!mgr.available) toast('Text-to-speech is off: replies will not be read aloud.', 4000);
  }
  frameTimer = setInterval(onFrame, FRAME_MS);
  dispatch({ type: 'enable' });
}

function disable() {
  dispatch({ type: 'disable' });
}

function teardown() {
  session++;
  if (frameTimer) { clearInterval(frameTimer); frameTimer = null; }
  stopCapture();
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  if (audioCtx) { try { audioCtx.close(); } catch (_) {} audioCtx = null; }
  analyser = null;
  samples = null;
  vad = null;
  const mgr = ttsManager();
  if (mgr && prevAutoPlay !== null) mgr.autoPlay = prevAutoPlay;
  prevAutoPlay = null;
  if (state !== 'off') { state = 'off'; }
  render();
}

// ── UI ──

function render() {
  if (!btn) return;
  const active = state !== 'off';
  btn.dataset.convState = state;
  btn.classList.toggle('active', active);
  btn.setAttribute('aria-pressed', active ? 'true' : 'false');
  const label = active ? CONVERSATION_LABELS[state] : '';
  if (labelEl) labelEl.textContent = label;
  btn.title = active
    ? label + ' Click or press Esc to end conversation mode.'
    : 'Conversation mode: talk hands-free (Esc to exit)';
}

function syncVisibility() {
  if (!btn) return;
  const configured = sttProvider() !== 'disabled';
  btn.hidden = !configured && state === 'off';
}

function injectStyles() {
  if (document.getElementById('conversation-mode-style')) return;
  const style = document.createElement('style');
  style.id = 'conversation-mode-style';
  style.textContent = `
.conversation-mode-btn { gap: 4px; }
.conversation-mode-btn[hidden] { display: none !important; }
.conversation-mode-btn .conversation-mode-label { font-size: 11px; white-space: nowrap; }
.conversation-mode-btn .conversation-mode-label:empty { display: none; }
.conversation-mode-btn[data-conv-state="listening"],
.conversation-mode-btn[data-conv-state="recording"] { color: var(--accent, var(--red, #e55)); }
.conversation-mode-btn[data-conv-state="recording"] svg { animation: conv-pulse 0.9s ease-in-out infinite; }
.conversation-mode-btn[data-conv-state="transcribing"],
.conversation-mode-btn[data-conv-state="thinking"] { opacity: 0.8; }
.conversation-mode-btn[data-conv-state="speaking"] { color: var(--fg); }
@keyframes conv-pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.18); } }
`;
  document.head.appendChild(style);
}

function init() {
  btn = document.getElementById('conversation-mode-btn');
  if (!btn) return;
  labelEl = btn.querySelector('.conversation-mode-label');
  injectStyles();
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    if (state === 'off') enable();
    else disable();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state !== 'off') disable();
  });
  window.addEventListener('beforeunload', () => { if (state !== 'off') disable(); });
  syncVisibility();
  // The STT provider arrives asynchronously and can change from Settings.
  setInterval(syncVisibility, 2000);
  render();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();

const voiceConversationModule = {
  enable,
  disable,
  get state() { return state; },
};
window.voiceConversationModule = voiceConversationModule;
export default voiceConversationModule;

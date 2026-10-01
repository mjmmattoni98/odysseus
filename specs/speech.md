# Speech

Last updated: dev@e71f8ce | 2026-08-25

## Scope

This spec covers speech behavior in:

- app service initialization and route registration in `app.py`;
- `services/stt/stt_service.py`;
- `services/tts/tts_service.py`;
- `routes/stt_routes.py`;
- `routes/tts_routes.py`;
- `src/upload_limits.py`;
- settings defaults/cache in `src/settings.py`;
- settings routes in `routes/auth_routes.py`;
- model endpoint cleanup in `routes/model_routes.py`;
- settings/tool aliases in `src/tool_implementations.py`;
- frontend modules `static/js/voiceRecorder.js`, `static/js/tts-ai.js`, `static/js/voiceConversation.js`, `static/js/voiceConversationCore.js`, `static/app.js`, `static/js/chat.js`, `static/js/slashCommands.js`, `static/js/keyboard-shortcuts.js`, `static/js/settings.js`, and `static/index.html`;
- optional dependency declarations in `requirements-optional.txt`;
- runtime cache path `data/tts_cache/`;
- tests covering speech service toggles, TTS speed/cache, STT temp cleanup, upload limits, settings scrubbing, and model endpoint cleanup.

## Current Call Sites Include

- chat mic/send button behavior;
- browser and server STT recording paths;
- chat message read-aloud buttons and streaming TTS queueing;
- `/tts` slash command playback;
- keyboard shortcut TTS activation;
- admin/settings API writes and `manage_settings` aliases;
- model endpoint deletion cleanup for `endpoint:<id>` speech providers.

## STT

`services.stt.STTService` owns speech-to-text provider behavior. `routes/stt_routes.py` owns `/api/stt/transcribe` and `/api/stt/stats`. `static/js/voiceRecorder.js` owns microphone capture, browser STT, server upload, and audio-attachment fallback.

Provider runtime:

- `disabled` returns unavailable and avoids provider calls;
- `browser` is client-side only through Web Speech API and does not call `/api/stt/transcribe`;
- `local` lazily imports `faster-whisper`, writes uploaded audio to a temporary WebM file, transcribes, and deletes the temp file in `finally`;
- `endpoint:<id>` resolves a `ModelEndpoint` and posts `audio.webm` to `/audio/transcriptions` with model and optional language.
- `stt_language` is normalized by `normalize_stt_language()` before local/endpoint calls: BCP-47 tags keep their primary subtag (`es-ES` -> `es`), a few English language names map to codes, and `''`/`auto`/unrecognized values mean auto-detect. Browser STT uses the raw setting as the recognition `lang`.

Route behavior:

- audio uploads are capped by the shared STT upload limit from `src.upload_limits`, including environment override validation;
- empty uploads return a route error;
- availability probes, transcription and stats run in the threadpool so model loading/transcription does not block the event loop;
- uploaded content type, extension, and magic bytes are not strongly validated today;
- endpoint providers report optimistic availability and fail at request time if offline/misconfigured.

Frontend behavior:

- browser recording needs secure context and microphone permissions;
- server transcription success inserts text into the input;
- failed server transcription can attach the recorded audio file to chat instead; empty transcription shows a no-speech message.

## Conversation Mode

`static/js/voiceConversation.js` owns the hands-free toggle (`#conversation-mode-btn`, next to the send/mic button, shown when STT is not disabled); its decisions live in the DOM-free `static/js/voiceConversationCore.js` (tested with `node --test tests/voice_conversation_core.test.mjs`).

- The mic stream (echo cancellation on) feeds a Web Audio analyser; an RMS energy VAD starts an utterance after ~200 ms of voice and ends it after 1.2 s of silence or 30 s total. Utterances under ~350 ms are discarded.
- Server STT records with `MediaRecorder` while listening and uploads the utterance to `/api/stt/transcribe`; browser STT runs Web Speech recognition with the `stt_language` tag.
- Empty, one-character, and known silence-hallucination transcripts ("Thank you.", Amara subtitles) are not sent; otherwise the transcript is appended to the composer and submitted like Enter (queued if a reply is still streaming).
- While active it forces `aiTTSManager.autoPlay` on (restored on exit), so replies are read sentence-by-sentence while streaming. The turn ends when the chat stream is idle and TTS drained for a short grace period (or a start timeout passes); the mic then re-arms.
- The mic is paused during playback. A louder barge-in threshold stays armed: speaking over the reply stops TTS and starts a new capture (the first ~200 ms of the barge-in utterance can be clipped).
- States are shown on the button (Listening / Hearing you / Transcribing / Thinking / Speaking). The toggle or Esc exits and releases the microphone.

## TTS

`services.tts.TTSService` owns text-to-speech provider behavior, speed parsing, cache behavior, and local/provider-specific synthesis. `routes/tts_routes.py` owns `/api/tts/stats`, `/api/tts/synthesize`, and cache clearing. `static/js/tts-ai.js` owns frontend playback, client object-URL caching, browser TTS, queueing, and streaming button state.

Provider runtime:

- `disabled` returns unavailable and avoids provider calls;
- `browser` is client-side only through `speechSynthesis`;
- `local` currently means Kokoro and requires `torch`, `kokoro`, `soundfile`, and CUDA/import availability. The Kokoro `lang_code` is the voice id's first letter (`a` en-US, `b` en-GB, `e` es, `f` fr, `h` hi, `i` it, `j` ja, `p` pt-BR, `z` zh); non-Kokoro voice names fall back to `af_heart`. One pipeline per language is built lazily, cached, and shares the first pipeline's model. A language whose G2P dependencies are missing (espeak-ng for es/fr/hi/it/pt, `misaki[ja]`/`misaki[zh]`) records a clear error, retried at most every 60 s, and fails only that synthesis; the engine stays available for other languages. Non-English text is split per sentence before G2P so long input is not truncated;
- `endpoint:<id>` resolves a `ModelEndpoint` and posts to `/audio/speech`.
- unknown or non-string `tts_provider` values are treated as unavailable rather
  than being parsed as endpoint strings.

Route behavior:

- `/api/tts/synthesize` supports binary `audio` responses and JSON `base64` responses;
- `/api/tts/voices` lists Kokoro voices grouped by language for the voice selector;
- synthesis, availability probes and stats run in the threadpool; a failed local synthesis returns 500 with the recorded reason (e.g. missing espeak-ng for Spanish), and local stats include `lang_code`, `language` and any `error`;
- binary responses choose WAV or MP3 MIME by audio magic bytes;
- synthesis input is passed to the service as submitted and capped there;
- malformed or nonpositive `tts_speed` falls back to `1.0`;
- provider unavailable returns 503; failed synthesis/transcription generally returns route-level failure.

## Settings, Endpoints, And Cache

Speech providers are global settings under `data/settings.json`, with defaults in `src/settings.py`. Settings reads are scrubbed for non-admin callers, writes are admin-only, and `manage_settings` can change non-secret speech settings through aliases.

The AI settings panel shows the TTS card (provider, model, voice, speed, preview) and an STT card (enable, provider, Whisper model, language). For the local provider the voice is a language-grouped Kokoro select; endpoint providers also list the Kokoro voice ids for Kokoro-compatible OpenAI servers. The overflow "TTS Mode" toggle stays hidden.

`routes.model_routes` clears `tts_provider` and `stt_provider` references when a referenced model endpoint is deleted.

TTS cache behavior:

- server cache lives under `data/tts_cache/`;
- cache keys include provider, model, voice, safe speed, and text;
- cache files are stored as MP3 or WAV;
- route stats expose global cache state;
- cache clear is global;
- frontend TTS has a separate object-URL cache.

`ODYSSEUS_TTS_CACHE_MAX_BYTES` bounds server cache growth and is forwarded by all Compose variants. The default is 500 MiB; invalid integers fall back to that default and values at or below zero disable eviction. After a cache write, enforcement scans only `.mp3`/`.wav`, ignores files that disappear or cannot be stated, and when over limit removes oldest-by-mtime entries toward 80% of the ceiling. Sort/stat/unlink failures are logged and do not fail synthesis.

## Security And Provenance

Speech routes rely on app-wide authentication and do not implement route-local admin or scope checks. Bearer-token callers that pass app auth can reach speech stats/synthesis/transcription/cache-clear surfaces using global speech settings.

Endpoint providers send user audio or assistant text to configured `ModelEndpoint` URLs with optional bearer keys. Endpoint lookup is by configured endpoint ID and currently does not enforce per-request owner filtering. `ModelEndpoint.api_key` is encrypted at rest and forwarded only process-side.

Microphone audio, uploaded audio, endpoint transcripts, and assistant text sent to TTS are untrusted/user/provider-visible data flows. Transcripts become user input; they are not trusted system instructions.

TTS cached audio can contain sensitive assistant text rendered as speech. The cache is global, has no owner partition or TTL, and is served inline/base64 by POST responses without a dedicated generated-file route.

## Degraded Behavior

- Optional local speech packages may be absent.
- Local STT can run CPU-only and tolerates missing/broken torch by falling back to CPU/int8 behavior.
- Local TTS/Kokoro extras are declared as `kokoro==0.9.4` plus `soundfile` only for Python 3.11-3.12; Python 3.13+ intentionally skips them because Kokoro excludes those runtimes, so the Python 3.14 Docker image has no local Kokoro (and no espeak-ng). Even where installed, local Kokoro remains unavailable without a CUDA-capable torch build/GPU. In Docker, use browser TTS or an OpenAI-compatible Kokoro server as an endpoint provider.
- External endpoint providers can be offline or misconfigured and may only fail at request time.
- Browser `speechSynthesis`, `SpeechRecognition`, `webkitSpeechRecognition`, secure context, and microphone permissions can be absent.
- Docker GPU overlays are passthrough-only and do not install speech engines by themselves.
- Optional dependency errors and route error wording are not fully consistent across STT and TTS.

## Testing Coverage

Existing coverage includes speech service toggles, malformed/non-string TTS provider and speed handling, cache stats plus configured eviction/disable/file filtering/error handling, STT temp cleanup, direct upload limits, model routes, and settings scrubbing.

Missing coverage includes route-level STT/TTS success and failure shapes, auth/API-token behavior, endpoint owner isolation, STT type/magic rejection, TTS request-size/no-store/cache privacy behavior, degraded optional dependency paths, and frontend recorder/TTS fallback states.

## Current Gaps

- Speech routes need a deliberate API-token/scope policy.
- Endpoint speech providers need owner-isolation or explicit global-settings documentation.
- TTS cache needs privacy policy: owner partition, TTL, no-store response headers, or accepted global cache semantics.
- STT upload validation needs content type/extension/magic-byte policy.
- Browser/compare STT mic behavior needs a product decision or regression test because compare can force send-button visuals while shared empty-input logic can start recording.

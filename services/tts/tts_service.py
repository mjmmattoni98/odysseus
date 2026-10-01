# src/tts_service.py
"""Multi-provider TTS service — dispatches to local Kokoro, OpenAI-compatible API, or browser."""

import io
import os
import re
import time
import wave
import logging
import hashlib
import threading
import httpx
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.constants import TTS_CACHE_DIR

logger = logging.getLogger(__name__)


def _safe_speed(value, default: float = 1.0) -> float:
    """Parse the stored tts_speed defensively. The settings layer tolerates
    corrupt/agent-written config, so a non-numeric or empty value (e.g. an agent
    setting "speech speed" = "fast", or a hand-edited settings.json) must not
    crash synthesis or the stats endpoint with a ValueError."""
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return default
    return speed if speed > 0 else default


# ── Kokoro voices / languages ──
# Kokoro-82M picks its G2P (phonemizer) per pipeline via `lang_code`, and every
# voice id starts with the letter of the language it was trained for
# (`ef_dora` -> 'e' Spanish). Reading Spanish text through the English pipeline
# gives English phonetics, so the pipeline language is derived from the voice.
KOKORO_LANGUAGES = {
    "a": "English (US)",
    "b": "English (UK)",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Portuguese (BR)",
    "z": "Chinese (Mandarin)",
}

KOKORO_VOICES = {
    "a": [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
        "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
        "am_onyx", "am_puck", "am_santa",
    ],
    "b": [
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ],
    "e": ["ef_dora", "em_alex", "em_santa"],
    "f": ["ff_siwis"],
    "h": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "i": ["if_sara", "im_nicola"],
    "j": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "p": ["pf_dora", "pm_alex", "pm_santa"],
    "z": [
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ],
}

DEFAULT_KOKORO_VOICE = "af_heart"

# `af_heart`, blends like `ef_dora,em_alex`, or a local `.pt` voice file.
_KOKORO_VOICE_RE = re.compile(r"^[a-z][fm]_[a-z0-9_]+(,[a-z][fm]_[a-z0-9_]+)*$")

# Sentence ends for languages whose Kokoro G2P does not chunk long input
# (everything but English): split there so no chunk exceeds the ~510-phoneme
# limit and gets silently truncated.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?;:\u2026])\s+|(?<=[\u3002\uff01\uff1f\uff1b])")


def resolve_kokoro_voice(voice) -> str:
    """Return a voice id Kokoro can load. Non-Kokoro names (e.g. the OpenAI
    default `alloy` left over from another provider) fall back to af_heart."""
    v = (voice if isinstance(voice, str) else "").strip()
    if v.lower().endswith(".pt"):
        return v
    v = v.lower()
    return v if _KOKORO_VOICE_RE.match(v) else DEFAULT_KOKORO_VOICE


def kokoro_lang_code(voice) -> str:
    """Kokoro `lang_code` for a voice: the first letter of the voice id when it
    names a supported language, else American English ('a')."""
    v = resolve_kokoro_voice(voice)
    first = os.path.basename(v)[:1].lower()
    return first if first in KOKORO_LANGUAGES else "a"


def kokoro_voice_groups() -> List[Dict[str, Any]]:
    """Kokoro voices grouped by language, in selector order."""
    return [
        {"lang_code": code, "language": name, "voices": list(KOKORO_VOICES.get(code, []))}
        for code, name in KOKORO_LANGUAGES.items()
    ]


def split_sentences_for_g2p(text: str) -> str:
    """Put each sentence on its own line; Kokoro splits on newlines by default."""
    return _SENTENCE_END_RE.sub("\n", text or "")


def _describe_kokoro_error(lang_code: str, exc: Exception) -> str:
    language = KOKORO_LANGUAGES.get(lang_code, lang_code)
    if lang_code == "j":
        hint = " Japanese needs the misaki Japanese extra: pip install 'misaki[ja]'."
    elif lang_code == "z":
        hint = " Mandarin needs the misaki Chinese extra: pip install 'misaki[zh]'."
    elif lang_code in ("a", "b"):
        hint = ""
    else:
        hint = (" This language uses the espeak-ng phonemizer: install espeak-ng "
                "(apt-get install espeak-ng) or pip install espeakng-loader phonemizer-fork.")
    return f"Kokoro could not load {language} (lang_code '{lang_code}'): {exc}.{hint}"


class TTSService:
    """Multi-provider TTS service.

    Reads provider config from data/settings.json on each call.
    Providers:
      "disabled"        — no TTS
      "browser"         — client-side Web Speech API (no server synthesis)
      "local"           — Kokoro-82M on GPU
      "endpoint:<id>"   — OpenAI-compatible /audio/speech via ModelEndpoint
    """

    def __init__(self, cache_dir: str = TTS_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._kokoro = None  # lazy-init
        
        try:
            self.max_cache_bytes = int(os.getenv("ODYSSEUS_TTS_CACHE_MAX_BYTES", 500 * 1024 * 1024))
        except ValueError:
            self.max_cache_bytes = 500 * 1024 * 1024

    # ── Settings ──

    def _load_settings(self) -> dict:
        from src.settings import load_settings
        saved = load_settings()
        return {
            "tts_enabled": saved.get("tts_enabled", True),
            "tts_provider": saved.get("tts_provider", "disabled"),
            "tts_model": saved.get("tts_model", "tts-1"),
            "tts_voice": saved.get("tts_voice", "alloy"),
            "tts_speed": saved.get("tts_speed", "1"),
        }

    @property
    def available(self) -> bool:
        settings = self._load_settings()
        if settings.get("tts_enabled") is False:
            return False
        provider = settings["tts_provider"]
        if provider == "disabled":
            return False
        if provider == "browser":
            return True  # handled client-side
        if provider == "local":
            kokoro = self._get_kokoro()
            return kokoro is not None and kokoro.available
        if isinstance(provider, str) and provider.startswith("endpoint:"):
            return True  # assume reachable; errors surface at synthesis time
        return False

    # ── Cache ──

    def _cache_key(self, text: str, provider: str, model: str, voice: str, speed: float = 1.0) -> str:
        raw = f"{provider}|{model}|{voice}|{speed}|{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[bytes]:
        for ext in (".mp3", ".wav"):
            path = self.cache_dir / f"{key}{ext}"
            if path.exists():
                return path.read_bytes()
        return None

    def _put_cache(self, key: str, data: bytes):
        ext = ".mp3" if (len(data) >= 3 and (data[:3] == b'ID3' or (data[0] == 0xff and (data[1] & 0xe0) == 0xe0))) else ".wav"
        (self.cache_dir / f"{key}{ext}").write_bytes(data)

        self._enforce_cache_limit()

    def _enforce_cache_limit(self):
            """Evicts oldest files if the cache exceeds the configured byte limit."""
            if self.max_cache_bytes <= 0:
                return

            try:
                files = []
                total_size = 0

                # Safely scan files and sum sizes, ignoring files deleted mid-scan
                for f in self.cache_dir.iterdir():
                    try:
                        if f.is_file() and f.suffix.lower() in (".mp3", ".wav"):
                            files.append(f)
                            total_size += f.stat().st_size
                    except OSError:
                        continue

                if total_size > self.max_cache_bytes:
                    logger.info(
                        f"TTS cache ({total_size} bytes) exceeded limit ({self.max_cache_bytes} bytes). Evicting oldest files."
                    )

                    # Sort files by modification time (oldest first)
                    try:
                        files.sort(key=lambda f: f.stat().st_mtime)
                    except OSError as e:
                        logger.warning(f"Failed to sort cache files by mtime: {e}")

                    # Trim down to 80% of max capacity
                    target_size = self.max_cache_bytes * 0.8

                    while files and total_size > target_size:
                        f = files.pop(0)
                        try:
                            size = f.stat().st_size
                            f.unlink()
                            total_size -= size
                        except OSError as e:
                            logger.warning(f"Failed to evict cache file {f}: {e}")
                            continue

            except Exception as e:
                logger.warning(f"Error enforcing TTS cache limit: {e}", exc_info=True)

    def clear_cache(self):
        count = 0
        for f in self.cache_dir.glob("*.*"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} cached TTS files")

    # ── Kokoro (local) ──

    def _get_kokoro(self):
        if self._kokoro is None:
            voice = self._load_settings().get("tts_voice", DEFAULT_KOKORO_VOICE)
            self._kokoro = _KokoroPipeline(kokoro_lang_code(voice))
        return self._kokoro

    def failure_reason(self) -> str:
        """Human-readable reason the configured provider cannot synthesize
        right now ("" when unknown). Derived from cached pipeline state, so it
        is safe to call after a failed synthesize() from any request."""
        try:
            settings = self._load_settings()
            if settings.get("tts_provider") != "local":
                return ""
            kokoro = self._kokoro
            if kokoro is None or not kokoro.available:
                return "Kokoro TTS is not available (needs the kokoro package and a CUDA-capable torch)."
            return kokoro.error_for(kokoro_lang_code(settings.get("tts_voice")))
        except Exception:
            return ""

    # ── API endpoint ──

    def _synthesize_api(self, text: str, endpoint_id: str, model: str, voice: str, speed: float = 1.0) -> Optional[bytes]:
        from src.database import SessionLocal, ModelEndpoint

        db = SessionLocal()
        try:
            ep = db.query(ModelEndpoint).filter(ModelEndpoint.id == endpoint_id).first()
            if not ep:
                logger.error(f"TTS endpoint {endpoint_id} not found")
                return None
            base_url = ep.base_url.rstrip("/")
            api_key = ep.api_key
        finally:
            db.close()

        url = base_url + "/audio/speech"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": "mp3",
            "speed": speed,
        }

        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            logger.info(f"API TTS: {len(r.content)} bytes from {base_url}")
            return r.content
        except Exception as e:
            logger.error(f"API TTS synthesis failed: {e}")
            return None

    # ── Public interface ──

    def synthesize(self, text: str, use_cache: bool = True) -> Optional[bytes]:
        settings = self._load_settings()
        if settings.get("tts_enabled") is False:
            return None
        provider = settings["tts_provider"]
        model = settings["tts_model"]
        voice = settings["tts_voice"]
        speed = _safe_speed(settings.get("tts_speed", "1"))

        if provider in ("disabled", "browser"):
            return None

        if len(text) > 5000:
            text = text[:5000]

        if use_cache:
            key = self._cache_key(text, provider, model, voice, speed)
            cached = self._get_cached(key)
            if cached:
                logger.info(f"TTS cache hit ({len(text)} chars)")
                return cached

        audio_data = None

        if provider == "local":
            kokoro = self._get_kokoro()
            if kokoro and kokoro.available:
                audio_data = kokoro.synthesize_raw(text, voice)
            else:
                logger.warning("Kokoro TTS not available")
                return None
        elif provider.startswith("endpoint:"):
            endpoint_id = provider.split(":", 1)[1]
            audio_data = self._synthesize_api(text, endpoint_id, model, voice, speed)
        else:
            logger.error(f"Unknown TTS provider: {provider}")
            return None

        if audio_data and use_cache:
            key = self._cache_key(text, provider, model, voice, speed)
            self._put_cache(key, audio_data)

        return audio_data

    def synthesize_to_base64(self, text: str) -> Optional[str]:
        import base64
        audio = self.synthesize(text)
        if audio:
            return base64.b64encode(audio).decode("utf-8")
        return None

    def set_voice(self, voice: str):
        """Legacy no-op — voice is now managed via admin settings."""

    def get_stats(self) -> Dict[str, Any]:
        settings = self._load_settings()
        provider = settings["tts_provider"]
        tts_enabled = settings.get("tts_enabled", True)

        cache_files = list(self.cache_dir.glob("*.wav")) + list(self.cache_dir.glob("*.mp3"))
        cache_size = sum(f.stat().st_size for f in cache_files)

        is_available = self.available and tts_enabled
        stats = {
            "available": is_available,
            "ready": is_available,
            "provider": provider,
            "model": settings["tts_model"],
            "voice": settings["tts_voice"],
            "speed": _safe_speed(settings.get("tts_speed", "1")),
            "cache_entries": len(cache_files),
            "cache_size_mb": round(cache_size / (1024 * 1024), 2),
        }

        if provider == "local":
            kokoro = self._get_kokoro()
            stats["model"] = "Kokoro-82M (GPU)" if (kokoro and kokoro.available) else "Kokoro (not loaded)"
            lang_code = kokoro_lang_code(settings["tts_voice"])
            stats["lang_code"] = lang_code
            stats["language"] = KOKORO_LANGUAGES.get(lang_code, "")
            error = kokoro.error_for(lang_code) if (kokoro and kokoro.available) else ""
            if error:
                stats["error"] = error
        elif provider == "browser":
            stats["model"] = "Browser (Web Speech API)"
        elif provider.startswith("endpoint:"):
            stats["endpoint_id"] = provider.split(":", 1)[1]

        return stats


class _KokoroPipeline:
    """Encapsulates the Kokoro-82M local GPU pipelines, one per language.

    A KPipeline binds one G2P language, so a Spanish voice needs a 'e'
    pipeline. Pipelines are built lazily per lang_code, cached, and share the
    first pipeline's model weights. A language whose G2P dependencies are
    missing records a clear error (retried at most every _RETRY_AFTER_S) and
    fails that request instead of crashing the engine.
    """

    _RETRY_AFTER_S = 60.0

    def __init__(self, lang_code: str = "a"):
        self.available = False
        self.device = None
        self._pipelines: Dict[str, Any] = {}
        self._errors: Dict[str, tuple] = {}  # lang_code -> (monotonic ts, message)
        self._lock = threading.Lock()
        self._init(lang_code)

    @property
    def pipeline(self):
        """First loaded pipeline (kept for callers of the single-pipeline API)."""
        return next(iter(self._pipelines.values()), None)

    def _init(self, lang_code: str):
        try:
            import torch
            import kokoro  # noqa: F401 — import probe; pipelines are built per language
        except ImportError as e:
            logger.warning(f"Kokoro TTS not available: {e}")
            logger.warning("Install with: pip install kokoro soundfile")
            return
        except Exception as e:
            logger.error(f"Kokoro init failed: {e}", exc_info=True)
            return
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available for Kokoro TTS")
                return
            self.device = torch.device("cuda:0")
        except Exception as e:
            logger.error(f"Kokoro init failed: {e}", exc_info=True)
            return
        if self.get_pipeline(lang_code) is None and lang_code != "a":
            # The configured language could not load (e.g. no espeak-ng); keep
            # the engine usable for English and report the language error.
            self.get_pipeline("a")
        self.available = bool(self._pipelines)
        if self.available:
            logger.info(f"Kokoro-82M TTS pipeline loaded ({', '.join(self._pipelines)})")

    def _build_pipeline(self, lang_code: str):
        import torch
        from kokoro import KPipeline

        shared = next(
            (p.model for p in self._pipelines.values() if getattr(p, "model", None) is not None),
            None,
        )
        with torch.cuda.device(0):
            if shared is not None:
                try:
                    return KPipeline(lang_code=lang_code, model=shared)
                except TypeError:
                    pass  # older kokoro without model sharing: load a copy
            pipeline = KPipeline(lang_code=lang_code)
            if getattr(pipeline, "model", None) is not None:
                pipeline.model = pipeline.model.to(self.device)
            return pipeline

    def get_pipeline(self, lang_code: str):
        """Cached pipeline for `lang_code`, building it on first use. Returns
        None (with the reason in error_for) when it cannot be built."""
        if lang_code not in KOKORO_LANGUAGES:
            lang_code = "a"
        with self._lock:
            pipeline = self._pipelines.get(lang_code)
            if pipeline is not None:
                return pipeline
            failed = self._errors.get(lang_code)
            if failed and time.monotonic() - failed[0] < self._RETRY_AFTER_S:
                return None
            try:
                pipeline = self._build_pipeline(lang_code)
            except Exception as e:
                message = _describe_kokoro_error(lang_code, e)
                self._errors[lang_code] = (time.monotonic(), message)
                logger.error(message)
                return None
            self._pipelines[lang_code] = pipeline
            self._errors.pop(lang_code, None)
            return pipeline

    def error_for(self, lang_code: str) -> str:
        failed = self._errors.get(lang_code)
        return failed[1] if failed else ""

    def synthesize_raw(self, text: str, voice: str = DEFAULT_KOKORO_VOICE) -> Optional[bytes]:
        if not self.available:
            return None
        voice = resolve_kokoro_voice(voice)
        lang_code = kokoro_lang_code(voice)
        pipeline = self.get_pipeline(lang_code)
        if pipeline is None:
            return None
        if lang_code not in ("a", "b"):
            text = split_sentences_for_g2p(text)
        try:
            import torch
            import numpy as np

            with torch.cuda.device(self.device):
                chunks = []
                for _, _, audio in pipeline(text, voice=voice):
                    if audio is not None:
                        chunks.append(audio)

            if not chunks:
                return None

            full = np.concatenate(chunks)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes((full * 32767).astype(np.int16).tobytes())
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Kokoro synthesis failed: {e}", exc_info=True)
            return None


# Module-level singleton
_tts_service = None

def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service

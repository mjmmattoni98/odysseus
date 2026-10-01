"""Kokoro local TTS picks its G2P language from the voice.

Kokoro's KPipeline binds one phonemizer language, so a Spanish voice
(`ef_dora`) read through the English ('a') pipeline sounds like English
phonetics. These tests pin the voice -> lang_code derivation, one cached
pipeline per language, the grouped voice list, and graceful failure when a
language's G2P dependencies (espeak-ng / misaki extras) are missing.

`torch` and `kokoro` are replaced by in-memory fakes: no model downloads, no GPU.
"""

import contextlib
import sys
import types

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes.tts_routes import setup_tts_routes
from services.tts import tts_service as tts_mod
from services.tts.tts_service import (
    KOKORO_LANGUAGES,
    TTSService,
    kokoro_lang_code,
    kokoro_voice_groups,
    resolve_kokoro_voice,
    split_sentences_for_g2p,
)


class _FakeModel:
    def to(self, device):
        return self


def _install_fake_kokoro(monkeypatch, failing_langs=()):
    """Install fake `torch` + `kokoro` modules; return the construction log."""
    built = []  # (lang_code, shared_model_passed)
    calls = []  # (lang_code, text, voice)

    class FakeKPipeline:
        def __init__(self, lang_code, model=True):
            if lang_code in failing_langs:
                raise ImportError("No module named 'espeakng_loader'")
            built.append((lang_code, isinstance(model, _FakeModel)))
            self.lang_code = lang_code
            self.model = model if isinstance(model, _FakeModel) else _FakeModel()

        def __call__(self, text, voice=None):
            calls.append((self.lang_code, text, voice))
            yield ("graphemes", "phonemes", np.zeros(240, dtype=np.float32))

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device=lambda _d: contextlib.nullcontext(),
    )
    fake_torch.device = lambda name: name
    fake_kokoro = types.ModuleType("kokoro")
    fake_kokoro.KPipeline = FakeKPipeline

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)
    return built, calls


def _local_service(tmp_path, monkeypatch, voice):
    service = TTSService(cache_dir=str(tmp_path))
    settings = {
        "tts_enabled": True,
        "tts_provider": "local",
        "tts_model": "kokoro",
        "tts_voice": voice,
        "tts_speed": "1",
    }
    monkeypatch.setattr(service, "_load_settings", lambda: dict(settings))
    return service, settings


@pytest.mark.parametrize("voice, expected", [
    ("ef_dora", "e"),
    ("em_alex", "e"),
    ("em_santa", "e"),
    ("EF_DORA", "e"),
    ("af_heart", "a"),
    ("bm_george", "b"),
    ("pf_dora", "p"),
    ("jf_alpha", "j"),
    ("ef_dora,em_alex", "e"),
    ("alloy", "a"),        # OpenAI voice left over from another provider
    ("", "a"),
    (None, "a"),
])
def test_lang_code_is_derived_from_voice_prefix(voice, expected):
    assert kokoro_lang_code(voice) == expected


def test_non_kokoro_voice_names_fall_back_to_default_voice():
    assert resolve_kokoro_voice("alloy") == "af_heart"
    assert resolve_kokoro_voice(" ef_dora ") == "ef_dora"


def test_voice_groups_cover_spanish_and_match_their_language_letter():
    groups = {g["lang_code"]: g for g in kokoro_voice_groups()}
    assert set(groups) == set(KOKORO_LANGUAGES)
    assert groups["e"]["language"] == "Spanish"
    assert groups["e"]["voices"] == ["ef_dora", "em_alex", "em_santa"]
    for code, group in groups.items():
        assert group["voices"], code
        assert all(kokoro_lang_code(v) == code for v in group["voices"]), code


def test_spanish_voice_uses_spanish_pipeline(tmp_path, monkeypatch):
    built, calls = _install_fake_kokoro(monkeypatch)
    service, _ = _local_service(tmp_path, monkeypatch, "ef_dora")

    audio = service.synthesize("Hola, ¿qué tal?", use_cache=False)

    assert audio and audio[:4] == b"RIFF"
    assert built == [("e", False)]
    assert calls[0][0] == "e" and calls[0][2] == "ef_dora"


def test_one_cached_pipeline_per_language_sharing_model(tmp_path, monkeypatch):
    built, calls = _install_fake_kokoro(monkeypatch)
    service, settings = _local_service(tmp_path, monkeypatch, "ef_dora")

    service.synthesize("Primera frase.", use_cache=False)
    service.synthesize("Segunda frase.", use_cache=False)
    settings["tts_voice"] = "af_heart"
    service.synthesize("Now in English.", use_cache=False)
    settings["tts_voice"] = "em_alex"
    service.synthesize("Otra vez en español.", use_cache=False)

    # Built once per language; the second language reuses the loaded weights.
    assert built == [("e", False), ("a", True)]
    assert [c[0] for c in calls] == ["e", "e", "a", "e"]


def test_spanish_text_is_split_per_sentence_for_g2p(tmp_path, monkeypatch):
    _, calls = _install_fake_kokoro(monkeypatch)
    service, _ = _local_service(tmp_path, monkeypatch, "ef_dora")

    service.synthesize("Hola. ¿Cómo estás? Muy bien, gracias.", use_cache=False)

    assert calls[0][1] == "Hola.\n¿Cómo estás?\nMuy bien, gracias."


def test_split_sentences_keeps_decimals_and_handles_cjk():
    assert split_sentences_for_g2p("Cuesta 3.5 euros. Vale") == "Cuesta 3.5 euros.\nVale"
    assert split_sentences_for_g2p("你好。再见") == "你好。\n再见"


def test_missing_g2p_deps_degrade_to_clear_error(tmp_path, monkeypatch):
    built, _ = _install_fake_kokoro(monkeypatch, failing_langs={"e"})
    service, settings = _local_service(tmp_path, monkeypatch, "ef_dora")

    # The engine stays usable (English fallback loaded at init)...
    assert service.available is True
    # ...but Spanish synthesis fails cleanly instead of raising.
    assert service.synthesize("Hola", use_cache=False) is None
    reason = service.failure_reason()
    assert "Spanish" in reason and "espeak-ng" in reason

    # The failed language is not rebuilt on every sentence.
    before = list(built)
    assert service.synthesize("Otra frase", use_cache=False) is None
    assert built == before

    settings["tts_voice"] = "af_heart"
    assert service.synthesize("Hello", use_cache=False)
    assert service.failure_reason() == ""


def test_failed_language_is_retried_after_backoff(tmp_path, monkeypatch):
    _install_fake_kokoro(monkeypatch, failing_langs={"e"})
    service, _ = _local_service(tmp_path, monkeypatch, "ef_dora")
    assert service.synthesize("Hola", use_cache=False) is None

    built, _ = _install_fake_kokoro(monkeypatch)  # deps now installed
    clock = [tts_mod.time.monotonic() + tts_mod._KokoroPipeline._RETRY_AFTER_S + 1]
    monkeypatch.setattr(tts_mod.time, "monotonic", lambda: clock[0])

    assert service.synthesize("Hola", use_cache=False)
    assert ("e", True) in built


def test_missing_kokoro_package_reports_unavailable(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "kokoro", None)  # import raises ImportError
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    service, _ = _local_service(tmp_path, monkeypatch, "ef_dora")

    assert service.available is False
    assert service.synthesize("Hola", use_cache=False) is None
    assert "not available" in service.failure_reason()


def test_synthesize_route_surfaces_failure_reason_and_lists_voices(tmp_path, monkeypatch):
    _install_fake_kokoro(monkeypatch, failing_langs={"e"})
    service, _ = _local_service(tmp_path, monkeypatch, "ef_dora")
    app = FastAPI()
    app.include_router(setup_tts_routes(service))
    client = TestClient(app)

    r = client.post("/api/tts/synthesize", json={"text": "Hola", "format": "audio"})
    assert r.status_code == 500
    assert "espeak-ng" in r.json()["detail"]["message"]

    stats = client.get("/api/tts/stats").json()
    assert stats["lang_code"] == "e" and "Spanish" in stats["error"]

    voices = client.get("/api/tts/voices").json()
    spanish = next(g for g in voices["groups"] if g["lang_code"] == "e")
    assert spanish["voices"] == ["ef_dora", "em_alex", "em_santa"]
    assert voices["default"] == "af_heart"

"""The stt_language setting reaches the transcription backend in a form it accepts.

faster-whisper (and OpenAI-compatible /audio/transcriptions) take a bare
ISO-639 code; the setting may hold a BCP-47 tag such as "es-ES" (what the
browser Web Speech API wants), so the service normalizes it before the call.
"""

import types

import pytest

from services.stt.stt_service import STTService, normalize_stt_language


@pytest.mark.parametrize("raw, expected", [
    ("es", "es"),
    ("es-ES", "es"),
    ("es_AR", "es"),
    (" ES ", "es"),
    ("Spanish", "es"),
    ("", ""),
    ("auto", ""),
    (None, ""),
    ("???", ""),
])
def test_normalize_stt_language(raw, expected):
    assert normalize_stt_language(raw) == expected


def _service(monkeypatch, provider, language):
    service = STTService()
    monkeypatch.setattr(service, "_load_settings", lambda: {
        "stt_enabled": True,
        "stt_provider": provider,
        "stt_model": "base",
        "stt_language": language,
    })
    return service


def test_local_whisper_receives_language(monkeypatch):
    service = _service(monkeypatch, "local", "es-ES")
    seen = {}

    class FakeWhisper:
        def transcribe(self, path, **kwargs):
            seen.update(kwargs)
            info = types.SimpleNamespace(language="es", language_probability=0.99)
            return iter([types.SimpleNamespace(text=" hola ")]), info

    monkeypatch.setattr(service, "_get_whisper", lambda: FakeWhisper())

    assert service.transcribe(b"audio") == "hola"
    assert seen == {"language": "es"}


def test_local_whisper_auto_detects_without_language(monkeypatch):
    service = _service(monkeypatch, "local", "auto")
    seen = {}

    class FakeWhisper:
        def transcribe(self, path, **kwargs):
            seen.update(kwargs)
            info = types.SimpleNamespace(language="en", language_probability=0.9)
            return iter([types.SimpleNamespace(text="hi")]), info

    monkeypatch.setattr(service, "_get_whisper", lambda: FakeWhisper())

    assert service.transcribe(b"audio") == "hi"
    assert "language" not in seen


def test_endpoint_provider_receives_language(monkeypatch):
    service = _service(monkeypatch, "endpoint:ep1", "es-MX")
    seen = {}

    def fake_api(audio_bytes, endpoint_id, model, language=""):
        seen.update(endpoint_id=endpoint_id, language=language)
        return "hola"

    monkeypatch.setattr(service, "_transcribe_api", fake_api)

    assert service.transcribe(b"audio") == "hola"
    assert seen == {"endpoint_id": "ep1", "language": "es"}

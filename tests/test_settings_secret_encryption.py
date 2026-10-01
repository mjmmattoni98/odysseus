"""Search/integration API keys in data/settings.json are encrypted at rest.

`SECRET_SETTING_KEYS` values are written with the app key (`enc:` prefix) by
`save_settings()`, handed back as plaintext by `load_settings()` (so readers
such as the search providers are unchanged), migrated from legacy plaintext,
masked in the admin settings API, and degrade to "unset" — never a crash —
when the app key is missing or rotated.
"""

import asyncio
import importlib
import json
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import routes.auth_routes as auth_routes
import routes.backup_routes as backup_routes
import src.settings as settings_mod
from src.settings_scrub import MASKED_SECRET

BRAVE = "BSA-test-brave-key-123"
TAVILY = "tvly-test-key-456"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point settings + app key at tmp files; return a helper namespace."""
    settings_file = tmp_path / "settings.json"
    # src.settings imports src.secret_storage lazily from sys.modules; some
    # test modules install a bare stub there, so make sure the real one is live.
    secret_storage = sys.modules.get("src.secret_storage")
    if secret_storage is None or not hasattr(secret_storage, "try_decrypt"):
        monkeypatch.delitem(sys.modules, "src.secret_storage", raising=False)
        secret_storage = importlib.import_module("src.secret_storage")
    monkeypatch.setattr(settings_mod, "SETTINGS_FILE", str(settings_file))
    monkeypatch.setattr(secret_storage, "_KEY_PATH", tmp_path / ".app_key")
    monkeypatch.setattr(secret_storage, "_fernet", None)
    monkeypatch.setattr(settings_mod, "_secret_warned", set())
    settings_mod._invalidate_caches()

    def raw():
        return json.loads(settings_file.read_text(encoding="utf-8"))

    def write_raw(data):
        settings_file.write_text(json.dumps(data), encoding="utf-8")
        settings_mod._invalidate_caches()

    yield SimpleNamespace(file=settings_file, raw=raw, write_raw=write_raw, tmp=tmp_path, ss=secret_storage)
    settings_mod._invalidate_caches()


def test_secret_keys_are_enumerated_in_one_place():
    assert {"brave_api_key", "google_pse_key", "tavily_api_key", "serper_api_key"} <= settings_mod.SECRET_SETTING_KEYS
    # Legacy flat email passwords are read raw by other modules: not encrypted here.
    assert "smtp_password" not in settings_mod.SECRET_SETTING_KEYS
    assert "google_pse_cx" not in settings_mod.SECRET_SETTING_KEYS


def test_round_trip_encrypts_on_disk_and_reads_plaintext(store):
    settings_mod.save_settings({"brave_api_key": BRAVE, "search_provider": "brave", "serper_api_key": ""})

    text = store.file.read_text(encoding="utf-8")
    assert BRAVE not in text
    raw = store.raw()
    assert raw["brave_api_key"].startswith("enc:")
    assert raw["search_provider"] == "brave"   # non-secrets untouched
    assert raw["serper_api_key"] == ""        # empty stays empty

    loaded = settings_mod.load_settings()
    assert loaded["brave_api_key"] == BRAVE
    assert settings_mod.get_setting("brave_api_key") == BRAVE


def test_save_does_not_mutate_callers_dict(store):
    data = {"brave_api_key": BRAVE}
    settings_mod.save_settings(data)
    assert data == {"brave_api_key": BRAVE}


def test_search_provider_reader_gets_plaintext(store):
    from services.search.providers import _get_provider_key

    settings_mod.save_settings({"tavily_api_key": TAVILY})
    assert store.raw()["tavily_api_key"].startswith("enc:")
    assert _get_provider_key("tavily") == TAVILY


def test_legacy_plaintext_is_readable_and_migrated_on_next_save(store):
    store.write_raw({"brave_api_key": BRAVE, "search_provider": "brave"})

    loaded = settings_mod.load_settings()
    assert loaded["brave_api_key"] == BRAVE          # readers work before migration

    loaded["search_result_count"] = 7                # unrelated edit
    settings_mod.save_settings(loaded)

    assert store.raw()["brave_api_key"].startswith("enc:")
    assert settings_mod.load_settings()["brave_api_key"] == BRAVE


def test_startup_migration_encrypts_without_materializing_defaults(store):
    store.write_raw({"brave_api_key": BRAVE, "tavily_api_key": TAVILY, "tts_voice": "ef_dora"})

    assert settings_mod.migrate_secret_settings() is True

    raw = store.raw()
    assert set(raw) == {"brave_api_key", "tavily_api_key", "tts_voice"}
    assert raw["brave_api_key"].startswith("enc:") and raw["tavily_api_key"].startswith("enc:")
    assert raw["tts_voice"] == "ef_dora"
    assert settings_mod.load_settings()["tavily_api_key"] == TAVILY
    assert settings_mod.migrate_secret_settings() is False  # idempotent


def test_startup_migration_without_settings_file_is_noop(store):
    assert settings_mod.migrate_secret_settings() is False
    assert not store.file.exists()


def test_rotated_app_key_reads_as_unset_with_one_warning(store, monkeypatch, caplog):
    settings_mod.save_settings({"brave_api_key": BRAVE})
    ciphertext = store.raw()["brave_api_key"]
    old_key = store.tmp / ".app_key"

    # Simulate a lost/rotated key: a different key file.
    monkeypatch.setattr(store.ss, "_KEY_PATH", store.tmp / "rotated.key")
    monkeypatch.setattr(store.ss, "_fernet", None)
    settings_mod._invalidate_caches()

    with caplog.at_level(logging.WARNING, logger="src.settings"):
        assert settings_mod.load_settings()["brave_api_key"] == ""
        settings_mod._invalidate_caches()
        assert settings_mod.load_settings()["brave_api_key"] == ""
    warnings = [r for r in caplog.records if "brave_api_key" in r.getMessage()]
    assert len(warnings) == 1

    # An unrelated save does not destroy the unreadable ciphertext...
    current = settings_mod.load_settings()
    current["search_result_count"] = 3
    settings_mod.save_settings(current)
    assert store.raw()["brave_api_key"] == ciphertext

    # ...so restoring the old key brings the secret back.
    monkeypatch.setattr(store.ss, "_KEY_PATH", old_key)
    monkeypatch.setattr(store.ss, "_fernet", None)
    settings_mod._invalidate_caches()
    assert settings_mod.load_settings()["brave_api_key"] == BRAVE


def test_new_value_replaces_unreadable_ciphertext(store, monkeypatch):
    settings_mod.save_settings({"brave_api_key": BRAVE})
    monkeypatch.setattr(store.ss, "_KEY_PATH", store.tmp / "rotated.key")
    monkeypatch.setattr(store.ss, "_fernet", None)
    settings_mod._invalidate_caches()

    settings_mod.save_settings({"brave_api_key": "BSA-new"})
    assert settings_mod.load_settings()["brave_api_key"] == "BSA-new"


def test_encryption_failure_keeps_value_instead_of_failing_save(store, monkeypatch):
    def broken_encrypt(_value):
        raise OSError("read-only data dir")

    monkeypatch.setattr(store.ss, "encrypt", broken_encrypt)
    settings_mod.save_settings({"brave_api_key": BRAVE})

    assert store.raw()["brave_api_key"] == BRAVE
    assert settings_mod.load_settings()["brave_api_key"] == BRAVE


def test_mask_placeholder_is_never_persisted(store):
    settings_mod.save_settings({"brave_api_key": BRAVE})
    settings_mod.save_settings({"brave_api_key": MASKED_SECRET, "tavily_api_key": MASKED_SECRET})

    raw = store.raw()
    assert MASKED_SECRET not in json.dumps(raw)
    assert "tavily_api_key" not in raw
    assert settings_mod.load_settings()["brave_api_key"] == BRAVE


# ── Settings API ──

class _AuthManager:
    def get_username_for_token(self, token):
        return {"admin-session": "admin", "user-session": "bob"}.get(token)

    def is_admin(self, username):
        return username == "admin"


def _request(body=None, *, user=None):
    cookies = {auth_routes.SESSION_COOKIE: f"{user}-session"} if user else {}

    async def _json():
        return body

    return SimpleNamespace(cookies=cookies, json=_json)


def _endpoint(router, path, method):
    return next(r.endpoint for r in router.routes if r.path == path and method in r.methods)


@pytest.fixture
def settings_api(store, monkeypatch):
    monkeypatch.setattr(auth_routes, "migrate_from_settings", lambda: None)
    router = auth_routes.setup_auth_routes(_AuthManager())
    return (
        _endpoint(router, "/api/auth/settings", "GET"),
        _endpoint(router, "/api/auth/settings", "POST"),
    )


def test_settings_api_masks_secrets_for_admin_and_blanks_for_others(store, settings_api):
    get_settings, _ = settings_api
    settings_mod.save_settings({"brave_api_key": BRAVE, "search_provider": "brave"})

    admin = asyncio.run(get_settings(_request(user="admin")))
    assert admin["brave_api_key"] == MASKED_SECRET     # "key set" stays visible
    assert admin["serper_api_key"] == ""               # unset stays unset
    assert admin["search_provider"] == "brave"
    assert BRAVE not in json.dumps(admin)

    for user in ("bob", None):
        other = asyncio.run(get_settings(_request(user=user)))
        assert other["brave_api_key"] == ""
        assert BRAVE not in json.dumps(other)


def test_settings_post_keeps_secret_when_mask_echoed_and_replaces_on_new_value(store, settings_api):
    _, set_settings = settings_api
    settings_mod.save_settings({"brave_api_key": BRAVE})

    # The Search tab posts the key field back with every change.
    resp = asyncio.run(set_settings(_request(
        {"brave_api_key": MASKED_SECRET, "search_result_count": 10}, user="admin")))
    assert resp["brave_api_key"] == MASKED_SECRET
    assert settings_mod.load_settings()["brave_api_key"] == BRAVE
    assert settings_mod.load_settings()["search_result_count"] == 10

    resp = asyncio.run(set_settings(_request({"brave_api_key": "BSA-rotated"}, user="admin")))
    assert "BSA-rotated" not in json.dumps(resp)
    assert store.raw()["brave_api_key"].startswith("enc:")
    assert settings_mod.load_settings()["brave_api_key"] == "BSA-rotated"

    asyncio.run(set_settings(_request({"brave_api_key": ""}, user="admin")))
    assert settings_mod.load_settings()["brave_api_key"] == ""


def test_settings_routes_setup_migrates_plaintext(store, monkeypatch):
    store.write_raw({"serper_api_key": "serper-plain"})
    monkeypatch.setattr(auth_routes, "migrate_from_settings", lambda: None)

    auth_routes.setup_auth_routes(_AuthManager())

    assert store.raw()["serper_api_key"].startswith("enc:")
    assert settings_mod.load_settings()["serper_api_key"] == "serper-plain"


# ── Backup export / import ──

def _backup_endpoints(monkeypatch):
    monkeypatch.setattr(backup_routes, "require_admin", lambda request: None)
    monkeypatch.setattr(backup_routes, "get_current_user", lambda request: "admin")
    import routes.prefs_routes as prefs_routes
    monkeypatch.setattr(prefs_routes, "_load_for_user", lambda user: {})
    memory, presets, skills = MagicMock(), MagicMock(), MagicMock()
    memory.load.return_value = []
    presets.get_all.return_value = {}
    skills.load.return_value = []
    router = backup_routes.setup_backup_routes(memory, presets, skills)
    return _endpoint(router, "/api/export", "GET"), _endpoint(router, "/api/import", "POST")


def test_backup_export_restores_secrets_through_import(store, monkeypatch):
    export, import_ = _backup_endpoints(monkeypatch)
    settings_mod.save_settings({"brave_api_key": BRAVE, "search_provider": "brave"})

    exported = json.loads(asyncio.run(export(SimpleNamespace())).body)
    # The admin-only backup stays restorable on another install (which has a
    # different app key), so it carries the usable value, not ciphertext.
    assert exported["settings"]["brave_api_key"] == BRAVE

    store.write_raw({})
    result = asyncio.run(import_(SimpleNamespace(json=_async({"settings": exported["settings"]}))))
    assert "settings" in result["imported"]
    assert store.raw()["brave_api_key"].startswith("enc:")
    assert settings_mod.load_settings()["brave_api_key"] == BRAVE


def _async(value):
    async def _inner():
        return value
    return _inner


def test_admin_mask_keeps_non_secret_capability_handles_visible():
    from src.settings_scrub import mask_settings

    out = mask_settings({
        "reminder_webhook_integration_id": "global-webhook",
        "google_pse_cx": "cx123",
        "brave_api_key": BRAVE,
        "nested": {"smtp_password": "pw"},
    })
    assert out["reminder_webhook_integration_id"] == "global-webhook"
    assert out["google_pse_cx"] == "cx123"
    assert out["brave_api_key"] == MASKED_SECRET
    assert out["nested"]["smtp_password"] == MASKED_SECRET

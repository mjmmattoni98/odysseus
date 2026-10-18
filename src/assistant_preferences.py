"""Conversation preferences and immutable, task-local turn settings."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal
import threading

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_LOCAL_CONTEXT_LIMIT = 32768
MIN_CONTEXT_LIMIT = 1024
MAX_CONTEXT_LIMIT = 262144
# Saved by the retired assistant profiles. Conversations stored before their
# removal still carry these keys; they are dropped on load and ignored on save.
RETIRED_PREFERENCE_FIELDS = frozenset({"profile", "web_mode"})


class AssistantPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # "auto" uses the Settings default (local_thinking_effort).
    thinking: Literal["auto", "off", "low", "medium", "high"] = "auto"
    instructions: str = Field("", max_length=10000)
    # Model IDs are exact: a limit for one model must not affect a fallback.
    context_limits: dict[str, int] = Field(default_factory=dict, max_length=100)


DEFAULT_PREFERENCES = AssistantPreferences()
_turn = ContextVar("assistant_turn", default=None)
_search_reports = ContextVar("assistant_search_reports", default=None)
_sources = ContextVar("assistant_sources", default=None)


class SourceRegistry:
    """Stable citation numbers across searches and fetches in one answer."""

    def __init__(self):
        self._numbers = {}
        self._lock = threading.Lock()

    def number(self, url):
        with self._lock:
            return self._numbers.setdefault(url, len(self._numbers) + 1)


def source_registry():
    return _sources.get()


@contextmanager
def assistant_turn(preferences, endpoint_url="", reports=None, sources=None):
    token = _turn.set((preferences, endpoint_url))
    report_token = _search_reports.set(reports if reports is not None else [])
    source_token = _sources.set(sources if sources is not None else SourceRegistry())
    try:
        yield
    finally:
        _turn.reset(token)
        _search_reports.reset(report_token)
        _sources.reset(source_token)


def search_reports():
    return list(_search_reports.get() or [])


def record_search_report(report):
    reports = _search_reports.get()
    if reports is not None:
        reports.append(dict(report))


def current_preferences():
    value = _turn.get()
    return value[0] if value else DEFAULT_PREFERENCES


def default_context_limit():
    """Configured default local context cap (``local_context_limit_default``).

    Used for native local requests outside a turn and as the per-model
    default inside one. Invalid stored values fall back to 32768; valid ones
    are clamped to the same range as per-conversation limits.
    """
    try:
        from src.settings import get_setting

        raw = get_setting("local_context_limit_default", DEFAULT_LOCAL_CONTEXT_LIMIT)
        if isinstance(raw, bool):
            return DEFAULT_LOCAL_CONTEXT_LIMIT
        value = int(raw)
    except Exception:
        return DEFAULT_LOCAL_CONTEXT_LIMIT
    return max(MIN_CONTEXT_LIMIT, min(value, MAX_CONTEXT_LIMIT))


def context_limit(url, model):
    from src.ollama_capabilities import ollama_api_root

    value = _turn.get()
    default = default_context_limit()
    root = ollama_api_root(url)
    if root and value and root == ollama_api_root(value[1]):
        return value[0].context_limits.get(model, default)
    return default


def parse_preferences(data):
    if isinstance(data, dict):
        data = {key: value for key, value in data.items() if key not in RETIRED_PREFERENCE_FIELDS}
    preferences = AssistantPreferences.model_validate(data)
    for model, limit in preferences.context_limits.items():
        if not model.strip() or len(model) > 256 or isinstance(limit, bool) or not MIN_CONTEXT_LIMIT <= limit <= MAX_CONTEXT_LIMIT:
            raise ValueError("Context limits must be between 1024 and 262144 tokens for a named model")
    return preferences


def load_preferences(session_id):
    from core.database import SessionLocal, Session

    with SessionLocal() as db:
        row = db.query(Session.assistant_preferences).filter(Session.id == session_id).first()
        return parse_preferences(row[0]) if row and row[0] else DEFAULT_PREFERENCES


def assistant_prompt(preferences):
    """The conversation's custom instructions, appended to the system prompt."""
    return preferences.instructions


def merge_sources(existing, incoming):
    """Keep sources from every search round; later fetch evidence replaces snippets."""
    result = {s["url"]: dict(s) for s in existing if s.get("url")}
    rank = {"snippet": 0, "failed": 1, "read": 2}
    for source in incoming:
        url = source.get("url")
        if not url:
            continue
        old = result.get(url)
        if old is None or rank.get(source.get("read_status"), -1) >= rank.get(old.get("read_status"), -1):
            result[url] = dict(source)
    return list(result.values())

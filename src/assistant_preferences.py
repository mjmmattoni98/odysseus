"""Conversation preferences and immutable, task-local turn settings."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal
import re
import threading

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_LOCAL_CONTEXT_LIMIT = 32768
MIN_CONTEXT_LIMIT = 1024
MAX_CONTEXT_LIMIT = 262144
READ_ONLY_TOOLS = frozenset({"web_search", "web_fetch", "search_chats", "read_file", "grep", "glob", "ls", "ask_user"})


class AssistantPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: Literal["legacy", "everyday", "research", "actions"] = "everyday"
    web_mode: Literal["off", "auto", "on"] = "auto"
    thinking: Literal["auto", "off", "low", "medium", "high"] = "off"
    instructions: str = Field("", max_length=10000)
    # Model IDs are exact: a limit for one model must not affect a fallback.
    context_limits: dict[str, int] = Field(default_factory=dict, max_length=100)

    @property
    def read_only(self) -> bool:
        return self.profile in {"everyday", "research"}


LEGACY_PREFERENCES = AssistantPreferences(profile="legacy", thinking="auto")
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
    return value[0] if value else LEGACY_PREFERENCES


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
    preferences = AssistantPreferences.model_validate(data)
    for model, limit in preferences.context_limits.items():
        if not model.strip() or len(model) > 256 or isinstance(limit, bool) or not MIN_CONTEXT_LIMIT <= limit <= MAX_CONTEXT_LIMIT:
            raise ValueError("Context limits must be between 1024 and 262144 tokens for a named model")
    return preferences


def load_preferences(session_id):
    from core.database import SessionLocal, Session

    with SessionLocal() as db:
        row = db.query(Session.assistant_preferences).filter(Session.id == session_id).first()
        return parse_preferences(row[0]) if row and row[0] else LEGACY_PREFERENCES


def web_enabled(preferences, message, *, explicitly_denied=False):
    if explicitly_denied or preferences.web_mode == "off":
        return False
    if preferences.web_mode == "on":
        return True
    # Auto is intentionally cheap: no extra model invocation for each message.
    # Web tools remain available for follow-up lookups on an Auto turn.
    return bool(re.search(
        r"https?://|\b(search|look\s*up|verify|fact.check|sources?|citations?|"
        r"latest|current|today|news|weather|forecast|prices?|buy|purchase|recommend|"
        r"recommendations?|available|availability|compare|comparison|reviews?|"
        r"buscar?|busca|verifica|fuentes?|actual|últim[oa]s?|hoy|noticias|"
        r"precios?|comprar|recomienda|compar[ae]|disponibilidad)\b",
        message or "", re.IGNORECASE,
    ))


def assistant_prompt(preferences):
    if preferences.profile == "legacy":
        return ""
    prompt = (
        "Help the user understand things and make well-supported decisions. "
        "Answer directly and explain the important reasons. Separate facts, assumptions, "
        "and preferences. For decisions, compare realistic options, recommend one, "
        "and explain its strongest drawback and what would change the recommendation. "
        "Ask for missing constraints only when necessary. Cite only sources supplied "
        "by tools or the user. Distinguish a page you read from a search snippet. "
        "If searching or fetching fails, say so; do not imply verification succeeded."
    )
    if preferences.read_only:
        prompt += " This conversation is for discussion and research. Do not execute commands or change data."
    if preferences.web_mode == "off":
        prompt += " Web access is off. Explain when an answer needs current verification."
    elif preferences.web_mode == "auto":
        prompt += " Use web tools when current facts, recommendations, or verification are needed."
    if preferences.profile == "research":
        prompt += " Read primary sources where possible, compare conflicting evidence, and identify unresolved questions."
    return prompt + ("\n\n" + preferences.instructions if preferences.instructions else "")


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

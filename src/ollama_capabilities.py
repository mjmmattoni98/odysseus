"""TTL-cached capability lookup for local Ollama endpoints.

Ollama reports per-model capabilities (``tools``, ``thinking``, ``vision``) on
its native ``/api/show`` endpoint. Odysseus' request paths historically
inferred those from hardcoded model-name substring lists, which miss every new
model family and forced users to hand-edit ``ModelEndpoint.supports_tools``.

This module turns the native report into a small cached oracle — safe to call
on request paths — with an explicit ``None`` ("unknown") result when the probe
fails, so callers can fall back to their name heuristics. Only ports known to
host Ollama are probed; other local servers return ``None`` before any network
call.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import FrozenSet, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 600.0
_FAILURE_TTL_SECONDS = 60.0
_PROBE_TIMEOUT_SECONDS = 1.5
_TOOL_TOKENS: FrozenSet[str] = frozenset({"tools", "tool"})
_THINKING_TOKENS: FrozenSet[str] = frozenset({"thinking", "reasoning"})

_cache: dict[tuple[str, str], tuple[float, Optional[FrozenSet[str]]]] = {}
_lock = threading.Lock()


def ollama_api_root(url: str) -> str:
    """Return ``scheme://host:port`` when the URL targets a known Ollama port.

    Returns an empty string otherwise, which makes every public helper a
    no-op (``None``) instead of probing unrelated local servers.
    """
    try:
        parsed = urlparse(str(url or "").strip())
    except Exception:
        return ""
    if not parsed.scheme or not parsed.netloc:
        return ""
    host = (parsed.hostname or "").lower()
    is_ollama = parsed.port == 11434 or host == "ollama.com" or host.endswith(".ollama.com")
    if not is_ollama:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def capability_tokens(url: str, model: str, *, timeout: float = _PROBE_TIMEOUT_SECONDS) -> Optional[FrozenSet[str]]:
    """Return the model's lowercase capability tokens, or ``None`` if unknown.

    Results are cached per ``(api root, model)``. Failures (unreachable host,
    old Ollama without ``capabilities``, HTTP error) are cached briefly so a
    down endpoint does not add latency to every request.
    """
    root = ollama_api_root(url)
    model = str(model or "").strip()
    if not root or not model:
        return None
    key = (root, model)
    now = time.time()
    with _lock:
        cached = _cache.get(key)
    if cached is not None:
        ts, tokens = cached
        ttl = _FAILURE_TTL_SECONDS if tokens is None else _CACHE_TTL_SECONDS
        if now - ts < ttl:
            return tokens

    tokens: Optional[FrozenSet[str]] = None
    try:
        response = httpx.post(
            f"{root}/api/show",
            json={"model": model},
            timeout=timeout,
        )
        if response.is_success:
            payload = response.json() or {}
            raw = payload.get("capabilities")
            if isinstance(raw, list):
                tokens = frozenset(
                    str(item).strip().lower() for item in raw if str(item).strip()
                )
    except Exception as exc:  # noqa: BLE001 - capability probe must never raise
        logger.debug("Ollama capability probe failed for %s on %s: %s", model, root, exc)

    with _lock:
        _cache[key] = (now, tokens)
    return tokens


def supports_tool_calls(url: str, model: str) -> Optional[bool]:
    """Whether the model advertises native tool calling; ``None`` if unknown."""
    tokens = capability_tokens(url, model)
    if tokens is None:
        return None
    return bool(tokens & _TOOL_TOKENS)


def supports_thinking(url: str, model: str) -> Optional[bool]:
    """Whether the model advertises a thinking/reasoning mode; ``None`` if unknown."""
    tokens = capability_tokens(url, model)
    if tokens is None:
        return None
    return bool(tokens & _THINKING_TOKENS)


def reset_cache() -> None:
    """Drop the cache (tests and endpoint reconfiguration)."""
    with _lock:
        _cache.clear()

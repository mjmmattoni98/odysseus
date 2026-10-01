"""TTL-cached capability lookup for local Ollama endpoints.

Ollama reports per-model capabilities (``tools``, ``thinking``, ``vision``,
``embedding``) on its native ``/api/show`` endpoint. Odysseus' request paths
historically inferred those from hardcoded model-name substring lists, which
miss every new model family and forced users to hand-edit
``ModelEndpoint.supports_tools``.

This module turns the native report into a small cached oracle — safe to call
on request paths — with an explicit ``None`` ("unknown") result when the probe
fails, so callers can fall back to their name heuristics.

It is also the single source of truth for "is this URL an Ollama server?"
(:func:`is_ollama_url`): port 11434, Ollama Cloud, endpoints registered with
``endpoint_kind == "ollama"`` (Cookbook serves Ollama on 11435+), and local
servers on other ports that answered ``GET /api/version`` like Ollama. Other
local servers (LM Studio, llama.cpp, vLLM) return ``None`` from every helper
before any capability probe.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import threading
import time
from typing import FrozenSet, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_PORT = 11434
_CACHE_TTL_SECONDS = 600.0
_FAILURE_TTL_SECONDS = 60.0
_PROBE_TIMEOUT_SECONDS = 1.5
# /api/version fingerprint of local servers on non-default ports. Answers are
# stable (a port rarely changes server software) so positives live long;
# negatives expire sooner so an Ollama started after Odysseus is picked up.
_VERSION_TTL_SECONDS = 600.0
_VERSION_FAILURE_TTL_SECONDS = 60.0
_VERSION_TIMEOUT_SECONDS = 0.75
# /api/ps residency changes with every model swap; keep it only briefly.
_PS_TTL_SECONDS = 5.0
_PS_TIMEOUT_SECONDS = 1.0
_TOOL_TOKENS: FrozenSet[str] = frozenset({"tools", "tool"})
_THINKING_TOKENS: FrozenSet[str] = frozenset({"thinking", "reasoning"})
_VISION_TOKENS: FrozenSet[str] = frozenset({"vision"})
_EMBEDDING_TOKENS: FrozenSet[str] = frozenset({"embedding"})
_TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")

_cache: dict[tuple[str, str], tuple[float, Optional[FrozenSet[str]]]] = {}
_lock = threading.Lock()
_context_windows: dict[tuple[str, str], int] = {}
_version_cache: dict[str, tuple[float, bool]] = {}
_version_warming: set[str] = set()
_ps_cache: dict[str, tuple[float, Optional[List[dict]]]] = {}
_ps_warming: set[str] = set()


def _in_event_loop() -> bool:
    """True when called on a thread that is running an asyncio loop."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _url_root(url: str) -> tuple[str, str, Optional[int]]:
    """Return ``(scheme://netloc, lowercase host, port)`` or empty values."""
    try:
        parsed = urlparse(str(url or "").strip())
        port = parsed.port
    except Exception:
        return "", "", None
    if not parsed.scheme or not parsed.netloc:
        return "", "", None
    host = (parsed.hostname or "").lower().rstrip(".")
    return f"{parsed.scheme}://{parsed.netloc}", host, port


def _is_cloud_host(host: str) -> bool:
    return host == "ollama.com" or host.endswith(".ollama.com")


def _is_probe_candidate(host: str) -> bool:
    """Hosts we may fingerprint: loopback/LAN/Tailscale/container names only.

    Public hosts are never probed, so an arbitrary remote URL cannot turn a
    provider-detection call into outbound traffic.
    """
    if not host:
        return False
    if host in {"localhost", "host.docker.internal"} or host.endswith((".local", ".localhost")):
        return True
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        # Dotless names are container/compose service names (``ollama``).
        return "." not in host
    return bool(
        ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified
        or ip in _TAILSCALE_CGNAT
    )


def _registered_kind(url: str) -> Optional[str]:
    """Kind of the enabled endpoint covering ``url`` (cached DB lookup)."""
    try:
        from src.model_context import _configured_endpoint_kind

        return _configured_endpoint_kind(url)
    except Exception:
        return None


def _registered_as_ollama(url: str) -> bool:
    """Whether an enabled endpoint covering ``url`` is registered as kind "ollama"."""
    return _registered_kind(url) == "ollama"


def _probe_version(root: str, timeout: float) -> Optional[str]:
    """``GET /api/version`` → Ollama's version string, else ``None``."""
    try:
        response = httpx.get(f"{root}/api/version", timeout=timeout)
        if not response.is_success:
            return None
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - fingerprinting must never raise
        logger.debug("Ollama version probe failed for %s: %s", root, exc)
        return None
    version = payload.get("version") if isinstance(payload, dict) else None
    return version.strip() if isinstance(version, str) and version.strip() else None


def _cached_version_answer(root: str) -> Optional[bool]:
    with _lock:
        cached = _version_cache.get(root)
    if cached is None:
        return None
    ts, answer = cached
    ttl = _VERSION_TTL_SECONDS if answer else _VERSION_FAILURE_TTL_SECONDS
    return answer if time.time() - ts < ttl else None


def _store_version_answer(root: str, answer: bool) -> None:
    with _lock:
        _version_cache[root] = (time.time(), bool(answer))


def _warm_version_in_background(root: str) -> None:
    with _lock:
        if root in _version_warming:
            return
        _version_warming.add(root)

    def _run():
        try:
            _store_version_answer(root, _probe_version(root, _VERSION_TIMEOUT_SECONDS) is not None)
        finally:
            with _lock:
                _version_warming.discard(root)

    threading.Thread(target=_run, name="ollama-version-probe", daemon=True).start()


def fingerprint_ollama(url: str, *, timeout: float = _VERSION_TIMEOUT_SECONDS) -> Optional[str]:
    """Probe ``url``'s server for Ollama now; return its version or ``None``.

    Blocking — for already-synchronous code (discovery scans, endpoint
    registration). The answer seeds the cache :func:`is_ollama_url` reads.
    """
    root, host, _port = _url_root(url)
    if not root or not (_is_probe_candidate(host) or _is_cloud_host(host)):
        return None
    version = _probe_version(root, timeout)
    _store_version_answer(root, version is not None)
    return version


def is_ollama_url(url: str, *, probe: bool = True) -> bool:
    """Single source of truth for "does this URL talk to an Ollama server?".

    True for port 11434, Ollama Cloud, endpoints registered with kind
    ``"ollama"``, and local servers that answered ``GET /api/version`` like
    Ollama. The path is not considered: callers decide native (``/api``)
    versus OpenAI-compatible (``/v1``) surfaces themselves.

    Never blocks an event loop: on a cache miss inside a running loop the
    fingerprint is warmed in a background thread and this call answers
    ``False`` ("not known to be Ollama"). Synchronous callers probe inline
    with a short timeout; failures are negative-cached. ``probe=False``
    answers from what is already known (port, kind, cached fingerprint).
    """
    root, host, port = _url_root(url)
    if not root:
        return False
    if port == OLLAMA_DEFAULT_PORT or _is_cloud_host(host):
        return True
    cached = _cached_version_answer(root)
    if cached:
        return True
    kind = _registered_kind(url)
    if kind == "ollama":
        return True
    # An endpoint explicitly configured as a remote API/proxy is never probed.
    if not probe or kind in ("api", "proxy") or cached is not None or not _is_probe_candidate(host):
        return False
    if _in_event_loop():
        _warm_version_in_background(root)
        return False
    return fingerprint_ollama(root) is not None


def ollama_api_root(url: str, *, probe: bool = True) -> str:
    """Return ``scheme://host:port`` when the URL targets an Ollama server.

    Returns an empty string otherwise, which makes every public helper a
    no-op (``None``) instead of probing unrelated local servers.
    """
    root, _host, _port = _url_root(url)
    return root if root and is_ollama_url(url, probe=probe) else ""


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
    window = None
    try:
        response = httpx.post(
            f"{root}/api/show",
            json={"model": model},
            timeout=timeout,
        )
        if response.is_success:
            payload = response.json() or {}
            windows = [v for k, v in (payload.get("model_info") or {}).items()
                       if (k == "context_length" or k.endswith(".context_length"))
                       and isinstance(v, int) and not isinstance(v, bool) and v > 0]
            window = min(windows) if windows else None
            raw = payload.get("capabilities")
            if isinstance(raw, list):
                tokens = frozenset(
                    str(item).strip().lower() for item in raw if str(item).strip()
                )
    except Exception as exc:  # noqa: BLE001 - capability probe must never raise
        logger.debug("Ollama capability probe failed for %s on %s: %s", model, root, exc)

    with _lock:
        _cache[key] = (now, tokens)
        if window:
            _context_windows[key] = window
        else:
            _context_windows.pop(key, None)
    return tokens


def cached_capability_tokens(url: str, model: str) -> Optional[FrozenSet[str]]:
    """Capability tokens already in the cache; never probes (cheap filters)."""
    root = ollama_api_root(url)
    with _lock:
        cached = _cache.get((root, str(model or "").strip())) if root else None
    if cached is None:
        return None
    ts, tokens = cached
    ttl = _FAILURE_TTL_SECONDS if tokens is None else _CACHE_TTL_SECONDS
    return tokens if time.time() - ts < ttl else None


def _has_any(url: str, model: str, wanted: FrozenSet[str]) -> Optional[bool]:
    tokens = capability_tokens(url, model)
    if tokens is None:
        return None
    return bool(tokens & wanted)


def supports_tool_calls(url: str, model: str) -> Optional[bool]:
    """Whether the model advertises native tool calling; ``None`` if unknown."""
    return _has_any(url, model, _TOOL_TOKENS)


def supports_thinking(url: str, model: str) -> Optional[bool]:
    """Whether the model advertises a thinking/reasoning mode; ``None`` if unknown."""
    return _has_any(url, model, _THINKING_TOKENS)


def supports_vision(url: str, model: str) -> Optional[bool]:
    """Whether the model accepts images (``vision`` capability); ``None`` if unknown."""
    return _has_any(url, model, _VISION_TOKENS)


def supports_embedding(url: str, model: str) -> Optional[bool]:
    """Whether the model advertises ``embedding``; ``None`` if unknown."""
    return _has_any(url, model, _EMBEDDING_TOKENS)


def is_embedding_only(tokens: Optional[FrozenSet[str]]) -> bool:
    """True for a capability report of an embedding model that cannot chat."""
    return bool(tokens) and bool(tokens & _EMBEDDING_TOKENS) and "completion" not in tokens


def model_context_window(url: str, model: str) -> Optional[int]:
    """Advertised maximum, distinct from the running allocation in /api/ps."""
    capability_tokens(url, model)
    with _lock:
        return _context_windows.get((ollama_api_root(url), str(model or "").strip()))


def _fetch_loaded_models(root: str) -> Optional[List[dict]]:
    try:
        response = httpx.get(f"{root}/api/ps", timeout=_PS_TIMEOUT_SECONDS)
        if not response.is_success:
            return None
        items = (response.json() or {}).get("models") or []
    except Exception as exc:  # noqa: BLE001 - residency probe must never raise
        logger.debug("Ollama /api/ps probe failed for %s: %s", root, exc)
        return None
    return [item for item in items if isinstance(item, dict)]


def loaded_models(url: str) -> Optional[List[dict]]:
    """Models Ollama currently keeps loaded (``/api/ps``), newest answer ≤5 s old.

    ``None`` means unknown (not Ollama, unreachable, or — inside a running
    event loop — not fetched yet: the probe is then warmed in a background
    thread so the loop never waits on the network).
    """
    root = ollama_api_root(url)
    if not root or _is_cloud_host(_url_root(root)[1]):
        return None
    now = time.time()
    with _lock:
        cached = _ps_cache.get(root)
    if cached is not None and now - cached[0] < _PS_TTL_SECONDS:
        return cached[1]
    if _in_event_loop():
        with _lock:
            if root in _ps_warming:
                return None
            _ps_warming.add(root)

        def _run():
            try:
                fetched = _fetch_loaded_models(root)
                with _lock:
                    _ps_cache[root] = (time.time(), fetched)
            finally:
                with _lock:
                    _ps_warming.discard(root)
        threading.Thread(target=_run, name="ollama-ps-probe", daemon=True).start()
        return None
    fetched = _fetch_loaded_models(root)
    with _lock:
        _ps_cache[root] = (now, fetched)
    return fetched


def loaded_model_names(url: str) -> Optional[List[str]]:
    """Names of the loaded models, most recently used first; ``None`` if unknown."""
    items = loaded_models(url)
    if items is None:
        return None
    # Ollama extends ``expires_at`` on every use, so the latest expiry is the
    # model that served the most recent request.
    ordered = sorted(items, key=lambda item: str(item.get("expires_at") or ""), reverse=True)
    names = []
    for item in ordered:
        name = str(item.get("name") or item.get("model") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def reset_cache() -> None:
    """Drop the cache (tests and endpoint reconfiguration)."""
    with _lock:
        _cache.clear()
        _context_windows.clear()
        _version_cache.clear()
        _ps_cache.clear()

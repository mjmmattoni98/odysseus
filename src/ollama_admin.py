"""Admin-side Ollama model management for the Cookbook "Ollama" tab.

Wraps the native Ollama HTTP API (``/api/tags``, ``/api/show``, ``/api/ps``,
``/api/pull``, ``/api/delete``, ``/api/generate`` keep-alive, ``/api/create``)
behind a small target allowlist:

* Ollama-looking rows in ``model_endpoints`` (admin-registered), and
* the configured/discovered local daemon (``OLLAMA_BASE_URL``/``OLLAMA_URL``/
  ``OLLAMA_HOST``, ``127.0.0.1:11434`` and, inside Docker,
  ``host.docker.internal:11434``).

Callers address a target by its opaque id from :func:`list_targets`; a raw
URL is never accepted, so these routes cannot be used as an SSRF proxy.
Redirects are not followed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_PORT = 11434
_PROBE_TIMEOUT = 1.5
_REQUEST_TIMEOUT = 15.0
_ANY_BIND_HOSTS = {"0.0.0.0", "::", "[::]", ""}

# Ollama model references (`qwen3:8b`, `library/llama3:latest`,
# `hf.co/org/repo:Q4_K_M`). Shell/URL-safe glyphs only.
_MODEL_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,200}$")
# New preset names: lowercase, one optional `:tag`.
_PRESET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}(?::[a-z0-9][a-z0-9._-]{0,47})?$")

# name -> (type, min, max). Mirrors the Modelfile PARAMETER list we expose.
PRESET_PARAMETER_RANGES: Dict[str, tuple] = {
    "num_ctx": (int, 256, 1_048_576),
    "temperature": (float, 0.0, 2.0),
    "top_p": (float, 0.0, 1.0),
    "top_k": (int, 0, 1000),
    "min_p": (float, 0.0, 1.0),
    "repeat_penalty": (float, 0.0, 2.0),
    "num_predict": (int, -2, 1_048_576),
}
_MAX_STOP_SEQUENCES = 8
_MAX_STOP_LEN = 64
_MAX_SYSTEM_LEN = 16_000

# Settings keys whose value is a model id, and the endpoint-id key that scopes
# it (when set). Fallback chains hold [{"endpoint_id", "model"}] entries.
_MODEL_SETTING_KEYS = (
    ("default_model", "default_endpoint_id"),
    ("utility_model", "utility_endpoint_id"),
    ("vision_model", None),
    ("research_model", "research_endpoint_id"),
    ("task_model", "task_endpoint_id"),
    ("teacher_model", None),
    ("image_model", None),
)
_FALLBACK_SETTING_KEYS = ("utility_model_fallbacks", "vision_model_fallbacks")


class OllamaAdminError(Exception):
    """Upstream/validation failure with an HTTP status suitable for the route."""

    def __init__(self, message: str, status_code: int = 502, **extra: Any):
        super().__init__(message)
        self.status_code = status_code
        self.extra = extra


@dataclass
class OllamaTarget:
    id: str
    label: str
    root: str
    source: str  # "endpoint" | "local"
    endpoint_id: str = ""
    api_key: str = field(default="", repr=False)

    @property
    def host(self) -> str:
        return (urlparse(self.root).hostname or "").lower()

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "url": self.root,
            "host": self.host,
            "source": self.source,
            "endpoint_id": self.endpoint_id,
        }


# ── URL helpers ─────────────────────────────────────────────────────────────


def ollama_root(url: str) -> str:
    """``http://host:port`` for an Ollama base URL (drops ``/v1``, ``/api``…)."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parsed = urlparse(raw)
    except Exception:
        return ""
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return ""
    host = parsed.hostname
    if host in _ANY_BIND_HOSTS:
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        return ""
    return f"{parsed.scheme}://{host}:{port}" if port else f"{parsed.scheme}://{host}"


def _is_ollama_cloud(host: str) -> bool:
    return host == "ollama.com" or host.endswith(".ollama.com")


def endpoint_looks_like_ollama(base_url: str, endpoint_kind: Optional[str] = None) -> bool:
    """Ollama check for endpoint rows (kind, default port, name, or a known fingerprint)."""
    try:
        parsed = urlparse(str(base_url or ""))
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except Exception:
        return False
    if not host or _is_ollama_cloud(host):
        return False
    if str(endpoint_kind or "").strip().lower() == "ollama":
        return True
    if port == DEFAULT_OLLAMA_PORT or "ollama" in host:
        return True
    # Cached /api/version fingerprint (e.g. a Cookbook-served daemon on
    # 11435+); probe=False keeps this listing free of network calls.
    from src.ollama_capabilities import is_ollama_url

    return is_ollama_url(base_url, probe=False)


def _running_in_container() -> bool:
    try:
        from src.host_docker_access import running_in_container
        return running_in_container()
    except Exception:
        return False


def local_candidate_roots() -> List[str]:
    """Configured + default local Ollama roots, in priority order."""
    out: List[str] = []

    def add(raw: str) -> None:
        root = ollama_root(raw)
        if root and root not in out:
            out.append(root)

    for env_name in ("OLLAMA_BASE_URL", "OLLAMA_URL", "OLLAMA_HOST"):
        value = os.getenv(env_name, "").strip()
        if value:
            if env_name == "OLLAMA_HOST" and ":" not in value.split("://")[-1]:
                value = f"{value}:{DEFAULT_OLLAMA_PORT}"
            add(value)
    add(f"http://127.0.0.1:{DEFAULT_OLLAMA_PORT}")
    if _running_in_container():
        add(f"http://host.docker.internal:{DEFAULT_OLLAMA_PORT}")
    return out


def _load_endpoint_rows() -> List[Any]:
    from core.database import SessionLocal, ModelEndpoint
    db = SessionLocal()
    try:
        rows = db.query(ModelEndpoint).order_by(ModelEndpoint.created_at).all()
        for row in rows:
            db.expunge(row)
        return rows
    finally:
        db.close()


def list_targets() -> List[OllamaTarget]:
    """Allowlisted Ollama targets. No network I/O."""
    targets: List[OllamaTarget] = []
    seen: Dict[str, OllamaTarget] = {}
    try:
        rows = _load_endpoint_rows()
    except Exception as exc:  # noqa: BLE001 - DB trouble must not break the tab
        logger.debug("Ollama admin: endpoint listing failed: %s", exc)
        rows = []
    for row in rows:
        if (getattr(row, "model_type", None) or "llm") == "image":
            continue
        if not endpoint_looks_like_ollama(row.base_url, getattr(row, "endpoint_kind", None)):
            continue
        root = ollama_root(row.base_url)
        if not root or root in seen:
            continue
        target = OllamaTarget(
            id=f"ep:{row.id}",
            label=row.name or root,
            root=root,
            source="endpoint",
            endpoint_id=row.id,
            api_key=getattr(row, "api_key", None) or "",
        )
        seen[root] = target
        targets.append(target)
    for root in local_candidate_roots():
        if root in seen:
            continue
        target = OllamaTarget(id=f"local:{root}", label=f"Local Ollama ({urlparse(root).netloc})", root=root, source="local")
        seen[root] = target
        targets.append(target)
    return targets


def resolve_target(target_id: str) -> OllamaTarget:
    """Resolve an opaque id from :func:`list_targets`; anything else is refused."""
    wanted = str(target_id or "").strip()
    for target in list_targets():
        if target.id == wanted:
            return target
    raise OllamaAdminError("Unknown Ollama server. Pick one from the Ollama tab's server list.", 404)


# ── HTTP plumbing ───────────────────────────────────────────────────────────


def _make_client(timeout: Optional[float] = _REQUEST_TIMEOUT) -> httpx.AsyncClient:
    """Factory seam (tests swap in an ``httpx.MockTransport``)."""
    return httpx.AsyncClient(timeout=timeout, follow_redirects=False)


def _headers(target: OllamaTarget) -> Dict[str, str]:
    return {"Authorization": f"Bearer {target.api_key}"} if target.api_key else {}


def _upstream_error(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])[:300]
    except Exception:
        pass
    return f"Ollama returned HTTP {resp.status_code}"


async def _request(target: OllamaTarget, method: str, path: str, *, json_body: Any = None,
                   timeout: Optional[float] = _REQUEST_TIMEOUT) -> httpx.Response:
    try:
        async with _make_client(timeout) as client:
            return await client.request(method, target.root + path, json=json_body, headers=_headers(target))
    except httpx.HTTPError as exc:
        raise OllamaAdminError(f"Ollama at {target.root} is not reachable: {exc.__class__.__name__}", 502) from exc


async def probe_version(root: str, *, timeout: float = _PROBE_TIMEOUT, api_key: str = "") -> Optional[str]:
    """Return the Ollama version string when ``root`` answers ``/api/version``."""
    try:
        async with _make_client(timeout) as client:
            resp = await client.get(root + "/api/version",
                                    headers={"Authorization": f"Bearer {api_key}"} if api_key else None)
        if resp.status_code != 200:
            return None
        data = resp.json()
        version = data.get("version") if isinstance(data, dict) else None
        return str(version) if version else None
    except Exception:
        return None


async def describe_targets() -> List[Dict[str, Any]]:
    """Targets plus reachability. Local candidates are listed only when up."""
    targets = list_targets()
    versions = await asyncio.gather(*(probe_version(t.root, api_key=t.api_key) for t in targets))
    out = []
    for target, version in zip(targets, versions):
        if target.source == "local" and not version:
            continue
        item = target.public()
        item["reachable"] = bool(version)
        item["version"] = version or ""
        out.append(item)
    return out


async def find_running_daemon(roots: List[str]) -> Optional[Dict[str, str]]:
    """First root in ``roots`` that already answers as an Ollama server."""
    if not roots:
        return None
    versions = await asyncio.gather(*(probe_version(r) for r in roots))
    for root, version in zip(roots, versions):
        if version:
            return {"url": root, "version": version}
    return None


# ── Model metadata ──────────────────────────────────────────────────────────

# (root, name, digest) -> normalized /api/show details. Keyed by digest so a
# re-pulled model refreshes automatically.
_show_cache: Dict[tuple, Dict[str, Any]] = {}
_SHOW_CACHE_MAX = 512


def _parse_modelfile_parameters(text: Any) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if not isinstance(text, str):
        return params
    for line in text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        key, raw = parts[0], parts[1].strip().strip('"')
        try:
            value: Any = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        if key == "stop":
            params.setdefault("stop", []).append(raw)
        else:
            params[key] = value
    return params


def _model_info_value(info: Dict[str, Any], suffix: str) -> Any:
    for key, value in info.items():
        if key == suffix or key.endswith("." + suffix):
            return value
    return None


def kv_cache_profile(model_info: Any) -> Optional[Dict[str, int]]:
    """Rough f16 KV-cache cost from GGUF ``model_info``.

    Per layer: ``kv_heads × (key_length + value_length) × 2 bytes`` per cached
    token. Full-attention layers cache every context token (``per_token``);
    sliding-window layers (``*.attention.sliding_window_pattern`` true, e.g.
    Gemma) cache at most ``sliding_window`` tokens (``swa_per_token``, using
    the ``*_swa`` key/value lengths). Hybrid models with
    ``*.full_attention_interval = N`` keep KV only on every Nth layer.
    Per-layer ``head_count_kv`` arrays are honoured. Returns ``None`` when the
    fields are missing. q8_0 KV is ~0.53× and q4_0 ~0.28× of these values.
    """
    if not isinstance(model_info, dict):
        return None
    blocks = _model_info_value(model_info, "block_count")
    heads = _model_info_value(model_info, "attention.head_count")
    kv_heads = _model_info_value(model_info, "attention.head_count_kv")
    key_len = _model_info_value(model_info, "attention.key_length")
    value_len = _model_info_value(model_info, "attention.value_length")
    embed = _model_info_value(model_info, "embedding_length")
    if not isinstance(blocks, int) or blocks <= 0:
        return None
    if isinstance(heads, list):
        heads = max((h for h in heads if isinstance(h, int)), default=0)
    if kv_heads is None:
        kv_heads = heads
    if not isinstance(key_len, int) or key_len <= 0:
        key_len = (embed // heads) if isinstance(embed, int) and isinstance(heads, int) and heads > 0 else 0
    if not isinstance(value_len, int) or value_len <= 0:
        value_len = key_len
    if key_len <= 0:
        return None
    key_swa = _model_info_value(model_info, "attention.key_length_swa")
    value_swa = _model_info_value(model_info, "attention.value_length_swa")
    key_swa = key_swa if isinstance(key_swa, int) and key_swa > 0 else key_len
    value_swa = value_swa if isinstance(value_swa, int) and value_swa > 0 else value_len
    window = _model_info_value(model_info, "attention.sliding_window")
    window = window if isinstance(window, int) and window > 0 else 0
    pattern = _model_info_value(model_info, "attention.sliding_window_pattern")
    pattern = pattern if isinstance(pattern, list) and window else []
    interval = _model_info_value(model_info, "full_attention_interval")
    interval = interval if isinstance(interval, int) and interval > 1 else 1
    if isinstance(kv_heads, list):
        layer_heads = [h if isinstance(h, int) and h > 0 else 0 for h in kv_heads[:blocks]]
    elif isinstance(kv_heads, int) and kv_heads > 0:
        layer_heads = [kv_heads] * blocks
    else:
        return None
    full = swa = 0
    for i, n_heads in enumerate(layer_heads):
        if (i + 1) % interval:
            continue  # recurrent/SSM layer in a hybrid model: no KV cache
        if i < len(pattern) and pattern[i] is True:
            swa += n_heads * (key_swa + value_swa)
        else:
            full += n_heads * (key_len + value_len)
    return {"per_token": full * 2, "swa_per_token": swa * 2, "sliding_window": window if swa else 0}


def kv_cache_bytes_per_token(model_info: Any) -> Optional[int]:
    """Full-attention f16 KV bytes per context token (see :func:`kv_cache_profile`)."""
    profile = kv_cache_profile(model_info)
    return profile["per_token"] if profile else None


def _context_length(model_info: Any) -> Optional[int]:
    if not isinstance(model_info, dict):
        return None
    values = [v for k, v in model_info.items()
              if (k == "context_length" or k.endswith(".context_length"))
              and isinstance(v, int) and not isinstance(v, bool) and v > 0]
    return min(values) if values else None


def _normalize_show(data: Any) -> Dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    info = data.get("model_info") or {}
    details = data.get("details") or {}
    caps = data.get("capabilities")
    return {
        "capabilities": [str(c).lower() for c in caps] if isinstance(caps, list) else None,
        "context_length": _context_length(info),
        "parameter_size": details.get("parameter_size") or "",
        "quantization": details.get("quantization_level") or "",
        "family": details.get("family") or "",
        "parameters": _parse_modelfile_parameters(data.get("parameters")),
        "kv_cache": kv_cache_profile(info),
        "has_system": bool(data.get("system")),
    }


async def _show(target: OllamaTarget, name: str, digest: str) -> Dict[str, Any]:
    key = (target.root, name, digest)
    cached = _show_cache.get(key)
    if cached is not None:
        return cached
    try:
        resp = await _request(target, "POST", "/api/show", json_body={"model": name}, timeout=8.0)
        details = _normalize_show(resp.json()) if resp.status_code == 200 else {}
    except Exception as exc:  # noqa: BLE001 - one bad model must not hide the list
        logger.debug("Ollama admin: /api/show %s failed: %s", name, exc)
        return {}
    if details:
        if len(_show_cache) >= _SHOW_CACHE_MAX:
            _show_cache.clear()
        _show_cache[key] = details
    return details


def reset_cache() -> None:
    _show_cache.clear()


async def list_installed(target: OllamaTarget) -> List[Dict[str, Any]]:
    """``/api/tags`` merged with cached ``/api/show`` details."""
    resp = await _request(target, "GET", "/api/tags")
    if resp.status_code != 200:
        raise OllamaAdminError(_upstream_error(resp), 502)
    items = (resp.json() or {}).get("models") or []
    rows = [m for m in items if isinstance(m, dict) and (m.get("name") or m.get("model"))]
    sem = asyncio.Semaphore(4)

    async def _one(item: Dict[str, Any]) -> Dict[str, Any]:
        name = str(item.get("name") or item.get("model"))
        digest = str(item.get("digest") or "")
        async with sem:
            show = await _show(target, name, digest)
        details = item.get("details") or {}
        caps = show.get("capabilities")
        if caps is None and isinstance(item.get("capabilities"), list):
            caps = [str(c).lower() for c in item["capabilities"]]
        ctx = show.get("context_length") or details.get("context_length")
        return {
            "name": name,
            "digest": digest,
            "size": int(item.get("size") or 0),
            "modified_at": item.get("modified_at") or "",
            "parameter_size": show.get("parameter_size") or details.get("parameter_size") or "",
            "quantization": show.get("quantization") or details.get("quantization_level") or "",
            "family": show.get("family") or details.get("family") or "",
            "parent_model": details.get("parent_model") or "",
            "capabilities": caps or [],
            "context_length": ctx if isinstance(ctx, int) and ctx > 0 else None,
            "parameters": show.get("parameters") or {},
            # f16 KV estimate: per_token × ctx + swa_per_token × min(ctx, sliding_window)
            "kv_bytes_per_token": (show.get("kv_cache") or {}).get("per_token"),
            "kv_swa_bytes_per_token": (show.get("kv_cache") or {}).get("swa_per_token") or 0,
            "kv_sliding_window": (show.get("kv_cache") or {}).get("sliding_window") or 0,
        }

    models = await asyncio.gather(*(_one(m) for m in rows))
    return sorted(models, key=lambda m: m["name"])


async def list_running(target: OllamaTarget) -> List[Dict[str, Any]]:
    resp = await _request(target, "GET", "/api/ps")
    if resp.status_code != 200:
        raise OllamaAdminError(_upstream_error(resp), 502)
    out = []
    for item in (resp.json() or {}).get("models") or []:
        if not isinstance(item, dict):
            continue
        size = int(item.get("size") or 0)
        size_vram = int(item.get("size_vram") or 0)
        ctx = item.get("context_length")
        out.append({
            "name": str(item.get("name") or item.get("model") or ""),
            "size": size,
            "size_vram": size_vram,
            "gpu_fraction": round(size_vram / size, 4) if size else 0.0,
            "fully_on_gpu": bool(size) and size_vram >= size,
            "context_length": ctx if isinstance(ctx, int) and ctx > 0 else None,
            "expires_at": item.get("expires_at") or "",
            "parameter_size": (item.get("details") or {}).get("parameter_size") or "",
            "quantization": (item.get("details") or {}).get("quantization_level") or "",
        })
    return out


# ── Mutations ───────────────────────────────────────────────────────────────


def validate_model_ref(model: Any) -> str:
    name = str(model or "").strip()
    if not _MODEL_REF_RE.fullmatch(name) or ".." in name:
        raise OllamaAdminError("Invalid Ollama model name.", 400)
    return name


async def stream_pull(target: OllamaTarget, model: str) -> AsyncIterator[Dict[str, Any]]:
    """Yield Ollama ``/api/pull`` NDJSON progress objects.

    Closing the generator (client disconnect) closes the upstream stream,
    which cancels the pull; Ollama resumes partial blobs on the next pull.
    """
    model = validate_model_ref(model)
    async with _make_client(httpx.Timeout(30.0, read=None)) as client:
        try:
            async with client.stream("POST", target.root + "/api/pull",
                                     json={"model": model, "stream": True},
                                     headers=_headers(target)) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    try:
                        err = json.loads(body or b"{}").get("error")
                    except Exception:
                        err = None
                    yield {"error": str(err or f"Ollama returned HTTP {resp.status_code}")[:300]}
                    return
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except httpx.HTTPError as exc:
            yield {"error": f"Ollama at {target.root} is not reachable: {exc.__class__.__name__}"}


async def delete_model(target: OllamaTarget, model: str) -> None:
    model = validate_model_ref(model)
    resp = await _request(target, "DELETE", "/api/delete", json_body={"model": model})
    if resp.status_code == 404:
        raise OllamaAdminError(f"Model {model} is not installed on this server.", 404)
    if resp.status_code != 200:
        raise OllamaAdminError(_upstream_error(resp), 502)
    _drop_show_cache(target, model)


def _drop_show_cache(target: OllamaTarget, model: str) -> None:
    for key in [k for k in _show_cache if k[0] == target.root and k[1] == model]:
        _show_cache.pop(key, None)


_KEEP_ALIVE_RE = re.compile(r"^-?\d{1,6}(?:\.\d+)?[smh]?$")


def normalize_keep_alive(value: Any) -> Any:
    """``0`` unloads, negative keeps loaded, ``"30m"``/seconds set a duration."""
    if isinstance(value, bool):
        raise OllamaAdminError("Invalid keep_alive value.", 400)
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else value
    text = str(value or "").strip()
    if not _KEEP_ALIVE_RE.fullmatch(text):
        raise OllamaAdminError("keep_alive must be seconds or a duration like 30m / 24h, 0 to unload, -1 to keep loaded.", 400)
    if text.lstrip("-").isdigit():
        return int(text)
    return text


async def set_keep_alive(target: OllamaTarget, model: str, keep_alive: Any) -> Dict[str, Any]:
    """POST ``/api/generate`` with only ``model`` + ``keep_alive`` (no prompt)."""
    model = validate_model_ref(model)
    keep = normalize_keep_alive(keep_alive)
    payload = {"model": model, "keep_alive": keep, "stream": False}
    # Loading a large model for "keep loaded" can take a while.
    resp = await _request(target, "POST", "/api/generate", json_body=payload,
                          timeout=_REQUEST_TIMEOUT if keep == 0 else 300.0)
    if resp.status_code == 404:
        raise OllamaAdminError(f"Model {model} is not installed on this server.", 404)
    if resp.status_code != 200:
        raise OllamaAdminError(_upstream_error(resp), 502)
    try:
        data = resp.json()
    except ValueError:
        data = {}
    return {"model": model, "keep_alive": keep, "done_reason": (data or {}).get("done_reason", "")}


def _coerce_number(name: str, raw: Any) -> Any:
    kind, low, high = PRESET_PARAMETER_RANGES[name]
    if isinstance(raw, bool) or raw is None or raw == "":
        raise OllamaAdminError(f"{name} must be a number.", 400)
    try:
        num = float(raw)
    except (TypeError, ValueError):
        raise OllamaAdminError(f"{name} must be a number.", 400)
    if not math.isfinite(num):
        raise OllamaAdminError(f"{name} must be a number.", 400)
    if kind is int:
        if not num.is_integer():
            raise OllamaAdminError(f"{name} must be a whole number.", 400)
        num = int(num)
    if num < low or num > high:
        raise OllamaAdminError(f"{name} must be between {low} and {high}.", 400)
    return num


def build_create_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a preset request and return the ``/api/create`` JSON body."""
    if not isinstance(body, dict):
        raise OllamaAdminError("Invalid preset request.", 400)
    name = str(body.get("name") or "").strip()
    if not _PRESET_NAME_RE.fullmatch(name):
        raise OllamaAdminError(
            "Preset name must be lowercase letters, digits, '.', '_', '-' with one optional ':tag'.", 400)
    base = validate_model_ref(body.get("from"))
    if name == base or (":" not in name and f"{name}:latest" == base):
        raise OllamaAdminError("The preset name must differ from the base model.", 400)
    raw_params = body.get("parameters") or {}
    if not isinstance(raw_params, dict):
        raise OllamaAdminError("parameters must be an object.", 400)
    params: Dict[str, Any] = {}
    for key, value in raw_params.items():
        if value is None or value == "" or value == []:
            continue
        if key == "stop":
            stops = value if isinstance(value, list) else [value]
            if len(stops) > _MAX_STOP_SEQUENCES:
                raise OllamaAdminError(f"At most {_MAX_STOP_SEQUENCES} stop sequences.", 400)
            clean = []
            for s in stops:
                if not isinstance(s, str) or not s or len(s) > _MAX_STOP_LEN:
                    raise OllamaAdminError(f"Stop sequences must be 1-{_MAX_STOP_LEN} characters.", 400)
                clean.append(s)
            params["stop"] = clean
            continue
        if key not in PRESET_PARAMETER_RANGES:
            raise OllamaAdminError(f"Unsupported parameter: {key}", 400)
        params[key] = _coerce_number(key, value)
    payload: Dict[str, Any] = {"model": name, "from": base, "stream": False}
    if params:
        payload["parameters"] = params
    system = body.get("system")
    if system not in (None, ""):
        if not isinstance(system, str) or len(system) > _MAX_SYSTEM_LEN:
            raise OllamaAdminError(f"System prompt must be text up to {_MAX_SYSTEM_LEN} characters.", 400)
        payload["system"] = system
    return payload


async def create_preset(target: OllamaTarget, body: Dict[str, Any], *, overwrite: bool = False) -> Dict[str, Any]:
    payload = build_create_payload(body)
    tags = await _request(target, "GET", "/api/tags")
    try:
        tag_items = (tags.json() or {}).get("models") or [] if tags.status_code == 200 else []
    except ValueError:
        tag_items = []
    installed = {str(m.get("name") or m.get("model")) for m in tag_items if isinstance(m, dict)}
    if installed:
        if payload["from"] not in installed and f"{payload['from']}:latest" not in installed:
            raise OllamaAdminError(f"Base model {payload['from']} is not installed on this server.", 404)
        if not overwrite and (payload["model"] in installed or f"{payload['model']}:latest" in installed):
            raise OllamaAdminError(f"A model named {payload['model']} already exists.", 409)
    resp = await _request(target, "POST", "/api/create", json_body=payload, timeout=300.0)
    if resp.status_code != 200:
        raise OllamaAdminError(_upstream_error(resp), 502)
    _drop_show_cache(target, payload["model"])
    try:
        status = (resp.json() or {}).get("status", "success")
    except ValueError:
        status = "success"
    return {"model": payload["model"], "from": payload["from"], "parameters": payload.get("parameters", {}),
            "status": status}


# ── In-use guard ────────────────────────────────────────────────────────────


def _same_model(a: str, b: str) -> bool:
    a, b = str(a or "").strip().lower(), str(b or "").strip().lower()
    if not a or not b:
        return False
    if ":" not in a:
        a += ":latest"
    if ":" not in b:
        b += ":latest"
    return a == b


def _endpoint_root(endpoint_id: str, rows: List[Any]) -> Optional[str]:
    for row in rows:
        if row.id == endpoint_id:
            return ollama_root(row.base_url)
    return None


def models_in_use(target: OllamaTarget, model: str) -> List[Dict[str, str]]:
    """Settings that point at ``model`` on ``target`` (unscoped entries count)."""
    try:
        from src.settings import load_settings
        settings = load_settings() or {}
    except Exception:
        settings = {}
    try:
        rows = _load_endpoint_rows()
    except Exception:
        rows = []

    def _on_target(endpoint_id: Any) -> bool:
        endpoint_id = str(endpoint_id or "").strip()
        if not endpoint_id:
            return True
        root = _endpoint_root(endpoint_id, rows)
        return root is None or root == target.root

    uses: List[Dict[str, str]] = []
    for key, ep_key in _MODEL_SETTING_KEYS:
        if _same_model(settings.get(key), model) and _on_target(settings.get(ep_key) if ep_key else ""):
            uses.append({"setting": key, "model": str(settings.get(key))})
    for key in _FALLBACK_SETTING_KEYS:
        for entry in settings.get(key) or []:
            if isinstance(entry, dict) and _same_model(entry.get("model"), model) and _on_target(entry.get("endpoint_id")):
                uses.append({"setting": key, "model": str(entry.get("model"))})
                break
    try:
        from src.embeddings import _load_persisted_endpoint
        emb = _load_persisted_endpoint() or {}
        if _same_model(emb.get("model"), model) and ollama_root(emb.get("url", "")) in ("", target.root):
            uses.append({"setting": "embedding_endpoint", "model": str(emb.get("model"))})
    except Exception:
        pass
    return uses

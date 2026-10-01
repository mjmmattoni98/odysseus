"""Per-model performance summary built from persisted chat message metrics.

Every assistant reply stores its metrics in ``ChatMessage.metadata``: TTFT,
tokens/second (``tps_source`` says whether the backend measured it), and for
native Ollama also ``prefill_tps`` and ``load_ms``/``prefill_ms``/``gen_ms``.
This read-only API aggregates those per (endpoint, model) over a recent
window so users can see which model is slow to start, slow to generate, or
keeps being reloaded (a large ``load_ms`` means Ollama swapped the model in).
"""
import json
import logging
import math
import statistics
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request

from core.database import ChatMessage as DbChatMessage, Session as DbSession, SessionLocal, utcnow_naive
from src.auth_helpers import require_user
from src.owner_identity import auth_disabled

logger = logging.getLogger(__name__)

# Load time above this is a real model (re)load rather than a warm hit.
RELOAD_THRESHOLD_MS = 500.0
DEFAULT_WINDOW_DAYS = 30
DEFAULT_MESSAGE_LIMIT = 2000
MAX_MESSAGE_LIMIT = 5000
_SPEED_KEYS = ("time_to_first_token", "tokens_per_second", "gen_tps", "prefill_tps", "load_ms")


def _positive(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def _non_negative(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile; ``None`` for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _rounded(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(value, digits)


def _endpoint_host(url: str) -> str:
    """Host[:port] only, so stored URLs never echo credentials or paths."""
    try:
        parsed = urlparse(url or "")
    except ValueError:
        return ""
    host = parsed.hostname or ""
    try:
        port = parsed.port
    except ValueError:
        port = None
    return f"{host}:{port}" if host and port else host


def _ttft_seconds(meta: Dict[str, Any]) -> Optional[float]:
    """Measured TTFT, else the backend's load + prompt time (plain chat has no TTFT)."""
    measured = _positive(meta.get("time_to_first_token"))
    if measured is not None:
        return measured
    prefill_ms = _non_negative(meta.get("prefill_ms"))
    if prefill_ms is None:
        return None
    return (prefill_ms + (_non_negative(meta.get("load_ms")) or 0.0)) / 1000


def aggregate_model_performance(
    rows: Iterable[Dict[str, Any]],
    *,
    reload_threshold_ms: float = RELOAD_THRESHOLD_MS,
) -> List[Dict[str, Any]]:
    """Summarize message metrics per (endpoint, model).

    Each row is ``{"metadata": dict, "timestamp": datetime|None,
    "endpoint_url": str, "model": str}`` where ``endpoint_url``/``model`` are
    the session's values, used only when the message metadata lacks them.
    Failed and stopped replies are skipped: their timings describe an error.
    """
    groups: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        meta = row.get("metadata")
        if not isinstance(meta, dict) or meta.get("failed") or meta.get("stopped"):
            continue
        if not any(key in meta for key in _SPEED_KEYS):
            continue
        model = meta.get("model") if isinstance(meta.get("model"), str) and meta.get("model").strip() else row.get("model")
        if not isinstance(model, str) or not model.strip():
            continue
        model = model.strip()
        endpoint_id = meta.get("endpoint_id") if isinstance(meta.get("endpoint_id"), str) else None
        label = meta.get("endpoint_label") if isinstance(meta.get("endpoint_label"), str) else ""
        host = _endpoint_host(row.get("endpoint_url") or "")
        key = (endpoint_id or label or host, model)
        group = groups.setdefault(key, {
            "model": model,
            "endpoint_id": endpoint_id,
            "endpoint_label": label or host or "Unknown endpoint",
            "messages": 0,
            "ttft": [],
            "gen_backend": [],
            "gen_computed": [],
            "prompt": [],
            "load": [],
            "last_used": None,
        })
        group["messages"] += 1
        ttft = _ttft_seconds(meta)
        if ttft is not None:
            group["ttft"].append(ttft)
        tps = _positive(meta.get("tokens_per_second"))
        backend_tps = _positive(meta.get("gen_tps")) or (tps if meta.get("tps_source") == "backend" else None)
        if backend_tps is not None:
            group["gen_backend"].append(backend_tps)
        elif tps is not None:
            group["gen_computed"].append(tps)
        prompt_tps = _positive(meta.get("prefill_tps"))
        if prompt_tps is not None:
            group["prompt"].append(prompt_tps)
        load_ms = _non_negative(meta.get("load_ms"))
        if load_ms is not None:
            group["load"].append(load_ms)
        stamp = row.get("timestamp")
        if stamp is not None and (group["last_used"] is None or stamp > group["last_used"]):
            group["last_used"] = stamp

    summaries = []
    for group in groups.values():
        # Wall-clock tok/s reads low (it includes prefill and tool time), so
        # only fall back to it when the backend never reported real speed.
        gen = group["gen_backend"] or group["gen_computed"]
        loads = group["load"]
        summaries.append({
            "model": group["model"],
            "endpoint_id": group["endpoint_id"],
            "endpoint_label": group["endpoint_label"],
            "messages": group["messages"],
            "ttft_median_s": _rounded(statistics.median(group["ttft"]) if group["ttft"] else None, 2),
            "ttft_p90_s": _rounded(_percentile(group["ttft"], 90), 2),
            "gen_tps_median": _rounded(statistics.median(gen) if gen else None, 1),
            "gen_tps_source": ("backend" if group["gen_backend"] else "computed") if gen else None,
            "prompt_tps_median": _rounded(statistics.median(group["prompt"]) if group["prompt"] else None, 1),
            "load_median_s": _rounded(statistics.median(loads) / 1000 if loads else None, 2),
            "load_avg_s": _rounded(sum(loads) / len(loads) / 1000 if loads else None, 2),
            "reloads": sum(1 for value in loads if value > reload_threshold_ms),
            "last_used": group["last_used"].isoformat() + "Z" if group["last_used"] else None,
        })
    summaries.sort(key=lambda item: (-item["messages"], item["model"].lower()))
    return summaries


def setup_model_performance_routes() -> APIRouter:
    router = APIRouter(prefix="/api", tags=["models"])

    @router.get("/model-performance")
    def model_performance(
        request: Request,
        days: int = DEFAULT_WINDOW_DAYS,
        limit: int = DEFAULT_MESSAGE_LIMIT,
    ) -> Dict[str, Any]:
        """Aggregate the caller's own recent reply metrics per endpoint/model."""
        user = require_user(request)
        if not user and not auth_disabled():
            # Loopback bypass / first-run callers are not an identified owner;
            # never aggregate other users' sessions for them.
            raise HTTPException(401, "Authentication required")
        days = max(1, min(int(days), 365))
        limit = max(1, min(int(limit), MAX_MESSAGE_LIMIT))
        since = utcnow_naive() - timedelta(days=days)
        db = SessionLocal()
        try:
            query = (
                db.query(
                    DbChatMessage.meta_data,
                    DbChatMessage.timestamp,
                    DbSession.endpoint_url,
                    DbSession.model,
                )
                .join(DbSession, DbSession.id == DbChatMessage.session_id)
                .filter(
                    DbChatMessage.role == "assistant",
                    DbChatMessage.meta_data != None,  # noqa: E711
                    DbChatMessage.timestamp >= since,
                )
            )
            # Same ownership rule as session history: a named user sees only
            # their own sessions; "" only reaches here with auth disabled.
            if user:
                query = query.filter(DbSession.owner == user)
            records = query.order_by(DbChatMessage.timestamp.desc()).limit(limit).all()
        finally:
            db.close()

        rows = []
        for meta_raw, stamp, endpoint_url, session_model in records:
            try:
                meta = json.loads(meta_raw) if meta_raw else None
            except (TypeError, ValueError):
                continue
            rows.append({
                "metadata": meta,
                "timestamp": stamp,
                "endpoint_url": endpoint_url,
                "model": session_model,
            })
        return {
            "days": days,
            "limit": limit,
            "messages_scanned": len(rows),
            "reload_threshold_ms": RELOAD_THRESHOLD_MS,
            "models": aggregate_model_performance(rows),
        }

    return router

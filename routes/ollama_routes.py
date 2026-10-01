"""Cookbook Ollama management routes (admin-only).

Installed/loaded model listing, streamed pulls, delete, unload/keep-alive and
parameter presets against allowlisted Ollama servers (see src/ollama_admin.py).
Servers are addressed by the opaque ``server`` id from ``/servers``; raw URLs
are never accepted.
"""

import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from core.middleware import require_admin
from src import ollama_admin

logger = logging.getLogger(__name__)


class OllamaModelRequest(BaseModel):
    server: str
    model: str


class OllamaKeepAliveRequest(BaseModel):
    server: str
    model: str
    keep_alive: Any = -1


class OllamaPresetRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    server: str
    name: str
    from_model: Optional[str] = Field(default=None, alias="from")
    parameters: Dict[str, Any] = {}
    system: Optional[str] = None
    overwrite: bool = False


def _http_error(exc: ollama_admin.OllamaAdminError) -> HTTPException:
    if exc.extra:
        return HTTPException(exc.status_code, {"message": str(exc), **exc.extra})
    return HTTPException(exc.status_code, str(exc))


def _target(server_id: str) -> ollama_admin.OllamaTarget:
    try:
        return ollama_admin.resolve_target(server_id)
    except ollama_admin.OllamaAdminError as exc:
        raise _http_error(exc)


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _default_context() -> int:
    try:
        from src.settings import get_setting
        value = int(get_setting("local_context_limit_default", 32768) or 32768)
        return value if value > 0 else 32768
    except Exception:
        return 32768


def setup_ollama_routes() -> APIRouter:
    router = APIRouter(prefix="/api/cookbook/ollama", tags=["cookbook-ollama"])

    @router.get("/servers")
    async def ollama_servers(request: Request):
        require_admin(request)
        return {"servers": await ollama_admin.describe_targets()}

    @router.get("/models")
    async def ollama_installed(request: Request, server: str):
        require_admin(request)
        target = _target(server)
        try:
            models = await ollama_admin.list_installed(target)
        except ollama_admin.OllamaAdminError as exc:
            raise _http_error(exc)
        return {"server": target.public(), "models": models, "default_context": _default_context()}

    @router.get("/running")
    async def ollama_running(request: Request, server: str):
        require_admin(request)
        target = _target(server)
        try:
            models = await ollama_admin.list_running(target)
        except ollama_admin.OllamaAdminError as exc:
            raise _http_error(exc)
        return {"server": target.public(), "models": models}

    @router.post("/pull")
    async def ollama_pull(request: Request, req: OllamaModelRequest):
        """Proxy Ollama's /api/pull NDJSON progress as Server-Sent Events.

        Closing the EventSource/fetch (Cancel) closes the upstream stream,
        which stops the pull; Ollama resumes it on the next pull."""
        require_admin(request)
        target = _target(req.server)
        try:
            model = ollama_admin.validate_model_ref(req.model)
        except ollama_admin.OllamaAdminError as exc:
            raise _http_error(exc)

        async def _events():
            yield _sse("start", {"model": model, "server": target.id})
            ok = False
            stream = ollama_admin.stream_pull(target, model)
            try:
                async for item in stream:
                    if await request.is_disconnected():
                        logger.info("Ollama pull of %s cancelled by client", model)
                        return
                    if item.get("error"):
                        yield _sse("error", {"error": str(item["error"])})
                        return
                    yield _sse("progress", {
                        "status": str(item.get("status") or ""),
                        "digest": str(item.get("digest") or ""),
                        "total": int(item.get("total") or 0),
                        "completed": int(item.get("completed") or 0),
                    })
                    if item.get("status") == "success":
                        ok = True
            finally:
                await stream.aclose()
            if ok:
                ollama_admin.reset_cache()
                yield _sse("done", {"model": model, "status": "success"})
            else:
                yield _sse("error", {"error": "Pull ended without a success status."})

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.delete("/models")
    async def ollama_delete(request: Request, server: str, model: str, force: bool = False):
        require_admin(request)
        target = _target(server)
        try:
            name = ollama_admin.validate_model_ref(model)
            uses = ollama_admin.models_in_use(target, name)
            if uses and not force:
                raise ollama_admin.OllamaAdminError(
                    f"{name} is configured as " + ", ".join(u["setting"] for u in uses)
                    + ". Pick another model in Settings first, or delete with force.",
                    409,
                    in_use=uses,
                )
            await ollama_admin.delete_model(target, name)
        except ollama_admin.OllamaAdminError as exc:
            raise _http_error(exc)
        return {"ok": True, "model": name, "was_in_use": uses}

    @router.post("/unload")
    async def ollama_unload(request: Request, req: OllamaModelRequest):
        require_admin(request)
        target = _target(req.server)
        try:
            result = await ollama_admin.set_keep_alive(target, req.model, 0)
        except ollama_admin.OllamaAdminError as exc:
            raise _http_error(exc)
        return {"ok": True, **result}

    @router.post("/keep-alive")
    async def ollama_keep_alive(request: Request, req: OllamaKeepAliveRequest):
        require_admin(request)
        target = _target(req.server)
        try:
            result = await ollama_admin.set_keep_alive(target, req.model, req.keep_alive)
        except ollama_admin.OllamaAdminError as exc:
            raise _http_error(exc)
        return {"ok": True, **result}

    @router.post("/create")
    async def ollama_create(request: Request, req: OllamaPresetRequest):
        require_admin(request)
        target = _target(req.server)
        body = {"name": req.name, "from": req.from_model, "parameters": req.parameters, "system": req.system}
        try:
            result = await ollama_admin.create_preset(target, body, overwrite=req.overwrite)
        except ollama_admin.OllamaAdminError as exc:
            raise _http_error(exc)
        return {"ok": True, **result}

    return router

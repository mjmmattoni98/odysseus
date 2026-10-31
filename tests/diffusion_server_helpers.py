"""Import scripts/diffusion_server.py without torch, diffusers, or a GPU.

The real module (routes, lifecycle, locks) is executed; only ``torch`` and
``uvicorn`` are replaced with small stubs. Callers replace ``load_model``.
"""
import importlib.util
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diffusion_server.py"
_XFORMERS = ("xformers", "xformers.ops", "xformers.ops.fmha")


def _torch_stub():
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float32 = "bfloat16", "float16", "float32"
    torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
        empty_cache=lambda: None,
    )
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    return torch


def load_diffusion_server(monkeypatch, **args):
    monkeypatch.setitem(sys.modules, "torch", _torch_stub())
    monkeypatch.setitem(sys.modules, "uvicorn", types.ModuleType("uvicorn"))
    saved = {name: sys.modules.get(name) for name in _XFORMERS}
    spec = importlib.util.spec_from_file_location(f"diffusion_server_{uuid.uuid4().hex}", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:  # the script installs fake xformers modules process-wide
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    defaults = dict(
        model="Tongyi-MAI/Z-Image-Turbo", steps=2, guidance_scale=0.0,
        negative_prompt="", width=32, height=32, idle_timeout=300,
    )
    module._args = SimpleNamespace(**{**defaults, **args})
    return module

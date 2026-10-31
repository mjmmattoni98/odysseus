"""The editing route must use the configured model's guidance, including retry."""
import ast
import asyncio
import base64
import io
import logging
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import pytest
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image


@pytest.mark.parametrize("guidance", [1.0, 3.5])
@pytest.mark.parametrize("retry", [False, True])
def test_edit_uses_configured_guidance(guidance, retry):
    source = Path(__file__).resolve().parents[1] / "scripts/diffusion_server.py"
    names = {
        "_start_progress", "_update_progress", "_finish_progress",
        "_run_pipeline_with_progress", "_parse_size", "_quality_steps",
        "_guidance_scale", "_default_negative_prompt", "_pipeline_accepts_arg",
        "_pipeline_call_kwargs", "_image_response", "edit_image",
    }
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    calls = []

    class Pipeline:
        attempted = False

        def __call__(self, prompt, image, *, guidance_scale, num_inference_steps,
                     width=None, height=None, output_type="pil", callback_on_step_end=None):
            if retry and not self.attempted:
                self.attempted = True
                raise TypeError("pipeline rejected explicit dimensions")
            calls.append(guidance_scale)
            return SimpleNamespace(images=[image.copy()])

    namespace = {
        "File": File, "Form": Form, "UploadFile": UploadFile, "HTTPException": HTTPException,
        "app": FastAPI(),
        "base64": base64, "io": io, "time": time, "uuid": uuid,
        "logger": logging.getLogger(__name__), "_PROGRESS": {}, "_pipe": Pipeline(),
        "_args": SimpleNamespace(steps=4, guidance_scale=guidance, negative_prompt="", width=32, height=32),
    }
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    data = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(data, format="PNG")
    result = asyncio.run(namespace["edit_image"](
        prompt="Make the shirt red", image=UploadFile(filename="source.png", file=io.BytesIO(data.getvalue())),
        model="", n=1, size="32x32", quality="medium", response_format="b64_json", request_id="test",
    ))
    assert calls == [guidance]
    assert Image.open(io.BytesIO(base64.b64decode(result["data"][0]["b64_json"]))).size == (32, 32)

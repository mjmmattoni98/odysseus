"""The editing route must use the configured model's guidance, including retry."""
import asyncio
import base64
import io
from types import SimpleNamespace

import pytest
from fastapi import UploadFile
from PIL import Image

from tests.diffusion_server_helpers import load_diffusion_server


@pytest.mark.parametrize("guidance", [1.0, 3.5])
@pytest.mark.parametrize("retry", [False, True])
def test_edit_uses_configured_guidance(monkeypatch, guidance, retry):
    server = load_diffusion_server(monkeypatch, steps=4, guidance_scale=guidance)
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

    server._pipe = Pipeline()
    data = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(data, format="PNG")
    result = asyncio.run(server.edit_image(
        prompt="Make the shirt red", image=UploadFile(filename="source.png", file=io.BytesIO(data.getvalue())),
        model="", n=1, size="32x32", quality="medium", response_format="b64_json", request_id="test",
    ))
    assert calls == [guidance]
    assert Image.open(io.BytesIO(base64.b64decode(result["data"][0]["b64_json"]))).size == (32, 32)

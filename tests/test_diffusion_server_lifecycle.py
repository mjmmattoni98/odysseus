"""On-demand image model lifecycle in scripts/diffusion_server.py.

Like an Ollama keep_alive: discovery never loads weights, inference loads them
once, and they are released after the idle timeout unless a request is active
or queued.
"""
import base64
import gc
import io
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.diffusion_server_helpers import load_diffusion_server


class FakePipe:
    """Stands in for a Diffusers pipeline; optionally blocks inside a call."""

    def __init__(self, gate=None, entered=None):
        self.gate, self.entered = gate, entered
        self.calls = 0
        self.hooks_removed = False

    def __call__(self, prompt, image=None, width=32, height=32, callback_on_step_end=None, **_kwargs):
        self.calls += 1
        if self.entered is not None:
            self.entered.release()
        if self.gate is not None:
            assert self.gate.wait(5), "test gate never opened"
        return SimpleNamespace(images=[Image.new("RGB", (width or 32, height or 32), "red")])

    def remove_all_hooks(self):
        self.hooks_removed = True


@pytest.fixture
def server(monkeypatch):
    def build(fail_loads=0, load_gate=None, **args):
        mod = load_diffusion_server(monkeypatch, **args)
        state = SimpleNamespace(loads=0, pipes=[], fail_loads=fail_loads, pipe_kwargs={}, mmap_pinned=0)
        # Record instead of changing this test process's malloc settings.
        mod._prefer_mmap_for_large_allocations = lambda: setattr(state, "mmap_pinned", state.mmap_pinned + 1)

        def fake_load_model():
            state.loads += 1
            if load_gate is not None:
                assert load_gate.wait(5), "load gate never opened"
            if state.fail_loads:
                state.fail_loads -= 1
                mod._pipe = FakePipe()  # partial state a failed load may leave
                raise RuntimeError("simulated out of memory")
            mod._pipe = FakePipe(**state.pipe_kwargs)
            state.pipes.append(mod._pipe)

        mod.load_model = fake_load_model
        mod._model_id = ""
        client = TestClient(mod.app, base_url="http://localhost", raise_server_exceptions=False)
        return mod, state, client

    return build


def _generate(client, prompt="a lighthouse"):
    return client.post("/v1/images/generations", json={"prompt": prompt, "size": "32x32"})


def _edit(client):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(buf, format="PNG")
    return client.post(
        "/v1/images/edits",
        data={"prompt": "make it red", "size": "32x32"},
        files={"image": ("source.png", buf.getvalue(), "image/png")},
    )


def _png_size(response):
    data = base64.b64decode(response.json()["data"][0]["b64_json"])
    return Image.open(io.BytesIO(data)).size


def test_startup_discovery_and_health_do_not_load_weights(server):
    mod, state, client = server()
    with client:
        assert client.get("/v1/models").json()["data"][0]["id"] == "Z-Image-Turbo"
        health = client.get("/health").json()
        assert client.get("/v1/images/progress/unknown").json()["status"] == "unknown"
    assert state.loads == 0
    assert state.mmap_pinned == 1  # freed weights must be returnable to the OS
    assert health == {"status": "ok", "model": "Z-Image-Turbo", "loaded": False,
                      "active_requests": 0, "idle_timeout": 300.0}


def test_zero_idle_timeout_keeps_eager_startup_load(server):
    mod, state, client = server(idle_timeout=0)
    with client:
        assert state.loads == 1
        assert _generate(client).status_code == 200
        assert mod._unload_if_idle(now=time.monotonic() + 10**6) is False
    assert state.loads == 1 and state.mmap_pinned == 0


def test_first_request_loads_once_and_later_requests_reuse(server):
    mod, state, client = server()
    with client:
        first, second = _generate(client), _generate(client)
        assert client.get("/health").json()["loaded"] is True
    assert (first.status_code, second.status_code) == (200, 200)
    assert _png_size(first) == (32, 32)
    assert state.loads == 1 and state.pipes[0].calls == 2


def test_idle_timeout_releases_all_pipelines_and_next_request_reloads(server):
    mod, state, client = server()
    with client:
        assert _generate(client).status_code == 200
        derived = FakePipe()
        mod._inpaint_pipe = mod._img2img_pipe = derived  # cached adapters share weights
        finished = mod._last_activity
        assert mod._unload_if_idle(now=finished + 299.9) is False
        assert mod._pipe is not None
        assert mod._unload_if_idle(now=finished + 300.01) is True
        assert (mod._pipe, mod._inpaint_pipe, mod._img2img_pipe) == (None, None, None)
        assert state.pipes[0].hooks_removed and derived.hooks_removed
        assert client.get("/health").json()["loaded"] is False
        assert _generate(client).status_code == 200
    assert state.loads == 2


def test_background_watcher_unloads_after_real_idle_interval(server):
    mod, state, client = server(idle_timeout=0.3)
    with client:
        assert _generate(client).status_code == 200
        deadline = time.monotonic() + 5
        while client.get("/health").json()["loaded"] and time.monotonic() < deadline:
            time.sleep(0.05)
        assert client.get("/health").json()["loaded"] is False
        assert _generate(client).status_code == 200
    assert state.loads == 2


def test_active_and_queued_requests_block_unload_and_share_one_load(server):
    load_gate, pipe_gate = threading.Event(), threading.Event()
    entered = threading.Semaphore(0)
    mod, state, client = server(load_gate=load_gate)
    state.pipe_kwargs = {"gate": pipe_gate, "entered": entered}
    with client, ThreadPoolExecutor(max_workers=3) as pool:
        # A cold generation and an async edit race for the same unloaded model.
        futures = [pool.submit(_generate, client), pool.submit(_edit, client), pool.submit(_generate, client)]
        deadline = time.monotonic() + 5
        while mod._active_requests < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert mod._active_requests == 3
        assert client.get("/v1/models").status_code == 200  # stays responsive while loading
        load_gate.set()
        assert entered.acquire(timeout=5)  # one request is inside the pipeline
        assert mod._unload_if_idle(now=time.monotonic() + 10**6) is False
        assert mod._pipe is not None
        pipe_gate.set()
        responses = [future.result(timeout=10) for future in futures]
    assert [r.status_code for r in responses] == [200, 200, 200]
    assert state.loads == 1 and state.pipes[0].calls == 3
    assert mod._active_requests == 0


def test_failed_load_clears_partial_state_and_next_request_retries(server):
    mod, state, client = server(fail_loads=1)
    with client:
        failed = _generate(client)
        assert failed.status_code == 503
        assert "simulated out of memory" in failed.text
        assert mod._pipe is None and mod._active_requests == 0
        assert _generate(client).status_code == 200
    assert state.loads == 2


def test_inference_error_releases_lease_so_idle_unload_still_happens(server):
    mod, state, client = server()

    def broken(*_args, **_kwargs):
        raise RuntimeError("HIP error")

    with client:
        assert _generate(client).status_code == 200
        state.pipes[0].__class__ = type("BrokenPipe", (FakePipe,), {"__call__": broken})
        assert _generate(client).status_code == 500
        assert mod._active_requests == 0
        assert mod._unload_if_idle(now=mod._last_activity + 300.01) is True


class Payload:
    pass


class FinalizedPipe(FakePipe):
    """Its finalizer creates new cyclic garbage, like offload hooks and tensors."""

    created = []

    def __del__(self):
        holder = {"payload": Payload()}
        holder["self"] = holder
        FinalizedPipe.created.append(weakref.ref(holder["payload"]))


def test_unload_frees_cyclic_pipelines_without_waiting_for_automatic_gc(server):
    mod, state, client = server()
    with client:
        assert _generate(client).status_code == 200
        first = state.pipes[0]
        first.self_ref = first  # real pipelines are cycles (components <-> hooks)
        derived = FinalizedPipe()
        derived.self_ref = derived
        mod._img2img_pipe = derived
        alive = [weakref.ref(first), weakref.ref(derived)]
        del first, derived
        state.pipes.clear()
        gc.disable()
        try:
            assert mod._unload_if_idle(now=mod._last_activity + 300.01) is True
            assert [ref() for ref in alive] == [None, None]
            assert [ref() for ref in FinalizedPipe.created] == [None]
        finally:
            gc.enable()


def test_failed_load_frees_partial_pipeline_before_returning(server):
    mod, state, client = server(fail_loads=1)
    fake_load = mod.load_model
    partial = []

    def load_and_track_partial():
        try:
            fake_load()  # leaves a partial FakePipe in _pipe, then raises
        finally:
            if mod._pipe is not None:
                mod._pipe.self_ref = mod._pipe
                partial.append(weakref.ref(mod._pipe))

    mod.load_model = load_and_track_partial
    with client:
        gc.disable()
        try:
            assert _generate(client).status_code == 503
            assert partial and partial[0]() is None
        finally:
            gc.enable()

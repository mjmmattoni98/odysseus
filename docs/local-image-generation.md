# Local image generation on AMD Linux

The optional ROCm service runs Odysseus's existing `scripts/diffusion_server.py`.
Chat's image tool and Gallery use it through the normal image provider settings;
no separate image UI or cloud image account is required.

The starting configuration targets an RX 7900 XTX with 24 GB VRAM and uses
Z-Image-Turbo in BF16, 9 steps, guidance 0, and model CPU offloading. It is the
only configured local image model. Keep room in system RAM for offloaded
components while it is loaded. The initial model download is about
31 GiB; the Python/ROCm image also needs several GiB. The Hugging Face cache is
persistent and shared with Cookbook. Downloads need internet; generation uses
the local GPU.

## Start the inference service

From the checkout root, with Docker Compose and a working AMD `/dev/kfd`:

```bash
export RENDER_GID="$(getent group render | cut -d: -f3)"
docker compose -f docker-compose.yml -f docker/image.amd.yml \
  up -d --build --no-deps image-generation
docker compose -f docker-compose.yml -f docker/image.amd.yml \
  logs --tail 50 -f image-generation
```

Use the same Compose project name and `APP_DATA_DIR` as the running Odysseus
instance. This command starts only image inference. Wait for Uvicorn's startup
completion; the service starts without loading weights. Exit log following
with Ctrl+C. The service restarts with Docker. `RENDER_GID` must also be
supplied when running later Compose commands.

## On-demand loading

The service follows the same lifecycle as an idle Ollama text model. Its
listener stays up, and `/health`, `/v1/models`, Odysseus model discovery, and
endpoint probes never load weights. The first image request (chat message to
`Z-Image-Turbo`, the image tool, or Gallery) loads the model; later requests
reuse it. After 300 seconds without an image request, the weights are released
from RAM and GPU memory. The next request loads them again.

Requests run one at a time; queued or running requests keep the model loaded.
A failed load returns HTTP 503 and the next request retries. `GET /health`
reports `loaded` and `active_requests`. Change the timeout with
`IMAGE_IDLE_TIMEOUT` (seconds) when starting the service; `0` restores the
old behaviour of loading at startup and never unloading.

Loading from a warm page cache took 5-15 seconds on the RX 7900 XTX. In a new
container, the first generation at each image size also prepares ROCm kernels:
about 1 minute at 512x512 and 3 minutes at 1024x1024. Reloads in the same
container skip that step. The whole request must finish within Odysseus's
300-second image read timeout.

CPU offloading keeps inactive components in RAM and uses the GPU only during
generation. Measured on this workstation, the server process used about
0.9 GiB of RAM before the first load, 22 GiB while loaded, and 2.3 GiB after
unloading. The remainder is ROCm runtime state, as is about 0.56 GiB of
VRAM held by the process. Stopping the service frees it.

The service has no host-published port. Inside the Compose network, its image
API is `http://image-generation:8110/v1`. Port 8100 remains available for the
existing host ChromaDB mapping. No GPU devices or packages are added to the
main application container.

## Connect Odysseus

In Settings > Models, add an enabled endpoint with:

- Name: `Local images (AMD)`
- Base URL: `http://image-generation:8110/v1`
- API key: empty
- Model type: `image`
- Endpoint kind: `local`
- Model: `Z-Image-Turbo` (discovered from `/v1/models`)

Image endpoint probes check the advertised `/v1/models` list, without sending
a chat completion or generating an image. Re-probing therefore does not load
weights or hide a listed model merely because it does not serve text chat.
This verifies availability, not generation quality or chat-tool support.

Enable image generation, select `Z-Image-Turbo`, and choose Medium quality.
Then ask chat to generate an image, or use Gallery's generation controls.
This model generates images from text; instruction-based photo editing is a
separate model capability and is not established by this setup.

## Verify the actual integration

The following calls Odysseus's production image tool, resolves the configured
endpoint, performs inference, and checks the saved PNG and Gallery row:

```bash
docker compose exec -T odysseus gosu odysseus \
  python - --owner YOUR_USERNAME < scripts/verify_local_image_generation.py
```

For a custom image, run Python as the application account with its HOME. This produces one
real Gallery image; `PASS` includes its URL, model, dimensions, and elapsed
time. It does not test the chat model's decision to invoke the image tool.

Stop the listener without stopping Odysseus or deleting weights:

```bash
docker compose -f docker-compose.yml -f docker/image.amd.yml stop image-generation
```

The default model is configurable with `IMAGE_MODEL`, but its steps, guidance,
VRAM needs, and editing support must be adjusted for the replacement model.

## Retired: FLUX.2 Klein 4B

FLUX.2 Klein 4B was evaluated as a second local service and has been retired;
its Compose overlay and endpoint were removed. Its cached weights and the
comparison results under `data/image-comparison/` were kept.
`scripts/benchmark_local_images.py` reproduces that comparison: six prompts,
two seeds, 1024x1024, BF16, CPU offloading, each model's recommended steps and
guidance, with one excluded warm-up, in disposable containers.

Sources: [Diffusers Z-Image](https://huggingface.co/docs/diffusers/api/pipelines/z_image),
[model settings](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo),
[AMD PyTorch installation](https://github.com/Comfy-Org/ComfyUI#amd-gpus-linux).

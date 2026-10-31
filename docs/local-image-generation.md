# Local image generation on AMD Linux

The optional ROCm service runs Odysseus's existing `scripts/diffusion_server.py`.
Chat's image tool and Gallery use it through the normal image provider settings;
no separate image UI or cloud image account is required.

The starting configuration targets an RX 7900 XTX with 24 GB VRAM and uses
Z-Image-Turbo in BF16, 9 steps, guidance 0, and model CPU offloading. Keep room
in system RAM for offloaded components. The initial model download is about
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
instance. This command starts only image inference. Wait for `Model loaded`
and Uvicorn's startup completion; the first download/load can take several
minutes. Exit log following with Ctrl+C. The service restarts with Docker.
`RENDER_GID` must also be supplied when running later Compose commands.

The first generation on a fresh ROCm kernel cache can also take several
minutes. Wait for the request to finish before submitting another. CPU
offloading moves inactive components to RAM; the tested service held about
24 GiB of system RAM while loaded, with GPU memory released between requests.

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

Stop inference without stopping Odysseus or deleting weights:

```bash
docker compose -f docker-compose.yml -f docker/image.amd.yml stop image-generation
```

The default model is configurable with `IMAGE_MODEL`, but its steps, guidance,
VRAM needs, and editing support must be adjusted for the replacement model.

## FLUX.2 Klein 4B: generation and instruction editing

This model has a separate optional service with the documented 4-step,
guidance-1 configuration. It uses the same ROCm image and model cache:

```bash
export RENDER_GID="$(getent group render | cut -d: -f3)"
docker compose -f docker-compose.yml -f docker/image.amd.yml \
  -f docker/image.flux-amd.yml up -d --build --no-deps image-flux
```

Add an image endpoint at `http://image-flux:8111/v1`, without an API key,
and select its discovered `FLUX.2-klein-4B` model at Medium quality. The
service provides generation and single-reference instruction editing through
Odysseus's existing tools. Multi-reference editing and native masked inpainting
have not been wired or tested by this setup.

Verify generation or editing without changing the global default:

```bash
docker compose exec -T odysseus gosu odysseus python - \
  --owner YOUR_USERNAME --model FLUX.2-klein-4B \
  < scripts/verify_local_image_generation.py

docker compose exec -T odysseus gosu odysseus python - \
  --owner YOUR_USERNAME --model FLUX.2-klein-4B \
  --source-image /app/data/path/to/photo.png \
  --prompt 'Change the blue shirt to red; preserve the person and background.' \
  < scripts/verify_local_image_generation.py
```

CPU offloading still keeps the model in system RAM. The workstation comparison
therefore runs one image model at a time. To switch from FLUX to Z-Image:

```bash
docker compose -f docker-compose.yml -f docker/image.amd.yml \
  -f docker/image.flux-amd.yml stop image-flux
docker compose -f docker-compose.yml -f docker/image.amd.yml \
  up -d --no-deps image-generation
```

Then enable the Z-Image endpoint, disable the stopped FLUX endpoint, and select
`Z-Image-Turbo` in image settings. Reverse the service stop/start and settings
changes to return to FLUX. Disabling a stopped endpoint also prevents model
resolution from waiting on its unreachable Docker hostname.

The reproducible benchmark is `scripts/benchmark_local_images.py`: six prompts,
two seeds, 1024x1024, BF16, CPU offloading, each model's recommended steps and
guidance, with one excluded warm-up. It calls the production diffusion-server
handlers in disposable containers; separate live checks exercise Odysseus's
provider resolution, HTTP transport, and Gallery persistence. Results and
original PNGs from this workstation are under `data/image-comparison/`.

Sources: [Diffusers Z-Image](https://huggingface.co/docs/diffusers/api/pipelines/z_image),
[model settings](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo),
[AMD PyTorch installation](https://github.com/Comfy-Org/ComfyUI#amd-gpus-linux).

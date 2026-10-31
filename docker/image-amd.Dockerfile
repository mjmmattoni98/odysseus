FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Keep the ROCm wheels separate from PyPI resolution: a generic torch install
# can otherwise select CUDA packages on this Linux host.
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/rocm7.2
COPY docker/requirements-image.txt /tmp/requirements-image.txt
RUN pip install --no-cache-dir -r /tmp/requirements-image.txt

# Required by the ROCm wheel's native libraries on Debian slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY scripts/diffusion_server.py /app/diffusion_server.py
USER 1000:1000
ENTRYPOINT ["python", "diffusion_server.py"]

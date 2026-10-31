#!/usr/bin/env python3
"""Compare the actual diffusion-server handlers with matched prompts and seeds.

Run one model per disposable image-service container, with no other image model
running. Mount this script, the server, a shared writable ROCm cache, and an
output directory. A warm-up is recorded separately. Existing completed samples
are reused only when their configuration matches. See the comparison report.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import importlib.util
import io
import json
from pathlib import Path
import resource
import statistics
import time
from types import SimpleNamespace


CASES = [
    ("watercolor", "A small red sailboat on a calm blue lake, green mountains in the background, warm morning sunlight, detailed watercolor illustration."),
    ("portrait", "Photorealistic waist-up studio portrait of an adult woman with short curly dark hair, freckles, wearing a plain blue cotton shirt, neutral gray background, soft window light, natural skin texture, both hands visible resting on a wooden table."),
    ("product", "Professional product photograph of a matte black ceramic coffee mug with one curved handle, next to a whole yellow lemon on a pale beige stone counter. Soft light from the left, realistic contact shadows, uncluttered composition, no lettering."),
    ("typography", 'A clean travel poster with the exact large headline "EXPLORE MORE" and the exact smaller subtitle "MADRID 2026". A red bicycle in front of a cream wall, two potted green plants, simple balanced graphic design. No other text.'),
    ("spatial", "A minimal still-life photograph on a plain white table: exactly three apples in one row, a red apple on the left, a green apple in the middle, a yellow apple on the right. One blue ceramic cup stands behind the green apple. No other objects."),
    ("landscape", "Photorealistic landscape of a small stone cottage beside a narrow stream in the Scottish Highlands, purple heather in the foreground, misty mountains behind, overcast morning, realistic stone and water detail, wide composition, no people."),
]


def load_server(path):
    spec = importlib.util.spec_from_file_location("image_benchmark_server", path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    return server


def save_result(payload, output):
    from PIL import Image, ImageStat
    raw = base64.b64decode(payload["data"][0]["b64_json"])
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if image.size != (1024, 1024):
            raise RuntimeError(f"Unexpected image dimensions: {image.size}")
        if max(ImageStat.Stat(image.convert("RGB")).stddev) < 1:
            raise RuntimeError("Effectively blank image")
    output.write_bytes(raw)


def run(args):
    import torch
    from starlette.datastructures import UploadFile

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("This comparison requires the AMD ROCm GPU")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    server = load_server(args.server)
    server._args = SimpleNamespace(
        model=args.model, dtype="bfloat16", cpu_offload=True,
        attention_slicing=False, vae_slicing=True, lora=None, lora_scale=1.0,
        width=1024, height=1024, steps=args.steps, guidance_scale=args.guidance,
        negative_prompt="", single_file_config="", harmonize_gpu=None,
    )
    config = {
        "model": args.model, "steps": args.steps, "guidance": args.guidance,
        "dtype": "bfloat16", "cpu_offload": True, "seeds": args.seeds,
        "resolution": "1024x1024", "cases": CASES,
        "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
    }
    # Normalize tuples before comparing an existing JSON record.
    config = json.loads(json.dumps(config))
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Output directory contains a different benchmark configuration")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    load_started = time.perf_counter()
    server.load_model()
    print(json.dumps({"event": "loaded", "seconds": round(time.perf_counter()-load_started, 3)}), flush=True)

    def sample(name, prompt, seed, source=None):
        record_path = output / (name + ".json")
        image_path = output / (name + ".png")
        if record_path.exists() and image_path.exists():
            return json.loads(record_path.read_text())
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if source is None:
            payload = server.generate_image(server.ImageRequest(
                prompt=prompt, size="1024x1024", quality="medium", request_id=name,
            ))
        else:
            upload = UploadFile(filename=Path(source).name, file=io.BytesIO(Path(source).read_bytes()))
            payload = asyncio.run(server.edit_image(
                prompt=prompt, image=upload, model="", n=1, size="1024x1024",
                quality="medium", response_format="b64_json", request_id=name,
            ))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        save_result(payload, image_path)
        record = {
            "name": name, "prompt": prompt, "seed": seed,
            "seconds": round(elapsed, 3), "image": str(image_path),
            "source": str(source) if source else None,
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated()/2**30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved()/2**30, 3),
            "process_peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, 3),
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        return record

    sample("warmup", CASES[0][1], 999)
    records = [sample(f"{name}-{seed}", prompt, seed) for name, prompt in CASES for seed in args.seeds]
    if args.edit_source:
        sample("edit-sail", "Change only the red sail to a vivid cobalt blue sail. Keep the hull red and preserve the boat, mountains, lake, composition and watercolor style.", 2026, args.edit_source)
    if args.portrait_source:
        sample("edit-shirt", "Change only the blue shirt to a bright red shirt. Preserve the woman's identity, face, hair, hands, pose, background, lighting and photographic style.", 2026, args.portrait_source)
    summary = {
        "model": args.model, "samples": len(records),
        "median_seconds": round(statistics.median(r["seconds"] for r in records), 3),
        "mean_seconds": round(statistics.mean(r["seconds"] for r in records), 3),
        "min_seconds": min(r["seconds"] for r in records),
        "max_seconds": max(r["seconds"] for r in records),
        "peak_allocated_gib": max(r["peak_allocated_gib"] for r in records),
        "peak_reserved_gib": max(r["peak_reserved_gib"] for r in records),
        "process_peak_rss_gib": max(r["process_peak_rss_gib"] for r in records),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", **summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="/app/diffusion_server.py")
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--guidance", required=True, type=float)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    parser.add_argument("--output", required=True)
    parser.add_argument("--edit-source")
    parser.add_argument("--portrait-source")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

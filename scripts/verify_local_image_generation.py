#!/usr/bin/env python3
"""Exercise the configured image provider and verify its persisted Gallery PNG.

Run in the same environment/data directory as Odysseus. For Docker, pipe this
file into the running app's Python; see docs/local-image-generation.md.
This uses real inference, creates one Gallery image, and does not change settings.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def verify(args):
    from PIL import Image, ImageStat
    from src.ai_interaction import do_edit_image, do_generate_image
    from src.constants import GENERATED_IMAGES_DIR
    from src.database import GalleryImage, SessionLocal

    started = time.monotonic()
    if args.source_image:
        result = await do_edit_image(
            args.prompt, args.source_image, model_spec=args.model or "",
            owner=args.owner, size=args.size, quality="medium",
        )
    else:
        content = json.dumps({"prompt": args.prompt, "model": args.model, "size": args.size, "quality": "medium"})
        result = await do_generate_image(content, owner=args.owner)
    if result.get("error"):
        raise RuntimeError(result["error"])
    if not result.get("image_id"):
        raise RuntimeError(f"Image was not saved to Gallery: {result}")
    with SessionLocal() as db:
        row = db.query(GalleryImage).filter_by(id=result["image_id"]).one()
        path = Path(GENERATED_IMAGES_DIR) / row.filename
        if row.owner != args.owner:
            raise RuntimeError("Gallery ownership does not match the request")
    with Image.open(path) as image:
        image.load()
        expected_size = tuple(int(part) for part in args.size.split("x"))
        if image.size != expected_size or image.format != "PNG":
            raise RuntimeError(f"Unexpected image: {image.format} {image.size}")
        if max(ImageStat.Stat(image.convert("RGB")).stddev) < 1:
            raise RuntimeError("Generated image is effectively blank")
    print(json.dumps({
        "status": "PASS",
        "model": result["image_model"],
        "image_url": result["image_url"],
        "image_id": result["image_id"],
        "path": str(path),
        "size": args.size,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", help="Account that should own the generated image")
    parser.add_argument("--model", help="Override the configured default image model")
    parser.add_argument("--source-image", help="Edit this image instead of generating a new one")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--prompt", default=(
        "A small red sailboat on a calm blue lake, green mountains in the "
        "background, warm morning sunlight, detailed watercolor illustration."
    ))
    args = parser.parse_args()
    asyncio.run(verify(args))


if __name__ == "__main__":
    main()

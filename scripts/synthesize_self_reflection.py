#!/usr/bin/env python3
"""Create a five-image synthetic reflection set from clean self-collected photos."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, default=ROOT / "self")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/custom_synth")
    parser.add_argument(
        "--reflection-dir",
        type=Path,
        default=ROOT / "repos/ERRNet/datasets/raw_data/real89/transmission_layer",
    )
    parser.add_argument(
        "--reflection-fallback-dir",
        type=Path,
        default=ROOT / "repos/ERRNet/datasets/raw_data/real89/blended",
    )
    parser.add_argument("--seed", type=int, default=2018)
    parser.add_argument("--max-long-edge", type=int, default=1024)
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def load_rgb(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def resize_long_edge(image: Image.Image, max_long_edge: int) -> Image.Image:
    long_edge = max(image.size)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / float(long_edge)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def resize_cover(image: Image.Image, size: tuple[int, int], rng: random.Random) -> Image.Image:
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(target_w, math.ceil(image.width * scale)), max(target_h, math.ceil(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = rng.randint(0, resized.width - target_w) if resized.width > target_w else 0
    top = rng.randint(0, resized.height - target_h) if resized.height > target_h else 0
    return resized.crop((left, top, left + target_w, top + target_h))


def make_mask(width: int, height: int, rng: random.Random) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx = rng.uniform(0.35, 0.65) * width
    cy = rng.uniform(0.30, 0.70) * height
    sx = rng.uniform(0.55, 0.95) * width
    sy = rng.uniform(0.55, 0.95) * height
    mask = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy)))
    stripe = 0.85 + 0.15 * np.sin((xx / max(width, 1)) * rng.uniform(2.5, 5.5) * math.pi + rng.random())
    mask = np.clip(mask * stripe, 0.25, 1.0)
    return mask[..., None].astype(np.float32)


def save_float_image(array: np.ndarray, path: Path) -> None:
    image = Image.fromarray(np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB")
    image.save(path)


def clear_image_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
            child.unlink()


def synthesize_pair(
    clean_path: Path,
    reflection_path: Path,
    name: str,
    out_dir: Path,
    rng: random.Random,
    max_long_edge: int,
) -> dict[str, object]:
    transmission = resize_long_edge(load_rgb(clean_path), max_long_edge)
    reflection = resize_cover(load_rgb(reflection_path), transmission.size, rng)

    sigma = rng.uniform(2.0, 6.0)
    alpha_t = rng.uniform(0.84, 0.93)
    alpha_r = rng.uniform(0.26, 0.42)
    reflection = reflection.filter(ImageFilter.GaussianBlur(radius=sigma))

    t = np.asarray(transmission, dtype=np.float32) / 255.0
    r = np.asarray(reflection, dtype=np.float32) / 255.0
    mask = make_mask(transmission.width, transmission.height, rng)
    reflection_layer = np.clip(alpha_r * r * mask, 0.0, 1.0)
    blended = np.clip(alpha_t * t + reflection_layer, 0.0, 1.0)

    save_float_image(blended, out_dir / "blended" / f"{name}.png")
    save_float_image(t, out_dir / "transmission_layer" / f"{name}.png")
    save_float_image(reflection_layer, out_dir / "reflection_layer" / f"{name}.png")

    return {
        "name": name,
        "clean": str(clean_path),
        "reflection_source": str(reflection_path),
        "width": transmission.width,
        "height": transmission.height,
        "sigma": sigma,
        "alpha_t": alpha_t,
        "alpha_r": alpha_r,
    }


def main() -> int:
    args = parse_args()
    clean_paths = list_images(args.clean_dir)
    if len(clean_paths) != 5:
        raise SystemExit(f"Expected exactly 5 clean self images in {args.clean_dir}, found {len(clean_paths)}")

    reflection_paths = list_images(args.reflection_dir)
    reflection_root = args.reflection_dir
    if not reflection_paths:
        reflection_paths = list_images(args.reflection_fallback_dir)
        reflection_root = args.reflection_fallback_dir
    if not reflection_paths:
        raise SystemExit("No reflection source images found in real89 transmission or blended folders")

    out_dir = args.output_dir
    for dirname in ("blended", "transmission_layer", "reflection_layer"):
        clear_image_dir(out_dir / dirname)

    rng = random.Random(args.seed)
    metadata = {
        "seed": args.seed,
        "max_long_edge": args.max_long_edge,
        "clean_dir": str(args.clean_dir),
        "reflection_dir": str(reflection_root),
        "formula": "M = clip(alpha_t * T + alpha_r * GaussianBlur(R) * mask, 0, 1)",
        "items": [],
    }
    chosen_reflections = rng.sample(reflection_paths, k=len(clean_paths))
    for index, (clean_path, reflection_path) in enumerate(zip(clean_paths, chosen_reflections)):
        name = f"self_{index}"
        metadata["items"].append(
            synthesize_pair(clean_path, reflection_path, name, out_dir, rng, args.max_long_edge)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote synthetic self-reflection data to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

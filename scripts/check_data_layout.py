#!/usr/bin/env python3
"""Check expected dataset counts for the DIP SIRR project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


EXPECTED = {
    "voc_train": ("VOCdevkit/VOC2012/PNGImages", 15287),
    "real_train_blended": ("real_train/blended", 89),
    "real_train_gt": ("real_train/transmission_layer", 89),
    "ceilnet_table2_blended": ("testdata_CEILNET_table2/blended", 100),
    "ceilnet_table2_gt": ("testdata_CEILNET_table2/transmission_layer", 100),
    "real20_blended": ("real20/blended", 20),
    "real20_gt": ("real20/transmission_layer", 20),
    "objects_blended": ("objects/blended", 200),
    "objects_gt": ("objects/transmission_layer", 200),
    "postcard_blended": ("postcard/blended", 179),
    "postcard_gt": ("postcard/transmission_layer", 179),
    "wild_blended": ("wild/blended", 101),
    "wild_gt": ("wild/transmission_layer", 101),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed dataset layout.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "repos/ERRNet/datasets/processed_data",
    )
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> int:
    args = parse_args()
    rows = []
    ok = True
    for name, (rel, expected) in EXPECTED.items():
        path = args.root / rel
        count = count_images(path)
        status = "ok" if count == expected else "missing"
        ok = ok and status == "ok"
        rows.append({"name": name, "path": str(path), "count": count, "expected": expected, "status": status})

    print(json.dumps(rows, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

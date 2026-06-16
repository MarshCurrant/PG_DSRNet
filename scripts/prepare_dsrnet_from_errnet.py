#!/usr/bin/env python3
"""Build the DSRNet base_dir layout from ERRNet processed_data."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DSRNet data layout.")
    parser.add_argument(
        "--errnet_processed",
        type=Path,
        default=ROOT / "repos/ERRNet/datasets/processed_data",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/processed/dsrnet_base",
    )
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    parser.add_argument("--force", action="store_true", help="Replace existing output layout.")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy:
        shutil.copytree(src, dst)
    else:
        dst.symlink_to(src.resolve(), target_is_directory=True)


def main() -> int:
    args = parse_args()
    src = args.errnet_processed.resolve()
    out = args.out.resolve()
    if out.exists():
        if not args.force:
            raise SystemExit(f"Output exists; pass --force to replace: {out}")
        shutil.rmtree(out)

    mappings = {
        src / "VOCdevkit/VOC2012/PNGImages": out / "train/VOCdevkit/VOC2012/PNGImages",
        src / "real_train": out / "train/real",
        src / "testdata_CEILNET_table2": out / "test/CEILNet_table2",
        src / "real20": out / "test/real20_420",
        src / "objects": out / "test/SIR2/SolidObjectDataset",
        src / "postcard": out / "test/SIR2/PostcardDataset",
        src / "wild": out / "test/SIR2/WildSceneDataset",
    }
    for source, dest in mappings.items():
        link_or_copy(source, dest, copy=args.copy)
        print(f"{source} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

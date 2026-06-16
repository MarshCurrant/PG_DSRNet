#!/usr/bin/env python3
"""Run ERRNet or DSRNet inference through a common command line."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ENV_PYTHONS = {
    "errnet": Path("/home/nebula/.conda/envs/dip-errnet/bin/python"),
    "dsrnet": Path("/home/nebula/.conda/envs/dip-dsrnet/bin/python"),
}
IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified SIRR inference wrapper.")
    parser.add_argument("--method", required=True, choices=("errnet", "dsrnet"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--python", type=Path, default=None, help="Override Python executable.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nThreads", type=int, default=4)
    parser.add_argument("--max_long_edge", type=int, default=None)
    parser.add_argument("--inet", default="auto", help="DSRNet architecture: auto, dsrnet_s, dsrnet_l.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def resolve_checkpoint(method: str, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint.resolve()
    if method == "errnet":
        default = ROOT / "repos/ERRNet/checkpoints/errnet/errnet_060_00463920.pt"
    else:
        default = ROOT / "repos/DSRNet/weights/dsrnet_s_epoch14.pt"
    if default.exists():
        return default
    raise SystemExit(f"Missing --checkpoint and default checkpoint was not found: {default}")


def infer_dsrnet_inet(checkpoint: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    name = checkpoint.name.lower()
    if "dsrnet_s" in name:
        return "dsrnet_s"
    return "dsrnet_l"


def run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    printable = " ".join(str(x) for x in cmd)
    print(f"[cwd] {cwd}")
    print(f"[cmd] {printable}")
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("TORCH_HOME", str(ROOT / "weights/torch"))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def copy_latest_dsrnet_output(repo: Path, name: str, output_dir: Path) -> None:
    candidates = sorted(
        (repo / "checkpoints" / name).glob("*/test"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"Could not locate DSRNet output under {repo / 'checkpoints' / name}")
    src = candidates[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        dst = output_dir / child.name
        if child.is_dir():
            shutil.copytree(child, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(child, dst)
    print(f"[copy] {src} -> {output_dir}")


def resize_image_dir(input_dir: Path, output_dir: Path, max_long_edge: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(input_dir.iterdir()):
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
            continue
        with Image.open(src) as img:
            img = img.convert("RGB")
            width, height = img.size
            long_edge = max(width, height)
            if long_edge > max_long_edge:
                scale = max_long_edge / float(long_edge)
                width = max(1, int(round(width * scale)))
                height = max(1, int(round(height * scale)))
                img = img.resize((width, height), Image.BICUBIC)
            img.save(output_dir / src.name)


def main() -> int:
    args = parse_args()
    repo = ROOT / ("repos/ERRNet" if args.method == "errnet" else "repos/DSRNet")
    python = (args.python or ENV_PYTHONS[args.method]).resolve()
    checkpoint = resolve_checkpoint(args.method, args.checkpoint)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    name = args.name or args.method

    if args.method == "errnet":
        cmd = [
            str(python),
            "test_errnet.py",
            "--name",
            name,
            "--dataset",
            "custom",
            "--input_dir",
            str(input_dir),
            "--result_dir",
            str(output_dir),
            "--save_subdir",
            ".",
            "-r",
            "--icnn_path",
            str(checkpoint),
            "--hyper",
            "--nThreads",
            str(args.nThreads),
        ]
        if args.cpu:
            cmd.extend(["--gpu_ids", "-1"])
        if args.max_long_edge is not None:
            cmd.extend(["--max_long_edge", str(args.max_long_edge)])
        cmd.extend(args.extra_args)
        run(cmd, repo, args.dry_run)
    else:
        inet = infer_dsrnet_inet(checkpoint, args.inet)
        temp_input = None
        dsrnet_input_dir = input_dir
        if args.max_long_edge is not None:
            temp_input = tempfile.TemporaryDirectory(prefix="dip_dsrnet_input_")
            dsrnet_input_dir = Path(temp_input.name)
            resize_image_dir(input_dir, dsrnet_input_dir, args.max_long_edge)
        cmd = [
            str(python),
            "test_sirs.py",
            "--inet",
            inet,
            "--model",
            "dsrnet_model_sirs",
            "--dataset",
            "sirs_dataset",
            "--name",
            name,
            "--if_align",
            "--resume",
            "--weight_path",
            str(checkpoint),
            "--base_dir",
            str(dsrnet_input_dir),
            "--nThreads",
            str(args.nThreads),
        ]
        if args.cpu:
            cmd.extend(["--gpu_ids", "-1"])
        cmd.extend(args.extra_args)
        try:
            run(cmd, repo, args.dry_run)
            if not args.dry_run:
                copy_latest_dsrnet_output(repo, name, output_dir)
        finally:
            if temp_input is not None:
                temp_input.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

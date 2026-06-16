#!/usr/bin/env python3
"""Evaluate single-image reflection removal outputs.

The script is intentionally repo-agnostic. It can score ERRNet/DSRNet result
folders as long as each prediction can be matched to a ground-truth image by
either the image stem or the parent folder stem.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_METRICS = ("psnr", "ssim", "ncc", "lmse")
SKIP_STEMS = {
    "m_input",
    "input",
    "t_label",
    "r_label",
    "reflection",
    "reflection_layer",
    "blended",
}
PRED_SUFFIXES = (
    "_t",
    "_l",
    "_pred",
    "_output",
    "_transmission",
    "_transmission_layer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PSNR/SSIM/NCC/LMSE for SIRR predictions."
    )
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--gt_dir", required=True, type=Path)
    parser.add_argument("--out_csv", required=True, type=Path)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric names: psnr,ssim,ncc,lmse",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=None,
        help="Optional path for aggregate metrics.",
    )
    parser.add_argument(
        "--align",
        choices=("crop-min", "resize-pred", "resize-gt", "skip"),
        default="crop-min",
        help="How to handle prediction/GT size mismatches.",
    )
    parser.add_argument(
        "--fail_on_missing",
        action="store_true",
        help="Exit with an error if any GT image has no prediction.",
    )
    return parser.parse_args()


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def strip_known_suffix(stem: str) -> str:
    out = stem
    for suffix in PRED_SUFFIXES:
        if out.lower().endswith(suffix):
            return out[: -len(suffix)]
    return out


def normalize_key(path: Path, root: Path, is_gt: bool) -> list[str]:
    stem = path.stem
    keys = [stem, strip_known_suffix(stem)]

    rel_parent = path.parent.relative_to(root) if path.parent != root else Path(".")
    if rel_parent != Path("."):
        keys.append(path.parent.name)

    # CEILNet raw names sometimes carry explicit label/input suffixes.
    for key in list(keys):
        keys.append(key.replace("-label1", "").replace("-input", ""))
        keys.append(key.replace("-g-", "-m-"))

    if is_gt and stem == "t_label" and rel_parent != Path("."):
        keys.insert(0, path.parent.name)

    seen = set()
    normalized = []
    for key in keys:
        clean = key.strip()
        if clean and clean not in seen:
            seen.add(clean)
            normalized.append(clean)
    return normalized


def prediction_priority(path: Path) -> int:
    stem = path.stem.lower()
    name = path.name.lower()
    if stem in SKIP_STEMS:
        return 999
    if stem.endswith("_r") or stem.endswith("_rr") or "_r_" in stem:
        return 999
    if stem.endswith("_t") or name.endswith("_t.png"):
        return 0
    if stem.endswith("_l") or name.endswith("_l.png"):
        return 1
    if "errnet" in stem or "dsrnet" in stem:
        return 2
    return 10


def build_index(root: Path, is_gt: bool) -> dict[str, Path]:
    indexed: dict[str, tuple[int, Path]] = {}
    for path in iter_images(root):
        priority = 0 if is_gt else prediction_priority(path)
        if priority >= 999:
            continue
        for key in normalize_key(path, root, is_gt=is_gt):
            previous = indexed.get(key)
            if previous is None or priority < previous[0]:
                indexed[key] = (priority, path)
    return {key: path for key, (_, path) in indexed.items()}


def preferred_gt_root(gt_dir: Path) -> Path:
    transmission = gt_dir / "transmission_layer"
    return transmission if transmission.exists() else gt_dir


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float64)


def align_pair(pred: np.ndarray, gt: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, str]:
    if pred.shape == gt.shape:
        return pred, gt, "same"

    if mode == "skip":
        raise ValueError(f"size mismatch pred={pred.shape} gt={gt.shape}")

    if mode == "crop-min":
        h = min(pred.shape[0], gt.shape[0])
        w = min(pred.shape[1], gt.shape[1])
        return pred[:h, :w], gt[:h, :w], f"crop-min:{w}x{h}"

    if mode == "resize-pred":
        resized = Image.fromarray(np.clip(pred, 0, 255).astype(np.uint8)).resize(
            (gt.shape[1], gt.shape[0]), Image.BICUBIC
        )
        return np.asarray(resized, dtype=np.float64), gt, "resize-pred"

    resized_gt = Image.fromarray(np.clip(gt, 0, 255).astype(np.uint8)).resize(
        (pred.shape[1], pred.shape[0]), Image.BICUBIC
    )
    return pred, np.asarray(resized_gt, dtype=np.float64), "resize-gt"


def ncc(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_c = pred - np.mean(pred)
    gt_c = gt - np.mean(gt)
    denom = math.sqrt(float(np.sum(pred_c * pred_c) * np.sum(gt_c * gt_c)))
    if denom <= 1e-12:
        return 1.0 if np.allclose(pred, gt) else float("nan")
    return float(np.sum(pred_c * gt_c) / denom)


def ssq_error(correct: np.ndarray, estimate: np.ndarray) -> float:
    if np.sum(estimate**2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate**2)
    else:
        alpha = 0.0
    return float(np.sum((correct - alpha * estimate) ** 2))


def lmse(pred: np.ndarray, gt: np.ndarray, window_size: int = 20, shift: int = 10) -> float:
    h, w, c = gt.shape
    if h < window_size or w < window_size:
        denom = float(np.sum(gt**2))
        return ssq_error(gt, pred) / denom if denom > 1e-12 else float("nan")

    ssq = 0.0
    total = 0.0
    for ch in range(c):
        for y in range(0, h - window_size + 1, shift):
            for x in range(0, w - window_size + 1, shift):
                gt_win = gt[y : y + window_size, x : x + window_size, ch]
                pred_win = pred[y : y + window_size, x : x + window_size, ch]
                ssq += ssq_error(gt_win, pred_win)
                total += float(np.sum(gt_win**2))
    return ssq / total if total > 1e-12 else float("nan")


def compute_metrics(pred: np.ndarray, gt: np.ndarray, metric_names: list[str]) -> dict[str, float]:
    out = {}
    if "psnr" in metric_names:
        out["psnr"] = float(peak_signal_noise_ratio(gt, pred, data_range=255))
    if "ssim" in metric_names:
        out["ssim"] = float(structural_similarity(gt, pred, channel_axis=-1, data_range=255))
    if "ncc" in metric_names:
        out["ncc"] = ncc(pred, gt)
    if "lmse" in metric_names:
        out["lmse"] = lmse(pred, gt)
    return out


def mean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return float("nan")
    if any(math.isinf(v) for v in finite):
        return float("inf")
    return float(np.mean(finite))


def main() -> int:
    args = parse_args()
    metric_names = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    invalid = sorted(set(metric_names) - set(DEFAULT_METRICS))
    if invalid:
        raise SystemExit(f"Unsupported metrics: {', '.join(invalid)}")

    pred_dir = args.pred_dir.resolve()
    gt_root = preferred_gt_root(args.gt_dir.resolve())
    pred_index = build_index(pred_dir, is_gt=False)
    gt_index = build_index(gt_root, is_gt=True)

    rows = []
    missing = []
    skipped = []
    for key, gt_path in sorted(gt_index.items()):
        pred_path = pred_index.get(key)
        if pred_path is None:
            missing.append(key)
            continue
        try:
            pred, gt, align_note = align_pair(load_rgb(pred_path), load_rgb(gt_path), args.align)
        except ValueError as exc:
            skipped.append({"key": key, "reason": str(exc)})
            continue

        row = {
            "key": key,
            "pred_path": str(pred_path),
            "gt_path": str(gt_path),
            "width": pred.shape[1],
            "height": pred.shape[0],
            "align": align_note,
        }
        row.update(compute_metrics(pred, gt, metric_names))
        rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["key", "pred_path", "gt_path", "width", "height", "align", *metric_names]
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_root),
        "count": len(rows),
        "missing": missing,
        "skipped": skipped,
        "metrics": {metric: mean([float(row[metric]) for row in rows]) for metric in metric_names},
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    print(json.dumps(summary, indent=2, allow_nan=True))
    if args.fail_on_missing and missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

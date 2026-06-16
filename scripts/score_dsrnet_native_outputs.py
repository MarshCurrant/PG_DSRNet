#!/usr/bin/env python3
"""Score official-script outputs with the bandwise SIRR metric convention."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute bandwise SIRR metrics for official output folders."
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--out_csv", required=True, type=Path)
    parser.add_argument("--summary_csv", required=True, type=Path)
    parser.add_argument("--summary_json", required=True, type=Path)
    parser.add_argument(
        "--pred_glob",
        default="*_t.png",
        help="Prediction filename glob relative to run_dir. DSRNet eval_sirs.py uses '*_t.png'.",
    )
    parser.add_argument(
        "--align",
        choices=("exact", "crop-min", "resize-pred", "resize-gt"),
        default="exact",
        help="How to handle saved prediction/GT shape mismatches.",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float64)


def bandwise_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(
        np.mean(
            [
                peak_signal_noise_ratio(gt[..., ch], pred[..., ch], data_range=255)
                for ch in range(gt.shape[-1])
            ]
        )
    )


def bandwise_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(
        np.mean(
            [
                structural_similarity(gt[..., ch], pred[..., ch], data_range=255)
                for ch in range(gt.shape[-1])
            ]
        )
    )


def ncc(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_c = gt - np.mean(gt)
    pred_c = pred - np.mean(pred)
    denom = float(np.std(gt) * np.std(pred))
    if denom <= 1e-12:
        return 1.0 if np.allclose(gt, pred) else float("nan")
    return float(np.mean(gt_c * pred_c) / denom)


def ssq_error(correct: np.ndarray, estimate: np.ndarray) -> float:
    if np.sum(estimate**2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate**2)
    else:
        alpha = 0.0
    return float(np.sum((correct - alpha * estimate) ** 2))


def lmse(gt: np.ndarray, pred: np.ndarray, window_size: int = 20, shift: int = 10) -> float:
    h, w, channels = gt.shape
    if h < window_size or w < window_size:
        denom = float(np.sum(gt**2))
        return ssq_error(gt, pred) / denom if denom > 1e-12 else float("nan")

    ssq = 0.0
    total = 0.0
    for ch in range(channels):
        for y in range(0, h - window_size + 1, shift):
            for x in range(0, w - window_size + 1, shift):
                gt_win = gt[y : y + window_size, x : x + window_size, ch]
                pred_win = pred[y : y + window_size, x : x + window_size, ch]
                ssq += ssq_error(gt_win, pred_win)
                total += float(np.sum(gt_win**2))
    return ssq / total if total > 1e-12 else float("nan")


def infer_split(path: Path, run_dir: Path) -> str:
    try:
        rel = path.relative_to(run_dir)
    except ValueError:
        return "unknown"
    return rel.parts[0] if rel.parts else "unknown"


def align_pair(pred: np.ndarray, gt: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, str]:
    if pred.shape == gt.shape:
        return pred, gt, "same"
    if mode == "exact":
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")
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


def score_pair(pred_path: Path, align: str) -> dict[str, object]:
    gt_path = pred_path.with_name("t_label.png")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing sibling t_label.png for {pred_path}")

    pred = load_rgb(pred_path)
    gt = load_rgb(gt_path)
    pred, gt, align_note = align_pair(pred, gt, align)

    return {
        "file": pred_path.parent.name,
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
        "width": pred.shape[1],
        "height": pred.shape[0],
        "align": align_note,
        "psnr": bandwise_psnr(gt, pred),
        "ssim": bandwise_ssim(gt, pred),
        "ncc": ncc(gt, pred),
        "lmse": lmse(gt, pred),
    }


def mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def read_metrics_txt(split_dir: Path) -> dict[str, float] | None:
    metrics_path = split_dir / "metrics.txt"
    if not metrics_path.exists():
        return None

    psnr_values = []
    ssim_values = []
    for line in metrics_path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        psnr_values.append(float(parts[1]))
        ssim_values.append(float(parts[2]))

    if not psnr_values:
        return None

    return {
        "metrics_txt_psnr": mean(psnr_values),
        "metrics_txt_ssim": mean(ssim_values),
    }


def main() -> None:
    args = parse_args()
    pred_paths = sorted(args.run_dir.rglob(args.pred_glob))
    if not pred_paths:
        raise SystemExit(f"No predictions matched {args.pred_glob!r} under {args.run_dir}")
    rows = []
    for pred_path in pred_paths:
        row = score_pair(pred_path, args.align)
        row["split"] = infer_split(pred_path, args.run_dir)
        rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "file",
        "width",
        "height",
        "align",
        "psnr",
        "ssim",
        "ncc",
        "lmse",
        "pred_path",
        "gt_path",
    ]
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        summaries.append(
            {
                "split": split,
                "count": len(split_rows),
                "psnr": mean([float(row["psnr"]) for row in split_rows]),
                "ssim": mean([float(row["ssim"]) for row in split_rows]),
                "ncc": mean([float(row["ncc"]) for row in split_rows]),
                "lmse": mean([float(row["lmse"]) for row in split_rows]),
            }
        )
        metrics_txt = read_metrics_txt(args.run_dir / split)
        if metrics_txt is not None:
            summaries[-1].update(metrics_txt)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "count",
                "metrics_txt_psnr",
                "metrics_txt_ssim",
                "psnr",
                "ssim",
                "ncc",
                "lmse",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w") as f:
        json.dump(
            {
                "run_dir": str(args.run_dir.resolve()),
                "count": len(rows),
                "splits": summaries,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(json.dumps({"count": len(rows), "splits": summaries}, indent=2))


if __name__ == "__main__":
    main()

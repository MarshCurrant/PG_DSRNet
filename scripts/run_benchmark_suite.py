#!/usr/bin/env python3
"""Run official-checkpoint inference and unified metrics for SIRR datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_PYTHONS = {
    "errnet": Path(os.environ.get("ERRNET_PY", "python")),
    "dsrnet": Path(os.environ.get("DSRNET_PY", "python")),
}

DATASETS = {
    "ceilnet_table2": {
        "input": ROOT / "repos/ERRNet/datasets/processed_data/testdata_CEILNET_table2/blended",
        "gt": ROOT / "repos/ERRNet/datasets/processed_data/testdata_CEILNET_table2",
    },
    "real20": {
        "input": ROOT / "repos/ERRNet/datasets/processed_data/real20/blended",
        "gt": ROOT / "repos/ERRNet/datasets/processed_data/real20",
        "errnet_max_long_edge": 512,
        "dsrnet_max_long_edge": 512,
    },
    "objects": {
        "input": ROOT / "repos/ERRNet/datasets/processed_data/objects/blended",
        "gt": ROOT / "repos/ERRNet/datasets/processed_data/objects",
    },
    "postcard": {
        "input": ROOT / "repos/ERRNet/datasets/processed_data/postcard/blended",
        "gt": ROOT / "repos/ERRNet/datasets/processed_data/postcard",
    },
    "wild": {
        "input": ROOT / "repos/ERRNet/datasets/processed_data/wild/blended",
        "gt": ROOT / "repos/ERRNet/datasets/processed_data/wild",
    },
}


def parse_csv(value: str, choices: set[str]) -> list[str]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(items) - choices)
    if invalid:
        raise SystemExit(f"Unsupported values: {', '.join(invalid)}")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SIRR benchmark suite.")
    parser.add_argument("--methods", default="errnet,dsrnet", help="Comma-separated: errnet,dsrnet")
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset keys.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=ROOT / "outputs/benchmarks/official",
    )
    parser.add_argument("--nThreads", type=int, default=4)
    parser.add_argument("--metrics", default="psnr,ssim,ncc,lmse")
    parser.add_argument("--errnet_checkpoint", type=Path, default=None)
    parser.add_argument("--dsrnet_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--dsrnet_inet",
        default="auto",
        choices=("auto", "dsrnet_s", "dsrnet_l"),
        help="DSRNet architecture passed through to scripts/infer_method.py.",
    )
    parser.add_argument(
        "--align",
        default="auto",
        choices=("auto", "crop-min", "resize-pred", "resize-gt", "skip"),
    )
    parser.add_argument("--skip_infer", action="store_true", help="Only recompute metrics from existing predictions.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def checked_path(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing path: {path}")
    return path


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("[cmd]", " ".join(str(x) for x in cmd))
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("TORCH_HOME", str(ROOT / "weights/torch"))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def run_infer(
    method: str,
    dataset: str,
    spec: dict[str, object],
    pred_dir: Path,
    args: argparse.Namespace,
) -> None:
    nthreads = args.nThreads
    dry_run = args.dry_run
    if not dry_run:
        if pred_dir.exists():
            shutil.rmtree(pred_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ENV_PYTHONS[method]),
        str(ROOT / "scripts/infer_method.py"),
        "--method",
        method,
        "--input_dir",
        str(checked_path(spec["input"])),
        "--output_dir",
        str(pred_dir),
        "--name",
        f"{method}_{dataset}_official",
        "--nThreads",
        str(nthreads),
    ]
    if method == "errnet" and spec.get("errnet_max_long_edge") is not None:
        cmd.extend(["--max_long_edge", str(spec["errnet_max_long_edge"])])
    if method == "dsrnet" and spec.get("dsrnet_max_long_edge") is not None:
        cmd.extend(["--max_long_edge", str(spec["dsrnet_max_long_edge"])])
    if method == "errnet" and getattr(args, "errnet_checkpoint", None) is not None:
        cmd.extend(["--checkpoint", str(args.errnet_checkpoint)])
    if method == "dsrnet":
        if getattr(args, "dsrnet_checkpoint", None) is not None:
            cmd.extend(["--checkpoint", str(args.dsrnet_checkpoint)])
        if getattr(args, "dsrnet_inet", "auto") != "auto":
            cmd.extend(["--inet", args.dsrnet_inet])
    run_cmd(cmd, dry_run=dry_run)


def resolve_align(method: str, dataset: str, args: argparse.Namespace) -> str:
    if args.align != "auto":
        return args.align
    if method == "dsrnet" or (method == "errnet" and dataset == "real20"):
        return "resize-gt"
    return "crop-min"


def run_metrics(
    method: str,
    dataset: str,
    spec: dict[str, object],
    pred_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    out_csv = out_dir / "metrics.csv"
    out_json = out_dir / "metrics.json"
    cmd = [
        str(ENV_PYTHONS[method]),
        str(ROOT / "scripts/eval_sirr.py"),
        "--pred_dir",
        str(pred_dir),
        "--gt_dir",
        str(checked_path(spec["gt"])),
        "--out_csv",
        str(out_csv),
        "--summary_json",
        str(out_json),
        "--metrics",
        args.metrics,
        "--align",
        resolve_align(method, dataset, args),
        "--fail_on_missing",
    ]
    run_cmd(cmd, dry_run=args.dry_run)
    if args.dry_run:
        return {}
    return json.loads(out_json.read_text())


def write_summary(rows: list[dict[str, object]], output_root: Path) -> None:
    if not rows:
        return
    metrics = sorted({metric for row in rows for metric in row.get("metrics", {})})
    fieldnames = ["method", "dataset", "count", "missing", "skipped", *metrics]
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "method": row["method"],
                "dataset": row["dataset"],
                "count": row.get("count", 0),
                "missing": len(row.get("missing", [])),
                "skipped": len(row.get("skipped", [])),
            }
            out.update(row.get("metrics", {}))
            writer.writerow(out)
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    methods = parse_csv(args.methods, set(ENV_PYTHONS))
    datasets = parse_csv(args.datasets, set(DATASETS))
    rows = []

    for method in methods:
        for dataset in datasets:
            spec = DATASETS[dataset]
            out_dir = args.output_root / method / dataset
            pred_dir = out_dir / "predictions"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n== {method} / {dataset} ==")
            if not args.skip_infer:
                run_infer(method, dataset, spec, pred_dir, args)
            summary = run_metrics(method, dataset, spec, pred_dir, out_dir, args)
            if summary:
                summary["method"] = method
                summary["dataset"] = dataset
                rows.append(summary)

    write_summary(rows, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

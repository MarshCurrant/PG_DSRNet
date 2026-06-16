#!/usr/bin/env python3
"""Collect PG-DSRNet ablation benchmark summaries into one CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANTS = [
    ("DSRNet-L reproduced", ROOT / "outputs/benchmarks/self_trained_dsrnet_l/summary.csv"),
    ("PG-DSRNet-L freq", ROOT / "outputs/benchmarks/pg_dsrnet_l_freq/summary.csv"),
    ("PG-DSRNet-L prior", ROOT / "outputs/benchmarks/pg_dsrnet_l_prior/summary.csv"),
    ("PG-DSRNet-L freq+prior", ROOT / "outputs/benchmarks/pg_dsrnet_l_freq_prior/summary.csv"),
]


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"Invalid --variant {value!r}; use label=summary.csv")
    label, path = value.split("=", 1)
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant as label=summary.csv. Defaults to the standard PG-DSRNet set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/benchmarks/pg_dsrnet_summary.csv",
    )
    return parser.parse_args()


def read_rows(label: str, path: Path) -> list[dict[str, str]]:
    path = path.resolve()
    if not path.is_file():
        print(f"[skip] missing {label}: {path}")
        return []
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "variant": label,
                    "dataset": row.get("dataset", ""),
                    "count": row.get("count", ""),
                    "psnr": row.get("psnr", ""),
                    "ssim": row.get("ssim", ""),
                    "ncc": row.get("ncc", ""),
                    "lmse": row.get("lmse", ""),
                    "source_csv": str(path),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    variants = [parse_variant(item) for item in args.variant] if args.variant else DEFAULT_VARIANTS
    rows = []
    for label, path in variants:
        rows.extend(read_rows(label, path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "dataset", "count", "psnr", "ssim", "ncc", "lmse", "source_csv"]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

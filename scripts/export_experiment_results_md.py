#!/usr/bin/env python3
"""Export benchmark summary CSV files into a single Markdown results table."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SOURCE_FILES = {
    "official": Path("outputs/benchmarks/official_protocol_summary.csv"),
    "unified": Path("outputs/benchmarks/unified_protocol_summary.csv"),
    "self_trained": Path("outputs/benchmarks/self_trained_summary.csv"),
    "pg": Path("outputs/benchmarks/pg_dsrnet_summary.csv"),
    "custom": Path("outputs/benchmarks/custom_synth_summary.csv"),
}

BENCHMARK_DATASET_ORDER = ["CEILNet table2", "real20", "Objects", "Postcard", "Wild"]
DATASET_ORDER = [*BENCHMARK_DATASET_ORDER, "Self-synth"]
PG_VARIANT_ORDER = [
    "DSRNet-L reproduced",
    "PG-DSRNet-L freq",
    "PG-DSRNet-L prior",
    "PG-DSRNet-L freq+prior",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize_dataset(name: str) -> str:
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "ceilnet_table2": "CEILNet table2",
        "ceilnet": "CEILNet table2",
        "real20": "real20",
        "real20_420": "real20_420",
        "objects": "Objects",
        "object": "Objects",
        "postcard": "Postcard",
        "wild": "Wild",
        "self_synth": "Self-synth",
        "self": "Self-synth",
    }
    return mapping.get(key, (name or "").strip())


def dataset_rank(name: str) -> tuple[int, str]:
    normalized = normalize_dataset(name)
    if normalized == "real20_420":
        return (DATASET_ORDER.index("real20"), normalized)
    if normalized in DATASET_ORDER:
        return (DATASET_ORDER.index(normalized), normalized)
    return (len(DATASET_ORDER), normalized)


def md_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", r"\|")


def relpath_text(value: str, root: Path) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    try:
        path = Path(text)
        if path.is_absolute():
            return str(path.relative_to(root))
    except (ValueError, OSError):
        pass
    prefix = str(root.resolve()) + "/"
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def metric(row: dict[str, str], name: str) -> float:
    return float(row[name])


def fmt_float(value: str | float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def fmt_delta(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}"


def fmt_count(value: str | int) -> str:
    return str(int(float(value)))


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_escape(cell) for cell in row) + " |")
    return lines


def sort_by_dataset(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: dataset_rank(row.get("dataset", "")))


def source_section(root: Path) -> list[str]:
    rows = [
        [
            "Official native protocol",
            str(SOURCE_FILES["official"]),
            "ERRNet native test script and DSRNet native eval protocol; not mixed with unified evaluator.",
        ],
        [
            "Unified official weights",
            str(SOURCE_FILES["unified"]),
            "Official ERRNet and DSRNet-L epoch18 evaluated by the shared wrapper/evaluator.",
        ],
        [
            "Self-trained models",
            str(SOURCE_FILES["self_trained"]),
            "ERRNet self-trained/fine-tuned and DSRNet-L Setting I self-trained checkpoints.",
        ],
        [
            "PG-DSRNet ablation",
            str(SOURCE_FILES["pg"]),
            "Self-trained DSRNet-L baseline plus frequency, prior, and frequency+prior fine-tuning runs.",
        ],
        [
            "Self-collected synthetic reflection",
            str(SOURCE_FILES["custom"]),
            "Five self-collected clean photos synthesized with reflection layers and evaluated with full-reference metrics.",
        ],
        [
            "Per-model summaries",
            "outputs/benchmarks/*/summary.csv and summary.json",
            "Detailed outputs used by the four aggregate CSV files above.",
        ],
    ]
    return markdown_table(["Group", "File", "Meaning"], rows)


def official_table(rows: list[dict[str, str]], root: Path) -> list[str]:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["method"],
                relpath_text(row["checkpoint"], root),
                row["protocol"],
                normalize_dataset(row["dataset"]),
                fmt_count(row["count"]),
                fmt_float(row["psnr"], 3),
                fmt_float(row["ssim"], 4),
                fmt_float(row["ncc"], 4),
                fmt_float(row["lmse"], 5),
                row.get("metric_source", ""),
                row.get("notes", ""),
            ]
        )
    return markdown_table(
        [
            "Method",
            "Checkpoint",
            "Protocol",
            "Dataset",
            "Count",
            "PSNR",
            "SSIM",
            "NCC",
            "LMSE",
            "Source",
            "Notes",
        ],
        table_rows,
    )


def unified_table(rows: list[dict[str, str]], root: Path) -> list[str]:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["method"],
                relpath_text(row["checkpoint"], root),
                row["protocol"],
                normalize_dataset(row["dataset"]),
                fmt_count(row["count"]),
                fmt_float(row["psnr"], 3),
                fmt_float(row["ssim"], 4),
                fmt_float(row["ncc"], 4),
                fmt_float(row["lmse"], 5),
                relpath_text(row.get("source", ""), root),
            ]
        )
    return markdown_table(
        [
            "Method",
            "Checkpoint",
            "Protocol",
            "Dataset",
            "Count",
            "PSNR",
            "SSIM",
            "NCC",
            "LMSE",
            "Source",
        ],
        table_rows,
    )


def self_trained_table(rows: list[dict[str, str]], root: Path) -> list[str]:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["method"],
                relpath_text(row["checkpoint"], root),
                row["training_protocol"],
                normalize_dataset(row["dataset"]),
                fmt_count(row["count"]),
                fmt_float(row["psnr"], 3),
                fmt_float(row["ssim"], 4),
                fmt_float(row["ncc"], 4),
                fmt_float(row["lmse"], 5),
                row.get("notes", ""),
            ]
        )
    return markdown_table(
        [
            "Method",
            "Checkpoint",
            "Training protocol",
            "Dataset",
            "Count",
            "PSNR",
            "SSIM",
            "NCC",
            "LMSE",
            "Notes",
        ],
        table_rows,
    )


def pg_table(rows: list[dict[str, str]], root: Path) -> list[str]:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["variant"],
                normalize_dataset(row["dataset"]),
                fmt_count(row["count"]),
                fmt_float(row["psnr"], 3),
                fmt_float(row["ssim"], 4),
                fmt_float(row["ncc"], 4),
                fmt_float(row["lmse"], 5),
                relpath_text(row.get("source_csv", ""), root),
            ]
        )
    return markdown_table(
        ["Variant", "Dataset", "Count", "PSNR", "SSIM", "NCC", "LMSE", "Source"],
        table_rows,
    )


def custom_table(rows: list[dict[str, str]], root: Path) -> list[str]:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["method"],
                normalize_dataset(row["dataset"]),
                fmt_count(row["count"]),
                fmt_float(row["psnr"], 3),
                fmt_float(row["ssim"], 4),
                fmt_float(row["ncc"], 4),
                fmt_float(row["lmse"], 5),
                relpath_text(row.get("source", ""), root),
            ]
        )
    return markdown_table(
        ["Method", "Dataset", "Count", "PSNR", "SSIM", "NCC", "LMSE", "Source"],
        table_rows,
    )


def order_pg_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    variant_rank = {name: i for i, name in enumerate(PG_VARIANT_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            variant_rank.get(row.get("variant", ""), len(PG_VARIANT_ORDER)),
            dataset_rank(row.get("dataset", "")),
        ),
    )


def delta_rows(pg_rows: list[dict[str, str]]) -> list[list[str]]:
    by_variant_dataset: dict[tuple[str, str], dict[str, str]] = {}
    for row in pg_rows:
        by_variant_dataset[(row["variant"], normalize_dataset(row["dataset"]))] = row

    table_rows: list[list[str]] = []
    for variant in PG_VARIANT_ORDER[1:]:
        for dataset in BENCHMARK_DATASET_ORDER:
            baseline = by_variant_dataset.get(("DSRNet-L reproduced", dataset))
            current = by_variant_dataset.get((variant, dataset))
            if baseline is None or current is None:
                raise KeyError(f"Missing PG delta pair for {variant} / {dataset}")
            table_rows.append(
                [
                    variant,
                    dataset,
                    fmt_delta(metric(current, "psnr") - metric(baseline, "psnr"), 3),
                    fmt_delta(metric(current, "ssim") - metric(baseline, "ssim"), 4),
                    fmt_delta(metric(current, "ncc") - metric(baseline, "ncc"), 4),
                    fmt_delta(metric(current, "lmse") - metric(baseline, "lmse"), 5),
                ]
            )
    return table_rows


def count_by_method(rows: list[dict[str, str]], method_key: str) -> Counter[str]:
    return Counter(row[method_key] for row in rows)


def validate_inputs(
    official: list[dict[str, str]],
    unified: list[dict[str, str]],
    self_trained: list[dict[str, str]],
    pg: list[dict[str, str]],
    custom: list[dict[str, str]],
) -> None:
    official_counts = count_by_method(official, "method")
    if official_counts["ERRNet"] != 5 or official_counts["DSRNet-L"] != 4:
        raise ValueError(f"Unexpected official protocol row counts: {official_counts}")

    unified_counts = count_by_method(unified, "method")
    if unified_counts["ERRNet"] != 5 or unified_counts["DSRNet-L"] != 5:
        raise ValueError(f"Unexpected unified official row counts: {unified_counts}")

    self_counts = count_by_method(self_trained, "method")
    if self_counts["ERRNet"] != 5 or self_counts["DSRNet-L"] != 5:
        raise ValueError(f"Unexpected self-trained row counts: {self_counts}")

    pg_counts = count_by_method(pg, "variant")
    expected_pg = {
        "DSRNet-L reproduced": 5,
        "PG-DSRNet-L freq": 5,
        "PG-DSRNet-L prior": 5,
        "PG-DSRNet-L freq+prior": 5,
    }
    if dict(pg_counts) != expected_pg:
        raise ValueError(f"Unexpected PG row counts: {pg_counts}")

    custom_counts = count_by_method(custom, "method")
    expected_custom = {
        "ERRNet self-trained": 1,
        "DSRNet-L self-trained": 1,
        "PG-DSRNet-L freq+prior": 1,
    }
    if dict(custom_counts) != expected_custom:
        raise ValueError(f"Unexpected custom self-synth row counts: {custom_counts}")


def build_markdown(root: Path) -> str:
    official = sort_by_dataset(read_csv(root / SOURCE_FILES["official"]))
    unified = sort_by_dataset(read_csv(root / SOURCE_FILES["unified"]))
    self_trained = sort_by_dataset(read_csv(root / SOURCE_FILES["self_trained"]))
    pg = order_pg_rows(read_csv(root / SOURCE_FILES["pg"]))
    custom = sort_by_dataset(read_csv(root / SOURCE_FILES["custom"]))

    validate_inputs(official, unified, self_trained, pg, custom)
    deltas = delta_rows(pg)

    lines: list[str] = []
    lines.extend(
        [
            "# Experiment Results",
            "",
            "This document is generated from the current benchmark summary CSV files. It keeps native official protocols, unified evaluator results, self-trained checkpoints, and PG-DSRNet ablations in separate tables so metric protocols are not mixed.",
            "",
            "## 1. Result Sources",
            "",
        ]
    )
    lines.extend(source_section(root))
    lines.extend(
        [
            "",
            "## 2. Official Protocol Results",
            "",
            "ERRNet rows come from the native ERRNet evaluation script. DSRNet rows come from the native DSRNet protocol with official all-in-one data; `real20_420` is intentionally kept distinct from unified `real20`.",
            "",
        ]
    )
    lines.extend(official_table(official, root))
    lines.extend(
        [
            "",
            "## 3. Unified Protocol: Official Weights",
            "",
            "Both methods are evaluated by the same wrapper/evaluator, using the same dataset naming and metric implementation.",
            "",
        ]
    )
    lines.extend(unified_table(unified, root))
    lines.extend(
        [
            "",
            "## 4. Self-Trained Results",
            "",
            "These rows use our trained checkpoints rather than only the released weights.",
            "",
        ]
    )
    lines.extend(self_trained_table(self_trained, root))
    lines.extend(
        [
            "",
            "## 5. PG-DSRNet Ablation",
            "",
            "`DSRNet-L reproduced` is the self-trained DSRNet-L baseline. `freq` and `prior` were fine-tuned to epoch55; `freq+prior` was fine-tuned to epoch60, so this is not a strict same-epoch ablation.",
            "",
        ]
    )
    lines.extend(pg_table(pg, root))
    lines.extend(
        [
            "",
            "## 6. Delta vs Self-Trained DSRNet-L",
            "",
            "Positive deltas mean the PG variant is higher than `DSRNet-L reproduced`; for LMSE, lower is better, so negative deltas are favorable.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Variant", "Dataset", "\u0394PSNR", "\u0394SSIM", "\u0394NCC", "\u0394LMSE"],
            deltas,
        )
    )
    lines.extend(
        [
            "",
            "## 7. Self-Collected Synthetic Reflection Results",
            "",
            "The five self-collected photos in `self/` are treated as clean transmission images. Reflection inputs are synthesized with a fixed seed and evaluated with full-reference metrics.",
            "",
        ]
    )
    lines.extend(custom_table(custom, root))
    lines.extend(
        [
            "",
            "## 8. Key Findings",
            "",
            "- PG-DSRNet improves CEILNet synthetic most clearly. The `freq+prior` run improves by +0.866 PSNR over the self-trained DSRNet-L baseline on CEILNet table2.",
            "- On real20 and most SIR2 subsets, the current PG variants degrade relative to self-trained DSRNet-L. The `freq+prior` real20 delta is -0.853 PSNR.",
            "- On the self-collected synthetic set, ERRNet is much stronger than DSRNet-style models under the current synthesis protocol; PG-DSRNet is only slightly above DSRNet-L. This should be discussed as a self-synthesis/domain-gap result, not as broad real-world superiority.",
            "- The official DSRNet-L epoch18 checkpoint remains stronger than the self-trained DSRNet-L baseline on several real/SIR2 rows under the unified evaluator, while the self-trained DSRNet-L baseline is strongest on CEILNet among the baseline rows.",
            "- Because `freq+prior` trained to epoch60 while `freq` and `prior` trained to epoch55, the PG table should be reported as completed ablation evidence, not definitive same-budget causal proof.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/experiment_results.md"),
        help="Markdown output path, relative to --root unless absolute.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    markdown = build_markdown(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

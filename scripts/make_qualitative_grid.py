#!/usr/bin/env python3
"""Build qualitative comparison grids from benchmark predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRS = {
    "ceilnet_table2": "testdata_CEILNET_table2",
    "real20": "real20",
    "objects": "objects",
    "postcard": "postcard",
    "wild": "wild",
}
DISPLAY_NAMES = {
    "ceilnet_table2": "CEILNet table2",
    "real20": "real20",
    "objects": "SIR2 Objects",
    "postcard": "SIR2 Postcard",
    "wild": "SIR2 Wild",
}
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
BASE_COLS = ("Input", "GT", "ERRNet self", "DSRNet self")


def natural_key(value: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def parse_samples(values: list[str]) -> dict[str, str]:
    samples: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --sample {value!r}; use dataset=sample")
        dataset, sample = value.split("=", 1)
        if dataset not in DATASET_DIRS:
            raise SystemExit(f"Unknown dataset in --sample: {dataset}")
        samples[dataset] = sample
    return samples


def image_resample() -> int:
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def find_image(directory: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    matches = sorted(directory.glob(f"{stem}.*"), key=lambda path: natural_key(path.name))
    for match in matches:
        if match.suffix.lower() in IMAGE_EXTS and match.is_file():
            return match
    raise FileNotFoundError(f"No image found for {stem!r} in {directory}")


def pred_paths(root: Path, method: str, dataset: str, sample: str) -> Path:
    if method == "errnet":
        filename = f"errnet_{dataset}_official.png"
    elif method == "dsrnet":
        filename = f"dsrnet_{dataset}_official_l.png"
    elif method == "pg":
        filename = f"dsrnet_{dataset}_official_l.png"
    else:
        raise ValueError(method)
    pred_method = "dsrnet" if method == "pg" else method
    path = root / pred_method / dataset / "predictions" / sample / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def choose_sample(
    dataset: str,
    errnet_root: Path,
    dsrnet_root: Path,
    processed_root: Path,
    pg_root: Path | None = None,
) -> str:
    err_dir = errnet_root / "errnet" / dataset / "predictions"
    dsr_dir = dsrnet_root / "dsrnet" / dataset / "predictions"
    if not err_dir.is_dir() or not dsr_dir.is_dir():
        raise FileNotFoundError(f"Missing prediction directory for {dataset}")

    common_names = {path.name for path in err_dir.iterdir() if path.is_dir()}
    common_names &= {path.name for path in dsr_dir.iterdir() if path.is_dir()}
    if pg_root is not None:
        pg_dir = pg_root / "dsrnet" / dataset / "predictions"
        if not pg_dir.is_dir():
            raise FileNotFoundError(f"Missing PG prediction directory for {dataset}: {pg_dir}")
        common_names &= {path.name for path in pg_dir.iterdir() if path.is_dir()}
    common = sorted(common_names, key=natural_key)
    data_dir = processed_root / DATASET_DIRS[dataset]
    for sample in common:
        try:
            find_image(data_dir / "blended", sample)
            find_image(data_dir / "transmission_layer", sample)
            pred_paths(errnet_root, "errnet", dataset, sample)
            pred_paths(dsrnet_root, "dsrnet", dataset, sample)
            if pg_root is not None:
                pred_paths(pg_root, "pg", dataset, sample)
        except FileNotFoundError:
            continue
        return sample
    raise FileNotFoundError(f"No complete qualitative sample found for {dataset}")


def load_tile(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail(size, image_resample())
        tile = Image.new("RGB", size, (245, 245, 245))
        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2
        tile.paste(image, (x, y))
        return tile


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    draw.text(xy, text, fill=fill, font=font)


def sample_paths(
    dataset: str,
    sample: str,
    errnet_root: Path,
    dsrnet_root: Path,
    processed_root: Path,
    pg_root: Path | None = None,
) -> dict[str, Path]:
    data_dir = processed_root / DATASET_DIRS[dataset]
    paths = {
        "Input": find_image(data_dir / "blended", sample),
        "GT": find_image(data_dir / "transmission_layer", sample),
        "ERRNet self": pred_paths(errnet_root, "errnet", dataset, sample),
        "DSRNet self": pred_paths(dsrnet_root, "dsrnet", dataset, sample),
    }
    if pg_root is not None:
        paths["PG-DSRNet"] = pred_paths(pg_root, "pg", dataset, sample)
    return paths


def build_grid(
    rows: list[tuple[str, str, dict[str, Path]]],
    out_path: Path,
    thumb_size: tuple[int, int],
    row_label_width: int,
    header_height: int,
    gutter: int,
    cols: tuple[str, ...],
) -> None:
    cell_w, cell_h = thumb_size
    width = row_label_width + len(cols) * cell_w + (len(cols) + 1) * gutter
    height = header_height + len(rows) * (cell_h + gutter) + gutter
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for col_index, col in enumerate(cols):
        x = row_label_width + gutter + col_index * (cell_w + gutter)
        draw_text(draw, (x, 12), col, (35, 35, 35))

    y = header_height
    for dataset, sample, paths in rows:
        label = f"{DISPLAY_NAMES[dataset]}\n{sample}"
        draw_text(draw, (16, y + 12), label, (35, 35, 35))
        for col_index, col in enumerate(cols):
            x = row_label_width + gutter + col_index * (cell_w + gutter)
            tile = load_tile(paths[col], thumb_size)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(215, 215, 215))
        y += cell_h + gutter

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errnet-root", type=Path, default=ROOT / "outputs/benchmarks/self_trained_errnet")
    parser.add_argument("--dsrnet-root", type=Path, default=ROOT / "outputs/benchmarks/self_trained_dsrnet_l")
    parser.add_argument("--pg-root", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=ROOT / "repos/ERRNet/datasets/processed_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures/qualitative")
    parser.add_argument("--datasets", default="ceilnet_table2,real20,objects,postcard,wild")
    parser.add_argument("--sample", action="append", default=[], help="Override sample as dataset=sample")
    parser.add_argument("--thumb-width", type=int, default=256)
    parser.add_argument("--thumb-height", type=int, default=192)
    args = parser.parse_args()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(DATASET_DIRS))
    if unknown:
        raise SystemExit(f"Unknown datasets: {', '.join(unknown)}")

    overrides = parse_samples(args.sample)
    cols = BASE_COLS + (("PG-DSRNet",) if args.pg_root is not None else ())
    rows: list[tuple[str, str, dict[str, Path]]] = []
    manifest = []
    for dataset in datasets:
        sample = overrides.get(dataset) or choose_sample(
            dataset, args.errnet_root, args.dsrnet_root, args.processed_root, args.pg_root
        )
        paths = sample_paths(
            dataset, sample, args.errnet_root, args.dsrnet_root, args.processed_root, args.pg_root
        )
        rows.append((dataset, sample, paths))
        manifest.append(
            {
                "dataset": dataset,
                "sample": sample,
                "paths": {name: str(path) for name, path in paths.items()},
            }
        )

        individual = args.output_dir / f"{dataset}_{sample}.png"
        build_grid(
            [(dataset, sample, paths)],
            individual,
            (args.thumb_width, args.thumb_height),
            row_label_width=160,
            header_height=36,
            gutter=10,
            cols=cols,
        )

    summary_name = "pg_dsrnet_summary_grid.png" if args.pg_root is not None else "self_trained_summary_grid.png"
    summary = args.output_dir / summary_name
    build_grid(
        rows,
        summary,
        (args.thumb_width, args.thumb_height),
        row_label_width=170,
        header_height=36,
        gutter=10,
        cols=cols,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Qualitative Figures\n\n"
        "Generated from self-trained ERRNet and self-trained DSRNet-L benchmark predictions.\n\n"
        f"- Summary grid: `{summary.name}`\n"
        f"- Columns: {', '.join(cols)}.\n"
        "- Self-collected synthetic figures are generated separately by `scripts/make_paper_figures.py`.\n",
        encoding="utf-8",
    )

    print(f"Wrote {summary}")
    for row in manifest:
        print(f"{row['dataset']}: {row['sample']}")


if __name__ == "__main__":
    main()

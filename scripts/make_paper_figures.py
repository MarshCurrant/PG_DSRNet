#!/usr/bin/env python3
"""Generate paper-ready figures from benchmark summaries and predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DATASET_ORDER = ["CEILNet table2", "real20", "Objects", "Postcard", "Wild", "Self-synth"]
SHORT_DATASET = {
    "CEILNet table2": "CEILNet",
    "real20": "real20",
    "Objects": "Objects",
    "Postcard": "Postcard",
    "Wild": "Wild",
    "Self-synth": "Self",
}
METHODS = ["ERRNet", "DSRNet-L", "PG-DSRNet"]
COLORS = {
    "ERRNet": (57, 115, 178),
    "DSRNet-L": (82, 160, 91),
    "PG-DSRNet": (213, 112, 66),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/cvpr2026_pg_dsrnet/figures")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize_dataset(name: str) -> str:
    mapping = {
        "ceilnet_table2": "CEILNet table2",
        "CEILNet table2": "CEILNet table2",
        "real20": "real20",
        "objects": "Objects",
        "Objects": "Objects",
        "postcard": "Postcard",
        "Postcard": "Postcard",
        "wild": "Wild",
        "Wild": "Wild",
        "Self-synth": "Self-synth",
        "self_synth": "Self-synth",
    }
    return mapping.get(name, name)


def font(size: int = 14) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def text_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill=(30, 30, 30), size=14) -> None:
    fnt = font(size)
    lines = text.split("\n")
    heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total_h = sum(heights) + 4 * (len(lines) - 1)
    y = box[1] + (box[3] - box[1] - total_h) // 2
    for line, h in zip(lines, heights):
        bbox = draw.textbbox((0, 0), line, font=fnt)
        x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, fill=fill, font=fnt)
        y += h + 4


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill=(70, 70, 70)) -> None:
    draw.line((start, end), fill=fill, width=3)
    x2, y2 = end
    draw.polygon([(x2, y2), (x2 - 10, y2 - 6), (x2 - 10, y2 + 6)], fill=fill)


def draw_pipeline(out_path: Path) -> None:
    canvas = Image.new("RGB", (1500, 620), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 28), "PG-DSRNet training-time prior losses", fill=(20, 20, 20), font=font(24))

    boxes = [
        ((60, 180, 260, 300), "Reflection image\nI"),
        ((340, 150, 590, 330), "DSRNet-L\nshared backbone"),
        ((690, 90, 930, 210), "Transmission\nT_hat"),
        ((690, 250, 930, 370), "Reflection\nR_hat"),
        ((1040, 70, 1350, 230), "Frequency loss\nL_freq = |B^K(T_hat)-B^K(T)|\n+ |H(T_hat)-H(T)|"),
        ((1040, 270, 1350, 450), "Prior-weighted loss\nw = clamp(1 + prior * H(I), 1, 2)\nL_prior = mean(w * residuals)"),
    ]
    for box, label in boxes:
        draw.rounded_rectangle(box, radius=8, fill=(245, 247, 250), outline=(105, 118, 135), width=2)
        text_center(draw, box, label, size=17)

    arrow(draw, (260, 240), (340, 240))
    arrow(draw, (590, 210), (690, 150))
    arrow(draw, (590, 270), (690, 310))
    arrow(draw, (930, 150), (1040, 145))
    arrow(draw, (930, 310), (1040, 360))
    draw.text(
        (360, 430),
        "Inference is unchanged: PG-DSRNet keeps the DSRNet-L architecture and only adds losses during fine-tuning.",
        fill=(65, 65, 65),
        font=font(18),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def collect_quant_data() -> dict[str, dict[str, dict[str, float]]]:
    self_rows = read_csv(ROOT / "outputs/benchmarks/self_trained_summary.csv")
    pg_rows = read_csv(ROOT / "outputs/benchmarks/pg_dsrnet_summary.csv")
    custom_rows = read_csv(ROOT / "outputs/benchmarks/custom_synth_summary.csv")
    data: dict[str, dict[str, dict[str, float]]] = {m: {} for m in METHODS}

    for row in self_rows:
        dataset = normalize_dataset(row["dataset"])
        method = "ERRNet" if row["method"] == "ERRNet" else "DSRNet-L"
        data[method][dataset] = {"psnr": float(row["psnr"]), "ssim": float(row["ssim"])}
    for row in pg_rows:
        if row["variant"] == "PG-DSRNet-L freq+prior":
            dataset = normalize_dataset(row["dataset"])
            data["PG-DSRNet"][dataset] = {"psnr": float(row["psnr"]), "ssim": float(row["ssim"])}
    for row in custom_rows:
        method_map = {
            "ERRNet self-trained": "ERRNet",
            "DSRNet-L self-trained": "DSRNet-L",
            "PG-DSRNet-L freq+prior": "PG-DSRNet",
        }
        data[method_map[row["method"]]]["Self-synth"] = {
            "psnr": float(row["psnr"]),
            "ssim": float(row["ssim"]),
        }
    return data


def draw_grouped_bars(
    out_path: Path,
    title: str,
    metric: str,
    ymax: float,
    data: dict[str, dict[str, dict[str, float]]],
) -> Image.Image:
    width, height = 1500, 440
    margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 90
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 24), title, fill=(20, 20, 20), font=font(23))

    for tick in range(0, 6):
        value = ymax * tick / 5
        y = margin_t + plot_h - int(plot_h * value / ymax)
        draw.line((margin_l, y, width - margin_r, y), fill=(225, 225, 225), width=1)
        draw.text((20, y - 8), f"{value:.1f}" if metric == "psnr" else f"{value:.2f}", fill=(80, 80, 80), font=font(13))

    group_w = plot_w / len(DATASET_ORDER)
    bar_w = int(group_w * 0.18)
    for i, dataset in enumerate(DATASET_ORDER):
        cx = margin_l + i * group_w + group_w / 2
        draw.text((int(cx - 34), height - margin_b + 25), SHORT_DATASET[dataset], fill=(50, 50, 50), font=font(14))
        for j, method in enumerate(METHODS):
            value = data[method][dataset][metric]
            x0 = int(cx + (j - 1) * (bar_w + 8) - bar_w / 2)
            x1 = x0 + bar_w
            y1 = margin_t + plot_h
            y0 = y1 - int(plot_h * min(value, ymax) / ymax)
            draw.rectangle((x0, y0, x1, y1), fill=COLORS[method])
            draw.text((x0 - 4, y0 - 18), f"{value:.1f}" if metric == "psnr" else f"{value:.2f}", fill=(45, 45, 45), font=font(11))

    legend_x = width - 470
    for idx, method in enumerate(METHODS):
        x = legend_x + idx * 150
        draw.rectangle((x, 30, x + 18, 48), fill=COLORS[method])
        draw.text((x + 26, 30), method, fill=(35, 35, 35), font=font(14))
    return canvas


def draw_quant(out_path: Path) -> None:
    data = collect_quant_data()
    psnr = draw_grouped_bars(out_path, "PSNR comparison", "psnr", 32.0, data)
    ssim = draw_grouped_bars(out_path, "SSIM comparison", "ssim", 1.0, data)
    canvas = Image.new("RGB", (1500, 880), (255, 255, 255))
    canvas.paste(psnr, (0, 0))
    canvas.paste(ssim, (0, 440))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def draw_delta(out_path: Path) -> None:
    rows = read_csv(ROOT / "outputs/benchmarks/pg_dsrnet_summary.csv")
    baseline = {}
    pg = {}
    for row in rows:
        dataset = normalize_dataset(row["dataset"])
        if row["variant"] == "DSRNet-L reproduced":
            baseline[dataset] = float(row["psnr"])
        elif row["variant"] == "PG-DSRNet-L freq+prior":
            pg[dataset] = float(row["psnr"])

    canvas = Image.new("RGB", (1200, 460), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 24), "PG-DSRNet PSNR delta vs self-trained DSRNet-L", fill=(20, 20, 20), font=font(22))
    margin_l, margin_t, plot_w, plot_h = 90, 80, 1050, 290
    zero_y = margin_t + plot_h // 2
    draw.line((margin_l, zero_y, margin_l + plot_w, zero_y), fill=(90, 90, 90), width=2)
    draw.text((25, zero_y - 8), "0.0", fill=(60, 60, 60), font=font(13))
    scale = plot_h / 2 / 1.5
    group_w = plot_w / 5
    for i, dataset in enumerate(DATASET_ORDER[:5]):
        delta = pg[dataset] - baseline[dataset]
        cx = margin_l + i * group_w + group_w / 2
        y = int(zero_y - delta * scale)
        color = (59, 142, 95) if delta >= 0 else (190, 83, 83)
        draw.rectangle((int(cx - 35), min(y, zero_y), int(cx + 35), max(y, zero_y)), fill=color)
        draw.text((int(cx - 34), 392), SHORT_DATASET[dataset], fill=(45, 45, 45), font=font(14))
        draw.text((int(cx - 28), y - 24 if delta >= 0 else y + 8), f"{delta:+.3f}", fill=(35, 35, 35), font=font(13))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def load_tile(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        tile = Image.new("RGB", size, (245, 245, 245))
        tile.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
        return tile


def find_prediction(root: Path, sample: str, suffix: str | None = None) -> Path:
    sample_dir = root / sample
    candidates = sorted(path for path in sample_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    for path in candidates:
        stem = path.stem.lower()
        if suffix and stem.endswith(suffix):
            return path
    for path in candidates:
        stem = path.stem.lower()
        if stem != "m_input" and not stem.endswith("_r") and not stem.endswith("_rr"):
            return path
    raise FileNotFoundError(f"No prediction found for {sample} under {root}")


def draw_custom_grid(out_path: Path) -> None:
    columns = ["Input", "GT", "ERRNet", "DSRNet-L", "PG-DSRNet"]
    samples = [f"self_{idx}" for idx in range(5)]
    paths = {
        "Input": ROOT / "data/custom_synth/blended",
        "GT": ROOT / "data/custom_synth/transmission_layer",
        "ERRNet": ROOT / "outputs/benchmarks/custom_synth/errnet/predictions",
        "DSRNet-L": ROOT / "outputs/benchmarks/custom_synth/dsrnet_l/predictions",
        "PG-DSRNet": ROOT / "outputs/benchmarks/custom_synth/pg_dsrnet_l_freq_prior/predictions",
    }
    tile_size = (220, 150)
    label_w, header_h, gutter = 86, 44, 10
    width = label_w + len(columns) * tile_size[0] + (len(columns) + 1) * gutter
    height = header_h + len(samples) * (tile_size[1] + gutter) + gutter
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for col_idx, col in enumerate(columns):
        x = label_w + gutter + col_idx * (tile_size[0] + gutter)
        draw.text((x + 6, 14), col, fill=(35, 35, 35), font=font(15))

    for row_idx, sample in enumerate(samples):
        y = header_h + row_idx * (tile_size[1] + gutter)
        draw.text((16, y + 60), sample, fill=(35, 35, 35), font=font(14))
        for col_idx, col in enumerate(columns):
            x = label_w + gutter + col_idx * (tile_size[0] + gutter)
            if col in {"Input", "GT"}:
                img_path = paths[col] / f"{sample}.png"
            elif col == "ERRNet":
                img_path = find_prediction(paths[col], sample)
            else:
                img_path = find_prediction(paths[col], sample, suffix="_l")
            canvas.paste(load_tile(img_path, tile_size), (x, y))
            draw.rectangle((x, y, x + tile_size[0] - 1, y + tile_size[1] - 1), outline=(210, 210, 210))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def copy_benchmark_grid(out_path: Path) -> None:
    src = ROOT / "outputs/figures/qualitative/pg_dsrnet_summary_grid.png"
    with Image.open(src) as image:
        ImageOps.exif_transpose(image).convert("RGB").save(out_path)


def main() -> int:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    draw_pipeline(out / "fig_method_pipeline.png")
    draw_quant(out / "fig_quant_psnr_ssim.png")
    draw_delta(out / "fig_pg_delta.png")
    copy_benchmark_grid(out / "fig_qual_benchmark.png")
    draw_custom_grid(out / "fig_qual_custom_synth.png")
    print(f"Wrote paper figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

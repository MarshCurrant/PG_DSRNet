#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ERRNET_PY="${ERRNET_PY:-/home/nebula/.conda/envs/dip-errnet/bin/python}"
DSRNET_PY="${DSRNET_PY:-/home/nebula/.conda/envs/dip-dsrnet/bin/python}"

DATA_DIR="${DATA_DIR:-$ROOT/data/custom_synth}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/benchmarks/custom_synth}"
MAX_LONG_EDGE="${MAX_LONG_EDGE:-1024}"

ERRNET_CKPT="${ERRNET_CKPT:-$ROOT/repos/ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt}"
DSRNET_CKPT="${DSRNET_CKPT:-$ROOT/repos/DSRNet/checkpoints/dsrnet_l_train_setting_i/weights/20260613-042722/dsrnet_l_train_setting_i_latest.pt}"
PG_DSRNET_CKPT="${PG_DSRNET_CKPT:-$ROOT/repos/DSRNet/checkpoints/pg_dsrnet_l_freq_prior_ft/weights/20260616-020451/pg_dsrnet_l_freq_prior_ft_latest.pt}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/weights/torch}"

mkdir -p "$OUT_DIR"

"$ERRNET_PY" "$ROOT/scripts/synthesize_self_reflection.py" \
  --clean-dir "$ROOT/self" \
  --output-dir "$DATA_DIR" \
  --seed 2018 \
  --max-long-edge "$MAX_LONG_EDGE"

run_eval() {
  local method_name="$1"
  local pred_dir="$2"
  local method_dir="$3"

  "$ERRNET_PY" "$ROOT/scripts/eval_sirr.py" \
    --pred_dir "$pred_dir" \
    --gt_dir "$DATA_DIR/transmission_layer" \
    --out_csv "$method_dir/metrics.csv" \
    --summary_json "$method_dir/metrics.json" \
    --align crop-min \
    --fail_on_missing
}

"$ERRNET_PY" "$ROOT/scripts/infer_method.py" \
  --method errnet \
  --checkpoint "$ERRNET_CKPT" \
  --input_dir "$DATA_DIR/blended" \
  --output_dir "$OUT_DIR/errnet/predictions" \
  --name custom_synth_errnet_self \
  --max_long_edge "$MAX_LONG_EDGE" \
  --nThreads 4
run_eval "ERRNet self-trained" "$OUT_DIR/errnet/predictions" "$OUT_DIR/errnet"

"$DSRNET_PY" "$ROOT/scripts/infer_method.py" \
  --method dsrnet \
  --checkpoint "$DSRNET_CKPT" \
  --input_dir "$DATA_DIR/blended" \
  --output_dir "$OUT_DIR/dsrnet_l/predictions" \
  --name custom_synth_dsrnet_l_self \
  --inet dsrnet_l \
  --max_long_edge "$MAX_LONG_EDGE" \
  --nThreads 4
run_eval "DSRNet-L self-trained" "$OUT_DIR/dsrnet_l/predictions" "$OUT_DIR/dsrnet_l"

"$DSRNET_PY" "$ROOT/scripts/infer_method.py" \
  --method dsrnet \
  --checkpoint "$PG_DSRNET_CKPT" \
  --input_dir "$DATA_DIR/blended" \
  --output_dir "$OUT_DIR/pg_dsrnet_l_freq_prior/predictions" \
  --name custom_synth_pg_dsrnet_l_freq_prior \
  --inet dsrnet_l \
  --max_long_edge "$MAX_LONG_EDGE" \
  --nThreads 4
run_eval "PG-DSRNet-L freq+prior" "$OUT_DIR/pg_dsrnet_l_freq_prior/predictions" "$OUT_DIR/pg_dsrnet_l_freq_prior"

"$ERRNET_PY" - "$OUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
methods = [
    ("ERRNet self-trained", "errnet"),
    ("DSRNet-L self-trained", "dsrnet_l"),
    ("PG-DSRNet-L freq+prior", "pg_dsrnet_l_freq_prior"),
]
rows = []
for method, subdir in methods:
    metrics_path = out_dir / subdir / "metrics.json"
    summary = json.loads(metrics_path.read_text())
    metrics = summary["metrics"]
    rows.append(
        {
            "method": method,
            "dataset": "Self-synth",
            "count": summary["count"],
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "ncc": metrics["ncc"],
            "lmse": metrics["lmse"],
            "source": str(out_dir / subdir / "metrics.csv"),
        }
    )

summary_csv = out_dir / "summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

(out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")

aggregate_csv = out_dir.parent / "custom_synth_summary.csv"
with aggregate_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(out_dir.parent / "custom_synth_summary.json").write_text(json.dumps(rows, indent=2) + "\n")

print(f"Wrote {summary_csv}")
print(f"Wrote {aggregate_csv}")
PY

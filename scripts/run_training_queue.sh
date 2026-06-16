#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ERRNET_PY="${ERRNET_PY:-python}"
DSRNET_PY="${DSRNET_PY:-python}"
TORCH_HOME_DIR="$ROOT/weights/torch"
MPLCONFIG_DIR="/tmp/matplotlib"
LOG_DIR="$ROOT/logs/training"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_ERRNET_UNALIGNED="${RUN_ERRNET_UNALIGNED:-1}"
RUN_DSRNET="${RUN_DSRNET:-1}"
SKIP_ERRNET_ALIGNED="${SKIP_ERRNET_ALIGNED:-0}"
NTHREADS="${NTHREADS:-8}"
WANDB_ENTITY="${WANDB_ENTITY:-3268587895-fudan-university-school-of-management}"
WANDB_PROJECT="${WANDB_PROJECT:-dip}"
WANDB_GROUP="${WANDB_GROUP:-dip-training-${RUN_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-$LOG_DIR/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-60}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p "$LOG_DIR" "$MPLCONFIG_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"
QUEUE_LOG="$LOG_DIR/training_queue_${RUN_ID}.log"
STATUS_FILE="$LOG_DIR/training_status_${RUN_ID}.txt"

exec >> "$QUEUE_LOG" 2>&1

write_status() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS_FILE"
}

on_error() {
  local rc="$?"
  write_status "FAILED rc=${rc} line=${BASH_LINENO[0]}"
  exit "$rc"
}
trap on_error ERR

run_step() {
  local name="$1"
  local rc
  shift
  write_status "START ${name}"
  printf '[cmd]'
  printf ' %q' "$@"
  printf '\n'
  set +e
  "$@"
  rc="$?"
  set -e
  if [[ "$rc" -ne 0 ]]; then
    write_status "FAILED step=${name} rc=${rc}"
    exit "$rc"
  fi
  write_status "DONE ${name}"
}

latest_checkpoint() {
  local dir="$1"
  local pattern="${2:-*.pt}"
  if [[ ! -d "$dir" ]]; then
    return 0
  fi
  find "$dir" -maxdepth 1 -type f -name "$pattern" -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

require_checkpoint() {
  local label="$1"
  local checkpoint="$2"
  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    write_status "FAILED missing_checkpoint label=${label} path=${checkpoint:-<empty>}"
    exit 1
  fi
  write_status "CHECKPOINT ${label}=${checkpoint}"
}

write_status "QUEUE_START run_id=${RUN_ID} nthreads=${NTHREADS} wandb_entity=${WANDB_ENTITY} wandb_project=${WANDB_PROJECT} wandb_mode=${WANDB_MODE}"

cd "$ROOT/repos/ERRNet"
if [[ "$SKIP_ERRNET_ALIGNED" == "1" ]]; then
  write_status "SKIP errnet_aligned_train"
else
  run_step "errnet_aligned_train" \
    env TORCH_HOME="$TORCH_HOME_DIR" MPLCONFIGDIR="$MPLCONFIG_DIR" PYTHONUNBUFFERED="$PYTHONUNBUFFERED" \
    WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" WANDB_GROUP="$WANDB_GROUP" WANDB_MODE="$WANDB_MODE" \
    WANDB_DIR="$WANDB_DIR" WANDB_CACHE_DIR="$WANDB_CACHE_DIR" WANDB_INIT_TIMEOUT="$WANDB_INIT_TIMEOUT" \
    WANDB_NAME="errnet_train_${RUN_ID}" WANDB_JOB_TYPE="errnet_aligned_train" \
    "$ERRNET_PY" train_errnet.py \
    --name errnet_train \
    --hyper \
    --nThreads "$NTHREADS"
fi

ERRNET_ALIGNED_CKPT="${ERRNET_ALIGNED_CKPT:-$(latest_checkpoint "$ROOT/repos/ERRNet/checkpoints/errnet_train" '*latest.pt')}"
if [[ -z "$ERRNET_ALIGNED_CKPT" ]]; then
  ERRNET_ALIGNED_CKPT="$(latest_checkpoint "$ROOT/repos/ERRNet/checkpoints/errnet_train" '*.pt')"
fi
require_checkpoint "errnet_aligned" "$ERRNET_ALIGNED_CKPT"

if [[ "$RUN_ERRNET_UNALIGNED" == "1" ]]; then
  run_step "errnet_unaligned_finetune" \
    env TORCH_HOME="$TORCH_HOME_DIR" MPLCONFIGDIR="$MPLCONFIG_DIR" PYTHONUNBUFFERED="$PYTHONUNBUFFERED" \
    WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" WANDB_GROUP="$WANDB_GROUP" WANDB_MODE="$WANDB_MODE" \
    WANDB_DIR="$WANDB_DIR" WANDB_CACHE_DIR="$WANDB_CACHE_DIR" WANDB_INIT_TIMEOUT="$WANDB_INIT_TIMEOUT" \
    WANDB_NAME="errnet_unaligned_ft_${RUN_ID}" WANDB_JOB_TYPE="errnet_unaligned_finetune" \
    "$ERRNET_PY" train_errnet_unaligned.py \
    --name errnet_unaligned_ft \
    --hyper \
    -r \
    --icnn_path "$ERRNET_ALIGNED_CKPT" \
    --unaligned_loss vgg \
    --nThreads "$NTHREADS"

  ERRNET_SELF_CKPT="$(latest_checkpoint "$ROOT/repos/ERRNet/checkpoints/errnet_unaligned_ft" '*latest.pt')"
  if [[ -z "$ERRNET_SELF_CKPT" ]]; then
    ERRNET_SELF_CKPT="$(latest_checkpoint "$ROOT/repos/ERRNet/checkpoints/errnet_unaligned_ft" '*.pt')"
  fi
else
  ERRNET_SELF_CKPT="$ERRNET_ALIGNED_CKPT"
fi

require_checkpoint "errnet_self" "$ERRNET_SELF_CKPT"

run_step "errnet_self_unified_benchmark" \
  env TORCH_HOME="$TORCH_HOME_DIR" MPLCONFIGDIR="$MPLCONFIG_DIR" \
  "$ERRNET_PY" "$ROOT/scripts/run_benchmark_suite.py" \
  --methods errnet \
  --datasets ceilnet_table2,real20,objects,postcard,wild \
  --output_root "$ROOT/outputs/benchmarks/self_trained_errnet" \
  --errnet_checkpoint "$ERRNET_SELF_CKPT" \
  --nThreads 2

if [[ "$RUN_DSRNET" == "1" ]]; then
  cd "$ROOT/repos/DSRNet"
  run_step "dsrnet_l_setting_i_train" \
    env TORCH_HOME="$TORCH_HOME_DIR" MPLCONFIGDIR="$MPLCONFIG_DIR" PYTHONUNBUFFERED="$PYTHONUNBUFFERED" \
    WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" WANDB_GROUP="$WANDB_GROUP" WANDB_MODE="$WANDB_MODE" \
    WANDB_DIR="$WANDB_DIR" WANDB_CACHE_DIR="$WANDB_CACHE_DIR" WANDB_INIT_TIMEOUT="$WANDB_INIT_TIMEOUT" \
    WANDB_NAME="dsrnet_l_setting_i_${RUN_ID}" WANDB_JOB_TYPE="dsrnet_l_setting_i_train" \
    "$DSRNET_PY" train_sirs.py \
    --inet dsrnet_l \
    --model dsrnet_model_sirs \
    --dataset sirs_dataset \
    --loss losses \
    --name dsrnet_l_train_setting_i \
    --lambda_vgg 0.01 \
    --lambda_rec 0.2 \
    --if_align \
    --seed 2018 \
    --base_dir "$ROOT/data/processed/dsrnet_official_extract/reflection-removal" \
    --nThreads "$NTHREADS"

  DSRNET_SELF_CKPT="$(find "$ROOT/repos/DSRNet/checkpoints/dsrnet_l_train_setting_i/weights" -type f -name 'dsrnet_l_train_setting_i_latest.pt' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -z "$DSRNET_SELF_CKPT" || ! -f "$DSRNET_SELF_CKPT" ]]; then
    write_status "FAILED missing_dsrnet_latest_checkpoint"
    exit 1
  fi
  write_status "CHECKPOINT dsrnet_self=${DSRNET_SELF_CKPT}"

  cd "$ROOT"
  run_step "dsrnet_self_unified_benchmark" \
    env TORCH_HOME="$TORCH_HOME_DIR" MPLCONFIGDIR="$MPLCONFIG_DIR" \
    "$ERRNET_PY" "$ROOT/scripts/run_benchmark_suite.py" \
    --methods dsrnet \
    --datasets ceilnet_table2,real20,objects,postcard,wild \
    --output_root "$ROOT/outputs/benchmarks/self_trained_dsrnet_l" \
    --dsrnet_checkpoint "$DSRNET_SELF_CKPT" \
    --dsrnet_inet dsrnet_l \
    --nThreads 2
fi

write_status "QUEUE_DONE run_id=${RUN_ID}"

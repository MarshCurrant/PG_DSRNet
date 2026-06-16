#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/nebula/qqm/DIP"
ERRNET_PY="/home/nebula/.conda/envs/dip-errnet/bin/python"
DSRNET_PY="/home/nebula/.conda/envs/dip-dsrnet/bin/python"
TORCH_HOME_DIR="$ROOT/weights/torch"
MPLCONFIG_DIR="/tmp/matplotlib"
LOG_DIR="$ROOT/logs/training"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
NTHREADS="${NTHREADS:-8}"
WANDB_ENTITY="${WANDB_ENTITY:-3268587895-fudan-university-school-of-management}"
WANDB_PROJECT="${WANDB_PROJECT:-dip}"
WANDB_GROUP="${WANDB_GROUP:-pg-dsrnet-${RUN_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-$LOG_DIR/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-60}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PG_BASE_CKPT="${PG_BASE_CKPT:-$ROOT/repos/DSRNet/checkpoints/dsrnet_l_train_setting_i/weights/20260613-042722/dsrnet_l_train_setting_i_latest.pt}"
PG_BASE_DIR="${PG_BASE_DIR:-$ROOT/data/processed/dsrnet_official_extract/reflection-removal}"
PG_LR="${PG_LR:-5e-5}"
PG_PRIOR_SOURCE="${PG_PRIOR_SOURCE:-target_r}"
PG_FREQ_LEVELS="${PG_FREQ_LEVELS:-3}"
RUN_FREQ="${RUN_FREQ:-1}"
RUN_PRIOR="${RUN_PRIOR:-1}"
RUN_FREQ_PRIOR="${RUN_FREQ_PRIOR:-1}"

mkdir -p "$LOG_DIR" "$MPLCONFIG_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"
QUEUE_LOG="$LOG_DIR/pg_training_queue_${RUN_ID}.log"
STATUS_FILE="$LOG_DIR/pg_training_status_${RUN_ID}.txt"

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

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    write_status "FAILED missing_file label=${label} path=${path}"
    exit 1
  fi
  write_status "FILE ${label}=${path}"
}

latest_pg_checkpoint() {
  local name="$1"
  find "$ROOT/repos/DSRNet/checkpoints/${name}/weights" -type f -name "${name}_latest.pt" -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

train_and_benchmark() {
  local key="$1"
  local name="$2"
  local n_epochs="$3"
  local lambda_freq="$4"
  local lambda_prior="$5"
  local output_root="$ROOT/outputs/benchmarks/${key}"
  local ckpt

  cd "$ROOT/repos/DSRNet"
  run_step "train_${name}" \
    env TORCH_HOME="$TORCH_HOME_DIR" MPLCONFIGDIR="$MPLCONFIG_DIR" PYTHONUNBUFFERED="$PYTHONUNBUFFERED" \
    WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" WANDB_GROUP="$WANDB_GROUP" WANDB_MODE="$WANDB_MODE" \
    WANDB_DIR="$WANDB_DIR" WANDB_CACHE_DIR="$WANDB_CACHE_DIR" WANDB_INIT_TIMEOUT="$WANDB_INIT_TIMEOUT" \
    WANDB_NAME="${name}_${RUN_ID}" WANDB_JOB_TYPE="pg_dsrnet_finetune" \
    "$DSRNET_PY" train_sirs.py \
    --inet dsrnet_l \
    --model dsrnet_model_sirs \
    --dataset sirs_dataset \
    --loss losses \
    --name "$name" \
    --lambda_vgg 0.01 \
    --lambda_rec 0.2 \
    --lambda_freq "$lambda_freq" \
    --lambda_prior "$lambda_prior" \
    --prior_source "$PG_PRIOR_SOURCE" \
    --freq_levels "$PG_FREQ_LEVELS" \
    --if_align \
    --seed 2018 \
    --base_dir "$PG_BASE_DIR" \
    --nThreads "$NTHREADS" \
    --lr "$PG_LR" \
    --nEpochs "$n_epochs" \
    --resume \
    --weight_path "$PG_BASE_CKPT"

  ckpt="$(latest_pg_checkpoint "$name")"
  require_file "${name}_checkpoint" "$ckpt"

  cd "$ROOT"
  run_step "benchmark_${name}" \
    env TORCH_HOME="$TORCH_HOME_DIR" MPLCONFIGDIR="$MPLCONFIG_DIR" \
    "$ERRNET_PY" "$ROOT/scripts/run_benchmark_suite.py" \
    --methods dsrnet \
    --datasets ceilnet_table2,real20,objects,postcard,wild \
    --output_root "$output_root" \
    --dsrnet_checkpoint "$ckpt" \
    --dsrnet_inet dsrnet_l \
    --nThreads 2

  write_status "RESULT ${name}_summary=${output_root}/summary.csv"
}

write_status "PG_QUEUE_START run_id=${RUN_ID} nthreads=${NTHREADS} base_ckpt=${PG_BASE_CKPT} wandb_entity=${WANDB_ENTITY} wandb_project=${WANDB_PROJECT} wandb_mode=${WANDB_MODE}"
require_file "pg_base_ckpt" "$PG_BASE_CKPT"

if [[ "$RUN_FREQ" == "1" ]]; then
  train_and_benchmark "pg_dsrnet_l_freq" "pg_dsrnet_l_freq_ft" 55 0.05 0.0
else
  write_status "SKIP pg_dsrnet_l_freq_ft"
fi

if [[ "$RUN_PRIOR" == "1" ]]; then
  train_and_benchmark "pg_dsrnet_l_prior" "pg_dsrnet_l_prior_ft" 55 0.0 0.10
else
  write_status "SKIP pg_dsrnet_l_prior_ft"
fi

if [[ "$RUN_FREQ_PRIOR" == "1" ]]; then
  train_and_benchmark "pg_dsrnet_l_freq_prior" "pg_dsrnet_l_freq_prior_ft" 60 0.05 0.10
else
  write_status "SKIP pg_dsrnet_l_freq_prior_ft"
fi

cd "$ROOT"
run_step "summarize_pg_dsrnet" \
  "$ERRNET_PY" "$ROOT/scripts/summarize_pg_dsrnet.py" \
  --output "$ROOT/outputs/benchmarks/pg_dsrnet_summary.csv"

write_status "PG_QUEUE_DONE run_id=${RUN_ID}"

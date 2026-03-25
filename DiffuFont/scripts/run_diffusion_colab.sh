#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/scratch/yangximing/miniconda3/envs/sg3/bin/python}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
LOG_MAX_LINES=1500

RUN_MODE="daemon"
LOG_FILE=""
PID_FILE=""
SAVE_DIR="checkpoints/flow_$(date '+%Y%m%d_%H%M%S')"

RESUME_CKPT="/scratch/yangximing/code/ass5-exp6/DiffuFont/checkpoints/flow_20260325_174039/ckpt_step_10000.pt"
DEVICE_ARG="cuda:0"
SEED=42
FONT_SPLIT="train"
FONT_SPLIT_SEED=""
FONT_TRAIN_RATIO="0.95"

TARGET_STEPS=150000
SAVE_EVERY=5000
SAMPLE_EVERY=300
LOG_EVERY=100
VAL_EVERY=100
VAL_MAX_BATCHES=16
LR="1e-4"
LR_WARMUP_STEPS=2000
LR_MIN_RATIO="0.1"
WEIGHT_DECAY="0.0"

STYLE_REF_COUNT=8
TRAIN_SAMPLING="grouped_char_font"
GROUPED_CHAR_COUNT=8
GROUPED_FONTS_PER_CHAR=0
BATCH_SIZE=128
NUM_WORKERS=8
MAX_FONTS=0
IMAGE_SIZE=128

PATCH_SIZE=16
ENCODER_HIDDEN_DIM=512
ENCODER_DEPTH=4
ENCODER_HEADS=8
ENCODER_MLP_RATIO="4.0"
STYLE_FEATURE_DIM=256
STYLE_MEMORY_K=8
PATCH_HIDDEN_DIM=512
PATCH_DEPTH=12
PATCH_HEADS=8
PATCH_MLP_RATIO="4.0"
PIXEL_HIDDEN_DIM=32
PIT_DEPTH=2
PIT_HEADS=8
PIT_MLP_RATIO="4.0"
STYLE_FUSION_START=4
STYLE_FUSION_END=8

FLOW_LAMBDA_RF="1.0"
FLOW_SAMPLE_STEPS=20
FLOW_SAMPLER="flow_dpm"
TIMESTEP_SAMPLING="logit_normal"

OOM_RETRY_SLEEP=60
MAX_OOM_RETRIES=0
EXTRA_TRAIN_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_diffusion_colab.sh [options] [-- extra train.py args]

Core options:
  --foreground | --daemon
  --save-dir PATH
  --resume PATH
  --device DEVICE
  --seed INT
  --target-steps INT
  --batch INT
  --train-sampling MODE
  --grouped-char-count INT
  --grouped-fonts-per-char INT
  --style-feature-dim INT
  --style-memory-k INT
  --style-fusion-start INT
  --style-fusion-end INT
  --oom-retry-sleep SEC
  --max-oom-retries N

This wrapper launches the unified style-memory flow training path and supports:
  1. process-tree cleanup on TERM/INT
  2. auto GPU selection when --device auto
  3. OOM-aware retry loops
  4. LR schedule: fixed warmup, hold max LR, final 20% cosine decay to lr_min_ratio
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --foreground) RUN_MODE="foreground"; shift ;;
    --daemon) RUN_MODE="daemon"; shift ;;
    --log-file) LOG_FILE="${2:?}"; shift 2 ;;
    --pid-file) PID_FILE="${2:?}"; shift 2 ;;
    --save-dir) SAVE_DIR="${2:?}"; shift 2 ;;
    --resume) RESUME_CKPT="${2:?}"; shift 2 ;;
    --device) DEVICE_ARG="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --font-split) FONT_SPLIT="${2:?}"; shift 2 ;;
    --font-split-seed) FONT_SPLIT_SEED="${2:?}"; shift 2 ;;
    --font-train-ratio) FONT_TRAIN_RATIO="${2:?}"; shift 2 ;;
    --target-steps) TARGET_STEPS="${2:?}"; shift 2 ;;
    --save-every-steps) SAVE_EVERY="${2:?}"; shift 2 ;;
    --sample-every-steps) SAMPLE_EVERY="${2:?}"; shift 2 ;;
    --log-every-steps) LOG_EVERY="${2:?}"; shift 2 ;;
    --val-every-steps) VAL_EVERY="${2:?}"; shift 2 ;;
    --val-max-batches) VAL_MAX_BATCHES="${2:?}"; shift 2 ;;
    --lr) LR="${2:?}"; shift 2 ;;
    --lr-warmup-steps) LR_WARMUP_STEPS="${2:?}"; shift 2 ;;
    --lr-min-ratio) LR_MIN_RATIO="${2:?}"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="${2:?}"; shift 2 ;;
    --style-ref-count) STYLE_REF_COUNT="${2:?}"; shift 2 ;;
    --train-sampling) TRAIN_SAMPLING="${2:?}"; shift 2 ;;
    --grouped-char-count) GROUPED_CHAR_COUNT="${2:?}"; shift 2 ;;
    --grouped-fonts-per-char) GROUPED_FONTS_PER_CHAR="${2:?}"; shift 2 ;;
    --batch) BATCH_SIZE="${2:?}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:?}"; shift 2 ;;
    --max-fonts) MAX_FONTS="${2:?}"; shift 2 ;;
    --image-size) IMAGE_SIZE="${2:?}"; shift 2 ;;
    --patch-size) PATCH_SIZE="${2:?}"; shift 2 ;;
    --encoder-hidden-dim) ENCODER_HIDDEN_DIM="${2:?}"; shift 2 ;;
    --encoder-depth) ENCODER_DEPTH="${2:?}"; shift 2 ;;
    --encoder-heads) ENCODER_HEADS="${2:?}"; shift 2 ;;
    --encoder-mlp-ratio) ENCODER_MLP_RATIO="${2:?}"; shift 2 ;;
    --style-feature-dim) STYLE_FEATURE_DIM="${2:?}"; shift 2 ;;
    --style-memory-k) STYLE_MEMORY_K="${2:?}"; shift 2 ;;
    --patch-hidden-dim) PATCH_HIDDEN_DIM="${2:?}"; shift 2 ;;
    --patch-depth) PATCH_DEPTH="${2:?}"; shift 2 ;;
    --patch-heads) PATCH_HEADS="${2:?}"; shift 2 ;;
    --patch-mlp-ratio) PATCH_MLP_RATIO="${2:?}"; shift 2 ;;
    --pixel-hidden-dim) PIXEL_HIDDEN_DIM="${2:?}"; shift 2 ;;
    --pit-depth) PIT_DEPTH="${2:?}"; shift 2 ;;
    --pit-heads) PIT_HEADS="${2:?}"; shift 2 ;;
    --pit-mlp-ratio) PIT_MLP_RATIO="${2:?}"; shift 2 ;;
    --style-fusion-start) STYLE_FUSION_START="${2:?}"; shift 2 ;;
    --style-fusion-end) STYLE_FUSION_END="${2:?}"; shift 2 ;;
    --flow-lambda-rf) FLOW_LAMBDA_RF="${2:?}"; shift 2 ;;
    --flow-sample-steps) FLOW_SAMPLE_STEPS="${2:?}"; shift 2 ;;
    --flow-sampler) FLOW_SAMPLER="${2:?}"; shift 2 ;;
    --timestep-sampling) TIMESTEP_SAMPLING="${2:?}"; shift 2 ;;
    --grad-clip-norm|--flow-lambda-img-l1|--flow-lambda-img-perc|--ema-decay)
      echo "[run_diffusion_colab] removed option: $1" >&2
      exit 2
      ;;
    --oom-retry-sleep) OOM_RETRY_SLEEP="${2:?}"; shift 2 ;;
    --max-oom-retries) MAX_OOM_RETRIES="${2:?}"; shift 2 ;;
    --) shift; EXTRA_TRAIN_ARGS+=("$@"); break ;;
    *) EXTRA_TRAIN_ARGS+=("$1"); shift ;;
  esac
done

cd "${ROOT}"
mkdir -p logs checkpoints

[[ -z "${LOG_FILE}" ]] && LOG_FILE="logs/$(basename "${SAVE_DIR}").log"
[[ -z "${PID_FILE}" ]] && PID_FILE="logs/$(basename "${SAVE_DIR}").pid"

if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then
  echo "[run_diffusion_colab] missing resume checkpoint: ${RESUME_CKPT}" >&2
  exit 2
fi

if [[ "${RUN_MODE}" == "daemon" ]]; then
  daemon_args=(
    --foreground
    --log-file "${LOG_FILE}"
    --pid-file "${PID_FILE}"
    --save-dir "${SAVE_DIR}"
    --device "${DEVICE_ARG}"
    --seed "${SEED}"
    --font-split "${FONT_SPLIT}"
    --font-train-ratio "${FONT_TRAIN_RATIO}"
    --target-steps "${TARGET_STEPS}"
    --save-every-steps "${SAVE_EVERY}"
    --sample-every-steps "${SAMPLE_EVERY}"
    --log-every-steps "${LOG_EVERY}"
    --val-every-steps "${VAL_EVERY}"
    --val-max-batches "${VAL_MAX_BATCHES}"
    --lr "${LR}"
    --lr-warmup-steps "${LR_WARMUP_STEPS}"
    --lr-min-ratio "${LR_MIN_RATIO}"
    --weight-decay "${WEIGHT_DECAY}"
    --style-ref-count "${STYLE_REF_COUNT}"
    --train-sampling "${TRAIN_SAMPLING}"
    --grouped-char-count "${GROUPED_CHAR_COUNT}"
    --grouped-fonts-per-char "${GROUPED_FONTS_PER_CHAR}"
    --batch "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --max-fonts "${MAX_FONTS}"
    --image-size "${IMAGE_SIZE}"
    --patch-size "${PATCH_SIZE}"
    --encoder-hidden-dim "${ENCODER_HIDDEN_DIM}"
    --encoder-depth "${ENCODER_DEPTH}"
    --encoder-heads "${ENCODER_HEADS}"
    --encoder-mlp-ratio "${ENCODER_MLP_RATIO}"
    --style-feature-dim "${STYLE_FEATURE_DIM}"
    --style-memory-k "${STYLE_MEMORY_K}"
    --patch-hidden-dim "${PATCH_HIDDEN_DIM}"
    --patch-depth "${PATCH_DEPTH}"
    --patch-heads "${PATCH_HEADS}"
    --patch-mlp-ratio "${PATCH_MLP_RATIO}"
    --pixel-hidden-dim "${PIXEL_HIDDEN_DIM}"
    --pit-depth "${PIT_DEPTH}"
    --pit-heads "${PIT_HEADS}"
    --pit-mlp-ratio "${PIT_MLP_RATIO}"
    --style-fusion-start "${STYLE_FUSION_START}"
    --style-fusion-end "${STYLE_FUSION_END}"
    --flow-lambda-rf "${FLOW_LAMBDA_RF}"
    --flow-sample-steps "${FLOW_SAMPLE_STEPS}"
    --flow-sampler "${FLOW_SAMPLER}"
    --timestep-sampling "${TIMESTEP_SAMPLING}"
    --oom-retry-sleep "${OOM_RETRY_SLEEP}"
    --max-oom-retries "${MAX_OOM_RETRIES}"
  )
  if [[ -n "${FONT_SPLIT_SEED}" ]]; then
    daemon_args+=(--font-split-seed "${FONT_SPLIT_SEED}")
  fi
  if [[ -n "${RESUME_CKPT}" ]]; then
    daemon_args+=(--resume "${RESUME_CKPT}")
  fi
  if [[ "${#EXTRA_TRAIN_ARGS[@]}" -gt 0 ]]; then
    daemon_args+=(-- "${EXTRA_TRAIN_ARGS[@]}")
  fi
  nohup bash "${SCRIPT_PATH}" "${daemon_args[@]}" > /dev/null 2>&1 < /dev/null &
  echo "$!" > "${PID_FILE}"
  echo "[run_diffusion_colab] started daemon pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
  exit 0
fi

SCRIPT_PID="$$"

list_child_pids() {
  local parent_pid="$1"
  ps -o pid= --ppid "${parent_pid}" 2>/dev/null | awk '{print $1}'
}

kill_descendants() {
  local parent_pid="$1"
  local signal="${2:-TERM}"
  local child_pid
  while read -r child_pid; do
    [[ -z "${child_pid}" ]] && continue
    kill_descendants "${child_pid}" "${signal}"
    kill -"${signal}" "${child_pid}" 2>/dev/null || true
  done < <(list_child_pids "${parent_pid}")
}

cleanup_pid_file() {
  if [[ -z "${PID_FILE}" || ! -f "${PID_FILE}" ]]; then
    return 0
  fi
  local recorded_pid
  recorded_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ "${recorded_pid}" == "${SCRIPT_PID}" ]]; then
    rm -f "${PID_FILE}"
  fi
}

handle_signal() {
  local signal="$1"
  local exit_code=143
  if [[ "${signal}" == "INT" ]]; then
    exit_code=130
  fi
  trap - TERM INT EXIT
  echo "[run_diffusion_colab] received ${signal}, terminating process tree"
  kill_descendants "${SCRIPT_PID}" TERM
  sleep 1
  kill_descendants "${SCRIPT_PID}" KILL
  cleanup_pid_file
  exit "${exit_code}"
}

handle_exit() {
  local status=$?
  trap - EXIT
  kill_descendants "${SCRIPT_PID}" TERM
  cleanup_pid_file
  exit "${status}"
}

trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT
trap 'handle_exit' EXIT

capped_logger() {
  "${PYTHON_BIN}" -u -c '
import os
import sys
from collections import deque

log_path = sys.argv[1]
max_lines = int(sys.argv[2])
buffer = deque(maxlen=max_lines)

if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            buffer.append(line)
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        f.writelines(buffer)

for line in sys.stdin:
    sys.stdout.write(line)
    sys.stdout.flush()
    buffer.append(line)
    tmp_path = log_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", errors="replace") as f:
        f.writelines(buffer)
    os.replace(tmp_path, log_path)
' "${LOG_FILE}" "${LOG_MAX_LINES}"
}

exec > >(capped_logger) 2>&1
echo "$$" > "${PID_FILE}"

select_launch_device() {
  if [[ "${DEVICE_ARG}" != "auto" ]]; then
    echo "${DEVICE_ARG}"
    return 0
  fi

  "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    print("[run_diffusion_colab] torch.cuda.is_available()=False, falling back to cpu", file=sys.stderr)
    print("cpu")
    raise SystemExit(0)

device_count = torch.cuda.device_count()
probe_count = min(4, device_count)
best_idx = 0
best_free = -1

for idx in range(probe_count):
    try:
        with torch.cuda.device(idx):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception as exc:
        print(f"[run_diffusion_colab] gpu_probe_failed cuda:{idx} error={exc}", file=sys.stderr)
        continue
    free_gb = free_bytes / float(1024 ** 3)
    total_gb = total_bytes / float(1024 ** 3)
    print(
        f"[run_diffusion_colab] gpu_probe cuda:{idx} free_gb={free_gb:.2f} total_gb={total_gb:.2f}",
        file=sys.stderr,
    )
    if free_bytes > best_free:
        best_free = free_bytes
        best_idx = idx

print(f"cuda:{best_idx}")
PY
}

is_oom_failure() {
  local attempt_marker="$1"
  if [[ ! -f "${LOG_FILE}" ]]; then
    return 1
  fi
  if [[ -n "${attempt_marker}" ]] && grep -Fq "${attempt_marker}" "${LOG_FILE}"; then
    awk -v marker="${attempt_marker}" '
      found { print; next }
      index($0, marker) { found = 1; print }
    ' "${LOG_FILE}" | grep -Eiq \
      'torch\.OutOfMemoryError|CUDA out of memory|out of memory|CUDACachingAllocator|cuda runtime error'
    return $?
  fi
  grep -Eiq \
    'torch\.OutOfMemoryError|CUDA out of memory|out of memory|CUDACachingAllocator|cuda runtime error' \
    "${LOG_FILE}"
}

cmd_common=(
  "${PYTHON_BIN}" -u train.py
  --data-root "${ROOT}"
  --save-dir "${SAVE_DIR}"
  --seed "${SEED}"
  --font-split "${FONT_SPLIT}"
  --font-train-ratio "${FONT_TRAIN_RATIO}"
  --lr "${LR}"
  --lr-warmup-steps "${LR_WARMUP_STEPS}"
  --lr-min-ratio "${LR_MIN_RATIO}"
  --weight-decay "${WEIGHT_DECAY}"
  --batch "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --style-ref-count "${STYLE_REF_COUNT}"
  --train-sampling "${TRAIN_SAMPLING}"
  --grouped-char-count "${GROUPED_CHAR_COUNT}"
  --grouped-fonts-per-char "${GROUPED_FONTS_PER_CHAR}"
  --max-fonts "${MAX_FONTS}"
  --image-size "${IMAGE_SIZE}"
  --patch-size "${PATCH_SIZE}"
  --encoder-hidden-dim "${ENCODER_HIDDEN_DIM}"
  --encoder-depth "${ENCODER_DEPTH}"
  --encoder-heads "${ENCODER_HEADS}"
  --encoder-mlp-ratio "${ENCODER_MLP_RATIO}"
  --style-feature-dim "${STYLE_FEATURE_DIM}"
  --style-memory-k "${STYLE_MEMORY_K}"
  --patch-hidden-dim "${PATCH_HIDDEN_DIM}"
  --patch-depth "${PATCH_DEPTH}"
  --patch-heads "${PATCH_HEADS}"
  --patch-mlp-ratio "${PATCH_MLP_RATIO}"
  --pixel-hidden-dim "${PIXEL_HIDDEN_DIM}"
  --pit-depth "${PIT_DEPTH}"
  --pit-heads "${PIT_HEADS}"
  --pit-mlp-ratio "${PIT_MLP_RATIO}"
  --style-fusion-start "${STYLE_FUSION_START}"
  --style-fusion-end "${STYLE_FUSION_END}"
  --flow-lambda-rf "${FLOW_LAMBDA_RF}"
  --flow-sample-steps "${FLOW_SAMPLE_STEPS}"
  --flow-sampler "${FLOW_SAMPLER}"
  --timestep-sampling "${TIMESTEP_SAMPLING}"
  --total-steps "${TARGET_STEPS}"
  --log-every-steps "${LOG_EVERY}"
  --val-every-steps "${VAL_EVERY}"
  --val-max-batches "${VAL_MAX_BATCHES}"
  --save-every-steps "${SAVE_EVERY}"
  --sample-every-steps "${SAMPLE_EVERY}"
)

if [[ -n "${FONT_SPLIT_SEED}" ]]; then
  cmd_common+=(--font-split-seed "${FONT_SPLIT_SEED}")
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  cmd_common+=(--resume "${RESUME_CKPT}")
fi
if [[ "${#EXTRA_TRAIN_ARGS[@]}" -gt 0 ]]; then
  cmd_common+=("${EXTRA_TRAIN_ARGS[@]}")
fi

echo "[run_diffusion_colab] stage=flow"
echo "[run_diffusion_colab] save_dir=${SAVE_DIR}"
echo "[run_diffusion_colab] log_file=${LOG_FILE}"
echo "[run_diffusion_colab] requested_device=${DEVICE_ARG} seed=${SEED}"
echo "[run_diffusion_colab] batch=${BATCH_SIZE} lr=${LR} target_steps=${TARGET_STEPS}"
echo "[run_diffusion_colab] train_sampling=${TRAIN_SAMPLING}"
echo "[run_diffusion_colab] grouped_char_count=${GROUPED_CHAR_COUNT} grouped_fonts_per_char=${GROUPED_FONTS_PER_CHAR}"
echo "[run_diffusion_colab] lr_schedule=warmup_hold_then_final20pct_cosine"
echo "[run_diffusion_colab] lr=${LR} lr_warmup_steps=${LR_WARMUP_STEPS} lr_min_ratio=${LR_MIN_RATIO} weight_decay=${WEIGHT_DECAY}"
echo "[run_diffusion_colab] save_every_steps=${SAVE_EVERY} sample_every_steps=${SAMPLE_EVERY}"
echo "[run_diffusion_colab] image_size=${IMAGE_SIZE} patch_size=${PATCH_SIZE}"
echo "[run_diffusion_colab] encoder_hidden_dim=${ENCODER_HIDDEN_DIM} encoder_depth=${ENCODER_DEPTH} encoder_heads=${ENCODER_HEADS}"
echo "[run_diffusion_colab] style_feature_dim=${STYLE_FEATURE_DIM} style_memory_k=${STYLE_MEMORY_K}"
echo "[run_diffusion_colab] patch_hidden_dim=${PATCH_HIDDEN_DIM} patch_depth=${PATCH_DEPTH} patch_heads=${PATCH_HEADS}"
echo "[run_diffusion_colab] pixel_hidden_dim=${PIXEL_HIDDEN_DIM} pit_depth=${PIT_DEPTH} pit_heads=${PIT_HEADS}"
echo "[run_diffusion_colab] style_fusion_start=${STYLE_FUSION_START} style_fusion_end=${STYLE_FUSION_END}"
echo "[run_diffusion_colab] flow_lambda_rf=${FLOW_LAMBDA_RF} flow_sample_steps=${FLOW_SAMPLE_STEPS}"
echo "[run_diffusion_colab] flow_sampler=${FLOW_SAMPLER} timestep_sampling=${TIMESTEP_SAMPLING}"
echo "[run_diffusion_colab] oom_retry_sleep=${OOM_RETRY_SLEEP} max_oom_retries=${MAX_OOM_RETRIES}"

attempt=1
while true; do
  launch_device="$(select_launch_device)"
  attempt_marker="__run_diffusion_attempt_${attempt}_$(date +%s)"
  cmd=("${cmd_common[@]}")
  cmd+=(--device "${launch_device}")

  echo "[run_diffusion_colab] ${attempt_marker}"
  echo "[run_diffusion_colab] attempt=${attempt} launch_device=${launch_device}"
  printf '[run_diffusion_colab] cmd='
  printf ' %q' "${cmd[@]}"
  printf '\n'

  set +e
  "${cmd[@]}" &
  child_pid=$!
  wait "${child_pid}"
  status=$?
  set -e

  if [[ ${status} -eq 0 ]]; then
    exit 0
  fi

  if is_oom_failure "${attempt_marker}"; then
    if [[ "${MAX_OOM_RETRIES}" != "0" && ${attempt} -ge ${MAX_OOM_RETRIES} ]]; then
      echo "[run_diffusion_colab] attempt=${attempt} hit max OOM retries, aborting"
      exit "${status}"
    fi
    echo "[run_diffusion_colab] attempt=${attempt} failed with OOM, sleeping ${OOM_RETRY_SLEEP}s before retry"
    attempt=$((attempt + 1))
    sleep "${OOM_RETRY_SLEEP}"
    continue
  fi

  echo "[run_diffusion_colab] attempt=${attempt} failed with non-OOM status=${status}, aborting"
  exit "${status}"
done

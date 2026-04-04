#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/yangximing/code/ass5-exp6/DiffuFont"
PYTHON_BIN="/scratch/yangximing/miniconda3/envs/sg3/bin/python"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
LOG_MAX_LINES=1500

RUN_MODE="daemon"
LOG_FILE=""
PID_FILE=""
SAVE_DIR="checkpoints/flow_$(date '+%Y%m%d_%H%M%S')"

RESUME_CKPT=""
DEVICE_ARG="cuda:1"
SEED=42
FONT_SPLIT="train"
FONT_SPLIT_SEED=""
FONT_TRAIN_RATIO="0.95"

EPOCHS=10000000
TARGET_STEPS=100000
SAVE_EVERY=5000
SAMPLE_EVERY=300
LOG_EVERY=100
VAL_EVERY=100
VAL_MAX_BATCHES=16
LR="1e-4"
LR_WARMUP_STEPS=0
LR_DECAY_START_STEP="40000"
LR_MIN_SCALE="0.1"
GRAD_CLIP_NORM="1.0"
GRAD_CLIP_MIN_NORM="0.5"

STYLE_REF_COUNT=0
STYLE_REF_COUNT_MIN=3
STYLE_REF_COUNT_MAX=6
BATCH_SIZE=256
NUM_WORKERS=8
MAX_FONTS=0
IMAGE_SIZE=128

PATCH_SIZE=16
ENCODER_HIDDEN_DIM=512
DIT_HIDDEN_DIM=512
DIT_DEPTH=12
DIT_HEADS=8
DIT_MLP_RATIO="4.0"
CONTENT_INJECTION_LAYERS="2,4,6,8"
STYLE_INJECTION_LAYERS="9,10,11,12"
DETAILER_BASE_CHANNELS=32
DETAILER_MAX_CHANNELS=256

FLOW_LAMBDA="1.0"
USE_CNN_PERCEPTOR="1"
PERCEPTOR_CHECKPOINT="/scratch/yangximing/code/ass5-exp6/DiffuFont/checkpoints/font_perceptor_20260328_123942/best.pt"
PERCEPTUAL_LOSS_LAMBDA="0.2"
STYLE_LOSS_LAMBDA="0.05"
STYLE_BATCH_SUPCON_LAMBDA="0.01"
PIXEL_LOSS_LAMBDA="0.05"
AUX_LOSS_T_LOGISTIC_STEEPNESS="8.0"
PERCEPTUAL_LOSS_T_MIDPOINT="0.35"
STYLE_LOSS_T_MIDPOINT="0.45"
PIXEL_LOSS_T_MIDPOINT="0.55"
FLOW_SAMPLE_STEPS=20
EMA_DECAY="0"
TRAIN_SAMPLING="cartesian_font_char"
CARTESIAN_FONTS_PER_BATCH=64
CARTESIAN_CHARS_PER_BATCH=4

EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --epochs) EPOCHS="${2:?}"; shift 2 ;;
    --target-steps) TARGET_STEPS="${2:?}"; shift 2 ;;
    --save-every-steps) SAVE_EVERY="${2:?}"; shift 2 ;;
    --sample-every-steps) SAMPLE_EVERY="${2:?}"; shift 2 ;;
    --log-every-steps) LOG_EVERY="${2:?}"; shift 2 ;;
    --val-every-steps) VAL_EVERY="${2:?}"; shift 2 ;;
    --val-max-batches) VAL_MAX_BATCHES="${2:?}"; shift 2 ;;
    --lr) LR="${2:?}"; shift 2 ;;
    --lr-warmup-steps) LR_WARMUP_STEPS="${2:?}"; shift 2 ;;
    --lr-decay-start-step) LR_DECAY_START_STEP="${2:?}"; shift 2 ;;
    --lr-min-scale) LR_MIN_SCALE="${2:?}"; shift 2 ;;
    --grad-clip-norm) GRAD_CLIP_NORM="${2:?}"; shift 2 ;;
    --grad-clip-min-norm) GRAD_CLIP_MIN_NORM="${2:?}"; shift 2 ;;
    --flow-lambda) FLOW_LAMBDA="${2:?}"; shift 2 ;;
    --use-cnn-perceptor) USE_CNN_PERCEPTOR="1"; shift ;;
    --no-use-cnn-perceptor) USE_CNN_PERCEPTOR="0"; shift ;;
    --perceptor-checkpoint) PERCEPTOR_CHECKPOINT="${2:?}"; shift 2 ;;
    --perceptual-loss-lambda) PERCEPTUAL_LOSS_LAMBDA="${2:?}"; shift 2 ;;
    --style-loss-lambda) STYLE_LOSS_LAMBDA="${2:?}"; shift 2 ;;
    --style-batch-supcon-lambda) STYLE_BATCH_SUPCON_LAMBDA="${2:?}"; shift 2 ;;
    --pixel-loss-lambda) PIXEL_LOSS_LAMBDA="${2:?}"; shift 2 ;;
    --aux-loss-t-logistic-steepness) AUX_LOSS_T_LOGISTIC_STEEPNESS="${2:?}"; shift 2 ;;
    --perceptual-loss-t-midpoint) PERCEPTUAL_LOSS_T_MIDPOINT="${2:?}"; shift 2 ;;
    --style-loss-t-midpoint) STYLE_LOSS_T_MIDPOINT="${2:?}"; shift 2 ;;
    --pixel-loss-t-midpoint) PIXEL_LOSS_T_MIDPOINT="${2:?}"; shift 2 ;;
    --flow-sample-steps) FLOW_SAMPLE_STEPS="${2:?}"; shift 2 ;;
    --ema-decay) EMA_DECAY="${2:?}"; shift 2 ;;
    --style-ref-count) STYLE_REF_COUNT="${2:?}"; shift 2 ;;
    --style-ref-count-min) STYLE_REF_COUNT_MIN="${2:?}"; shift 2 ;;
    --style-ref-count-max) STYLE_REF_COUNT_MAX="${2:?}"; shift 2 ;;
    --batch) BATCH_SIZE="${2:?}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:?}"; shift 2 ;;
    --max-fonts) MAX_FONTS="${2:?}"; shift 2 ;;
    --image-size) IMAGE_SIZE="${2:?}"; shift 2 ;;
    --patch-size) PATCH_SIZE="${2:?}"; shift 2 ;;
    --encoder-hidden-dim) ENCODER_HIDDEN_DIM="${2:?}"; shift 2 ;;
    --dit-hidden-dim) DIT_HIDDEN_DIM="${2:?}"; shift 2 ;;
    --dit-depth) DIT_DEPTH="${2:?}"; shift 2 ;;
    --dit-heads) DIT_HEADS="${2:?}"; shift 2 ;;
    --dit-mlp-ratio) DIT_MLP_RATIO="${2:?}"; shift 2 ;;
    --content-injection-layers) CONTENT_INJECTION_LAYERS="${2:?}"; shift 2 ;;
    --style-injection-layers) STYLE_INJECTION_LAYERS="${2:?}"; shift 2 ;;
    --detailer-base-channels) DETAILER_BASE_CHANNELS="${2:?}"; shift 2 ;;
    --detailer-max-channels) DETAILER_MAX_CHANNELS="${2:?}"; shift 2 ;;
    --train-sampling) TRAIN_SAMPLING="${2:?}"; shift 2 ;;
    --cartesian-fonts-per-batch) CARTESIAN_FONTS_PER_BATCH="${2:?}"; shift 2 ;;
    --cartesian-chars-per-batch) CARTESIAN_CHARS_PER_BATCH="${2:?}"; shift 2 ;;
    --) shift; EXTRA_TRAIN_ARGS+=("$@"); break ;;
    *) EXTRA_TRAIN_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${FONT_SPLIT_SEED}" ]]; then
  FONT_SPLIT_SEED="${SEED}"
fi
if [[ "${LR_DECAY_START_STEP}" == "-1" ]]; then
  LR_DECAY_START_STEP="$(( TARGET_STEPS * 8 / 10 ))"
fi

cd "${ROOT}"
mkdir -p logs checkpoints

[[ -z "${LOG_FILE}" ]] && LOG_FILE="logs/$(basename "${SAVE_DIR}").log"
[[ -z "${PID_FILE}" ]] && PID_FILE="logs/$(basename "${SAVE_DIR}").pid"

mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${PID_FILE}")" "$(dirname "${SAVE_DIR}")"

if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then
  echo "[run_diffusion_colab] missing resume checkpoint: ${RESUME_CKPT}" >&2
  exit 2
fi
if [[ "${USE_CNN_PERCEPTOR}" == "1" && -n "${PERCEPTOR_CHECKPOINT}" && ! -f "${PERCEPTOR_CHECKPOINT}" ]]; then
  echo "[run_diffusion_colab] missing perceptor checkpoint: ${PERCEPTOR_CHECKPOINT}" >&2
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
    --font-split-seed "${FONT_SPLIT_SEED}"
    --font-train-ratio "${FONT_TRAIN_RATIO}"
    --epochs "${EPOCHS}"
    --target-steps "${TARGET_STEPS}"
    --save-every-steps "${SAVE_EVERY}"
    --sample-every-steps "${SAMPLE_EVERY}"
    --log-every-steps "${LOG_EVERY}"
    --val-every-steps "${VAL_EVERY}"
    --val-max-batches "${VAL_MAX_BATCHES}"
    --lr "${LR}"
    --lr-warmup-steps "${LR_WARMUP_STEPS}"
    --lr-decay-start-step "${LR_DECAY_START_STEP}"
    --lr-min-scale "${LR_MIN_SCALE}"
    --grad-clip-norm "${GRAD_CLIP_NORM}"
    --grad-clip-min-norm "${GRAD_CLIP_MIN_NORM}"
    --flow-lambda "${FLOW_LAMBDA}"
    --perceptual-loss-lambda "${PERCEPTUAL_LOSS_LAMBDA}"
    --style-loss-lambda "${STYLE_LOSS_LAMBDA}"
    --style-batch-supcon-lambda "${STYLE_BATCH_SUPCON_LAMBDA}"
    --flow-sample-steps "${FLOW_SAMPLE_STEPS}"
    --ema-decay "${EMA_DECAY}"
    --style-ref-count "${STYLE_REF_COUNT}"
    --style-ref-count-min "${STYLE_REF_COUNT_MIN}"
    --style-ref-count-max "${STYLE_REF_COUNT_MAX}"
    --batch "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --max-fonts "${MAX_FONTS}"
    --image-size "${IMAGE_SIZE}"
    --patch-size "${PATCH_SIZE}"
    --encoder-hidden-dim "${ENCODER_HIDDEN_DIM}"
    --dit-hidden-dim "${DIT_HIDDEN_DIM}"
    --dit-depth "${DIT_DEPTH}"
    --dit-heads "${DIT_HEADS}"
    --dit-mlp-ratio "${DIT_MLP_RATIO}"
    --content-injection-layers "${CONTENT_INJECTION_LAYERS}"
    --style-injection-layers "${STYLE_INJECTION_LAYERS}"
    --detailer-base-channels "${DETAILER_BASE_CHANNELS}"
    --detailer-max-channels "${DETAILER_MAX_CHANNELS}"
    --train-sampling "${TRAIN_SAMPLING}"
    --cartesian-fonts-per-batch "${CARTESIAN_FONTS_PER_BATCH}"
    --cartesian-chars-per-batch "${CARTESIAN_CHARS_PER_BATCH}"
  )
  if [[ -n "${RESUME_CKPT}" ]]; then
    daemon_args+=(--resume "${RESUME_CKPT}")
  fi
  if [[ "${USE_CNN_PERCEPTOR}" == "1" ]]; then
    daemon_args+=(--use-cnn-perceptor)
  else
    daemon_args+=(--no-use-cnn-perceptor)
  fi
  if [[ "${USE_CNN_PERCEPTOR}" == "1" && -n "${PERCEPTOR_CHECKPOINT}" ]]; then
    daemon_args+=(--perceptor-checkpoint "${PERCEPTOR_CHECKPOINT}")
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
import sys
import torch

if not torch.cuda.is_available():
    print("[run_diffusion_colab] torch.cuda.is_available()=False, falling back to cpu", file=sys.stderr)
    print("cpu")
    raise SystemExit(0)

device_count = torch.cuda.device_count()
probe_count = min(2, device_count)
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
      'torch\.OutOfMemoryError|CUDA out of memory|out of memory|NVML_SUCCESS == DriverAPI::get\(\)->nvmlInit_v2_\(\)|CUDACachingAllocator\.cpp'
    return $?
  fi
  grep -Eiq \
    'torch\.OutOfMemoryError|CUDA out of memory|out of memory|NVML_SUCCESS == DriverAPI::get\(\)->nvmlInit_v2_\(\)|CUDACachingAllocator\.cpp' \
    "${LOG_FILE}"
}

cmd_common=(
  "${PYTHON_BIN}" -u train.py
  --data-root "${ROOT}"
  --save-dir "${SAVE_DIR}"
  --seed "${SEED}"
  --font-split "${FONT_SPLIT}"
  --font-split-seed "${FONT_SPLIT_SEED}"
  --font-train-ratio "${FONT_TRAIN_RATIO}"
  --lr "${LR}"
  --lr-warmup-steps "${LR_WARMUP_STEPS}"
  --lr-decay-start-step "${LR_DECAY_START_STEP}"
  --lr-min-scale "${LR_MIN_SCALE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --grad-clip-min-norm "${GRAD_CLIP_MIN_NORM}"
  --batch "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --style-ref-count "${STYLE_REF_COUNT}"
  --style-ref-count-min "${STYLE_REF_COUNT_MIN}"
  --style-ref-count-max "${STYLE_REF_COUNT_MAX}"
  --max-fonts "${MAX_FONTS}"
  --image-size "${IMAGE_SIZE}"
  --patch-size "${PATCH_SIZE}"
  --encoder-hidden-dim "${ENCODER_HIDDEN_DIM}"
  --dit-hidden-dim "${DIT_HIDDEN_DIM}"
  --dit-depth "${DIT_DEPTH}"
  --dit-heads "${DIT_HEADS}"
  --dit-mlp-ratio "${DIT_MLP_RATIO}"
  --content-injection-layers "${CONTENT_INJECTION_LAYERS}"
  --style-injection-layers "${STYLE_INJECTION_LAYERS}"
  --detailer-base-channels "${DETAILER_BASE_CHANNELS}"
  --detailer-max-channels "${DETAILER_MAX_CHANNELS}"
  --train-sampling "${TRAIN_SAMPLING}"
  --cartesian-fonts-per-batch "${CARTESIAN_FONTS_PER_BATCH}"
  --cartesian-chars-per-batch "${CARTESIAN_CHARS_PER_BATCH}"
  --flow-lambda "${FLOW_LAMBDA}"
  --perceptual-loss-lambda "${PERCEPTUAL_LOSS_LAMBDA}"
  --style-loss-lambda "${STYLE_LOSS_LAMBDA}"
  --style-batch-supcon-lambda "${STYLE_BATCH_SUPCON_LAMBDA}"
  --pixel-loss-lambda "${PIXEL_LOSS_LAMBDA}"
  --aux-loss-t-logistic-steepness "${AUX_LOSS_T_LOGISTIC_STEEPNESS}"
  --perceptual-loss-t-midpoint "${PERCEPTUAL_LOSS_T_MIDPOINT}"
  --style-loss-t-midpoint "${STYLE_LOSS_T_MIDPOINT}"
  --pixel-loss-t-midpoint "${PIXEL_LOSS_T_MIDPOINT}"
  --flow-sample-steps "${FLOW_SAMPLE_STEPS}"
  --ema-decay "${EMA_DECAY}"
  --epochs "${EPOCHS}"
  --total-steps "${TARGET_STEPS}"
  --log-every-steps "${LOG_EVERY}"
  --val-every-steps "${VAL_EVERY}"
  --val-max-batches "${VAL_MAX_BATCHES}"
  --save-every-steps "${SAVE_EVERY}"
  --sample-every-steps "${SAMPLE_EVERY}"
)

if [[ -n "${RESUME_CKPT}" ]]; then
  cmd_common+=(--resume "${RESUME_CKPT}")
fi
if [[ "${USE_CNN_PERCEPTOR}" == "1" ]]; then
  cmd_common+=(--use-cnn-perceptor)
else
  cmd_common+=(--no-use-cnn-perceptor)
fi
if [[ "${USE_CNN_PERCEPTOR}" == "1" && -n "${PERCEPTOR_CHECKPOINT}" ]]; then
  cmd_common+=(--perceptor-checkpoint "${PERCEPTOR_CHECKPOINT}")
fi
if [[ "${#EXTRA_TRAIN_ARGS[@]}" -gt 0 ]]; then
  cmd_common+=("${EXTRA_TRAIN_ARGS[@]}")
fi

echo "[run_diffusion_colab] mode=pixel_flow"
echo "[run_diffusion_colab] save_dir=${SAVE_DIR}"
echo "[run_diffusion_colab] log_file=${LOG_FILE}"
echo "[run_diffusion_colab] requested_device=${DEVICE_ARG} seed=${SEED}"
echo "[run_diffusion_colab] resume=${RESUME_CKPT:-<none>}"
echo "[run_diffusion_colab] use_cnn_perceptor=${USE_CNN_PERCEPTOR}"
echo "[run_diffusion_colab] perceptor_checkpoint=${PERCEPTOR_CHECKPOINT:-<none>}"
echo "[run_diffusion_colab] batch=${BATCH_SIZE} lr=${LR} lr_warmup_steps=${LR_WARMUP_STEPS} lr_decay_start_step=${LR_DECAY_START_STEP} lr_min_scale=${LR_MIN_SCALE} grad_clip_norm=${GRAD_CLIP_NORM}"
echo "[run_diffusion_colab] style_ref_count=${STYLE_REF_COUNT} style_ref_count_min=${STYLE_REF_COUNT_MIN} style_ref_count_max=${STYLE_REF_COUNT_MAX}"
echo "[run_diffusion_colab] patch_size=${PATCH_SIZE} image_size=${IMAGE_SIZE} flow_sample_steps=${FLOW_SAMPLE_STEPS} flow_lambda=${FLOW_LAMBDA} ema_decay=${EMA_DECAY}"
echo "[run_diffusion_colab] dit_heads=${DIT_HEADS}"
echo "[run_diffusion_colab] perceptual_loss_lambda=${PERCEPTUAL_LOSS_LAMBDA} style_loss_lambda=${STYLE_LOSS_LAMBDA} style_batch_supcon_lambda=${STYLE_BATCH_SUPCON_LAMBDA} pixel_loss_lambda=${PIXEL_LOSS_LAMBDA} aux_loss_t_logistic_steepness=${AUX_LOSS_T_LOGISTIC_STEEPNESS} perceptual_loss_t_midpoint=${PERCEPTUAL_LOSS_T_MIDPOINT} style_loss_t_midpoint=${STYLE_LOSS_T_MIDPOINT} pixel_loss_t_midpoint=${PIXEL_LOSS_T_MIDPOINT}"
echo "[run_diffusion_colab] detailer_base_channels=${DETAILER_BASE_CHANNELS} detailer_max_channels=${DETAILER_MAX_CHANNELS}"
echo "[run_diffusion_colab] content_injection_layers=${CONTENT_INJECTION_LAYERS} style_injection_layers=${STYLE_INJECTION_LAYERS}"
echo "[run_diffusion_colab] train_sampling=${TRAIN_SAMPLING} cartesian_fonts_per_batch=${CARTESIAN_FONTS_PER_BATCH} cartesian_chars_per_batch=${CARTESIAN_CHARS_PER_BATCH}"

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
    echo "[run_diffusion_colab] attempt=${attempt} failed with OOM, sleeping 60s before retry"
    attempt=$((attempt + 1))
    sleep 60
    continue
  fi

  echo "[run_diffusion_colab] attempt=${attempt} failed with non-OOM status=${status}, aborting"
  exit "${status}"
done

#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/yangximing/code/ass5-exp6/DiffuFont"
PYTHON_BIN="/scratch/yangximing/miniconda3/envs/sg3/bin/python"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
LOG_MAX_LINES=1500

RUN_MODE="daemon"
LOG_FILE=""
PID_FILE=""
SAVE_DIR="checkpoints/font_perceptor_$(date '+%Y%m%d_%H%M%S')"

RESUME_CKPT=""
DEVICE_ARG="auto"
SEED=42
FONT_SPLIT="train"
FONT_SPLIT_SEED=""
FONT_TRAIN_RATIO="0.95"

EPOCHS=1000000
TARGET_STEPS=50000
SAVE_EVERY=5000
LOG_EVERY=100
VAL_EVERY=100
VAL_MAX_BATCHES=16
LR="3e-4"
GRAD_CLIP_NORM="1.0"

BATCH_SIZE=256
NUM_WORKERS=8
MAX_FONTS=0
IMAGE_SIZE=128

BASE_CHANNELS=32
STYLE_PROJ_DIM=128
DROPOUT="0.0"
FEATURE_STAGES="stage1,stage2,stage3,stage4"

TRAIN_SAMPLING="cartesian_font_char"
CARTESIAN_FONTS_PER_BATCH=16
CARTESIAN_CHARS_PER_BATCH=16

STYLE_SUPCON_LAMBDA="0.2"
STYLE_TEMPERATURE="0.07"
QUALIFY_MIN_CHAR_ACC="0.70"
QUALIFY_MIN_STYLE_MARGIN="0.10"

EXTRA_ARGS=()

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
    --log-every-steps) LOG_EVERY="${2:?}"; shift 2 ;;
    --val-every-steps) VAL_EVERY="${2:?}"; shift 2 ;;
    --val-max-batches) VAL_MAX_BATCHES="${2:?}"; shift 2 ;;
    --lr) LR="${2:?}"; shift 2 ;;
    --grad-clip-norm) GRAD_CLIP_NORM="${2:?}"; shift 2 ;;
    --batch) BATCH_SIZE="${2:?}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:?}"; shift 2 ;;
    --max-fonts) MAX_FONTS="${2:?}"; shift 2 ;;
    --image-size) IMAGE_SIZE="${2:?}"; shift 2 ;;
    --base-channels) BASE_CHANNELS="${2:?}"; shift 2 ;;
    --style-proj-dim) STYLE_PROJ_DIM="${2:?}"; shift 2 ;;
    --dropout) DROPOUT="${2:?}"; shift 2 ;;
    --feature-stages) FEATURE_STAGES="${2:?}"; shift 2 ;;
    --train-sampling) TRAIN_SAMPLING="${2:?}"; shift 2 ;;
    --cartesian-fonts-per-batch) CARTESIAN_FONTS_PER_BATCH="${2:?}"; shift 2 ;;
    --cartesian-chars-per-batch) CARTESIAN_CHARS_PER_BATCH="${2:?}"; shift 2 ;;
    --style-supcon-lambda) STYLE_SUPCON_LAMBDA="${2:?}"; shift 2 ;;
    --style-temperature) STYLE_TEMPERATURE="${2:?}"; shift 2 ;;
    --qualify-min-char-acc) QUALIFY_MIN_CHAR_ACC="${2:?}"; shift 2 ;;
    --qualify-min-style-margin) QUALIFY_MIN_STYLE_MARGIN="${2:?}"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${FONT_SPLIT_SEED}" ]]; then
  FONT_SPLIT_SEED="${SEED}"
fi

cd "${ROOT}"
mkdir -p logs checkpoints

[[ -z "${LOG_FILE}" ]] && LOG_FILE="logs/$(basename "${SAVE_DIR}").log"
[[ -z "${PID_FILE}" ]] && PID_FILE="logs/$(basename "${SAVE_DIR}").pid"
mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${PID_FILE}")" "$(dirname "${SAVE_DIR}")"

if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then
  echo "[run_font_perceptor_pretrain_colab] missing resume checkpoint: ${RESUME_CKPT}" >&2
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
    --log-every-steps "${LOG_EVERY}"
    --val-every-steps "${VAL_EVERY}"
    --val-max-batches "${VAL_MAX_BATCHES}"
    --lr "${LR}"
    --grad-clip-norm "${GRAD_CLIP_NORM}"
    --batch "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --max-fonts "${MAX_FONTS}"
    --image-size "${IMAGE_SIZE}"
    --base-channels "${BASE_CHANNELS}"
    --style-proj-dim "${STYLE_PROJ_DIM}"
    --dropout "${DROPOUT}"
    --feature-stages "${FEATURE_STAGES}"
    --train-sampling "${TRAIN_SAMPLING}"
    --cartesian-fonts-per-batch "${CARTESIAN_FONTS_PER_BATCH}"
    --cartesian-chars-per-batch "${CARTESIAN_CHARS_PER_BATCH}"
    --style-supcon-lambda "${STYLE_SUPCON_LAMBDA}"
    --style-temperature "${STYLE_TEMPERATURE}"
    --qualify-min-char-acc "${QUALIFY_MIN_CHAR_ACC}"
    --qualify-min-style-margin "${QUALIFY_MIN_STYLE_MARGIN}"
  )
  if [[ -n "${RESUME_CKPT}" ]]; then
    daemon_args+=(--resume "${RESUME_CKPT}")
  fi
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    daemon_args+=(-- "${EXTRA_ARGS[@]}")
  fi
  nohup bash "${SCRIPT_PATH}" "${daemon_args[@]}" > /dev/null 2>&1 < /dev/null &
  echo "$!" > "${PID_FILE}"
  echo "[run_font_perceptor_pretrain_colab] started daemon pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
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
  echo "[run_font_perceptor_pretrain_colab] received ${signal}, terminating process tree"
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
    print("[run_font_perceptor_pretrain_colab] torch.cuda.is_available()=False, falling back to cpu", file=sys.stderr)
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
        print(f"[run_font_perceptor_pretrain_colab] gpu_probe_failed cuda:{idx} error={exc}", file=sys.stderr)
        continue
    free_gb = free_bytes / float(1024 ** 3)
    total_gb = total_bytes / float(1024 ** 3)
    print(
        f"[run_font_perceptor_pretrain_colab] gpu_probe cuda:{idx} free_gb={free_gb:.2f} total_gb={total_gb:.2f}",
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
    ' "${LOG_FILE}" | grep -Eiq 'torch\.OutOfMemoryError|CUDA out of memory|out of memory'
    return $?
  fi
  grep -Eiq 'torch\.OutOfMemoryError|CUDA out of memory|out of memory' "${LOG_FILE}"
}

cmd_common=(
  "${PYTHON_BIN}" -u train_font_perceptor.py
  --data-root "${ROOT}"
  --save-dir "${SAVE_DIR}"
  --seed "${SEED}"
  --font-split "${FONT_SPLIT}"
  --font-split-seed "${FONT_SPLIT_SEED}"
  --font-train-ratio "${FONT_TRAIN_RATIO}"
  --lr "${LR}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --batch "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --max-fonts "${MAX_FONTS}"
  --image-size "${IMAGE_SIZE}"
  --base-channels "${BASE_CHANNELS}"
  --style-proj-dim "${STYLE_PROJ_DIM}"
  --dropout "${DROPOUT}"
  --feature-stages "${FEATURE_STAGES}"
  --train-sampling "${TRAIN_SAMPLING}"
  --cartesian-fonts-per-batch "${CARTESIAN_FONTS_PER_BATCH}"
  --cartesian-chars-per-batch "${CARTESIAN_CHARS_PER_BATCH}"
  --style-supcon-lambda "${STYLE_SUPCON_LAMBDA}"
  --style-temperature "${STYLE_TEMPERATURE}"
  --qualify-min-char-acc "${QUALIFY_MIN_CHAR_ACC}"
  --qualify-min-style-margin "${QUALIFY_MIN_STYLE_MARGIN}"
  --epochs "${EPOCHS}"
  --total-steps "${TARGET_STEPS}"
  --log-every-steps "${LOG_EVERY}"
  --val-every-steps "${VAL_EVERY}"
  --val-max-batches "${VAL_MAX_BATCHES}"
  --save-every-steps "${SAVE_EVERY}"
)

if [[ -n "${FONT_SPLIT_SEED}" ]]; then
  cmd_common+=(--font-split-seed "${FONT_SPLIT_SEED}")
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  cmd_common+=(--resume "${RESUME_CKPT}")
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd_common+=("${EXTRA_ARGS[@]}")
fi

echo "[run_font_perceptor_pretrain_colab] mode=font_perceptor"
echo "[run_font_perceptor_pretrain_colab] save_dir=${SAVE_DIR}"
echo "[run_font_perceptor_pretrain_colab] log_file=${LOG_FILE}"
echo "[run_font_perceptor_pretrain_colab] requested_device=${DEVICE_ARG} seed=${SEED}"
echo "[run_font_perceptor_pretrain_colab] resume=${RESUME_CKPT:-<none>}"
echo "[run_font_perceptor_pretrain_colab] batch=${BATCH_SIZE} lr=${LR} grad_clip_norm=${GRAD_CLIP_NORM}"
echo "[run_font_perceptor_pretrain_colab] base_channels=${BASE_CHANNELS} style_proj_dim=${STYLE_PROJ_DIM} dropout=${DROPOUT}"
echo "[run_font_perceptor_pretrain_colab] feature_stages=${FEATURE_STAGES}"
echo "[run_font_perceptor_pretrain_colab] train_sampling=${TRAIN_SAMPLING} cartesian_fonts_per_batch=${CARTESIAN_FONTS_PER_BATCH} cartesian_chars_per_batch=${CARTESIAN_CHARS_PER_BATCH}"
echo "[run_font_perceptor_pretrain_colab] style_supcon_lambda=${STYLE_SUPCON_LAMBDA} style_temperature=${STYLE_TEMPERATURE}"
echo "[run_font_perceptor_pretrain_colab] qualify_min_char_acc=${QUALIFY_MIN_CHAR_ACC} qualify_min_style_margin=${QUALIFY_MIN_STYLE_MARGIN}"

attempt=1
while true; do
  launch_device="$(select_launch_device)"
  attempt_marker="__run_font_perceptor_attempt_${attempt}_$(date +%s)"
  cmd=("${cmd_common[@]}")
  cmd+=(--device "${launch_device}")

  echo "[run_font_perceptor_pretrain_colab] ${attempt_marker}"
  echo "[run_font_perceptor_pretrain_colab] attempt=${attempt} launch_device=${launch_device}"
  printf '[run_font_perceptor_pretrain_colab] cmd='
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
    echo "[run_font_perceptor_pretrain_colab] attempt=${attempt} failed with OOM, sleeping 60s before retry"
    attempt=$((attempt + 1))
    sleep 60
    continue
  fi

  echo "[run_font_perceptor_pretrain_colab] attempt=${attempt} failed with non-OOM status=${status}, aborting"
  exit "${status}"
done

#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/yangximing/code/ass5-exp6/DiffuFont"
PYTHON_BIN="/scratch/yangximing/miniconda3/envs/sg3/bin/python"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
LOG_MAX_LINES=1500

RUN_MODE="daemon"
LOG_FILE=""
PID_FILE=""
SAVE_DIR="checkpoints/vae_pretrain_$(date '+%Y%m%d_%H%M%S')"

RESUME_CKPT=""
DEVICE_ARG="auto"
SEED=42
FONT_SPLIT="train"
FONT_SPLIT_SEED=""
FONT_TRAIN_RATIO="0.95"

TARGET_STEPS=80000
SAVE_EVERY=2000
SAMPLE_EVERY=500
LR="2e-4"
LR_WARMUP_STEPS=2000
LR_MIN_SCALE="0.1"
GRAD_CLIP_NORM="1.0"

STYLE_REF_COUNT=8
STYLE_REF_COUNT_MIN=8
STYLE_REF_COUNT_MAX=8
BATCH_SIZE=64
NUM_WORKERS=8
MAX_FONTS=0
IMAGE_SIZE=128

LATENT_CHANNELS=6
LATENT_SIZE=16
VAE_BOTTLENECK_CHANNELS=192
VAE_ENCODER_16X16_BLOCKS=2
VAE_DECODER_16X16_BLOCKS=2
VAE_DECODER_TAIL_BLOCKS=1
ENCODER_PATCH_SIZE=8
ENCODER_HIDDEN_DIM=512
ENCODER_DEPTH=4
ENCODER_HEADS=8
DIT_HIDDEN_DIM=512
DIT_DEPTH=12
DIT_HEADS=8
DIT_MLP_RATIO="4.0"
CONTENT_FUSION_START=0
CONTENT_FUSION_END=8
STYLE_FUSION_START=6
STYLE_FUSION_END=12
TRAIN_SAMPLING="shuffle"
CARTESIAN_FONTS_PER_BATCH=8
CARTESIAN_CHARS_PER_BATCH=8

VAE_LAMBDA_REC="1.0"
VAE_LAMBDA_PERC="0.18"
VAE_LAMBDA_KL="2e-4"
VAE_KL_WARMUP_STEPS=10000
VAE_LATENT_MEAN_WEIGHT="0.001"
VAE_LATENT_STD_WEIGHT="0.001"
VAE_LATENT_CORR_WEIGHT="0.0005"
VAE_LATENT_STD_TARGET="1.0"
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
    --target-steps) TARGET_STEPS="${2:?}"; shift 2 ;;
    --save-every-steps) SAVE_EVERY="${2:?}"; shift 2 ;;
    --sample-every-steps) SAMPLE_EVERY="${2:?}"; shift 2 ;;
    --lr) LR="${2:?}"; shift 2 ;;
    --lr-warmup-steps) LR_WARMUP_STEPS="${2:?}"; shift 2 ;;
    --lr-min-scale) LR_MIN_SCALE="${2:?}"; shift 2 ;;
    --grad-clip-norm) GRAD_CLIP_NORM="${2:?}"; shift 2 ;;
    --style-ref-count) STYLE_REF_COUNT="${2:?}"; shift 2 ;;
    --style-ref-count-min) STYLE_REF_COUNT_MIN="${2:?}"; shift 2 ;;
    --style-ref-count-max) STYLE_REF_COUNT_MAX="${2:?}"; shift 2 ;;
    --batch) BATCH_SIZE="${2:?}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:?}"; shift 2 ;;
    --max-fonts) MAX_FONTS="${2:?}"; shift 2 ;;
    --image-size) IMAGE_SIZE="${2:?}"; shift 2 ;;
    --latent-channels) LATENT_CHANNELS="${2:?}"; shift 2 ;;
    --latent-size) LATENT_SIZE="${2:?}"; shift 2 ;;
    --vae-bottleneck-channels) VAE_BOTTLENECK_CHANNELS="${2:?}"; shift 2 ;;
    --vae-encoder-16x16-blocks) VAE_ENCODER_16X16_BLOCKS="${2:?}"; shift 2 ;;
    --vae-decoder-16x16-blocks) VAE_DECODER_16X16_BLOCKS="${2:?}"; shift 2 ;;
    --vae-decoder-tail-blocks) VAE_DECODER_TAIL_BLOCKS="${2:?}"; shift 2 ;;
    --encoder-patch-size) ENCODER_PATCH_SIZE="${2:?}"; shift 2 ;;
    --encoder-hidden-dim) ENCODER_HIDDEN_DIM="${2:?}"; shift 2 ;;
    --encoder-depth) ENCODER_DEPTH="${2:?}"; shift 2 ;;
    --encoder-heads) ENCODER_HEADS="${2:?}"; shift 2 ;;
    --dit-hidden-dim) DIT_HIDDEN_DIM="${2:?}"; shift 2 ;;
    --dit-depth) DIT_DEPTH="${2:?}"; shift 2 ;;
    --dit-heads) DIT_HEADS="${2:?}"; shift 2 ;;
    --dit-mlp-ratio) DIT_MLP_RATIO="${2:?}"; shift 2 ;;
    --content-fusion-start) CONTENT_FUSION_START="${2:?}"; shift 2 ;;
    --content-fusion-end) CONTENT_FUSION_END="${2:?}"; shift 2 ;;
    --style-fusion-start) STYLE_FUSION_START="${2:?}"; shift 2 ;;
    --style-fusion-end) STYLE_FUSION_END="${2:?}"; shift 2 ;;
    --train-sampling) TRAIN_SAMPLING="${2:?}"; shift 2 ;;
    --cartesian-fonts-per-batch) CARTESIAN_FONTS_PER_BATCH="${2:?}"; shift 2 ;;
    --cartesian-chars-per-batch) CARTESIAN_CHARS_PER_BATCH="${2:?}"; shift 2 ;;
    --vae-lambda-rec) VAE_LAMBDA_REC="${2:?}"; shift 2 ;;
    --vae-lambda-perc) VAE_LAMBDA_PERC="${2:?}"; shift 2 ;;
    --vae-lambda-kl) VAE_LAMBDA_KL="${2:?}"; shift 2 ;;
    --vae-kl-warmup-steps) VAE_KL_WARMUP_STEPS="${2:?}"; shift 2 ;;
    --vae-latent-mean-weight) VAE_LATENT_MEAN_WEIGHT="${2:?}"; shift 2 ;;
    --vae-latent-std-weight) VAE_LATENT_STD_WEIGHT="${2:?}"; shift 2 ;;
    --vae-latent-corr-weight) VAE_LATENT_CORR_WEIGHT="${2:?}"; shift 2 ;;
    --vae-latent-std-target) VAE_LATENT_STD_TARGET="${2:?}"; shift 2 ;;
    --) shift; EXTRA_TRAIN_ARGS+=("$@"); break ;;
    *) EXTRA_TRAIN_ARGS+=("$1"); shift ;;
  esac
done

cd "${ROOT}"
mkdir -p logs checkpoints

[[ -z "${LOG_FILE}" ]] && LOG_FILE="logs/$(basename "${SAVE_DIR}").log"
[[ -z "${PID_FILE}" ]] && PID_FILE="logs/$(basename "${SAVE_DIR}").pid"

if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then
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
    --lr "${LR}"
    --lr-warmup-steps "${LR_WARMUP_STEPS}"
    --lr-min-scale "${LR_MIN_SCALE}"
    --grad-clip-norm "${GRAD_CLIP_NORM}"
    --style-ref-count "${STYLE_REF_COUNT}"
    --style-ref-count-min "${STYLE_REF_COUNT_MIN}"
    --style-ref-count-max "${STYLE_REF_COUNT_MAX}"
    --batch "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --max-fonts "${MAX_FONTS}"
    --image-size "${IMAGE_SIZE}"
    --latent-channels "${LATENT_CHANNELS}"
    --latent-size "${LATENT_SIZE}"
    --vae-bottleneck-channels "${VAE_BOTTLENECK_CHANNELS}"
    --vae-encoder-16x16-blocks "${VAE_ENCODER_16X16_BLOCKS}"
    --vae-decoder-16x16-blocks "${VAE_DECODER_16X16_BLOCKS}"
    --vae-decoder-tail-blocks "${VAE_DECODER_TAIL_BLOCKS}"
    --encoder-patch-size "${ENCODER_PATCH_SIZE}"
    --encoder-hidden-dim "${ENCODER_HIDDEN_DIM}"
    --encoder-depth "${ENCODER_DEPTH}"
    --encoder-heads "${ENCODER_HEADS}"
    --dit-hidden-dim "${DIT_HIDDEN_DIM}"
    --dit-depth "${DIT_DEPTH}"
    --dit-heads "${DIT_HEADS}"
    --dit-mlp-ratio "${DIT_MLP_RATIO}"
    --content-fusion-start "${CONTENT_FUSION_START}"
    --content-fusion-end "${CONTENT_FUSION_END}"
    --style-fusion-start "${STYLE_FUSION_START}"
    --style-fusion-end "${STYLE_FUSION_END}"
    --train-sampling "${TRAIN_SAMPLING}"
    --cartesian-fonts-per-batch "${CARTESIAN_FONTS_PER_BATCH}"
    --cartesian-chars-per-batch "${CARTESIAN_CHARS_PER_BATCH}"
    --vae-lambda-rec "${VAE_LAMBDA_REC}"
    --vae-lambda-perc "${VAE_LAMBDA_PERC}"
    --vae-lambda-kl "${VAE_LAMBDA_KL}"
    --vae-kl-warmup-steps "${VAE_KL_WARMUP_STEPS}"
    --vae-latent-mean-weight "${VAE_LATENT_MEAN_WEIGHT}"
    --vae-latent-std-weight "${VAE_LATENT_STD_WEIGHT}"
    --vae-latent-corr-weight "${VAE_LATENT_CORR_WEIGHT}"
    --vae-latent-std-target "${VAE_LATENT_STD_TARGET}"
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
  echo "[run_vae_pretrain_colab] started daemon pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
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
  echo "[run_vae_pretrain_colab] received ${signal}, terminating process tree"
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
    print("[run_vae_pretrain_colab] torch.cuda.is_available()=False, falling back to cpu", file=sys.stderr)
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
        print(f"[run_vae_pretrain_colab] gpu_probe_failed cuda:{idx} error={exc}", file=sys.stderr)
        continue
    free_gb = free_bytes / float(1024 ** 3)
    total_gb = total_bytes / float(1024 ** 3)
    print(
        f"[run_vae_pretrain_colab] gpu_probe cuda:{idx} free_gb={free_gb:.2f} total_gb={total_gb:.2f}",
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
  --stage vae
  --data-root "${ROOT}"
  --save-dir "${SAVE_DIR}"
  --seed "${SEED}"
  --font-split "${FONT_SPLIT}"
  --font-train-ratio "${FONT_TRAIN_RATIO}"
  --lr "${LR}"
  --lr-warmup-steps "${LR_WARMUP_STEPS}"
  --lr-min-scale "${LR_MIN_SCALE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --batch "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --style-ref-count "${STYLE_REF_COUNT}"
  --style-ref-count-min "${STYLE_REF_COUNT_MIN}"
  --style-ref-count-max "${STYLE_REF_COUNT_MAX}"
  --max-fonts "${MAX_FONTS}"
  --image-size "${IMAGE_SIZE}"
  --latent-channels "${LATENT_CHANNELS}"
  --latent-size "${LATENT_SIZE}"
  --vae-bottleneck-channels "${VAE_BOTTLENECK_CHANNELS}"
  --vae-encoder-16x16-blocks "${VAE_ENCODER_16X16_BLOCKS}"
  --vae-decoder-16x16-blocks "${VAE_DECODER_16X16_BLOCKS}"
  --vae-decoder-tail-blocks "${VAE_DECODER_TAIL_BLOCKS}"
  --encoder-patch-size "${ENCODER_PATCH_SIZE}"
  --encoder-hidden-dim "${ENCODER_HIDDEN_DIM}"
  --encoder-depth "${ENCODER_DEPTH}"
  --encoder-heads "${ENCODER_HEADS}"
  --dit-hidden-dim "${DIT_HIDDEN_DIM}"
  --dit-depth "${DIT_DEPTH}"
  --dit-heads "${DIT_HEADS}"
  --dit-mlp-ratio "${DIT_MLP_RATIO}"
  --content-fusion-start "${CONTENT_FUSION_START}"
  --content-fusion-end "${CONTENT_FUSION_END}"
  --style-fusion-start "${STYLE_FUSION_START}"
  --style-fusion-end "${STYLE_FUSION_END}"
  --train-sampling "${TRAIN_SAMPLING}"
  --cartesian-fonts-per-batch "${CARTESIAN_FONTS_PER_BATCH}"
  --cartesian-chars-per-batch "${CARTESIAN_CHARS_PER_BATCH}"
  --epochs 1000000
  --total-steps "${TARGET_STEPS}"
  --log-every-steps 100
  --val-every-steps 100
  --save-every-steps "${SAVE_EVERY}"
  --sample-every-steps "${SAMPLE_EVERY}"
  --vae-lambda-rec "${VAE_LAMBDA_REC}"
  --vae-lambda-perc "${VAE_LAMBDA_PERC}"
  --vae-lambda-kl "${VAE_LAMBDA_KL}"
  --vae-kl-warmup-steps "${VAE_KL_WARMUP_STEPS}"
  --vae-latent-mean-weight "${VAE_LATENT_MEAN_WEIGHT}"
  --vae-latent-std-weight "${VAE_LATENT_STD_WEIGHT}"
  --vae-latent-corr-weight "${VAE_LATENT_CORR_WEIGHT}"
  --vae-latent-std-target "${VAE_LATENT_STD_TARGET}"
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

echo "[run_vae_pretrain_colab] stage=vae"
echo "[run_vae_pretrain_colab] save_dir=${SAVE_DIR}"
echo "[run_vae_pretrain_colab] log_file=${LOG_FILE}"
echo "[run_vae_pretrain_colab] requested_device=${DEVICE_ARG} seed=${SEED}"
echo "[run_vae_pretrain_colab] batch=${BATCH_SIZE} lr=${LR} lr_warmup_steps=${LR_WARMUP_STEPS} lr_min_scale=${LR_MIN_SCALE} grad_clip_norm=${GRAD_CLIP_NORM}"
echo "[run_vae_pretrain_colab] style_ref_count=${STYLE_REF_COUNT} style_ref_count_min=${STYLE_REF_COUNT_MIN} style_ref_count_max=${STYLE_REF_COUNT_MAX}"
echo "[run_vae_pretrain_colab] vae_lambda_rec=${VAE_LAMBDA_REC} vae_lambda_perc=${VAE_LAMBDA_PERC} vae_lambda_kl=${VAE_LAMBDA_KL}"
echo "[run_vae_pretrain_colab] vae_kl_warmup_steps=${VAE_KL_WARMUP_STEPS} latent_mean_weight=${VAE_LATENT_MEAN_WEIGHT} latent_std_weight=${VAE_LATENT_STD_WEIGHT} latent_corr_weight=${VAE_LATENT_CORR_WEIGHT}"
echo "[run_vae_pretrain_colab] content_layers=[${CONTENT_FUSION_START},${CONTENT_FUSION_END}) style_layers=[${STYLE_FUSION_START},${STYLE_FUSION_END})"
echo "[run_vae_pretrain_colab] train_sampling=${TRAIN_SAMPLING} cartesian_fonts_per_batch=${CARTESIAN_FONTS_PER_BATCH} cartesian_chars_per_batch=${CARTESIAN_CHARS_PER_BATCH}"

attempt=1
while true; do
  launch_device="$(select_launch_device)"
  attempt_marker="__run_vae_pretrain_attempt_${attempt}_$(date +%s)"
  cmd=("${cmd_common[@]}")
  cmd+=(--device "${launch_device}")

  echo "[run_vae_pretrain_colab] ${attempt_marker}"
  echo "[run_vae_pretrain_colab] attempt=${attempt} launch_device=${launch_device}"
  printf '[run_vae_pretrain_colab] cmd='
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
    echo "[run_vae_pretrain_colab] attempt=${attempt} failed with OOM, sleeping 60s before retry"
    attempt=$((attempt + 1))
    sleep 60
    continue
  fi

  echo "[run_vae_pretrain_colab] attempt=${attempt} failed with non-OOM status=${status}, aborting"
  exit "${status}"
done

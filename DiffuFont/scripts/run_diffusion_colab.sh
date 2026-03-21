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
VAE_CKPT="/scratch/yangximing/code/ass5-exp6/DiffuFont/checkpoints/vae_pretrain_20260321_202403/best.pt"
STYLE_CKPT="/scratch/yangximing/code/ass5-exp6/DiffuFont/checkpoints/style_pretrain_20260321_101735/best.pt"
DEVICE_ARG="auto"
SEED=42
FONT_SPLIT="train"
FONT_SPLIT_SEED=""
FONT_TRAIN_RATIO="0.95"

TARGET_STEPS=150000
SAVE_EVERY=5000
SAMPLE_EVERY=300
LR="1e-4"
LR_WARMUP_STEPS=2000
LR_MIN_SCALE="0.1"

STYLE_REF_COUNT=8
BATCH_SIZE=64
NUM_WORKERS=8
MAX_FONTS=0
IMAGE_SIZE=128

LATENT_CHANNELS=10
LATENT_SIZE=16
ENCODER_PATCH_SIZE=8
ENCODER_HIDDEN_DIM=512
ENCODER_DEPTH=4
ENCODER_HEADS=8
DIT_HIDDEN_DIM=512
DIT_DEPTH=16
DIT_HEADS=8
DIT_MLP_RATIO="4.0"

STYLE_MID_TOKENS_PER_REF=12
LOCAL_STYLE_TOKENS_PER_REF=24
STYLE_RESIDUAL_TOKENS=8
STYLE_RESIDUAL_GATE_INIT="0.3"
CONTENT_CROSS_ATTN_LAYERS="8"
STYLE_CROSS_ATTN_EVERY_N_LAYERS=1

TRAIN_VAE_JOINTLY="0"
TRAIN_STYLE_JOINTLY="1"
STYLE_LR_SCALE="1.0"
STYLE_LR_WARMUP_STEPS=10000
FLOW_LAMBDA_IMG_L1="0.2"
FLOW_LAMBDA_IMG_PERC="0.02"
FLOW_DIFFICULTY_WARMUP_STEPS=10000
FLOW_DIFFICULTY_EMA_DECAY="0.99"
FLOW_DIFFICULTY_ALPHA="0.5"
FLOW_DIFFICULTY_MIN_WEIGHT="0.7"
FLOW_DIFFICULTY_MAX_WEIGHT="1.5"
FLOW_DIFFICULTY_REFRESH_EVERY_STEPS=1000
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) RUN_MODE="foreground"; shift ;;
    --daemon) RUN_MODE="daemon"; shift ;;
    --log-file) LOG_FILE="${2:?}"; shift 2 ;;
    --pid-file) PID_FILE="${2:?}"; shift 2 ;;
    --save-dir) SAVE_DIR="${2:?}"; shift 2 ;;
    --resume) RESUME_CKPT="${2:?}"; shift 2 ;;
    --vae-checkpoint) VAE_CKPT="${2:?}"; shift 2 ;;
    --style-checkpoint) STYLE_CKPT="${2:?}"; shift 2 ;;
    --train-vae-jointly) TRAIN_VAE_JOINTLY="1"; shift ;;
    --train-style-jointly) TRAIN_STYLE_JOINTLY="1"; shift ;;
    --device) DEVICE_ARG="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --font-split) FONT_SPLIT="${2:?}"; shift 2 ;;
    --font-split-seed) FONT_SPLIT_SEED="${2:?}"; shift 2 ;;
    --font-train-ratio) FONT_TRAIN_RATIO="${2:?}"; shift 2 ;;
    --target-steps) TARGET_STEPS="${2:?}"; shift 2 ;;
    --save-every-steps) SAVE_EVERY="${2:?}"; shift 2 ;;
    --sample-every-steps) SAMPLE_EVERY="${2:?}"; shift 2 ;;
    --lr-warmup-steps) LR_WARMUP_STEPS="${2:?}"; shift 2 ;;
    --lr-min-scale) LR_MIN_SCALE="${2:?}"; shift 2 ;;
    --style-lr-scale) STYLE_LR_SCALE="${2:?}"; shift 2 ;;
    --style-lr-warmup-steps) STYLE_LR_WARMUP_STEPS="${2:?}"; shift 2 ;;
    --flow-lambda-img-l1) FLOW_LAMBDA_IMG_L1="${2:?}"; shift 2 ;;
    --flow-lambda-img-perc) FLOW_LAMBDA_IMG_PERC="${2:?}"; shift 2 ;;
    --flow-difficulty-warmup-steps) FLOW_DIFFICULTY_WARMUP_STEPS="${2:?}"; shift 2 ;;
    --flow-difficulty-ema-decay) FLOW_DIFFICULTY_EMA_DECAY="${2:?}"; shift 2 ;;
    --flow-difficulty-alpha) FLOW_DIFFICULTY_ALPHA="${2:?}"; shift 2 ;;
    --flow-difficulty-min-weight) FLOW_DIFFICULTY_MIN_WEIGHT="${2:?}"; shift 2 ;;
    --flow-difficulty-max-weight) FLOW_DIFFICULTY_MAX_WEIGHT="${2:?}"; shift 2 ;;
    --flow-difficulty-refresh-every-steps) FLOW_DIFFICULTY_REFRESH_EVERY_STEPS="${2:?}"; shift 2 ;;
    --style-ref-count) STYLE_REF_COUNT="${2:?}"; shift 2 ;;
    --batch) BATCH_SIZE="${2:?}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:?}"; shift 2 ;;
    --max-fonts) MAX_FONTS="${2:?}"; shift 2 ;;
    --image-size) IMAGE_SIZE="${2:?}"; shift 2 ;;
    --latent-channels) LATENT_CHANNELS="${2:?}"; shift 2 ;;
    --latent-size) LATENT_SIZE="${2:?}"; shift 2 ;;
    --encoder-patch-size) ENCODER_PATCH_SIZE="${2:?}"; shift 2 ;;
    --encoder-hidden-dim) ENCODER_HIDDEN_DIM="${2:?}"; shift 2 ;;
    --encoder-depth) ENCODER_DEPTH="${2:?}"; shift 2 ;;
    --encoder-heads) ENCODER_HEADS="${2:?}"; shift 2 ;;
    --dit-hidden-dim) DIT_HIDDEN_DIM="${2:?}"; shift 2 ;;
    --dit-depth) DIT_DEPTH="${2:?}"; shift 2 ;;
    --dit-heads) DIT_HEADS="${2:?}"; shift 2 ;;
    --dit-mlp-ratio) DIT_MLP_RATIO="${2:?}"; shift 2 ;;
    --style-mid-tokens-per-ref) STYLE_MID_TOKENS_PER_REF="${2:?}"; shift 2 ;;
    --local-style-tokens-per-ref) LOCAL_STYLE_TOKENS_PER_REF="${2:?}"; shift 2 ;;
    --style-residual-tokens) STYLE_RESIDUAL_TOKENS="${2:?}"; shift 2 ;;
    --style-residual-gate-init) STYLE_RESIDUAL_GATE_INIT="${2:?}"; shift 2 ;;
    --content-cross-attn-layers) CONTENT_CROSS_ATTN_LAYERS="${2:?}"; shift 2 ;;
    --style-cross-attn-every-n-layers) STYLE_CROSS_ATTN_EVERY_N_LAYERS="${2:?}"; shift 2 ;;
    --lr) LR="${2:?}"; shift 2 ;;
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
if [[ -n "${VAE_CKPT}" && ! -f "${VAE_CKPT}" ]]; then
  exit 2
fi
if [[ -n "${STYLE_CKPT}" && ! -f "${STYLE_CKPT}" ]]; then
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
    --style-ref-count "${STYLE_REF_COUNT}"
    --batch "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --max-fonts "${MAX_FONTS}"
    --image-size "${IMAGE_SIZE}"
    --latent-channels "${LATENT_CHANNELS}"
    --latent-size "${LATENT_SIZE}"
    --encoder-patch-size "${ENCODER_PATCH_SIZE}"
    --encoder-hidden-dim "${ENCODER_HIDDEN_DIM}"
    --encoder-depth "${ENCODER_DEPTH}"
    --encoder-heads "${ENCODER_HEADS}"
    --dit-hidden-dim "${DIT_HIDDEN_DIM}"
    --dit-depth "${DIT_DEPTH}"
    --dit-heads "${DIT_HEADS}"
    --dit-mlp-ratio "${DIT_MLP_RATIO}"
    --style-mid-tokens-per-ref "${STYLE_MID_TOKENS_PER_REF}"
    --local-style-tokens-per-ref "${LOCAL_STYLE_TOKENS_PER_REF}"
    --style-residual-tokens "${STYLE_RESIDUAL_TOKENS}"
    --style-residual-gate-init "${STYLE_RESIDUAL_GATE_INIT}"
    --content-cross-attn-layers "${CONTENT_CROSS_ATTN_LAYERS}"
    --style-cross-attn-every-n-layers "${STYLE_CROSS_ATTN_EVERY_N_LAYERS}"
    --lr "${LR}"
    --lr-warmup-steps "${LR_WARMUP_STEPS}"
    --lr-min-scale "${LR_MIN_SCALE}"
    --style-lr-scale "${STYLE_LR_SCALE}"
    --style-lr-warmup-steps "${STYLE_LR_WARMUP_STEPS}"
    --flow-lambda-img-l1 "${FLOW_LAMBDA_IMG_L1}"
    --flow-lambda-img-perc "${FLOW_LAMBDA_IMG_PERC}"
    --flow-difficulty-warmup-steps "${FLOW_DIFFICULTY_WARMUP_STEPS}"
    --flow-difficulty-ema-decay "${FLOW_DIFFICULTY_EMA_DECAY}"
    --flow-difficulty-alpha "${FLOW_DIFFICULTY_ALPHA}"
    --flow-difficulty-min-weight "${FLOW_DIFFICULTY_MIN_WEIGHT}"
    --flow-difficulty-max-weight "${FLOW_DIFFICULTY_MAX_WEIGHT}"
    --flow-difficulty-refresh-every-steps "${FLOW_DIFFICULTY_REFRESH_EVERY_STEPS}"
  )
  if [[ -n "${FONT_SPLIT_SEED}" ]]; then
    daemon_args+=(--font-split-seed "${FONT_SPLIT_SEED}")
  fi
  if [[ -n "${RESUME_CKPT}" ]]; then
    daemon_args+=(--resume "${RESUME_CKPT}")
  fi
  if [[ -n "${VAE_CKPT}" ]]; then
    daemon_args+=(--vae-checkpoint "${VAE_CKPT}")
  fi
  if [[ -n "${STYLE_CKPT}" ]]; then
    daemon_args+=(--style-checkpoint "${STYLE_CKPT}")
  fi
  if [[ "${TRAIN_VAE_JOINTLY}" == "1" ]]; then
    daemon_args+=(--train-vae-jointly)
  fi
  if [[ "${TRAIN_STYLE_JOINTLY}" == "1" ]]; then
    daemon_args+=(--train-style-jointly)
  fi
  if [[ "${#EXTRA_TRAIN_ARGS[@]}" -gt 0 ]]; then
    daemon_args+=(-- "${EXTRA_TRAIN_ARGS[@]}")
  fi
  nohup bash "${SCRIPT_PATH}" "${daemon_args[@]}" > /dev/null 2>&1 < /dev/null &
  echo "$!" > "${PID_FILE}"
  echo "[run_diffusion_colab] started daemon pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
  exit 0
fi

if [[ "${TRAIN_VAE_JOINTLY}" != "1" && -z "${RESUME_CKPT}" && -z "${VAE_CKPT}" ]]; then
  exit 2
fi
if [[ "${TRAIN_STYLE_JOINTLY}" != "1" && -z "${RESUME_CKPT}" && -z "${STYLE_CKPT}" ]]; then
  exit 2
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
  --stage flow
  --data-root "${ROOT}"
  --save-dir "${SAVE_DIR}"
  --seed "${SEED}"
  --font-split "${FONT_SPLIT}"
  --font-train-ratio "${FONT_TRAIN_RATIO}"
  --lr "${LR}"
  --lr-warmup-steps "${LR_WARMUP_STEPS}"
  --lr-min-scale "${LR_MIN_SCALE}"
  --batch "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --style-ref-count "${STYLE_REF_COUNT}"
  --max-fonts "${MAX_FONTS}"
  --image-size "${IMAGE_SIZE}"
  --latent-channels "${LATENT_CHANNELS}"
  --latent-size "${LATENT_SIZE}"
  --encoder-patch-size "${ENCODER_PATCH_SIZE}"
  --encoder-hidden-dim "${ENCODER_HIDDEN_DIM}"
  --encoder-depth "${ENCODER_DEPTH}"
  --encoder-heads "${ENCODER_HEADS}"
  --dit-hidden-dim "${DIT_HIDDEN_DIM}"
  --dit-depth "${DIT_DEPTH}"
  --dit-heads "${DIT_HEADS}"
  --dit-mlp-ratio "${DIT_MLP_RATIO}"
  --style-mid-tokens-per-ref "${STYLE_MID_TOKENS_PER_REF}"
  --local-style-tokens-per-ref "${LOCAL_STYLE_TOKENS_PER_REF}"
  --style-residual-tokens "${STYLE_RESIDUAL_TOKENS}"
  --style-residual-gate-init "${STYLE_RESIDUAL_GATE_INIT}"
  --content-cross-attn-layers "${CONTENT_CROSS_ATTN_LAYERS}"
  --style-cross-attn-every-n-layers "${STYLE_CROSS_ATTN_EVERY_N_LAYERS}"
  --style-lr-scale "${STYLE_LR_SCALE}"
  --style-lr-warmup-steps "${STYLE_LR_WARMUP_STEPS}"
  --flow-lambda-img-l1 "${FLOW_LAMBDA_IMG_L1}"
  --flow-lambda-img-perc "${FLOW_LAMBDA_IMG_PERC}"
  --flow-difficulty-warmup-steps "${FLOW_DIFFICULTY_WARMUP_STEPS}"
  --flow-difficulty-ema-decay "${FLOW_DIFFICULTY_EMA_DECAY}"
  --flow-difficulty-alpha "${FLOW_DIFFICULTY_ALPHA}"
  --flow-difficulty-min-weight "${FLOW_DIFFICULTY_MIN_WEIGHT}"
  --flow-difficulty-max-weight "${FLOW_DIFFICULTY_MAX_WEIGHT}"
  --flow-difficulty-refresh-every-steps "${FLOW_DIFFICULTY_REFRESH_EVERY_STEPS}"
  --epochs 1000000
  --total-steps "${TARGET_STEPS}"
  --log-every-steps 100
  --val-every-steps 100
  --save-every-steps "${SAVE_EVERY}"
  --sample-every-steps "${SAMPLE_EVERY}"
)

if [[ -n "${FONT_SPLIT_SEED}" ]]; then
  cmd_common+=(--font-split-seed "${FONT_SPLIT_SEED}")
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  cmd_common+=(--resume "${RESUME_CKPT}")
fi
if [[ -n "${VAE_CKPT}" ]]; then
  cmd_common+=(--vae-checkpoint "${VAE_CKPT}")
fi
if [[ -n "${STYLE_CKPT}" ]]; then
  cmd_common+=(--style-checkpoint "${STYLE_CKPT}")
fi
if [[ "${TRAIN_VAE_JOINTLY}" == "1" ]]; then
  cmd_common+=(--train-vae-jointly)
fi
if [[ "${TRAIN_STYLE_JOINTLY}" == "1" ]]; then
  cmd_common+=(--train-style-jointly)
fi
if [[ "${#EXTRA_TRAIN_ARGS[@]}" -gt 0 ]]; then
  cmd_common+=("${EXTRA_TRAIN_ARGS[@]}")
fi

echo "[run_diffusion_colab] stage=flow"
echo "[run_diffusion_colab] save_dir=${SAVE_DIR}"
echo "[run_diffusion_colab] log_file=${LOG_FILE}"
echo "[run_diffusion_colab] requested_device=${DEVICE_ARG} seed=${SEED}"
echo "[run_diffusion_colab] batch=${BATCH_SIZE} lr=${LR} lr_warmup_steps=${LR_WARMUP_STEPS} lr_min_scale=${LR_MIN_SCALE}"
echo "[run_diffusion_colab] latent_channels=${LATENT_CHANNELS} latent_size=${LATENT_SIZE}"
echo "[run_diffusion_colab] vae_checkpoint=${VAE_CKPT:-<none>}"
echo "[run_diffusion_colab] style_checkpoint=${STYLE_CKPT:-<none>}"
echo "[run_diffusion_colab] style_lr_scale=${STYLE_LR_SCALE} style_lr_warmup_steps=${STYLE_LR_WARMUP_STEPS}"
echo "[run_diffusion_colab] flow_lambda_img_l1=${FLOW_LAMBDA_IMG_L1} flow_lambda_img_perc=${FLOW_LAMBDA_IMG_PERC}"
echo "[run_diffusion_colab] flow_difficulty_warmup_steps=${FLOW_DIFFICULTY_WARMUP_STEPS} flow_difficulty_refresh_every_steps=${FLOW_DIFFICULTY_REFRESH_EVERY_STEPS} flow_difficulty_ema_decay=${FLOW_DIFFICULTY_EMA_DECAY} flow_difficulty_alpha=${FLOW_DIFFICULTY_ALPHA} flow_difficulty_weight_range=[${FLOW_DIFFICULTY_MIN_WEIGHT},${FLOW_DIFFICULTY_MAX_WEIGHT}]"
echo "[run_diffusion_colab] style_mid_tokens_per_ref=${STYLE_MID_TOKENS_PER_REF} local_style_tokens_per_ref=${LOCAL_STYLE_TOKENS_PER_REF} style_residual_tokens=${STYLE_RESIDUAL_TOKENS} style_residual_gate_init=${STYLE_RESIDUAL_GATE_INIT}"
echo "[run_diffusion_colab] content_cross_attn_layers=${CONTENT_CROSS_ATTN_LAYERS} style_cross_attn_every_n_layers=${STYLE_CROSS_ATTN_EVERY_N_LAYERS}"

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

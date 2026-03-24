#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/yangximing/code/ass5-exp6/DiffuFont"
PYTHON_BIN="/scratch/yangximing/miniconda3/envs/sg3/bin/python"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

RUN_MODE="daemon"
LOG_FILE=""
PID_FILE=""
SAVE_DIR="checkpoints/style_pretrain_$(date '+%Y%m%d_%H%M%S')"

RESUME_CKPT=""
DEVICE_ARG="cuda:0"
SEED=42
FONT_SPLIT="train"
FONT_SPLIT_SEED=""
FONT_TRAIN_RATIO="0.95"

TARGET_STEPS=40000
SAVE_EVERY=5000
LR="2e-4"

BATCH_SIZE=64
NUM_WORKERS=8
MAX_FONTS=0
STYLE_REF_COUNT=8
IMAGE_SIZE=128

LATENT_CHANNELS=12
LATENT_SIZE=16
ENCODER_PATCH_SIZE=8
ENCODER_HIDDEN_DIM=512
ENCODER_DEPTH=4
ENCODER_HEADS=8
DIT_HIDDEN_DIM=512
DIT_DEPTH=12
DIT_HEADS=8
DIT_MLP_RATIO="4.0"
STYLE_TOKENS_PER_REF=8
CONTENT_CROSS_ATTN_INDICES="0,1,2,3,4,5,8,10"
STYLE_TOKEN_CROSS_ATTN_INDICES="6,7,8,9,10,11"
VAE_BOTTLENECK_CHANNELS=192
VAE_ENCODER_16X16_BLOCKS=2
VAE_DECODER_16X16_BLOCKS=2
VAE_DECODER_TAIL_BLOCKS=1
CONTRASTIVE_PROJ_DIM=128
CONTRASTIVE_TEMPERATURE="0.1"
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
    --lr) LR="${2:?}"; shift 2 ;;
    --batch) BATCH_SIZE="${2:?}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:?}"; shift 2 ;;
    --max-fonts) MAX_FONTS="${2:?}"; shift 2 ;;
    --style-ref-count) STYLE_REF_COUNT="${2:?}"; shift 2 ;;
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
    --style-tokens-per-ref) STYLE_TOKENS_PER_REF="${2:?}"; shift 2 ;;
    --content-cross-attn-indices) CONTENT_CROSS_ATTN_INDICES="${2:?}"; shift 2 ;;
    --style-token-cross-attn-indices) STYLE_TOKEN_CROSS_ATTN_INDICES="${2:?}"; shift 2 ;;
    --vae-bottleneck-channels) VAE_BOTTLENECK_CHANNELS="${2:?}"; shift 2 ;;
    --vae-encoder-16x16-blocks) VAE_ENCODER_16X16_BLOCKS="${2:?}"; shift 2 ;;
    --vae-decoder-16x16-blocks) VAE_DECODER_16X16_BLOCKS="${2:?}"; shift 2 ;;
    --vae-decoder-tail-blocks) VAE_DECODER_TAIL_BLOCKS="${2:?}"; shift 2 ;;
    --contrastive-proj-dim) CONTRASTIVE_PROJ_DIM="${2:?}"; shift 2 ;;
    --contrastive-temperature) CONTRASTIVE_TEMPERATURE="${2:?}"; shift 2 ;;
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
    --lr "${LR}"
    --batch "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --max-fonts "${MAX_FONTS}"
    --style-ref-count "${STYLE_REF_COUNT}"
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
    --style-tokens-per-ref "${STYLE_TOKENS_PER_REF}"
    --content-cross-attn-indices "${CONTENT_CROSS_ATTN_INDICES}"
    --style-token-cross-attn-indices "${STYLE_TOKEN_CROSS_ATTN_INDICES}"
    --vae-bottleneck-channels "${VAE_BOTTLENECK_CHANNELS}"
    --vae-encoder-16x16-blocks "${VAE_ENCODER_16X16_BLOCKS}"
    --vae-decoder-16x16-blocks "${VAE_DECODER_16X16_BLOCKS}"
    --vae-decoder-tail-blocks "${VAE_DECODER_TAIL_BLOCKS}"
    --contrastive-proj-dim "${CONTRASTIVE_PROJ_DIM}"
    --contrastive-temperature "${CONTRASTIVE_TEMPERATURE}"
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
  echo "[run_style_pretrain_colab] started daemon pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
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
  echo "[run_style_pretrain_colab] received ${signal}, terminating process tree"
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

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "$$" > "${PID_FILE}"

cmd=(
  "${PYTHON_BIN}" -u train.py
  --stage style
  --data-root "${ROOT}"
  --save-dir "${SAVE_DIR}"
  --device "${DEVICE_ARG}"
  --seed "${SEED}"
  --font-split "${FONT_SPLIT}"
  --font-train-ratio "${FONT_TRAIN_RATIO}"
  --lr "${LR}"
  --batch "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --max-fonts "${MAX_FONTS}"
  --style-ref-count "${STYLE_REF_COUNT}"
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
  --style-tokens-per-ref "${STYLE_TOKENS_PER_REF}"
  --content-cross-attn-indices "${CONTENT_CROSS_ATTN_INDICES}"
  --style-token-cross-attn-indices "${STYLE_TOKEN_CROSS_ATTN_INDICES}"
  --vae-bottleneck-channels "${VAE_BOTTLENECK_CHANNELS}"
  --vae-encoder-16x16-blocks "${VAE_ENCODER_16X16_BLOCKS}"
  --vae-decoder-16x16-blocks "${VAE_DECODER_16X16_BLOCKS}"
  --vae-decoder-tail-blocks "${VAE_DECODER_TAIL_BLOCKS}"
  --contrastive-proj-dim "${CONTRASTIVE_PROJ_DIM}"
  --contrastive-temperature "${CONTRASTIVE_TEMPERATURE}"
  --epochs 1000000
  --total-steps "${TARGET_STEPS}"
  --log-every-steps 100
  --val-every-steps 100
  --save-every-steps "${SAVE_EVERY}"
)

if [[ -n "${FONT_SPLIT_SEED}" ]]; then
  cmd+=(--font-split-seed "${FONT_SPLIT_SEED}")
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  cmd+=(--resume "${RESUME_CKPT}")
fi
if [[ "${#EXTRA_TRAIN_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_TRAIN_ARGS[@]}")
fi

echo "[run_style_pretrain_colab] stage=style"
echo "[run_style_pretrain_colab] save_dir=${SAVE_DIR}"
echo "[run_style_pretrain_colab] log_file=${LOG_FILE}"
echo "[run_style_pretrain_colab] device=${DEVICE_ARG} seed=${SEED}"
echo "[run_style_pretrain_colab] batch=${BATCH_SIZE} lr=${LR} total_steps=${TARGET_STEPS}"
echo "[run_style_pretrain_colab] style_ref_count=${STYLE_REF_COUNT} style_tokens_per_ref=${STYLE_TOKENS_PER_REF} total_style_tokens=$((STYLE_REF_COUNT * STYLE_TOKENS_PER_REF)) content_cross_attn_indices=${CONTENT_CROSS_ATTN_INDICES} vae_bottleneck_channels=${VAE_BOTTLENECK_CHANNELS} vae_encoder_16x16_blocks=${VAE_ENCODER_16X16_BLOCKS} vae_decoder_16x16_blocks=${VAE_DECODER_16X16_BLOCKS} vae_decoder_tail_blocks=${VAE_DECODER_TAIL_BLOCKS}"
echo "[run_style_pretrain_colab] contrastive_proj_dim=${CONTRASTIVE_PROJ_DIM} contrastive_temperature=${CONTRASTIVE_TEMPERATURE}"
printf '[run_style_pretrain_colab] cmd='
printf ' %q' "${cmd[@]}"
printf '\n'

set +e
"${cmd[@]}" &
child_pid=$!
wait "${child_pid}"
status=$?
set -e

exit "${status}"

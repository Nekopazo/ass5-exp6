#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/yangximing/code/ass5-exp6/DiffuFont"
PYTHON_BIN="/scratch/yangximing/miniconda3/envs/sg3/bin/python"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

RUN_MODE="daemon"
LOG_FILE=""
PID_FILE=""
SAVE_DIR="checkpoints/vae_pretrain_$(date '+%Y%m%d_%H%M%S')"

RESUME_CKPT=""
DEVICE_ARG="cuda:0"
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

BATCH_SIZE=64
NUM_WORKERS=8
MAX_FONTS=0
IMAGE_SIZE=128

LATENT_CHANNELS=4
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
LATENT_NORMALIZE_FOR_DIT=1
STYLE_REF_COUNT=8
VAE_LAMBDA_REC="1.0"
VAE_LAMBDA_PERC="0.18"
VAE_LAMBDA_KL="2e-4"
VAE_KL_WARMUP_STEPS=10000
VAE_LATENT_MEAN_WEIGHT="1e-3"
VAE_LATENT_STD_WEIGHT="1e-3"
VAE_LATENT_CORR_WEIGHT="5e-4"
VAE_LATENT_STD_TARGET="1.0"
VAE_DIFFICULTY_WARMUP_STEPS=2000
VAE_DIFFICULTY_EMA_DECAY="0.95"
VAE_DIFFICULTY_ALPHA="1.0"
VAE_DIFFICULTY_MIN_WEIGHT="0.5"
VAE_DIFFICULTY_MAX_WEIGHT="2.0"
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
    --style-tokens-per-ref) STYLE_TOKENS_PER_REF="${2:?}"; shift 2 ;;
    --content-cross-attn-indices) CONTENT_CROSS_ATTN_INDICES="${2:?}"; shift 2 ;;
    --style-token-cross-attn-indices) STYLE_TOKEN_CROSS_ATTN_INDICES="${2:?}"; shift 2 ;;
    --vae-bottleneck-channels) VAE_BOTTLENECK_CHANNELS="${2:?}"; shift 2 ;;
    --vae-encoder-16x16-blocks) VAE_ENCODER_16X16_BLOCKS="${2:?}"; shift 2 ;;
    --vae-decoder-16x16-blocks) VAE_DECODER_16X16_BLOCKS="${2:?}"; shift 2 ;;
    --vae-decoder-tail-blocks) VAE_DECODER_TAIL_BLOCKS="${2:?}"; shift 2 ;;
    --latent-normalize-for-dit) LATENT_NORMALIZE_FOR_DIT=1; shift ;;
    --no-latent-normalize-for-dit) LATENT_NORMALIZE_FOR_DIT=0; shift ;;
    --style-ref-count) STYLE_REF_COUNT="${2:?}"; shift 2 ;;
    --vae-lambda-rec) VAE_LAMBDA_REC="${2:?}"; shift 2 ;;
    --vae-lambda-perc) VAE_LAMBDA_PERC="${2:?}"; shift 2 ;;
    --vae-lambda-kl) VAE_LAMBDA_KL="${2:?}"; shift 2 ;;
    --vae-kl-warmup-steps) VAE_KL_WARMUP_STEPS="${2:?}"; shift 2 ;;
    --vae-latent-mean-weight) VAE_LATENT_MEAN_WEIGHT="${2:?}"; shift 2 ;;
    --vae-latent-std-weight) VAE_LATENT_STD_WEIGHT="${2:?}"; shift 2 ;;
    --vae-latent-corr-weight) VAE_LATENT_CORR_WEIGHT="${2:?}"; shift 2 ;;
    --vae-latent-std-target) VAE_LATENT_STD_TARGET="${2:?}"; shift 2 ;;
    --vae-difficulty-warmup-steps) VAE_DIFFICULTY_WARMUP_STEPS="${2:?}"; shift 2 ;;
    --vae-difficulty-ema-decay) VAE_DIFFICULTY_EMA_DECAY="${2:?}"; shift 2 ;;
    --vae-difficulty-alpha) VAE_DIFFICULTY_ALPHA="${2:?}"; shift 2 ;;
    --vae-difficulty-min-weight) VAE_DIFFICULTY_MIN_WEIGHT="${2:?}"; shift 2 ;;
    --vae-difficulty-max-weight) VAE_DIFFICULTY_MAX_WEIGHT="${2:?}"; shift 2 ;;
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
    --style-tokens-per-ref "${STYLE_TOKENS_PER_REF}"
    --content-cross-attn-indices "${CONTENT_CROSS_ATTN_INDICES}"
    --style-token-cross-attn-indices "${STYLE_TOKEN_CROSS_ATTN_INDICES}"
    --vae-bottleneck-channels "${VAE_BOTTLENECK_CHANNELS}"
    --vae-encoder-16x16-blocks "${VAE_ENCODER_16X16_BLOCKS}"
    --vae-decoder-16x16-blocks "${VAE_DECODER_16X16_BLOCKS}"
    --vae-decoder-tail-blocks "${VAE_DECODER_TAIL_BLOCKS}"
    --style-ref-count "${STYLE_REF_COUNT}"
    --vae-lambda-rec "${VAE_LAMBDA_REC}"
    --vae-lambda-perc "${VAE_LAMBDA_PERC}"
    --vae-lambda-kl "${VAE_LAMBDA_KL}"
    --vae-kl-warmup-steps "${VAE_KL_WARMUP_STEPS}"
    --vae-latent-mean-weight "${VAE_LATENT_MEAN_WEIGHT}"
    --vae-latent-std-weight "${VAE_LATENT_STD_WEIGHT}"
    --vae-latent-corr-weight "${VAE_LATENT_CORR_WEIGHT}"
    --vae-latent-std-target "${VAE_LATENT_STD_TARGET}"
    --vae-difficulty-warmup-steps "${VAE_DIFFICULTY_WARMUP_STEPS}"
    --vae-difficulty-ema-decay "${VAE_DIFFICULTY_EMA_DECAY}"
    --vae-difficulty-alpha "${VAE_DIFFICULTY_ALPHA}"
    --vae-difficulty-min-weight "${VAE_DIFFICULTY_MIN_WEIGHT}"
    --vae-difficulty-max-weight "${VAE_DIFFICULTY_MAX_WEIGHT}"
  )
  if [[ "${LATENT_NORMALIZE_FOR_DIT}" == "1" ]]; then
    daemon_args+=(--latent-normalize-for-dit)
  else
    daemon_args+=(--no-latent-normalize-for-dit)
  fi
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

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "$$" > "${PID_FILE}"

cmd=(
  "${PYTHON_BIN}" -u train.py
  --stage vae
  --data-root "${ROOT}"
  --save-dir "${SAVE_DIR}"
  --device "${DEVICE_ARG}"
  --seed "${SEED}"
  --font-split "${FONT_SPLIT}"
  --font-train-ratio "${FONT_TRAIN_RATIO}"
  --lr "${LR}"
  --lr-warmup-steps "${LR_WARMUP_STEPS}"
  --lr-min-scale "${LR_MIN_SCALE}"
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
  --style-tokens-per-ref "${STYLE_TOKENS_PER_REF}"
  --content-cross-attn-indices "${CONTENT_CROSS_ATTN_INDICES}"
  --style-token-cross-attn-indices "${STYLE_TOKEN_CROSS_ATTN_INDICES}"
  --vae-bottleneck-channels "${VAE_BOTTLENECK_CHANNELS}"
  --vae-encoder-16x16-blocks "${VAE_ENCODER_16X16_BLOCKS}"
  --vae-decoder-16x16-blocks "${VAE_DECODER_16X16_BLOCKS}"
  --vae-decoder-tail-blocks "${VAE_DECODER_TAIL_BLOCKS}"
  --style-ref-count "${STYLE_REF_COUNT}"
  --vae-lambda-rec "${VAE_LAMBDA_REC}"
  --vae-lambda-perc "${VAE_LAMBDA_PERC}"
  --vae-lambda-kl "${VAE_LAMBDA_KL}"
  --vae-kl-warmup-steps "${VAE_KL_WARMUP_STEPS}"
  --vae-latent-mean-weight "${VAE_LATENT_MEAN_WEIGHT}"
  --vae-latent-std-weight "${VAE_LATENT_STD_WEIGHT}"
  --vae-latent-corr-weight "${VAE_LATENT_CORR_WEIGHT}"
  --vae-latent-std-target "${VAE_LATENT_STD_TARGET}"
  --vae-difficulty-warmup-steps "${VAE_DIFFICULTY_WARMUP_STEPS}"
  --vae-difficulty-ema-decay "${VAE_DIFFICULTY_EMA_DECAY}"
  --vae-difficulty-alpha "${VAE_DIFFICULTY_ALPHA}"
  --vae-difficulty-min-weight "${VAE_DIFFICULTY_MIN_WEIGHT}"
  --vae-difficulty-max-weight "${VAE_DIFFICULTY_MAX_WEIGHT}"
  --epochs 1000000
  --total-steps "${TARGET_STEPS}"
  --log-every-steps 100
  --val-every-steps 100
  --save-every-steps "${SAVE_EVERY}"
  --sample-every-steps "${SAMPLE_EVERY}"
)
if [[ "${LATENT_NORMALIZE_FOR_DIT}" == "1" ]]; then
  cmd+=(--latent-normalize-for-dit)
else
  cmd+=(--no-latent-normalize-for-dit)
fi

if [[ -n "${FONT_SPLIT_SEED}" ]]; then
  cmd+=(--font-split-seed "${FONT_SPLIT_SEED}")
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  cmd+=(--resume "${RESUME_CKPT}")
fi
if [[ "${#EXTRA_TRAIN_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_TRAIN_ARGS[@]}")
fi

echo "[run_vae_pretrain_colab] stage=vae"
echo "[run_vae_pretrain_colab] save_dir=${SAVE_DIR}"
echo "[run_vae_pretrain_colab] log_file=${LOG_FILE}"
echo "[run_vae_pretrain_colab] device=${DEVICE_ARG} seed=${SEED}"
echo "[run_vae_pretrain_colab] batch=${BATCH_SIZE} lr=${LR} lr_warmup_steps=${LR_WARMUP_STEPS} lr_min_scale=${LR_MIN_SCALE}"
echo "[run_vae_pretrain_colab] style_tokens_per_ref=${STYLE_TOKENS_PER_REF} total_style_tokens=$((STYLE_REF_COUNT * STYLE_TOKENS_PER_REF)) content_cross_attn_indices=${CONTENT_CROSS_ATTN_INDICES} vae_bottleneck_channels=${VAE_BOTTLENECK_CHANNELS} vae_encoder_16x16_blocks=${VAE_ENCODER_16X16_BLOCKS} vae_decoder_16x16_blocks=${VAE_DECODER_16X16_BLOCKS} vae_decoder_tail_blocks=${VAE_DECODER_TAIL_BLOCKS} latent_normalize_for_dit=${LATENT_NORMALIZE_FOR_DIT}"
echo "[run_vae_pretrain_colab] total_steps=${TARGET_STEPS} vae_lambda_rec=${VAE_LAMBDA_REC} vae_lambda_perc=${VAE_LAMBDA_PERC} vae_lambda_kl=${VAE_LAMBDA_KL} vae_kl_warmup_steps=${VAE_KL_WARMUP_STEPS}"
echo "[run_vae_pretrain_colab] vae_latent_mean_weight=${VAE_LATENT_MEAN_WEIGHT} vae_latent_std_weight=${VAE_LATENT_STD_WEIGHT} vae_latent_corr_weight=${VAE_LATENT_CORR_WEIGHT} vae_latent_std_target=${VAE_LATENT_STD_TARGET}"
echo "[run_vae_pretrain_colab] vae_difficulty_warmup_steps=${VAE_DIFFICULTY_WARMUP_STEPS} vae_difficulty_ema_decay=${VAE_DIFFICULTY_EMA_DECAY} vae_difficulty_alpha=${VAE_DIFFICULTY_ALPHA} vae_difficulty_min_weight=${VAE_DIFFICULTY_MIN_WEIGHT} vae_difficulty_max_weight=${VAE_DIFFICULTY_MAX_WEIGHT}"
printf '[run_vae_pretrain_colab] cmd='
printf ' %q' "${cmd[@]}"
printf '\n'

set +e
"${cmd[@]}" &
child_pid=$!
wait "${child_pid}"
status=$?
set -e

exit "${status}"

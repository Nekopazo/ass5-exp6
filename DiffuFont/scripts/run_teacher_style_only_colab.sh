#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/yangximing/code/ass5-exp6/DiffuFont"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_MODE="daemon"
LOG_FILE=""
PID_FILE=""
RESUME_CKPT=""
SAVE_DIR_OVERRIDE=""
PRETRAIN_STYLE_CKPT="/scratch/yangximing/code/ass5-exp6/DiffuFont/checkpoints/pretrain_style_only_site_dropout_20260308/style_encoder_pretrain.pt"
DEVICE_ARG="cuda:1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) RUN_MODE="foreground"; shift ;;
    --daemon)     RUN_MODE="daemon";     shift ;;
    --log-file)   LOG_FILE="${2:?--log-file requires a value}"; shift 2 ;;
    --pid-file)   PID_FILE="${2:?--pid-file requires a value}"; shift 2 ;;
    --resume)     RESUME_CKPT="${2:?--resume requires a value}"; shift 2 ;;
    --save-dir)   SAVE_DIR_OVERRIDE="${2:?--save-dir requires a value}"; shift 2 ;;
    --pretrained-style-encoder) PRETRAIN_STYLE_CKPT="${2:?--pretrained-style-encoder requires a value}"; shift 2 ;;
    --device)     DEVICE_ARG="${2:?--device requires a value}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--foreground|--daemon] [--log-file PATH] [--pid-file PATH] [--resume CKPT] [--save-dir DIR] [--pretrained-style-encoder CKPT] [--device DEV]"
      exit 0 ;;
    *) echo "[teacher_style_only] unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "${ROOT}"
mkdir -p logs checkpoints

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
SAVE_DIR="checkpoints/teacher_style_only_${RUN_TS}"
if [[ -n "${SAVE_DIR_OVERRIDE}" ]]; then
  SAVE_DIR="${SAVE_DIR_OVERRIDE}"
fi

[[ -z "${LOG_FILE}" ]] && LOG_FILE="logs/teacher_style_only_${RUN_TS}.log"
[[ -z "${PID_FILE}" ]] && PID_FILE="logs/teacher_style_only.pid"
if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then
  echo "[teacher_style_only] resume checkpoint not found: ${RESUME_CKPT}" >&2
  exit 2
fi

if [[ "${RUN_MODE}" == "daemon" ]]; then
  _daemon_args=(--foreground --log-file "${LOG_FILE}" --pid-file "${PID_FILE}")
  if [[ -n "${RESUME_CKPT}" ]]; then
    _daemon_args+=(--resume "${RESUME_CKPT}")
  fi
  if [[ -n "${SAVE_DIR_OVERRIDE}" ]]; then
    _daemon_args+=(--save-dir "${SAVE_DIR_OVERRIDE}")
  fi
  if [[ -n "${PRETRAIN_STYLE_CKPT}" ]]; then
    _daemon_args+=(--pretrained-style-encoder "${PRETRAIN_STYLE_CKPT}")
  fi
  if [[ -n "${DEVICE_ARG}" ]]; then
    _daemon_args+=(--device "${DEVICE_ARG}")
  fi
  nohup bash "${SCRIPT_PATH}" "${_daemon_args[@]}" \
    > /dev/null 2>&1 < /dev/null &
  DAEMON_PID=$!
  echo "${DAEMON_PID}" > "${PID_FILE}"
  echo "[teacher_style_only] started in background pid=${DAEMON_PID}"
  echo "[teacher_style_only] pid_file=${ROOT}/${PID_FILE}"
  echo "[teacher_style_only] log_file=${ROOT}/${LOG_FILE}"
  exit 0
fi

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "$$" > "${PID_FILE}"

echo "[teacher_style_only] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "[teacher_style_only] root=${ROOT} pid=$$ device=${DEVICE_ARG} save_dir=${SAVE_DIR}"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "[teacher_style_only] resume_ckpt=${RESUME_CKPT} (will continue from checkpoint step)"
fi

if [[ -z "${PRETRAIN_STYLE_CKPT}" ]]; then
  if [[ -f "checkpoints/style_encoder_pretrain_best.pt" ]]; then
    PRETRAIN_STYLE_CKPT="checkpoints/style_encoder_pretrain_best.pt"
  else
    PRETRAIN_STYLE_CKPT="$(ls -1t checkpoints/style_encoder_pretrain_*.pt 2>/dev/null | head -n 1 || true)"
  fi
fi
if [[ -z "${PRETRAIN_STYLE_CKPT}" || ! -f "${PRETRAIN_STYLE_CKPT}" ]]; then
  echo "[teacher_style_only] pretrained style checkpoint not found. pass --pretrained-style-encoder CKPT" >&2
  exit 2
fi
echo "[teacher_style_only] pretrained_style_ckpt=${PRETRAIN_STYLE_CKPT}"


TARGET_STEPS=60000
if [[ -n "${RESUME_CKPT}" ]]; then
  set -- --resume "${RESUME_CKPT}"
else
  set --
fi

python -u train.py \
  --teacher-line style_only \
  --trainer diffusion \
  --device "${DEVICE_ARG}" \
  --precision bf16 \
  --batch 32 \
  --grad-accum 1 \
  --lr 2e-4 \
  --total-steps "${TARGET_STEPS}" \
  --val-ratio 0.1 \
  --style-ref-count 12 \
  --style-ref-drop-prob 0.15 \
  --style-ref-drop-min-keep 4 \
  --style-site-drop-prob 0.15 \
  --style-site-drop-min-keep 1 \
  --aux-loss-warmup-steps 5000 \
  --lambda-slot-nce 0.02 \
  --lambda-cons 0.0 \
  --lambda-div 0.0 \
  --lambda-proxy-low 0.05 \
  --lambda-proxy-mid 0.05 \
  --lambda-proxy-high 0.05 \
  --lambda-attn-sep 0.02 \
  --lambda-attn-order 0.0 \
  --lambda-attn-role 0.01 \
  --pretrained-style-encoder "${PRETRAIN_STYLE_CKPT}" \
  --num-workers 8 \
  --sample-every-steps 300 \
  --log-every-steps 100 \
  --save-every-steps 3000 \
  --save-dir "${SAVE_DIR}" \
  "$@"

echo "[teacher_style_only] done $(date '+%Y-%m-%d %H:%M:%S')"

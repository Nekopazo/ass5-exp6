#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/yangximing/code/ass5-exp6/DiffuFont"
PYTHON_BIN="/scratch/yangximing/miniconda3/envs/sg3/bin/python"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_MODE="daemon"
LOG_FILE=""
PID_FILE=""
RESUME_CKPT=""
SAVE_DIR_OVERRIDE=""
PRETRAIN_STYLE_CKPT="/scratch/yangximing/code/ass5-exp6/DiffuFont/checkpoints/style_encoder_pretrain_midmem_5000.pt"
DEVICE_ARG="cuda:1"
TARGET_STEPS=100000
STYLE_REF_COUNT=8
ROUTER_TEMPERATURE="1.0"
REFERENCE_TOPK="3"
LAMBDA_PROXY_LOW="0.05"
LAMBDA_PROXY_MID="0.05"
LAMBDA_PROXY_HIGH="0.05"
LAMBDA_ATTN_SEP="0.01"
LAMBDA_ATTN_ORDER="0.0"
LAMBDA_ATTN_ROLE="0.0"
LAMBDA_ROUTE_SPARSE="0.002"
LAMBDA_ROUTE_BALANCE="0.005"
LAMBDA_ROUTE_DIV="0.01"
LAMBDA_ROUTE_GATE="0.0"
LAMBDA_REF_SPARSE="0.002"
LAMBDA_REF_BALANCE="0.002"
LAMBDA_REF_DIV="0.01"

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
echo "[teacher_style_only] root=${ROOT} pid=$$ device=${DEVICE_ARG} python=${PYTHON_BIN} save_dir=${SAVE_DIR}"
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
if [[ -n "${RESUME_CKPT}" ]]; then
  set -- --resume "${RESUME_CKPT}"
else
  set --
fi

echo "[teacher_style_only] adaptive_style_routing=on router_temperature=${ROUTER_TEMPERATURE}"
echo "[teacher_style_only] route_loss=(sparse:${LAMBDA_ROUTE_SPARSE},balance:${LAMBDA_ROUTE_BALANCE},div:${LAMBDA_ROUTE_DIV},gate:${LAMBDA_ROUTE_GATE})"
echo "[teacher_style_only] ref_loss=(sparse:${LAMBDA_REF_SPARSE},balance:${LAMBDA_REF_BALANCE},div:${LAMBDA_REF_DIV}) reference_topk=${REFERENCE_TOPK}"
echo "[teacher_style_only] proxy_loss=(low:${LAMBDA_PROXY_LOW},mid:${LAMBDA_PROXY_MID},high:${LAMBDA_PROXY_HIGH}) attn=(sep:${LAMBDA_ATTN_SEP},order:${LAMBDA_ATTN_ORDER},role:${LAMBDA_ATTN_ROLE})"

"${PYTHON_BIN}" -u train.py \
  --teacher-line style_only \
  --trainer diffusion \
  --device "${DEVICE_ARG}" \
  --batch 32 \
  --grad-accum 1 \
  --lr 2e-4 \
  --total-steps "${TARGET_STEPS}" \
  --val-ratio 0.1 \
  --style-ref-count "${STYLE_REF_COUNT}" \
  --style-ref-drop-prob 0.15 \
  --style-ref-drop-min-keep 4 \
  --style-site-drop-prob 0.10 \
  --style-site-drop-min-keep 1 \
  --aux-loss-warmup-steps 5000 \
  --adaptive-style-routing \
  --router-temperature "${ROUTER_TEMPERATURE}" \
  --reference-topk "${REFERENCE_TOPK}" \
  --lambda-proxy-low "${LAMBDA_PROXY_LOW}" \
  --lambda-proxy-mid "${LAMBDA_PROXY_MID}" \
  --lambda-proxy-high "${LAMBDA_PROXY_HIGH}" \
  --lambda-attn-sep "${LAMBDA_ATTN_SEP}" \
  --lambda-attn-order "${LAMBDA_ATTN_ORDER}" \
  --lambda-attn-role "${LAMBDA_ATTN_ROLE}" \
  --lambda-route-sparse "${LAMBDA_ROUTE_SPARSE}" \
  --lambda-route-balance "${LAMBDA_ROUTE_BALANCE}" \
  --lambda-route-div "${LAMBDA_ROUTE_DIV}" \
  --lambda-route-gate "${LAMBDA_ROUTE_GATE}" \
  --lambda-ref-sparse "${LAMBDA_REF_SPARSE}" \
  --lambda-ref-balance "${LAMBDA_REF_BALANCE}" \
  --lambda-ref-div "${LAMBDA_REF_DIV}" \
  --pretrained-style-encoder "${PRETRAIN_STYLE_CKPT}" \
  --num-workers 8 \
  --sample-every-steps 300 \
  --log-every-steps 100 \
  --save-every-steps 5000 \
  --save-dir "${SAVE_DIR}" \
  "$@"

echo "[teacher_style_only] done $(date '+%Y-%m-%d %H:%M:%S')"

#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/yangximing/code/ass5-exp6/DiffuFont"
PYTHON_BIN="/scratch/yangximing/miniconda3/envs/sg3/bin/python"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_MODE="daemon"
LOG_FILE=""
PID_FILE=""
SAVE_OUT=""
DEVICE_ARG="cuda:0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) RUN_MODE="foreground"; shift ;;
    --daemon)     RUN_MODE="daemon"; shift ;;
    --log-file)   LOG_FILE="${2:?--log-file requires a value}"; shift 2 ;;
    --pid-file)   PID_FILE="${2:?--pid-file requires a value}"; shift 2 ;;
    --out)        SAVE_OUT="${2:?--out requires a value}"; shift 2 ;;
    --device)     DEVICE_ARG="${2:?--device requires a value}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--foreground|--daemon] [--log-file PATH] [--pid-file PATH] [--out CKPT] [--device DEV]"
      exit 0 ;;
    *) echo "[style_pretrain] unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "${ROOT}"
mkdir -p logs checkpoints

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
OUT_PATH="checkpoints/pretrain_style_only_${RUN_TS}/style_encoder_pretrain.pt"
if [[ -n "${SAVE_OUT}" ]]; then
  OUT_PATH="${SAVE_OUT}"
fi
OUT_DIR="$(dirname "${OUT_PATH}")"

[[ -z "${LOG_FILE}" ]] && LOG_FILE="${OUT_DIR}/style_encoder_pretrain.log"
[[ -z "${PID_FILE}" ]] && PID_FILE="logs/style_pretrain.pid"

mkdir -p "${OUT_DIR}"

if [[ "${RUN_MODE}" == "daemon" ]]; then
  _daemon_args=(--foreground --log-file "${LOG_FILE}" --pid-file "${PID_FILE}" --out "${OUT_PATH}")
  if [[ -n "${DEVICE_ARG}" ]]; then
    _daemon_args+=(--device "${DEVICE_ARG}")
  fi
  nohup bash "${SCRIPT_PATH}" "${_daemon_args[@]}" \
    > /dev/null 2>&1 < /dev/null &
  DAEMON_PID=$!
  echo "${DAEMON_PID}" > "${PID_FILE}"
  echo "[style_pretrain] started in background pid=${DAEMON_PID}"
  echo "[style_pretrain] pid_file=${ROOT}/${PID_FILE}"
  echo "[style_pretrain] log_file=${ROOT}/${LOG_FILE}"
  echo "[style_pretrain] out=${ROOT}/${OUT_PATH}"
  exit 0
fi

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "$$" > "${PID_FILE}"

echo "[style_pretrain] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "[style_pretrain] root=${ROOT} pid=$$ device=${DEVICE_ARG} python=${PYTHON_BIN} out=${OUT_PATH}"

"${PYTHON_BIN}" -u scripts/pretrain_style_encoder.py \
  --project-root "${ROOT}" \
  --out "${OUT_PATH}" \
  --log-file "${LOG_FILE}" \
  --metrics-jsonl "${OUT_DIR}/style_encoder_pretrain.metrics.jsonl" \
  --steps 5000 \
  --batch-size 32 \
  --ref-per-style 12 \
  --p-ref-drop 0.15 \
  --min-keep 4 \
  --device "$( [[ "${DEVICE_ARG}" == cuda:* ]] && echo cuda || echo "${DEVICE_ARG}" )" \
  --log-every 50

echo "[style_pretrain] done $(date '+%Y-%m-%d %H:%M:%S')"

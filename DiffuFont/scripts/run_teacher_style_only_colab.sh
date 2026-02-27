#!/usr/bin/env bash
set -euo pipefail

ROOT="/content/drive/MyDrive/ass5/ass5-exp6/DiffuFont"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_MODE="foreground"
LOG_FILE=""
PID_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) RUN_MODE="foreground"; shift ;;
    --daemon)     RUN_MODE="daemon";     shift ;;
    --log-file)   LOG_FILE="${2:?--log-file requires a value}"; shift 2 ;;
    --pid-file)   PID_FILE="${2:?--pid-file requires a value}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--foreground|--daemon] [--log-file PATH] [--pid-file PATH]"
      exit 0 ;;
    *) echo "[teacher_style_only] unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "${ROOT}"
mkdir -p logs checkpoints

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
SAVE_DIR="checkpoints/teacher_style_only_${RUN_TS}"

[[ -z "${LOG_FILE}" ]] && LOG_FILE="logs/teacher_style_only_${RUN_TS}.log"
[[ -z "${PID_FILE}" ]] && PID_FILE="logs/teacher_style_only.pid"

if [[ "${RUN_MODE}" == "daemon" ]]; then
  nohup bash "${SCRIPT_PATH}" --foreground --log-file "${LOG_FILE}" --pid-file "${PID_FILE}" \
    >> "${LOG_FILE}" 2>&1 < /dev/null &
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
echo "[teacher_style_only] root=${ROOT} pid=$$ device=auto save_dir=${SAVE_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TARGET_STEPS=50000
EPOCHS=20

python -u train.py \
  --stage teacher \
  --teacher-line style_only \
  --trainer diffusion \
  --device auto \
  --precision bf16 \
  --batch 64 \
  --grad-accum 1 \
  --lr 4e-4 \
  --epochs "${EPOCHS}" \
  --total-steps "${TARGET_STEPS}" \
  --lambda-diff 1.0 \
  --lambda-nce 0.0 \
  --cfg-drop-prob 0.1 \
  --part-drop-prob 0.0 \
  --num-workers 4 \
  --sample-every-steps 100 \
  --log-every-steps 50 \
  --save-every-steps 1000 \
  --save-dir "${SAVE_DIR}" \
  --attn-scales 16,32

echo "[teacher_style_only] done $(date '+%Y-%m-%d %H:%M:%S')"

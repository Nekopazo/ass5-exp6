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
    *) echo "[teacher_part_style] unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "${ROOT}"
mkdir -p logs checkpoints

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
SAVE_DIR="checkpoints/teacher_part_style_${RUN_TS}"

[[ -z "${LOG_FILE}" ]] && LOG_FILE="logs/teacher_part_style_${RUN_TS}.log"
[[ -z "${PID_FILE}" ]] && PID_FILE="logs/teacher_part_style.pid"

if [[ "${RUN_MODE}" == "daemon" ]]; then
  nohup bash "${SCRIPT_PATH}" --foreground --log-file "${LOG_FILE}" --pid-file "${PID_FILE}" \
    >> "${LOG_FILE}" 2>&1 < /dev/null &
  DAEMON_PID=$!
  echo "${DAEMON_PID}" > "${PID_FILE}"
  echo "[teacher_part_style] started in background pid=${DAEMON_PID}"
  echo "[teacher_part_style] pid_file=${ROOT}/${PID_FILE}"
  echo "[teacher_part_style] log_file=${ROOT}/${LOG_FILE}"
  exit 0
fi

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "$$" > "${PID_FILE}"

echo "[teacher_part_style] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "[teacher_part_style] root=${ROOT} pid=$$ device=auto save_dir=${SAVE_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TARGET_STEPS=100000
EPOCHS=39

python -u train.py \
  --stage teacher \
  --teacher-line part_style \
  --trainer diffusion \
  --device auto \
  --precision bf16 \
  --batch 16 \
  --grad-accum 2 \
  --lr 2e-4 \
  --epochs "${EPOCHS}" \
  --total-steps "${TARGET_STEPS}" \
  --lambda-diff 1.0 \
  --lambda-nce 0.05 \
  --nce-warmup-steps 5000 \
  --cfg-drop-prob 0.1 \
  --part-set-max 8 \
  --part-set-min 1 \
  --part-drop-prob 0.2 \
  --lambda-cons 0.1 \
  --num-workers 0 \
  --sample-every-steps 300 \
  --log-every-steps 100 \
  --save-every-steps 3000 \
  --save-dir "${SAVE_DIR}" \
  --attn-scales 16,32

echo "[teacher_part_style] done $(date '+%Y-%m-%d %H:%M:%S')"

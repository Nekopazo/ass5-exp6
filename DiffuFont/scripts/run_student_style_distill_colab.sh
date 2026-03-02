#!/usr/bin/env bash
set -euo pipefail

ROOT="/content/drive/MyDrive/ass5/ass5-exp6/DiffuFont"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_MODE="foreground"
LOG_FILE=""
PID_FILE=""
TEACHER_CKPT=""
TEACHER_MODE="part_style"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) RUN_MODE="foreground"; shift ;;
    --daemon)     RUN_MODE="daemon";     shift ;;
    --log-file)   LOG_FILE="${2:?--log-file requires a value}"; shift 2 ;;
    --pid-file)   PID_FILE="${2:?--pid-file requires a value}"; shift 2 ;;
    --teacher-ckpt) TEACHER_CKPT="${2:?--teacher-ckpt requires a value}"; shift 2 ;;
    --teacher-mode) TEACHER_MODE="${2:?--teacher-mode requires a value}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --teacher-ckpt PATH [--teacher-mode MODE] [--foreground|--daemon] [--log-file PATH] [--pid-file PATH]"
      exit 0 ;;
    *) echo "[student_style] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${TEACHER_CKPT}" ]]; then
  echo "[student_style] --teacher-ckpt is required" >&2
  exit 2
fi
if [[ ! -f "${TEACHER_CKPT}" ]]; then
  echo "[student_style] teacher checkpoint not found: ${TEACHER_CKPT}" >&2
  exit 2
fi

cd "${ROOT}"
mkdir -p logs checkpoints

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
SAVE_DIR="checkpoints/student_style_${RUN_TS}"

[[ -z "${LOG_FILE}" ]] && LOG_FILE="logs/student_style_${RUN_TS}.log"
[[ -z "${PID_FILE}" ]] && PID_FILE="logs/student_style.pid"

if [[ "${RUN_MODE}" == "daemon" ]]; then
  nohup bash "${SCRIPT_PATH}" --foreground --log-file "${LOG_FILE}" --pid-file "${PID_FILE}" \
    --teacher-ckpt "${TEACHER_CKPT}" --teacher-mode "${TEACHER_MODE}" \
    >> "${LOG_FILE}" 2>&1 < /dev/null &
  DAEMON_PID=$!
  echo "${DAEMON_PID}" > "${PID_FILE}"
  echo "[student_style] started in background pid=${DAEMON_PID}"
  echo "[student_style] pid_file=${ROOT}/${PID_FILE}"
  echo "[student_style] log_file=${ROOT}/${LOG_FILE}"
  exit 0
fi

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "$$" > "${PID_FILE}"

echo "[student_style] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "[student_style] root=${ROOT} pid=$$ device=auto save_dir=${SAVE_DIR}"
echo "[student_style] teacher_ckpt=${TEACHER_CKPT} teacher_mode=${TEACHER_MODE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TARGET_STEPS=15000
EPOCHS=6

python -u train.py \
  --stage student \
  --teacher-ckpt "${TEACHER_CKPT}" \
  --teacher-distill-mode "${TEACHER_MODE}" \
  --trainer diffusion \
  --device auto \
  --precision bf16 \
  --batch 64 \
  --grad-accum 1 \
  --lr 4e-4 \
  --epochs "${EPOCHS}" \
  --total-steps "${TARGET_STEPS}" \
  --lambda-diff 1.0 \
  --lambda-kd 1.0 \
  --lambda-nce 0.0 \
  --num-workers 4 \
  --sample-every-steps 100 \
  --log-every-steps 50 \
  --save-every-steps 1000 \
  --save-dir "${SAVE_DIR}" \
  --attn-scales 16,32

echo "[student_style] done $(date '+%Y-%m-%d %H:%M:%S')"

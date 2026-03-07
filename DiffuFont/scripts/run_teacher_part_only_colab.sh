#!/usr/bin/env bash
set -euo pipefail

ROOT="/content/drive/MyDrive/ass5/ass5-exp6/DiffuFont"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_MODE="foreground"
LOG_FILE=""
PID_FILE=""
RESUME_CKPT="/content/drive/MyDrive/ass5/ass5-exp6/DiffuFont/checkpoints/teacher_part_only_20260302_041234/ckpt_step_17500.pt"
SAVE_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) RUN_MODE="foreground"; shift ;;
    --daemon)     RUN_MODE="daemon";     shift ;;
    --log-file)   LOG_FILE="${2:?--log-file requires a value}"; shift 2 ;;
    --pid-file)   PID_FILE="${2:?--pid-file requires a value}"; shift 2 ;;
    --resume)     RESUME_CKPT="${2:?--resume requires a value}"; shift 2 ;;
    --save-dir)   SAVE_DIR_OVERRIDE="${2:?--save-dir requires a value}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--foreground|--daemon] [--log-file PATH] [--pid-file PATH] [--resume CKPT] [--save-dir DIR]"
      exit 0 ;;
    *) echo "[teacher_part_only] unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "${ROOT}"
mkdir -p logs checkpoints

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
SAVE_DIR="checkpoints/teacher_part_only_${RUN_TS}"
if [[ -n "${SAVE_DIR_OVERRIDE}" ]]; then
  SAVE_DIR="${SAVE_DIR_OVERRIDE}"
fi

[[ -z "${LOG_FILE}" ]] && LOG_FILE="logs/teacher_part_only_${RUN_TS}.log"
[[ -z "${PID_FILE}" ]] && PID_FILE="logs/teacher_part_only.pid"
if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then
  echo "[teacher_part_only] resume checkpoint not found: ${RESUME_CKPT}" >&2
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
  nohup bash "${SCRIPT_PATH}" "${_daemon_args[@]}" \
    >> "${LOG_FILE}" 2>&1 < /dev/null &
  DAEMON_PID=$!
  echo "${DAEMON_PID}" > "${PID_FILE}"
  echo "[teacher_part_only] started in background pid=${DAEMON_PID}"
  echo "[teacher_part_only] pid_file=${ROOT}/${PID_FILE}"
  echo "[teacher_part_only] log_file=${ROOT}/${LOG_FILE}"
  exit 0
fi

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "$$" > "${PID_FILE}"

echo "[teacher_part_only] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "[teacher_part_only] root=${ROOT} pid=$$ device=auto save_dir=${SAVE_DIR}"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "[teacher_part_only] resume_ckpt=${RESUME_CKPT} (will continue from checkpoint step)"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TARGET_STEPS=30000

if [[ -n "${RESUME_CKPT}" ]]; then
  set -- --resume "${RESUME_CKPT}"
else
  set --
fi

python -u train.py \
  --teacher-line part_only \
  --trainer diffusion \
  --device auto \
  --precision bf16 \
  --batch 64 \
  --grad-accum 1 \
  --lr 4e-4 \
  --total-steps "${TARGET_STEPS}" \
  --val-ratio 0.1 \
  --lambda-diff 1.0 \
  --lambda-nce 0.0 \
  --lambda-cons 0.0 \
  --lambda-div 0.0 \
  --style-nce-temp 0.07 \
  --nce-warmup-steps 5000 \
  --num-workers 8 \
  --sample-every-steps 200 \
  --log-every-steps 100 \
  --save-every-steps 500 \
  --save-dir "${SAVE_DIR}" \
  --attn-scales 16,32,64 \
  "$@"

echo "[teacher_part_only] done $(date '+%Y-%m-%d %H:%M:%S')"

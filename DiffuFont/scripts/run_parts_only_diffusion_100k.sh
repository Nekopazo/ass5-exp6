#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/yangximing/code/ass5-exp6/DiffuFont"
CONDA_SH="/scratch/yangximing/miniconda3/etc/profile.d/conda.sh"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_MODE="daemon"
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
    *) echo "[parts_only] unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "${ROOT}"
mkdir -p logs checkpoints

[[ -z "${LOG_FILE}" ]] && LOG_FILE="logs/parts_only_diffusion_100k_$(date '+%Y%m%d_%H%M%S').log"
[[ -z "${PID_FILE}" ]] && PID_FILE="logs/parts_only_diffusion_100k.pid"

if [[ "${RUN_MODE}" == "daemon" ]]; then
  nohup bash "${SCRIPT_PATH}" --foreground --log-file "${LOG_FILE}" --pid-file "${PID_FILE}" \
    >> "${LOG_FILE}" 2>&1 < /dev/null &
  DAEMON_PID=$!
  echo "${DAEMON_PID}" > "${PID_FILE}"
  echo "[parts_only] started in background  pid=${DAEMON_PID}"
  echo "[parts_only] pid_file=${ROOT}/${PID_FILE}"
  echo "[parts_only] log_file=${ROOT}/${LOG_FILE}"
  exit 0
fi

exec >> "${LOG_FILE}" 2>&1
echo "$$" > "${PID_FILE}"

source "${CONDA_SH}"
conda activate sg3

echo "[parts_only] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "[parts_only] root=${ROOT}  pid=$$  device=cuda:1"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
TARGET_STEPS=100000
EPOCHS=39

RETRY=0
while true; do
  set +e
  python -u train.py \
    --trainer diffusion \
    --device cuda:1 \
    --precision bf16 \
    --batch 32 \
    --lr 2e-4 \
    --epochs "${EPOCHS}" \
    --total-steps "${TARGET_STEPS}" \
    --val-ratio 0.1 \
    --conditioning-profile parts_vector_only \
    --lambda-nce 0.0 \
    --lambda-cons 0.0 \
    --lambda-div 0.0 \
    --style-nce-temp 0.07 \
    --style-ref-count 8 \
    --style-token-count 3 \
    --pretrained-style-encoder checkpoints/style_encoder_pretrain_best.pt \
    --num-workers 8 \
    --sample-every-steps 300 \
    --sample-inference-steps 20 \
    --log-every-steps 100 \
    --detailed-log \
    --save-every-steps 5000 \
    --save-dir checkpoints/parts_only_diffusion_100k \
    --attn-scales 16,32,64
  RC=$?
  set -e

  if [[ $RC -eq 0 ]]; then
    break
  fi

  # Check if the failure was an OOM error
  if tail -5 "${LOG_FILE}" | grep -q "OutOfMemoryError"; then
    RETRY=$((RETRY + 1))
    echo "[parts_only] OOM detected (attempt ${RETRY}), waiting 60s before retry..."
    sleep 60
    echo "[parts_only] retrying $(date '+%Y-%m-%d %H:%M:%S')"
  else
    echo "[parts_only] non-OOM error (exit code ${RC}), aborting."
    exit $RC
  fi
done

echo "[parts_only] done $(date '+%Y-%m-%d %H:%M:%S')"

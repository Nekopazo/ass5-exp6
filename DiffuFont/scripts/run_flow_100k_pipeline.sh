#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/yangximing/code/ass5-exp6/DiffuFont"
CONDA_SH="/scratch/yangximing/miniconda3/etc/profile.d/conda.sh"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_MODE="daemon"
LOG_FILE=""
PID_FILE=""

usage() {
  cat <<'EOF'
Usage: run_flow_100k_pipeline.sh [--foreground|--daemon] [--log-file PATH] [--pid-file PATH]

Options:
  --foreground   Run in foreground.
  --daemon       Run in background (default).
  --log-file     Log file path. Default: logs/pipeline_flow_100k_YYYYmmdd_HHMMSS.log
  --pid-file     PID file path. Default: logs/pipeline_flow_100k.pid
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground)
      RUN_MODE="foreground"
      shift
      ;;
    --daemon)
      RUN_MODE="daemon"
      shift
      ;;
    --log-file)
      LOG_FILE="${2:-}"
      if [[ -z "${LOG_FILE}" ]]; then
        echo "[pipeline] --log-file requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --pid-file)
      PID_FILE="${2:-}"
      if [[ -z "${PID_FILE}" ]]; then
        echo "[pipeline] --pid-file requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[pipeline] unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

cd "${ROOT}"
mkdir -p logs checkpoints

if [[ -z "${LOG_FILE}" ]]; then
  LOG_FILE="logs/pipeline_flow_100k_$(date '+%Y%m%d_%H%M%S').log"
fi
if [[ -z "${PID_FILE}" ]]; then
  PID_FILE="logs/pipeline_flow_100k.pid"
fi

if [[ "${RUN_MODE}" == "daemon" ]]; then
  nohup bash "${SCRIPT_PATH}" --foreground --log-file "${LOG_FILE}" --pid-file "${PID_FILE}" >> "${LOG_FILE}" 2>&1 < /dev/null &
  DAEMON_PID=$!
  echo "${DAEMON_PID}" > "${PID_FILE}"
  echo "[pipeline] started in background"
  echo "[pipeline] pid=${DAEMON_PID}"
  echo "[pipeline] pid_file=${ROOT}/${PID_FILE}"
  echo "[pipeline] log_file=${ROOT}/${LOG_FILE}"
  exit 0
fi

exec >> "${LOG_FILE}" 2>&1
echo "$$" > "${PID_FILE}"

source "${CONDA_SH}"
conda activate sg3

echo "[pipeline] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "[pipeline] root=${ROOT}"
echo "[pipeline] pid=$$"
echo "[pipeline] log_file=${ROOT}/${LOG_FILE}"

# Fixed 100k-step schedule under current default data config.
TARGET_STEPS=100000
EPOCHS=39
echo "[pipeline] target_steps=${TARGET_STEPS} epochs=${EPOCHS} (fixed default schedule)"

# Start main diffusion training with batch fallback.
for BATCH_SIZE in 8; do
  echo "[pipeline] try batch=${BATCH_SIZE}"
  set +e
  python train.py \
    --trainer diffusion \
    --device cuda:1 \
    --precision bf16 \
    --batch "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --total-steps "${TARGET_STEPS}" \
    --conditioning-profile full \
    --part-set-max 8 \
    --part-set-min 1 \
    --lambda-nce 0.05 \
    --num-workers 0 \
    --part-image-cache-size 20000 \
    --lmdb-decode-cache-size 50000 \
    --save-dir checkpoints/fulldiffusion_default_100k \
    --attn-scales 16,32 \
    --log-every-steps 10 
  RC=$?
  set -e
  if [[ "${RC}" -eq 0 ]]; then
    break
  fi
  echo "[pipeline] batch=${BATCH_SIZE} failed (rc=${RC}); fallback to next."
done

echo "[pipeline] done $(date '+%Y-%m-%d %H:%M:%S')"

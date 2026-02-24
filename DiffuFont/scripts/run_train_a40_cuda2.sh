#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/scratch/yangximing/miniconda3/envs/sg3/bin/python"

RUN_NAME="run_a40_cuda2_bf16_bs48_adamw_dbg4_lrfix_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs"
CKPT_DIR="$PROJECT_ROOT/checkpoints/$RUN_NAME"
LOG_FILE="$CKPT_DIR/train.log"
LOG_LINK="$LOG_DIR/$RUN_NAME.log"

mkdir -p "$LOG_DIR" "$CKPT_DIR"
cd "$PROJECT_ROOT"

CMD=(
  "$PYTHON_BIN" -u train.py
  --data-root .
  --device cuda:1
  --precision bf16
  --font-mode random
  --conditioning-profile baseline
  --batch 48
  --num-workers 0
  --epochs 50
  --sample-every-steps 300
  --log-every-steps 100
  --detailed-log
  --save-every-steps 5000
  --save-every-epochs 0
  --save-dir "$CKPT_DIR"
)

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

setsid "${CMD[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!

ln -sfn "$LOG_FILE" "$LOG_LINK"

echo "pid=$PID"
echo "run=$RUN_NAME"
echo "log=$LOG_FILE"
echo "log_link=$LOG_LINK"
echo "save=$CKPT_DIR"

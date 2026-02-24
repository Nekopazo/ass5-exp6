#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/scratch/yangximing/miniconda3/envs/sg3/bin/python"

RUN_NAME="run_a40_cuda1_bf16_bs48_adamw_noedge_dbg4_lrfix_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs"
CKPT_DIR="$PROJECT_ROOT/checkpoints/$RUN_NAME"
LOG_FILE="$CKPT_DIR/train.log"
LOG_LINK="$LOG_DIR/$RUN_NAME.log"
EP_CKPT="${EP_CKPT:-checkpoints/e_p_font_encoder_best.pt}"

mkdir -p "$LOG_DIR" "$CKPT_DIR"
cd "$PROJECT_ROOT"

CMD=(
  "$PYTHON_BIN" -u train.py
  --data-root .
  --device cuda:0
  --precision bf16
  --font-mode random
  --conditioning-profile full
  --batch 48
  --num-workers 0
  --epochs 50
  --part-set-min-size 2
  --part-set-size 10
  --part-retrieval-mode font_softmax_top1
  --part-retrieval-ep-ckpt "$EP_CKPT"
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

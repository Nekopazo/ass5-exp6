#!/usr/bin/env bash
set -euo pipefail

cd /scratch/yangximing/code/ass5-exp6/DiffuFont

PYTHON_BIN=/scratch/yangximing/miniconda3/envs/sg3/bin/python

runs=(
  "rope:checkpoints/xpred_20260511_132328_rope"
  "12fonts:checkpoints/xpred_20260512_170049_12fonts"
  "8fonts:checkpoints/xpred_20260513_084056_8fonts"
)

for item in "${runs[@]}"; do
  name="${item%%:*}"
  run_dir="${item#*:}"
  out_dir="$run_dir/eval_full_val_latest"

  echo "[$(date -Is)] START ${name} run_dir=${run_dir} checkpoint=latest.pt"
  "${PYTHON_BIN}" evaluate_full_val_grid.py \
    --run-dir "${run_dir}" \
    --checkpoint latest.pt \
    --output-dir "${out_dir}" \
    --device cuda:1 \
    --eval-fonts-per-batch 8 \
    --eval-chars-per-batch 48 \
    --num-workers 2 \
    --inference-steps 20 \
    --style-ref-count 8 \
    --log-every 1

  "${PYTHON_BIN}" - "${out_dir}/metrics.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as fp:
    data = json.load(fp)
result = data["result"]
run_name = data["run_dir"].rstrip("/").split("/")[-1]
print(
    f"[{run_name}] samples={result['samples']} "
    f"fid={result['fid']:.6f} "
    f"l1={result['l1']:.6f} "
    f"ssim={result['ssim']:.6f} "
    f"lpips={result['lpips']:.6f}",
    flush=True,
)
PY
  echo "[$(date -Is)] DONE ${name}"
done

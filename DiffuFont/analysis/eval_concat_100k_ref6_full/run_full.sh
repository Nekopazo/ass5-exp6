#!/usr/bin/env bash
set -euo pipefail

source /scratch/yangximing/miniconda3/etc/profile.d/conda.sh
conda activate sg3
cd /scratch/yangximing/code/ass5-exp6/DiffuFont

python scripts/eval_0424_ref_counts_by_font.py \
  --checkpoint checkpoints/xpred_20260426_012228/ckpt_step_100000.pt \
  --checkpoint-label concat_step100000 \
  --ref-counts 6 \
  --data-root /scratch/yangximing/code/ass5-exp6/DiffuFont \
  --output-dir analysis/eval_concat_100k_ref6_full \
  --device auto \
  --samples-per-font 30 \
  --max-train-fonts 0 \
  --max-test-fonts 0 \
  --eval-batch-size 32 \
  --inference-steps 20 \
  --font-split-seed 42 \
  --font-train-ratio 0.95 \
  --seed 42

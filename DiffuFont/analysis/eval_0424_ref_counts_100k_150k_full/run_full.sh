#!/usr/bin/env bash
set -euo pipefail

source /scratch/yangximing/miniconda3/etc/profile.d/conda.sh
conda activate sg3
cd /scratch/yangximing/code/ass5-exp6/DiffuFont

python scripts/eval_0424_ref_counts_by_font.py \
  --checkpoint checkpoints/xpred_20260424_182507/ckpt_step_100000.pt \
  --checkpoint-label step100000 \
  --checkpoint checkpoints/xpred_20260424_182507/ckpt_step_150000.pt \
  --checkpoint-label step150000 \
  --ref-counts 3 6 9 12 \
  --data-root /scratch/yangximing/code/ass5-exp6/DiffuFont \
  --output-dir analysis/eval_0424_ref_counts_100k_150k_full \
  --device auto \
  --samples-per-font 30 \
  --max-train-fonts 0 \
  --max-test-fonts 0 \
  --eval-batch-size 32 \
  --inference-steps 20 \
  --font-split-seed 42 \
  --font-train-ratio 0.95 \
  --seed 42

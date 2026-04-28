#!/usr/bin/env bash
set -euo pipefail

source /scratch/yangximing/miniconda3/etc/profile.d/conda.sh
conda activate sg3
cd /scratch/yangximing/code/ass5-exp6/DiffuFont

python scripts/compare_train_infer_trim_by_font_70k.py \
  --drop-trained-checkpoint checkpoints/xpred_20260423_230719/ckpt_step_70000.pt \
  --mean-trained-checkpoint checkpoints/xpred_20260424_182507/ckpt_step_70000.pt \
  --data-root /scratch/yangximing/code/ass5-exp6/DiffuFont \
  --output-dir analysis/compare_train_infer_trim_by_font_70k_full \
  --device auto \
  --samples-per-font 30 \
  --max-train-fonts 0 \
  --max-test-fonts 0 \
  --eval-batch-size 32 \
  --inference-steps 20 \
  --style-ref-count 6 \
  --font-split-seed 42 \
  --font-train-ratio 0.95 \
  --seed 42

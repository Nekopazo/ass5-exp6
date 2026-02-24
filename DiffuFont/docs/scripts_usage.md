# Script Usage And Default Parameters

This file documents all runnable entry scripts in this repo:

- `train.py`
- `scripts/*.py`
- `scripts/run_train_a40_cuda1.sh`
- `scripts/run_train_a40_cuda2.sh`

For all scripts, run `python <script> --help` for the authoritative interface.

## 1) `train.py`

Purpose: diffusion training with four conditioning profiles.

Minimal:

```bash
python train.py --data-root . --conditioning-profile full --part-retrieval-ep-ckpt checkpoints/e_p_font_encoder_best.pt
```

Default-focused command:

```bash
python train.py \
  --data-root . \
  --epochs 50 \
  --batch 64 \
  --num-workers 0 \
  --device cuda:0 \
  --precision fp32 \
  --lr 2e-4 \
  --lambda-cons 0.05 \
  --diffusion-steps 1000 \
  --lr-tmax-steps 0 \
  --style-k 3 \
  --font-index 0 \
  --font-name None \
  --font-mode random \
  --max-fonts 0 \
  --auto-select-font \
  --no-include-target-in-style \
  --component-guided-style \
  --style-overlap-topk \
  --decomposition-json CharacterData/decomposition.json \
  --conditioning-profile full \
  --part-bank-manifest DataPreparation/PartBank/manifest.json \
  --part-retrieval-mode font_softmax_top1 \
  --part-retrieval-ep-ckpt <required for non-baseline> \
  --part-set-size 10 \
  --part-set-min-size 2 \
  --part-set-sampling random \
  --no-part-target-char-priority \
  --part-image-size 64 \
  --sample-every-steps 300 \
  --log-every-steps 100 \
  --detailed-log \
  --overlap-report-samples 0 \
  --overlap-report-seed 42 \
  --overlap-report-json None \
  --save-every-epochs 0 \
  --save-every-steps 5000 \
  --save-dir checkpoints \
  --resume None
```

Notes:

- `--conditioning-profile baseline` disables both token and RSI.
- `--conditioning-profile token_only` enables token only.
- `--conditioning-profile rsi_only` enables RSI only.
- `--conditioning-profile full` enables both token and RSI.
- When token is disabled, `lambda-cons` is forced to `0`.

## 2) `scripts/prepare_common_charset.py`

Purpose: create `CharList.json` and `ReferenceCharList.json`.

Minimal:

```bash
python scripts/prepare_common_charset.py
```

Defaults:

- `--out-dir CharacterData`
- `--char-count 2000`
- `--ref-count 300`

## 3) `scripts/build_part_bank.py`

Purpose: build part patches and `manifest.json`.

Minimal:

```bash
python scripts/build_part_bank.py --project-root .
```

Defaults:

- `--project-root .`
- `--fonts-dir fonts`
- `--font-list-json DataPreparation/FontList.json`
- `--font-indices None`
- `--max-fonts 0`
- `--charset-json CharacterData/ReferenceCharList.json`
- `--max-chars 0`
- `--output-dir DataPreparation/PartBank`
- `--parts-per-font 48`
- `--patch-size 64`
- `--min-ink-ratio 0.02`
- `--max-ink-ratio 0.70`
- `--min-edge-ratio 0.05`
- `--max-candidates 6000`
- `--location-dedupe` (toggle with `--no-location-dedupe`)
- `--sim-dedupe-cos-threshold 0.995`
- `--sim-dedupe-anchor-limit 1500`
- `--medoid-pool-size 1200`
- `--kmedoids-n-init 4`
- `--kmedoids-max-iter 25`
- `--detector sift` (`sift|auto|orb`)
- `--canvas-size 256`
- `--char-size 224`
- `--x-offset 0`
- `--y-offset 0`
- `--random-seed 42`

## 4) `scripts/pretrain_joint_style_encoders.py`

Purpose: pretrain `E_p` font classifier.

Minimal:

```bash
python scripts/pretrain_joint_style_encoders.py --project-root .
```

Defaults:

- `--project-root .`
- `--train-lmdb DataPreparation/LMDB/TrainFont.lmdb`
- `--out checkpoints/e_p_font_encoder.pt`
- `--best-out checkpoints/e_p_font_encoder_best.pt`
- `--metrics-jsonl checkpoints/e_p_font_encoder.metrics.jsonl`
- `--log-file checkpoints/e_p_font_encoder.log`
- `--steps 20000`
- `--batch-size 48`
- `--val-ratio 0.1`
- `--val-batches 8`
- `--log-every 100`
- `--monitor val_loss`
- `--early-stop-patience 30`
- `--early-stop-min-delta 1e-4`
- `--num-font-classes 0`
- `--backbone resnet18` (`resnet18|resnet34|light_cnn`)
- `--label-smoothing 0.05`
- `--image-size 256`
- `--lr 1e-4`
- `--weight-decay 1e-4`
- `--seed 42`
- `--device auto` (`auto|cpu|cuda`)
- `--lmdb-scan-limit 0`
- `--min-chars-per-font 8`

## 5) `scripts/retrieve_parts_by_style.py`

Purpose: style image -> `E_p` softmax -> top1 font -> sampled parts.

Minimal:

```bash
python scripts/retrieve_parts_by_style.py --ep-ckpt checkpoints/e_p_font_encoder_best.pt --style-image path/to/style.png
```

Defaults:

- `--project-root .`
- `--manifest DataPreparation/PartBank/manifest.json`
- `--train-lmdb DataPreparation/LMDB/TrainFont.lmdb`
- `--style-image-size 256`
- `--font-topk 3`
- `--part-min-size 2`
- `--part-max-size 10`
- `--seed 42`
- `--out-json checkpoints/retrieved_parts.json`
- `--device auto` (`auto|cpu|cuda`)

## 6) `scripts/build_font_retrieval_cache.py`

Purpose: build cached top1 retrieval results per dataset sample.

Minimal:

```bash
python scripts/build_font_retrieval_cache.py --ep-ckpt checkpoints/e_p_font_encoder_best.pt
```

Defaults:

- `--project-root .`
- `--train-lmdb DataPreparation/LMDB/TrainFont.lmdb`
- `--manifest DataPreparation/PartBank/manifest.json`
- `--out-npz checkpoints/font_retrieval_cache.npz`
- `--out-json checkpoints/font_retrieval_cache.summary.json`
- `--image-size 256`
- `--batch-size 128`
- `--chars-per-font 48`
- `--min-chars-per-font 8`
- `--lmdb-scan-limit 0`
- `--seed 42`
- `--device auto` (`auto|cpu|cuda`)

## 7) `scripts/build_part_vector_index.py`

Purpose: encode part patches and build vector index (`faiss` / `annoy`).

Minimal:

```bash
python scripts/build_part_vector_index.py --encoder-ckpt checkpoints/e_p_font_encoder_best.pt
```

Defaults:

- `--project-root .`
- `--manifest DataPreparation/PartBank/manifest.json`
- `--out-dir checkpoints/part_vector_index`
- `--image-size 64`
- `--embedding-dim 256`
- `--batch-size 256`
- `--backend faiss,annoy`
- `--annoy-trees 50`
- `--device auto` (`auto|cpu|cuda`)

## 8) `scripts/pretrain_part_style_encoder.py`

Purpose: legacy contrastive pretrain for part style encoder.

Minimal:

```bash
python scripts/pretrain_part_style_encoder.py --project-root .
```

Defaults:

- `--project-root .`
- `--manifest DataPreparation/PartBank/manifest.json`
- `--out checkpoints/part_style_encoder_pretrain.pt`
- `--best-out checkpoints/part_style_encoder_pretrain_best.pt`
- `--log-file checkpoints/part_style_encoder_pretrain.log`
- `--metrics-jsonl checkpoints/part_style_encoder_pretrain.metrics.jsonl`
- `--steps 10000`
- `--batch-size 64`
- `--min-set-size 1`
- `--max-set-size 8`
- `--warmup-max-set-size 4`
- `--warmup-steps 4000`
- `--val-ratio 0.1`
- `--val-batches 8`
- `--monitor val_loss` (`val_loss|val_acc|train_loss|train_acc`)
- `--early-stop-patience 20`
- `--early-stop-min-delta 1e-4`
- `--patch-size 64`
- `--style-dim 256`
- `--lr 1e-4`
- `--temperature 0.4`
- `--seed 42`
- `--device auto` (`auto|cpu|cuda`)
- `--log-every 50`

## 9) `scripts/analyze_component_overlap.py`

Purpose: analyze component-guided style sampling overlap stats.

Minimal:

```bash
python scripts/analyze_component_overlap.py --project-root .
```

Defaults:

- `--project-root .`
- `--font-index 0`
- `--font-name None`
- `--font-mode random` (`fixed|random`)
- `--max-fonts 0`
- `--style-k 3`
- `--include-target-in-style` off by default
- `--component-guided-style` on by default
- `--decomposition-json CharacterData/decomposition.json`
- `--samples 4000`
- `--seed 42`
- `--top-char-k 20`
- `--top-pair-k 80`
- `--out-dir checkpoints/overlap_stats`
- `--json-name component_overlap_report.json`
- `--pairs-csv-name component_overlap_top_pairs.csv`
- `--plot-name component_overlap_hist.png`
- `--skip-plot` off by default

## 10) Runner Shell Scripts

### `scripts/run_train_a40_cuda1.sh`

Default launch profile:

- `--device cuda:0`
- `--precision bf16`
- `--conditioning-profile full`
- `--batch 48`
- `--part-retrieval-mode font_softmax_top1`
- `--part-retrieval-ep-ckpt ${EP_CKPT:-checkpoints/e_p_font_encoder_best.pt}`
- `--sample-every-steps 300`
- `--log-every-steps 100`
- `--save-every-steps 5000`
- `--save-every-epochs 0`

### `scripts/run_train_a40_cuda2.sh`

Default launch profile:

- `--device cuda:1`
- `--precision bf16`
- `--conditioning-profile baseline`
- `--batch 48`
- `--sample-every-steps 300`
- `--log-every-steps 100`
- `--save-every-steps 5000`
- `--save-every-epochs 0`

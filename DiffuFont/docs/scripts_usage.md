# Script Usage And Default Parameters

This file documents runnable entry scripts in this repo:

- `train.py`
- `scripts/*.py`
- `scripts/run_train_a40_cuda1.sh`
- `scripts/run_train_a40_cuda2.sh`

For all scripts, run `python <script> --help` for authoritative args.

## 1) `train.py`

Purpose: diffusion training with four conditioning profiles and online PartBank retrieval.

Minimal (current default direction: parts on, RSI off):

```bash
python train.py \
  --data-root . \
  --conditioning-profile parts_vector_only \
  --part-retrieval-ep-ckpt checkpoints/e_p_font_encoder_best.pt
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
  --diffusion-steps 1000 \
  --lr-tmax-steps 0 \
  --font-index 0 \
  --font-mode random \
  --max-fonts 0 \
  --auto-select-font \
  --no-include-target-in-style \
  --conditioning-profile parts_vector_only \
  --attn-scales 16,32 \
  --part-bank-manifest DataPreparation/PartBank/manifest.json \
  --part-retrieval-ep-ckpt checkpoints/e_p_font_encoder_best.pt \
  --part-set-size 32 \
  --part-set-min-size 32 \
  --part-set-sampling random \
  --style-token-count 8 \
  --style-token-dim 256 \
  --part-image-size 64 \
  --sample-every-steps 300 \
  --log-every-steps 100 \
  --detailed-log \
  --save-every-epochs 0 \
  --save-every-steps 5000 \
  --save-dir checkpoints \
  --resume None
```

Notes:

- `--conditioning-profile baseline`: parts OFF, RSI OFF.
- `--conditioning-profile parts_vector_only`: parts ON, RSI OFF (default).
- `--conditioning-profile rsi_only`: parts OFF, RSI ON.
- `--conditioning-profile full`: parts ON, RSI ON.
- Part retrieval policy is fixed to `top1 gate + top3 weighted mix`.
- Retrieval thresholds are internal constants in `dataset.py`.
- If parts conditioning is enabled, `--part-retrieval-ep-ckpt` is required.
- `--attn-scales` controls where style attention is active; current design default is `16,32`.
- `disable-self-attn` and `lite-daca` flags have been removed; model block structure is fixed.
- Part sampling requests 32 parts by default, but may return variable length if unique candidates are fewer.

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

Purpose: style image -> `E_p` softmax -> top1/top3 font retrieval -> sampled parts.

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

Purpose: build cached retrieval results per dataset sample.

Minimal:

```bash
python scripts/build_font_retrieval_cache.py --ep-ckpt checkpoints/e_p_font_encoder_best.pt
```

## 7) `scripts/build_part_vector_index.py`

Purpose: encode part patches and build vector index (`faiss` / `annoy`).

Minimal:

```bash
python scripts/build_part_vector_index.py --encoder-ckpt checkpoints/e_p_font_encoder_best.pt
```

## 8) `scripts/pretrain_part_style_encoder.py`

Purpose: legacy contrastive pretrain for part style encoder.

Minimal:

```bash
python scripts/pretrain_part_style_encoder.py --project-root .
```

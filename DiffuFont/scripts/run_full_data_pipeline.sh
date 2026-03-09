#!/usr/bin/env bash
set -euo pipefail

# Full data pipeline: render grayscale -> LMDB -> style pretrain
ROOT="DiffuFont/"

cd "${ROOT}"
export PYTHONUNBUFFERED=1
echo "=========================================="
echo "[pipeline] Step 1: Render ContentFont (grayscale)"
echo "=========================================="
python DiffuFont/DataPreparation/generate_font_images.py \
    --project-root "${ROOT}" \
    --char-list-json DiffuFont/CharacterData/CharList.json \
    --font-list-json DiffuFont/DataPreparation/ContentFontList.json \
    --font-dir DiffuFont/DataPreparation/Font \
    --out-dir DiffuFont/DataPreparation/Generated/ContentFont

echo "=========================================="
echo "[pipeline] Step 2: Render TrainFonts (grayscale)"
echo "=========================================="
python DiffuFont/DataPreparation/generate_font_images.py \
    --project-root "${ROOT}" \
    --char-list-json DiffuFont/CharacterData/CharList.json \
    --font-list-json DiffuFont/DataPreparation/FontList.json \
    --font-dir DiffuFont/DataPreparation/Font \
    --out-dir DiffuFont/DataPreparation/Generated/TrainFonts

echo "=========================================="
echo "[pipeline] Step 3: Build ContentFont LMDB"
echo "=========================================="
python DiffuFont/DataPreparation/images_to_lmdb.py \
    --project-root "${ROOT}" \
    --img-roots DiffuFont/DataPreparation/Generated/ContentFont \
    --lmdb-path DiffuFont/DataPreparation/LMDB/ContentFont.lmdb \
    --overwrite

echo "=========================================="
echo "[pipeline] Step 4: Build TrainFont LMDB"
echo "=========================================="
python DiffuFont/DataPreparation/images_to_lmdb.py \
    --project-root "${ROOT}" \
    --img-roots DiffuFont/DataPreparation/Generated/TrainFonts \
    --lmdb-path DiffuFont/DataPreparation/LMDB/TrainFont.lmdb \
    --overwrite

echo "=========================================="
echo "[pipeline] Step 5: Pretrain style encoder (contrastive, docs setup)"
echo "=========================================="
python scripts/pretrain_style_encoder.py \
    --project-root "${ROOT}" \
    --steps 5000 \
    --batch-size 32 \
    --ref-per-style 12 \
    --style-token-count 3 \
    --p-ref-drop 0.15 \
    --min-keep 4 \
    --lambda-slot-nce 0.02 \
    --lambda-cons 0.0 \
    --lambda-div 0.0 \
    --lambda-proxy-low 0.05 \
    --lambda-proxy-mid 0.05 \
    --lambda-proxy-high 0.05 \
    --lambda-attn-sep 0.02 \
    --lambda-attn-order 0.0 \
    --lambda-attn-role 0.0 \
    --device auto \
    --log-every 50

echo "=========================================="
echo "[pipeline] DONE $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

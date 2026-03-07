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
    --steps 10000 \
    --batch-size 64 \
    --ref-per-style 8 \
    --style-token-count 3 \
    --device auto \
    --log-every 50

echo "=========================================="
echo "[pipeline] DONE $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

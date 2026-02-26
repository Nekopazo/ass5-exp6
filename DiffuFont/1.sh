#!/usr/bin/env bash

# Full data pipeline: render grayscale → LMDB → PartBank → PartBank LMDB → pretrain
ROOT="DiffuFont/"

export PYTHONUNBUFFERED=1

pip install lmdb
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

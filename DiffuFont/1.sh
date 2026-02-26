#!/usr/bin/env bash

# Full data pipeline: render grayscale → LMDB → PartBank → PartBank LMDB → pretrain
ROOT="DiffuFont/"

export PYTHONUNBUFFERED=1
nproc
lscpu
grep -c ^processor /proc/cpuinfo
pip install lmdb pillow opencv-python fonttools
echo "=========================================="
echo "[pipeline] Step 1: Render ContentFont (grayscale)"
echo "=========================================="
python DiffuFont/DataPreparation/generate_font_images.py \
    --project-root "${ROOT}" \
    --char-list-json CharacterData/CharList.json \
    --font-list-json DataPreparation/ContentFontList.json \
    --font-dir DataPreparation/Font \
    --out-dir DataPreparation/Generated/ContentFont

echo "=========================================="
echo "[pipeline] Step 2: Render TrainFonts (grayscale)"
echo "=========================================="
python DiffuFont/DataPreparation/generate_font_images.py \
    --project-root "${ROOT}" \
    --char-list-json CharacterData/CharList.json \
    --font-list-json DataPreparation/FontList.json \
    --font-dir Fonts \
    --out-dir DataPreparation/Generated/TrainFonts

echo "=========================================="
echo "[pipeline] Step 3: Build ContentFont LMDB"
echo "=========================================="
python DiffuFont/DataPreparation/images_to_lmdb.py \
    --project-root "${ROOT}" \
    --img-roots DataPreparation/Generated/ContentFont \
    --lmdb-path DataPreparation/LMDB/ContentFont.lmdb \
    --overwrite

echo "=========================================="
echo "[pipeline] Step 4: Build TrainFont LMDB"
echo "=========================================="
python DiffuFont/DataPreparation/images_to_lmdb.py \
    --project-root "${ROOT}" \
    --img-roots DataPreparation/Generated/TrainFonts \
    --lmdb-path DataPreparation/LMDB/TrainFont.lmdb \
    --overwrite

echo "=========================================="
echo "[pipeline] Step 5: Build PartBank (component-aware, grayscale)"
echo "=========================================="
python DiffuFont/scripts/build_part_bank_component_aware_from_images.py \
    --project-root "${ROOT}" \
    --glyph-root DataPreparation/Generated/TrainFonts \
    --output-dir DataPreparation/PartBank \
    --parts-per-font 32 \
    --reference-char-json Character/ReferenceCharList_200_balanced_v3.json\
    --workers 4

echo "=========================================="
echo "[pipeline] Step 6: Build PartBank LMDB"
echo "=========================================="
python DiffuFont/scripts/build_part_bank_lmdb.py \
    --project-root "${ROOT}" \
    --manifest DataPreparation/PartBank/manifest.json \
    --out-lmdb DataPreparation/LMDB/PartBank.lmdb

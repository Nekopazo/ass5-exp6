#!/usr/bin/env bash
set -euo pipefail

# Full data pipeline: render grayscale → LMDB → PartBank → PartBank LMDB → pretrain
ROOT="/scratch/yangximing/code/ass5-exp6/DiffuFont"
CONDA_SH="/scratch/yangximing/miniconda3/etc/profile.d/conda.sh"
source "${CONDA_SH}"
conda activate sg3

cd "${ROOT}"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "[pipeline] Step 1: Render ContentFont (grayscale)"
echo "=========================================="
python DataPreparation/generate_font_images.py \
    --project-root "${ROOT}" \
    --char-list-json CharacterData/CharList.json \
    --font-list-json DataPreparation/ContentFontList.json \
    --font-dir DataPreparation/Font \
    --out-dir DataPreparation/Generated/ContentFont

echo "=========================================="
echo "[pipeline] Step 2: Render TrainFonts (grayscale)"
echo "=========================================="
python DataPreparation/generate_font_images.py \
    --project-root "${ROOT}" \
    --char-list-json CharacterData/CharList.json \
    --font-list-json DataPreparation/FontList.json \
    --font-dir DataPreparation/Font \
    --out-dir DataPreparation/Generated/TrainFonts

echo "=========================================="
echo "[pipeline] Step 3: Build ContentFont LMDB"
echo "=========================================="
python DataPreparation/images_to_lmdb.py \
    --project-root "${ROOT}" \
    --img-roots DataPreparation/Generated/ContentFont \
    --lmdb-path DataPreparation/LMDB/ContentFont.lmdb \
    --overwrite

echo "=========================================="
echo "[pipeline] Step 4: Build TrainFont LMDB"
echo "=========================================="
python DataPreparation/images_to_lmdb.py \
    --project-root "${ROOT}" \
    --img-roots DataPreparation/Generated/TrainFonts \
    --lmdb-path DataPreparation/LMDB/TrainFont.lmdb \
    --overwrite

echo "=========================================="
echo "[pipeline] Step 5: Build PartBank (component-aware, grayscale)"
echo "=========================================="
python scripts/build_part_bank_component_aware_from_images.py \
    --project-root "${ROOT}" \
    --glyph-root DataPreparation/Generated/TrainFonts \
    --output-dir DataPreparation/PartBank \
    --parts-per-font 32 \
    --workers 48

echo "=========================================="
echo "[pipeline] Step 6: Build PartBank LMDB"
echo "=========================================="
python scripts/build_part_bank_lmdb.py \
    --project-root "${ROOT}" \
    --manifest DataPreparation/PartBank/manifest.json \
    --out-lmdb DataPreparation/LMDB/PartBank.lmdb

echo "=========================================="
echo "[pipeline] Step 7: Pretrain part style encoder (contrastive)"
echo "=========================================="
python scripts/pretrain_part_style_encoder.py \
    --project-root "${ROOT}" \
    --manifest DataPreparation/PartBank/manifest.json \
    --steps 10000 \
    --batch-size 64 \
    --device auto \
    --log-every 50

echo "=========================================="
echo "[pipeline] DONE $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

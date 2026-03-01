#!/usr/bin/env bash
set -euo pipefail

# Full data pipeline: render grayscale → LMDB → PartBank → PartBank LMDB → pretrain
ROOT="DiffuFont/"

cd "${ROOT}"
export PYTHONUNBUFFERED=1
PRETRAIN_PART_SET_MIN="${PRETRAIN_PART_SET_MIN:-8}"
PRETRAIN_PART_SET_MAX="${PRETRAIN_PART_SET_MAX:-8}"
PRETRAIN_PART_SAMPLE_WITH_REPLACEMENT="${PRETRAIN_PART_SAMPLE_WITH_REPLACEMENT:-0}"
PRETRAIN_PART_SAMPLE_ARGS=(--part-set-min "${PRETRAIN_PART_SET_MIN}" --part-set-max "${PRETRAIN_PART_SET_MAX}")
if [[ "${PRETRAIN_PART_SAMPLE_WITH_REPLACEMENT}" == "1" ]]; then
  PRETRAIN_PART_SAMPLE_ARGS+=(--part-sample-with-replacement)
else
  PRETRAIN_PART_SAMPLE_ARGS+=(--no-part-sample-with-replacement)
fi

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
echo "[pipeline] Step 5: Build PartBank (component-aware, grayscale)"
echo "=========================================="
python DiffuFont/scripts/build_part_bank_component_aware_from_images.py \
    --project-root "${ROOT}" \
    --glyph-root DiffuFont/DataPreparation/Generated/TrainFonts \
    --output-dir DiffuFont/DataPreparation/PartBank \
    --parts-per-font 32 \
    --workers 24

echo "=========================================="
echo "[pipeline] Step 6: Build PartBank LMDB"
echo "=========================================="
python DiffuFont/scripts/build_part_bank_lmdb.py \
    --project-root "${ROOT}" \
    --manifest DiffuFont/DataPreparation/PartBank/manifest.json \
    --out-lmdb DiffuFont/DataPreparation/LMDB/PartBank.lmdb

echo "=========================================="
echo "[pipeline] Step 7: Pretrain part style encoder (contrastive)"
echo "=========================================="
python scripts/pretrain_part_style_encoder.py \
    --project-root "${ROOT}" \
    --steps 10000 \
    --batch-size 64 \
    "${PRETRAIN_PART_SAMPLE_ARGS[@]}" \
    --device auto \
    --log-every 50

echo "=========================================="
echo "[pipeline] DONE $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

#!/usr/bin/env bash

# Full data pipeline: render grayscale → LMDB → PartBank → PartBank LMDB → pretrain
ROOT="DiffuFont/"

cd DiffuFont

python DataPreparation/generate_font_images.py \
    --project-root "DiffuFont/" \
    --char-list-json DiffuFont/CharacterData/CharList.json \
    --font-list-json DiffuFont/DataPreparation/ContentFontList.json \
    --font-dir DiffuFont/DataPreparation/Font \
    --out-dir DiffuFont/DataPreparation/Generated/ContentFont

python DataPreparation/generate_font_images.py \
    --project-root "DiffuFont/" \
    --char-list-json DiffuFont/CharacterData/CharList.json \
    --font-list-json DiffuFont/DataPreparation/FontList.json \
    --font-dir DiffuFont/DataPreparation/Font \
    --out-dir DiffuFont/DataPreparation/Generated/TrainFonts
    
python DataPreparation/images_to_lmdb.py \
    --project-root "DiffuFont/" \
    --img-roots DiffuFont/DataPreparation/Generated/ContentFont \
    --lmdb-path DiffuFont/DataPreparation/LMDB/ContentFont.lmdb \
    --overwrite

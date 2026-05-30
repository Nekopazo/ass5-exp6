#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/scratch/yangximing/miniconda3/envs/sg3/bin/python}"
NUM_WORKERS="${NUM_WORKERS:-48}"
MAP_SIZE="${MAP_SIZE:-1099511627776}"
KEEP_GENERATED="${KEEP_GENERATED:-0}"

build_one() {
  local size="$1"
  local char_size="$2"
  local base_dir="${PROJECT_ROOT}/DataPreparation/MultiRes/${size}"
  local gen_dir="${base_dir}/Generated"
  local lmdb_dir="${base_dir}/LMDB"

  echo "[build] size=${size} char_size=${char_size} base_dir=${base_dir}" >&2
  rm -rf "${gen_dir}" "${lmdb_dir}"
  mkdir -p "${gen_dir}" "${lmdb_dir}"

  "${PYTHON_BIN}" "${PROJECT_ROOT}/DataPreparation/generate_font_images.py" \
    --project-root "${PROJECT_ROOT}" \
    --char-list-json "CharacterData/CharList.json" \
    --font-list-json "DataPreparation/ContentFontList.json" \
    --font-dir "DataPreparation/Font" \
    --char-size "${char_size}" \
    --canvas-size "${size}" \
    --out-dir "DataPreparation/MultiRes/${size}/Generated/ContentFont" \
    --num-workers "${NUM_WORKERS}"

  "${PYTHON_BIN}" "${PROJECT_ROOT}/DataPreparation/generate_font_images.py" \
    --project-root "${PROJECT_ROOT}" \
    --char-list-json "CharacterData/CharList.json" \
    --font-list-json "DataPreparation/FontList.json" \
    --font-dir "DataPreparation/Font" \
    --char-size "${char_size}" \
    --canvas-size "${size}" \
    --out-dir "DataPreparation/MultiRes/${size}/Generated/TrainFonts" \
    --num-workers "${NUM_WORKERS}"

  "${PYTHON_BIN}" "${PROJECT_ROOT}/DataPreparation/images_to_lmdb.py" \
    --project-root "${PROJECT_ROOT}" \
    --img-roots "DataPreparation/MultiRes/${size}/Generated/ContentFont" \
    --lmdb-path "DataPreparation/MultiRes/${size}/LMDB/ContentFont.lmdb" \
    --map-size "${MAP_SIZE}" \
    --overwrite

  "${PYTHON_BIN}" "${PROJECT_ROOT}/DataPreparation/images_to_lmdb.py" \
    --project-root "${PROJECT_ROOT}" \
    --img-roots "DataPreparation/MultiRes/${size}/Generated/TrainFonts" \
    --lmdb-path "DataPreparation/MultiRes/${size}/LMDB/TrainFont.lmdb" \
    --map-size "${MAP_SIZE}" \
    --overwrite

  if [[ "${KEEP_GENERATED}" != "1" ]]; then
    rm -rf "${gen_dir}"
  fi

  echo "[done] size=${size} lmdb_dir=${lmdb_dir}" >&2
}

build_one 64 60
build_one 96 90

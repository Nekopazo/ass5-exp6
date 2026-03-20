#!/usr/bin/env bash
set -euo pipefail

ROOT="/content/drive/MyDrive/ass5/ass5-exp6/DiffuFont"
PY_SCRIPT="${ROOT}/inference_flexible.py"

EXAMPLE="style_lmdb"
CHECKPOINT="/content/drive/MyDrive/ass5/ass5-exp6/DiffuFont/checkpoints/teacher_style_only_20260304_070516/ckpt_step_30000.pt"
DEVICE="cpu"
OUTPUT_DIR="${ROOT}/output"
TRAINER="diffusion"
INFER_STEPS="8"
SEED="123"
DATA_ROOT="${ROOT}"

# LMDB defaults
CONTENT_CHARS="你,好,世,界"
CONTENT_FONT=""
STYLE_FONT="方正中等线简体"
STYLE_CHARS="春,风"
# Folder defaults (only used in style_folder example)
CONTENT_FOLDER=""
STYLE_FOLDER=""

print_help() {
  cat <<'EOF'
Usage:
  bash scripts/run_inference_flexible.sh [options] [-- extra inference args]

Options:
  --example NAME         style_lmdb | style_folder (default: style_lmdb)
  --checkpoint PATH      required, model checkpoint path
  --device DEV           e.g. cuda:0 / cpu (default: cuda:0)
  --output-dir DIR       output root dir (default: DiffuFont/output)
  --trainer TYPE         diffusion | flow_matching (default: diffusion)
  --inference-steps N    sampling steps (default: 20)
  --seed N               random seed for reproducible inference (default: 42)
  --data-root DIR        project root (default: DiffuFont)

  --content-chars CSV    for lmdb content input, e.g. "你,好,世,界"
  --content-font NAME    load original column from TrainFont.lmdb: <content_font>@<content_char>
  --style-font NAME      for style_lmdb
  --style-chars CSV      for style_lmdb
  --content-folder DIR   for style_folder
  --style-folder DIR     for style_folder

  -h, --help             show help

Examples:
  1) style_only + LMDB
     bash scripts/run_inference_flexible.sh \
       --example style_lmdb \
       --checkpoint checkpoints/xxx/ckpt_step_30000.pt \
       --style-font "方正中等线简体" \
       --style-chars "春,风" \
       --content-font "方正中等线简体" \
       --content-chars "你,好,世,界"

  2) style_only + folder
     bash scripts/run_inference_flexible.sh \
       --example style_folder \
       --checkpoint checkpoints/xxx/ckpt_step_30000.pt \
       --content-folder /path/to/content_imgs \
       --style-folder /path/to/style_imgs

Pass-through:
  Append extra args after '--', they are forwarded to inference_flexible.py.
  Example:
    bash scripts/run_inference_flexible.sh --checkpoint ckpt.pt -- --max-samples 8 --strict-load
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --example) EXAMPLE="${2:?--example requires a value}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:?--checkpoint requires a value}"; shift 2 ;;
    --device) DEVICE="${2:?--device requires a value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --trainer) TRAINER="${2:?--trainer requires a value}"; shift 2 ;;
    --inference-steps) INFER_STEPS="${2:?--inference-steps requires a value}"; shift 2 ;;
    --seed) SEED="${2:?--seed requires a value}"; shift 2 ;;
    --data-root) DATA_ROOT="${2:?--data-root requires a value}"; shift 2 ;;
    --content-chars) CONTENT_CHARS="${2:?--content-chars requires a value}"; shift 2 ;;
    --content-font) CONTENT_FONT="${2:?--content-font requires a value}"; shift 2 ;;
    --style-font) STYLE_FONT="${2:?--style-font requires a value}"; shift 2 ;;
    --style-chars) STYLE_CHARS="${2:?--style-chars requires a value}"; shift 2 ;;
    --content-folder) CONTENT_FOLDER="${2:?--content-folder requires a value}"; shift 2 ;;
    --style-folder) STYLE_FOLDER="${2:?--style-folder requires a value}"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "[run_inference_flexible] unknown arg: $1" >&2; print_help; exit 2 ;;
  esac
done

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[run_inference_flexible] --checkpoint is required" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[run_inference_flexible] checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[run_inference_flexible] python script not found: ${PY_SCRIPT}" >&2
  exit 2
fi

BASE_ARGS=(
  "${PY_SCRIPT}"
  --data-root "${DATA_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --device "${DEVICE}"
  --trainer "${TRAINER}"
  --inference-steps "${INFER_STEPS}"
  --seed "${SEED}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${CONTENT_FONT}" ]]; then
  BASE_ARGS+=(--content-font "${CONTENT_FONT}")
fi

case "${EXAMPLE}" in
  style_lmdb)
    echo "[run_inference_flexible] example=style_lmdb"
    set -x
    python "${BASE_ARGS[@]}" \
      --conditioning-profile style_only \
      --content-source lmdb \
      --content-chars "${CONTENT_CHARS}" \
      --style-source lmdb \
      --style-font "${STYLE_FONT}" \
      --style-chars "${STYLE_CHARS}" \
      "${EXTRA_ARGS[@]}"
    set +x
    ;;

  style_folder)
    echo "[run_inference_flexible] example=style_folder"
    if [[ -z "${CONTENT_FOLDER}" || -z "${STYLE_FOLDER}" ]]; then
      echo "[run_inference_flexible] style_folder needs --content-folder and --style-folder" >&2
      exit 2
    fi
    set -x
    python "${BASE_ARGS[@]}" \
      --conditioning-profile style_only \
      --content-source folder \
      --content-folder "${CONTENT_FOLDER}" \
      --style-source folder \
      --style-folder "${STYLE_FOLDER}" \
      "${EXTRA_ARGS[@]}"
    set +x
    ;;

  *)
    echo "[run_inference_flexible] unsupported --example: ${EXAMPLE}" >&2
    exit 2
    ;;
esac

echo "[run_inference_flexible] done. output_dir=${OUTPUT_DIR}"

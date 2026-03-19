#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/yangximing/code/ass5-exp6/DiffuFont"
WRAPPER="${ROOT}/scripts/run_diffusion_colab.sh"

echo "[deprecated] run_style_pretrain_colab.sh 当前为了兼容历史用法，会转到 diffusion 主模型训练。请改用 scripts/run_diffusion_colab.sh" >&2
exec bash "${WRAPPER}" "$@"

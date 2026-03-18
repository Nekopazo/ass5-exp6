#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/yangximing/code/ass5-exp6/DiffuFont"
WRAPPER="${ROOT}/scripts/run_teacher_style_only_colab.sh"

exec bash "${WRAPPER}" "$@"

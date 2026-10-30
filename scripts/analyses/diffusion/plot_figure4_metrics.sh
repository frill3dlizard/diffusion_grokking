#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY_SCRIPT_PATH="${ROOT_DIR}/analyses/diffusion/plot_figure4_metrics.py"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash scripts/analyses/diffusion/plot_figure4_metrics.sh RUN_DIR [OUTPUT_PNG]" >&2
    exit 2
fi

cd "${ROOT_DIR}"
source venv/bin/activate

args=(--run-dir "$1")
if [[ $# -eq 2 ]]; then
    args+=(--output "$2")
fi

export PYTHONUNBUFFERED=1
python -u "${PY_SCRIPT_PATH}" "${args[@]}"

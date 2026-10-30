#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY_SCRIPT_PATH="${ROOT_DIR}/analyses/diffusion/plot_figure4_metrics.py"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash scripts/analyses/diffusion/plot_figure4_metrics.sh RUN_DIR_OR_METRICS_JSONL [OUTPUT_PNG]" >&2
    exit 2
fi

# Accept the accidental `timestamp path` value produced by
# `find ... -printf '%T@ %h\n' | sort -nr | head -n1`.
if [[ ! -e "$1" && "$1" == *" "* ]]; then
    candidate_path="${1#* }"
    if [[ -e "${candidate_path}" ]]; then
        set -- "${candidate_path}" "${@:2}"
    fi
fi

cd "${ROOT_DIR}"
source venv/bin/activate

args=(--run-dir "$1")
if [[ $# -eq 2 ]]; then
    args+=(--output "$2")
fi

export PYTHONUNBUFFERED=1
python -u "${PY_SCRIPT_PATH}" "${args[@]}"

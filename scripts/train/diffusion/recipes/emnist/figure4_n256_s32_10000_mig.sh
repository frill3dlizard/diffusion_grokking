#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
BASE_RECIPE="${SCRIPT_DIR}/figure4_n256_s32_10000.sh"

if [[ ! -f "${BASE_RECIPE}" ]]; then
    echo "Missing base Figure 4 recipe: ${BASE_RECIPE}" >&2
    exit 1
fi

export BATCH_SIZE="${BATCH_SIZE:-512}"

echo "Using conservative MIG batch size: ${BATCH_SIZE}"
exec "${BASE_RECIPE}" "$@"

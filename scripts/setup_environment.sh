#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/venv}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to a Python 3 executable and rerun." >&2
    exit 1
fi

"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating virtual environment: ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"

echo
echo "Environment ready."
echo "Activate it with: source \"${VENV_DIR}/bin/activate\""

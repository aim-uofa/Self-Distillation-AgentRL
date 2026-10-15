#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(realpath -m -- "${PROJECT_ROOT:-${REPO_ROOT}/..}")"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B}"
SOURCE_MODEL="${SOURCE_MODEL:-}"
if [[ -n "${SOURCE_MODEL}" ]]; then
    SOURCE_MODEL="$(realpath -e -- "${SOURCE_MODEL}")"
    DEFAULT_MODEL_NAME="$(basename -- "${SOURCE_MODEL}")"
else
    DEFAULT_MODEL_NAME="${MODEL_ID##*/}"
fi
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/${DEFAULT_MODEL_NAME}}"
# shellcheck source=common.sh
REQUIRE_MODEL=0 source "${SCRIPT_DIR}/common.sh"

ARGS=(--model-id "${MODEL_ID}" --output-dir "${MODEL_PATH}")
if [[ -n "${SOURCE_MODEL}" ]]; then ARGS+=(--source "${SOURCE_MODEL}"); fi
python "${SCRIPT_DIR}/prepare_model.py" "${ARGS[@]}"

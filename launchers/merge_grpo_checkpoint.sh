#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MERGE_METHOD="${MERGE_METHOD:-grpo}"
CHECKPOINT_PATH="${1:-}"
if [[ -z "${CHECKPOINT_PATH}" ]]; then
    echo "Usage: $0 <global_step_N directory | actor directory> [merged model directory]" >&2
    exit 2
fi

CHECKPOINT_PATH="$(realpath -e -- "${CHECKPOINT_PATH}")"
if [[ "$(basename -- "${CHECKPOINT_PATH}")" == "actor" ]]; then
    ACTOR_DIR="${CHECKPOINT_PATH}"
    STEP_DIR="$(dirname -- "${CHECKPOINT_PATH}")"
else
    ACTOR_DIR="${CHECKPOINT_PATH}/actor"
    STEP_DIR="${CHECKPOINT_PATH}"
fi

if [[ ! -d "${ACTOR_DIR}" ]]; then
    echo "Actor checkpoint directory not found: ${ACTOR_DIR}" >&2
    exit 1
fi

TARGET_DIR="$(realpath -m -- "${2:-${STEP_DIR}/merged_hf}")"
if [[ -e "${TARGET_DIR}" && ! -d "${TARGET_DIR}" ]]; then
    echo "Merge target exists and is not a directory: ${TARGET_DIR}" >&2
    exit 1
fi

# Resolve caller-relative paths before common.sh changes to the repository root.
# shellcheck source=common.sh
REQUIRE_MODEL=0 source "${SCRIPT_DIR}/common.sh"

python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "${ACTOR_DIR}" \
    --target_dir "${TARGET_DIR}"

echo "Merged Hugging Face model: ${TARGET_DIR}"
echo "Example: bash launchers/test_${MERGE_METHOD}_alfworld.sh ${TARGET_DIR}"

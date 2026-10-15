#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:-}"
if [[ -n "${TASK}" ]]; then
    shift
fi
MODEL_DIR="${1:-${GRPO_MODEL_PATH:-}}"

if [[ -z "${TASK}" || -z "${MODEL_DIR}" ]]; then
    echo "Usage: $0 <alfworld|webshop|search|scienceworld> <merged GRPO model directory> [Hydra overrides ...]" >&2
    exit 2
fi
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "Merged Hugging Face model not found at ${MODEL_DIR}" >&2
    echo "Run launchers/merge_grpo_checkpoint.sh first." >&2
    exit 1
fi

export MODEL_PATH="$(realpath -e -- "${MODEL_DIR}")"
export EVAL_KIND=grpo
export EVAL_DESCRIPTION="merged GRPO model"
export EVAL_RUNNER="${SCRIPT_DIR}/run_grpo_1gpu.sh"
export TRAIN_LOG_FILE=/dev/null

exec "${SCRIPT_DIR}/test_pretrained_1gpu.sh" "${TASK}" "$@"

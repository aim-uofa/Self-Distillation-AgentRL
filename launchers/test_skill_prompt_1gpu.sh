#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:-}"
if [[ -z "${TASK}" ]]; then
    echo "Usage: $0 <alfworld|webshop|search> [Hydra overrides ...]" >&2
    exit 2
fi
shift

export EVAL_KIND=skill_prompt
export EVAL_DESCRIPTION="${EVAL_DESCRIPTION:-pretrained model with retrieved skills}"
export EVAL_RUNNER="${SCRIPT_DIR}/eval_skill_prompt_1gpu.sh"
export TRAIN_LOG_FILE=/dev/null

exec "${SCRIPT_DIR}/test_pretrained_1gpu.sh" "${TASK}" "$@"

#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Evaluate the complete selected Search-QA pool (default: 500/source, 3125 total).
export EVAL_SIZE="${EVAL_SIZE:-all}"
if [[ -z "${EVAL_BATCH_SIZE:-}" ]]; then
    if [[ "${EVAL_SIZE}" == "all" ]]; then
        EVAL_BATCH_SIZE=512
    else
        EVAL_BATCH_SIZE="${EVAL_SIZE}"
    fi
fi
export EVAL_BATCH_SIZE

exec "${SCRIPT_DIR}/test_pretrained_1gpu.sh" search "$@"

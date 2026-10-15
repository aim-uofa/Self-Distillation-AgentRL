#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_SIZE="${EVAL_SIZE:-all}"
if [[ -z "${EVAL_BATCH_SIZE:-}" ]]; then
    if [[ "${EVAL_SIZE}" == "all" ]]; then
        EVAL_BATCH_SIZE=512
    else
        EVAL_BATCH_SIZE="${EVAL_SIZE}"
    fi
fi
export EVAL_BATCH_SIZE
export EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-false}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
exec "${SCRIPT_DIR}/test_rlsd_model.sh" search "$@"

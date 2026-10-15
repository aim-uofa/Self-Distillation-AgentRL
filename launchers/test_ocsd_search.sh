#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_SIZE="${EVAL_SIZE:-all}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
export EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-false}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
exec "${SCRIPT_DIR}/test_ocsd_model.sh" search "$@"

#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Paper mode uses the study's full pool; otherwise retain the 128-game default.
DEFAULT_EVAL_SIZE=128
if [[ "${PAPER_CONFIG:-0}" == "1" ]]; then DEFAULT_EVAL_SIZE=140; fi
export EVAL_SIZE="${EVAL_SIZE:-${DEFAULT_EVAL_SIZE}}"
export EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-true}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.4}"

exec "${SCRIPT_DIR}/test_pretrained_1gpu.sh" alfworld "$@"

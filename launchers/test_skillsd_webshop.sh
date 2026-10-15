#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_SIZE="${EVAL_SIZE:-128}"
export EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-true}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.4}"
exec "${SCRIPT_DIR}/test_skillsd_model.sh" webshop "$@"

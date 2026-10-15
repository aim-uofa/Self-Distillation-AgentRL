#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Match the evaluation protocol used by the official Qwen3 WebShop launcher.
export EVAL_SIZE="${EVAL_SIZE:-128}"
export EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-true}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.4}"

exec "${SCRIPT_DIR}/test_pretrained_1gpu.sh" webshop "$@"

#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MERGE_METHOD=adrs
exec "${SCRIPT_DIR}/merge_grpo_checkpoint.sh" "$@"

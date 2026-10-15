#!/usr/bin/env bash

# Internal evaluation runner used by test_skill_prompt_1gpu.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_skill_method_1gpu.sh" skill_prompt "$@"

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQUIRE_MODEL=0 source "${SCRIPT_DIR}/common.sh"
# Data-only setup. Dependency installation is separate and never upgrades the
# training environment implicitly. Downloads are pinned and checksum verified.
exec python "${SCRIPT_DIR}/prepare_scienceworld.py" --data-root "${DATA_ROOT}" --download "$@"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
REQUIRE_MODEL=0 source "${SCRIPT_DIR}/common.sh"

mkdir -p "${ALFWORLD_DATA}"

ALFWORLD_READY=true
for required_path in \
    "${ALFWORLD_DATA}/json_2.1.1/train" \
    "${ALFWORLD_DATA}/json_2.1.1/valid_seen" \
    "${ALFWORLD_DATA}/json_2.1.1/valid_unseen" \
    "${ALFWORLD_DATA}/logic/alfred.pddl" \
    "${ALFWORLD_DATA}/logic/alfred.twl2" \
    "${ALFWORLD_DATA}/detectors/mrcnn_alfred_objects_sep13_004.pth"; do
    if [[ ! -e "${required_path}" ]]; then
        ALFWORLD_READY=false
        break
    fi
done

if [[ -z "$(find "${ALFWORLD_DATA}/json_2.1.1/train" -name game.tw-pddl -print -quit 2>/dev/null)" ]]; then
    ALFWORLD_READY=false
fi

if [[ "${ALFWORLD_READY}" == "true" ]]; then
    echo "ALFWorld assets already exist; skipping download."
else
    alfworld-download --data-dir "${ALFWORLD_DATA}"
fi

echo "ALFWorld data is ready under ${ALFWORLD_DATA}"

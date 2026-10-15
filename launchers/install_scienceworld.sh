#!/usr/bin/env bash
# Add only the two optional ScienceWorld packages; never resolve/upgrade Torch,
# Transformers, vLLM, NumPy, or the dependencies of an existing benchmark.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQUIRE_MODEL=0 source "${SCRIPT_DIR}/common.sh"
python -m pip install --no-deps --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" \
    scienceworld==1.2.3 py4j==0.10.9.9
python - <<'PY'
from scienceworld import ScienceWorldEnv
from agent_system.scienceworld_protocol import verify_simulator_catalog
env = ScienceWorldEnv()
try:
    verify_simulator_catalog(env)
    print("ScienceWorld simulator and AgentGym task-ID catalog verified.")
finally:
    env.close()
PY

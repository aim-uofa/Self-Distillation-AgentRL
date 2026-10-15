#!/usr/bin/env bash

set -euo pipefail

LOCAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${LOCAL_SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(realpath -m -- "${PROJECT_ROOT:-${REPO_ROOT}/..}")"
export PROJECT_ROOT

CONDA_ENV="${CONDA_ENV:-agentrl}"
MODEL_PATH="$(realpath -m -- "${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3.5-2B}")"
DATA_ROOT="$(realpath -m -- "${DATA_ROOT:-${PROJECT_ROOT}/data}")"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/agentrl_ray_${UID}}"

# shellcheck source=conda.sh
source "${LOCAL_SCRIPT_DIR}/conda.sh"
agentrl_activate_conda

# Keep child Bash shells in the environment activated above.
export BASH_ENV=/dev/null

if [[ "${REQUIRE_MODEL:-1}" == 1 && ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Hugging Face model not found at ${MODEL_PATH}; set MODEL_PATH or run bash launchers/prepare_model.sh." >&2
    exit 1
fi

mkdir -p \
    "${DATA_ROOT}" \
    "${PROJECT_ROOT}/checkpoints" \
    "${HF_HOME:-${PROJECT_ROOT}/hf_cache}" \
    "${PROJECT_ROOT}/logs" \
    "${RAY_TMPDIR}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${DATA_ROOT}/alfworld}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"
export RAY_TMPDIR
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"

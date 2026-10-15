#!/usr/bin/env bash
# Reconstruct the reported Qwen3.5 environment in a fresh Conda environment.
# Does not download models/data or change training/evaluation settings.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONSTRAINTS="${REPO_ROOT}/constraints.txt"
CONDA_ENV="${CONDA_ENV:-agentrl}"
REUSE_ENV="${REUSE_ENV:-0}"
INSTALL_RETRIEVER="${INSTALL_RETRIEVER:-1}"

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "The reference profile targets Linux x86_64; other platforms are not validated." >&2
    exit 1
fi
if [[ ! "${CONDA_ENV}" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ || "${CONDA_ENV}" == base || "${CONDA_ENV}" == root ]]; then
    echo "CONDA_ENV must be a named, non-base environment (not a filesystem path)." >&2
    exit 1
fi
if [[ ! "${REUSE_ENV}" =~ ^[01]$ || ! "${INSTALL_RETRIEVER}" =~ ^[01]$ ]]; then
    echo "REUSE_ENV and INSTALL_RETRIEVER must be 0 or 1." >&2
    exit 1
fi
# shellcheck source=conda.sh
source "${SCRIPT_DIR}/conda.sh"
agentrl_init_conda

# Check all registered prefixes, including custom envs_dirs, before any install.
# Command/JSON errors must stop here, not be mistaken for a nonexistent env.
CONDA_ENVS_JSON="$(conda env list --json)"
ENV_EXISTS="$(printf '%s' "${CONDA_ENVS_JSON}" | "${CONDA_BASE}/bin/python" -c \
    'import json, pathlib, sys; print(int(any(pathlib.Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"])))' "${CONDA_ENV}")"
if [[ "${ENV_EXISTS}" == 1 || -e "${CONDA_BASE}/envs/${CONDA_ENV}" ]]; then
    if [[ "${REUSE_ENV}" != 1 ]]; then
        echo "Environment ${CONDA_ENV} already exists; no packages were changed." >&2
        echo "Choose a fresh CONDA_ENV. REUSE_ENV=1 is only for intentionally resuming an installation." >&2
        exit 1
    fi
    echo "REUSE_ENV=1: updating packages in ${CONDA_ENV} to the reference versions."
else
    # OpenJDK came from defaults in the supplied environment; CUDA runtimes from
    # nvidia. These runtime packages do not provide the nvcc build toolchain.
    conda create -n "${CONDA_ENV}" --override-channels \
        -c conda-forge -c defaults -c nvidia \
        "python=3.10.20" "defaults::openjdk=11.0.27" pip \
        "nvidia::cuda-cudart=12.9.79" "nvidia::cuda-nvrtc=12.9.86" \
        "nvidia::libcublas=12.9.2.10" -y
fi

set +u
conda activate "${CONDA_ENV}"
set -u
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export JAVA_HOME="${CONDA_PREFIX}"
hash -r
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "This profile requires Python 3.10"'

python -m pip install -c "${CONSTRAINTS}" \
    "pip==26.2" "setuptools==83.0.0" "wheel==0.47.0" \
    ninja packaging pybind11 Cython

# Use the default PyPI Torch 2.10 build (reference: wheel build 3, CUDA 12.8),
# not an unrelated cu130/CPU wheel. Mirror/index selection remains user-controlled.
python -m pip install -c "${CONSTRAINTS}" \
    "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0"
python -m pip install -c "${CONSTRAINTS}" -r "${REPO_ROOT}/requirements.txt"

# Build/download extensions only after Torch and their Python dependencies exist.
# Matching wheels or a compatible CUDA toolkit (nvcc) and C++ compiler are needed.
# MAX_JOBS limits build RAM; TORCH_CUDA_ARCH_LIST/CUDA_HOME may be supplied by caller.
MAX_JOBS="${MAX_JOBS:-8}" python -m pip install \
    -c "${CONSTRAINTS}" --no-build-isolation --no-deps \
    -r "${REPO_ROOT}/requirements-kernels.txt"

if [[ "${INSTALL_RETRIEVER}" == 1 ]]; then
    python -m pip install -c "${CONSTRAINTS}" "faiss-gpu==1.15.0"
fi
# spaCy models are release assets, not ordinary PyPI packages.
python -m pip install -c "${CONSTRAINTS}" \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# Both local distributions come from the checkout, never a same-named PyPI project.
python -m pip install -c "${CONSTRAINTS}" --no-build-isolation \
    -e "${REPO_ROOT}" \
    -e "${REPO_ROOT}/agent_system/environments/env_package/search/third_party"

# Install the prebuilt metadata-corrected wheels before final validation.
COMPAT_WHEEL_DIR="${REPO_ROOT}/compat_wheels"
if [[ ! -f "${COMPAT_WHEEL_DIR}/requirements.txt" ]]; then
    echo "Missing compatibility wheels under ${COMPAT_WHEEL_DIR}" >&2
    echo "The repository distribution must include these artifacts." >&2
    exit 1
fi
(
    cd "${COMPAT_WHEEL_DIR}"
    python -m pip install \
        -c "${CONSTRAINTS}" \
        --no-index \
        --no-deps \
        --force-reinstall \
        --require-hashes \
        -r requirements.txt
)

python -m pip check
python -c 'import decord; import textworld; print("decord/textworld imports OK")'
java -version
python - <<'PY'
from importlib.metadata import version
import torch
import flash_attn
import causal_conv1d
import verl

for name in ("torch", "vllm", "transformers", "ray", "numpy", "flash-attn", "causal-conv1d", "scienceworld"):
    print(f"{name}=={version(name)}")
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Repository import:", verl.__file__)
PY

echo "Environment installed: ${CONDA_ENV}"
echo "Activate with: conda activate ${CONDA_ENV}"
echo "Dependency/import checks do not replace a GPU training/evaluation smoke test."

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${1:-}"
TASK="${2:-}"
if [[ -z "${METHOD}" || -z "${TASK}" ]]; then
    echo "Usage: $0 <agentopsd|adrs|sdar|icsd|pcsd|ocsd|ovcsd|grpo_opsd|skillsd|rlsd> <alfworld|webshop|search|scienceworld> <merged model directory> [Hydra overrides ...]" >&2
    exit 2
fi
shift 2

case "${METHOD}" in
    agentopsd)
        MODEL_DIR="${1:-${AGENTOPSD_MODEL_PATH:-}}"
        EVAL_DESCRIPTION_DEFAULT="merged AgentOPSD model"
        # Privileged skills and recursive credit are training-only.
        EVAL_RUNNER_DEFAULT="${SCRIPT_DIR}/run_grpo_1gpu.sh"
        MERGE_SCRIPT=merge_agentopsd_checkpoint.sh
        ;;
    adrs)
        MODEL_DIR="${1:-${ADRS_MODEL_PATH:-}}"
        EVAL_DESCRIPTION_DEFAULT="merged ADRS model"
        # TVA reward shaping is training-only.
        EVAL_RUNNER_DEFAULT="${SCRIPT_DIR}/run_grpo_1gpu.sh"
        MERGE_SCRIPT=merge_adrs_checkpoint.sh
        ;;
    ocsd)
        MODEL_DIR="${1:-${OCSD_MODEL_PATH:-}}"
        EVAL_DESCRIPTION_DEFAULT="merged OCSD model"
        # Future-evidence replay is training-only; evaluation uses the same
        # ordinary rollout protocol and task selection as the other methods.
        EVAL_RUNNER_DEFAULT="${SCRIPT_DIR}/run_grpo_1gpu.sh"
        MERGE_SCRIPT=merge_ocsd_checkpoint.sh
        ;;
    ovcsd)
        MODEL_DIR="${1:-${OVCSD_MODEL_PATH:-}}"
        EVAL_DESCRIPTION_DEFAULT="merged OVCSD model"
        # Privileged skills and intervention branches are training-only.
        EVAL_RUNNER_DEFAULT="${SCRIPT_DIR}/run_grpo_1gpu.sh"
        MERGE_SCRIPT=merge_ovcsd_checkpoint.sh
        ;;
    sdar)
        MODEL_DIR="${1:-${SDAR_MODEL_PATH:-}}"
        EVAL_DESCRIPTION_DEFAULT="merged SDAR model"
        EVAL_RUNNER_DEFAULT="${SCRIPT_DIR}/run_sdar_1gpu.sh"
        MERGE_SCRIPT=merge_sdar_checkpoint.sh
        ;;
    icsd)
        MODEL_DIR="${1:-${ICSD_MODEL_PATH:-}}"
        EVAL_DESCRIPTION_DEFAULT="merged ICSD model"
        # Privileged skills and influence allocation are training-only.
        EVAL_RUNNER_DEFAULT="${SCRIPT_DIR}/run_grpo_1gpu.sh"
        MERGE_SCRIPT=merge_icsd_checkpoint.sh
        ;;
    pcsd)
        MODEL_DIR="${1:-${PCSD_MODEL_PATH:-}}"
        EVAL_DESCRIPTION_DEFAULT="merged PCSD model"
        # Persistent weighting and the frozen skill teacher are training-only.
        EVAL_RUNNER_DEFAULT="${SCRIPT_DIR}/run_grpo_1gpu.sh"
        MERGE_SCRIPT=merge_pcsd_checkpoint.sh
        ;;
    grpo_opsd)
        MODEL_DIR="${1:-${GRPO_OPSD_MODEL_PATH:-}}"
        EVAL_DESCRIPTION_DEFAULT="merged GRPO+OPSD model"
        EVAL_RUNNER_DEFAULT="${SCRIPT_DIR}/run_grpo_opsd_1gpu.sh"
        MERGE_SCRIPT=merge_grpo_opsd_checkpoint.sh
        ;;
    skillsd)
        MODEL_DIR="${1:-${SKILLSD_MODEL_PATH:-}}"
        EVAL_DESCRIPTION_DEFAULT="merged Skill-SD model"
        EVAL_RUNNER_DEFAULT="${SCRIPT_DIR}/run_skillsd_1gpu.sh"
        MERGE_SCRIPT=merge_skillsd_checkpoint.sh
        ;;
    rlsd)
        MODEL_DIR="${1:-${RLSD_MODEL_PATH:-}}"
        EVAL_DESCRIPTION_DEFAULT="merged RLSD model"
        EVAL_RUNNER_DEFAULT="${SCRIPT_DIR}/run_rlsd_1gpu.sh"
        MERGE_SCRIPT=merge_rlsd_checkpoint.sh
        ;;
    *)
        echo "Unsupported method: ${METHOD}" >&2
        exit 2
        ;;
esac

if [[ -z "${MODEL_DIR}" ]]; then
    echo "Usage: $0 ${METHOD} <alfworld|webshop|search|scienceworld> <merged model directory> [Hydra overrides ...]" >&2
    exit 2
fi
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "Merged Hugging Face model not found at ${MODEL_DIR}" >&2
    echo "Run launchers/${MERGE_SCRIPT} first." >&2
    exit 1
fi

export MODEL_PATH="$(realpath -e -- "${MODEL_DIR}")"
export EVAL_KIND="${METHOD}"
export EVAL_DESCRIPTION="${EVAL_DESCRIPTION:-${EVAL_DESCRIPTION_DEFAULT}}"
export EVAL_RUNNER="${EVAL_RUNNER_DEFAULT}"
export TRAIN_LOG_FILE=/dev/null

exec "${SCRIPT_DIR}/test_pretrained_1gpu.sh" "${TASK}" "$@"

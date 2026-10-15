#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

TASK="${1:-}"
if [[ -z "${TASK}" ]]; then
    echo "Usage: $0 <alfworld|webshop|search|scienceworld> [Hydra overrides ...]" >&2
    exit 2
fi
shift

if [[ "${PAPER_CONFIG:-0}" == "1" ]]; then
    # Standalone evaluation keeps the same default for every model size.
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
fi
if [[ "${TASK}" == "alfworld" && "${PAPER_CONFIG:-0}" == "1" ]]; then
    EVAL_SIZE="${EVAL_SIZE:-140}"
fi
EVAL_SIZE="${EVAL_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-false}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"

if [[ "${EVAL_SIZE}" == "all" && "${TASK}" != "search" && "${TASK}" != "scienceworld" ]]; then
    echo "EVAL_SIZE=all is only supported for Search-QA and ScienceWorld." >&2
    exit 2
fi
if [[ "${EVAL_SIZE}" != "all" && ! "${EVAL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_SIZE must be a positive integer or 'all' for Search-QA/ScienceWorld, got: ${EVAL_SIZE}" >&2
    exit 2
fi
if [[ -z "${EVAL_BATCH_SIZE}" ]]; then
    if [[ "${EVAL_SIZE}" == "all" ]]; then
        EVAL_BATCH_SIZE=512
        if [[ "${TASK}" == "scienceworld" ]]; then EVAL_BATCH_SIZE=32; fi
    else
        EVAL_BATCH_SIZE="${EVAL_SIZE}"
    fi
fi
if [[ ! "${EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_BATCH_SIZE must be a positive integer, got: ${EVAL_BATCH_SIZE}" >&2
    exit 2
fi

case "${TASK}" in
    alfworld)
        EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-50}"
        ;;
    webshop)
        EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-15}"
        ;;
    search)
        EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-4}"
        ;;
    scienceworld)
        EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-20}"
        ;;
    *)
        echo "Unsupported task: ${TASK}" >&2
        exit 2
        ;;
esac

# The selected trainer still constructs the training-side dataloader and envs.
# Keep that unused side at one sample; val_only returns after validation.
export TRAIN_DATA_SIZE=1
export VAL_DATA_SIZE="${EVAL_BATCH_SIZE}"
export GROUP_SIZE=1
export PPO_MINI_BATCH_SIZE=1
export PPO_MICRO_BATCH_SIZE=1
export LOG_PROB_MICRO_BATCH_SIZE=1
export TOTAL_TRAINING_STEPS=1
export VAL_BEFORE_TRAIN=true
export MAX_ENV_STEPS="${EVAL_MAX_STEPS}"
export MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-512}"

DATA_OVERRIDES=()
if [[ "${TASK}" == "scienceworld" ]]; then
    SCIENCEWORLD_TEST_DATA="$(python "${SCRIPT_DIR}/prepare_scienceworld.py" \
        --data-root "${DATA_ROOT}" --eval-size "${EVAL_SIZE}")"
    DATA_OVERRIDES+=("data.val_files=${SCIENCEWORLD_TEST_DATA}")
fi
if [[ "${TASK}" == "search" ]]; then
    SEARCH_SOURCE_DIR="${DATA_ROOT}/searchR1_processed_direct"
    SEARCH_EVAL_DIR="${DATA_ROOT}/searchR1_pretrained_eval"
    SEARCH_TRAIN_SUBSET="${SEARCH_EVAL_DIR}/train_1.parquet"

    for required_path in \
        "${SEARCH_SOURCE_DIR}/train.parquet" \
        "${SEARCH_SOURCE_DIR}/test.parquet"; do
        if [[ ! -s "${required_path}" ]]; then
            echo "Missing ${required_path}" >&2
            echo "Run: bash launchers/prepare_search.sh qa" >&2
            exit 1
        fi
    done

    SEARCH_TEST_DATA="$(python "${SCRIPT_DIR}/prepare_search_eval_subset.py" \
        --data-root "${DATA_ROOT}" --eval-size "${EVAL_SIZE}" \
        --eval-seed "${EVAL_SEED}" --prepare-train-subset)"

    SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:${PORT:-19080}/retrieve}"
    if [[ "${SKIP_RETRIEVER_CHECK:-0}" != "1" ]]; then
        if ! curl -fsS --max-time 30 \
            -H "Content-Type: application/json" \
            -d '{"query":"What is retrieval?","topk":1}' \
            "${SEARCH_URL}" >/dev/null; then
            echo "Search retriever is not ready at ${SEARCH_URL}" >&2
            echo "Run in another terminal: bash launchers/start_search_retriever.sh" >&2
            exit 1
        fi
    fi

    DATA_OVERRIDES+=(
        "data.train_files=${SEARCH_TRAIN_SUBSET}"
        "data.val_files=${SEARCH_TEST_DATA}"
    )
fi

EVAL_KIND="${EVAL_KIND:-pretrained}"
MODEL_NAME="$(basename -- "${MODEL_PATH%/}")"
MODEL_NAME="${MODEL_NAME//[^a-zA-Z0-9_.-]/_}"
EVAL_MODEL_TAG="${EVAL_MODEL_TAG:-${MODEL_NAME}}"
EVAL_DESCRIPTION="${EVAL_DESCRIPTION:-untouched pretrained model}"
EVAL_RUNNER="${EVAL_RUNNER:-${SCRIPT_DIR}/run_sdar_1gpu.sh}"

LOG_DIR="${PROJECT_ROOT}/logs/${EVAL_KIND}_eval"
mkdir -p "${LOG_DIR}"
ROLLOUT_SEED_SUFFIX=""
if [[ -n "${EVAL_ROLLOUT_SEED:-}" ]]; then
    ROLLOUT_SEED_SUFFIX="_rseed${EVAL_ROLLOUT_SEED}"
fi
LOG_TASK_NAME="${TASK}"
if [[ "${TASK}" == "search" ]]; then
    LOG_TASK_NAME=searchQA
fi
LOG_FILE="${LOG_DIR}/${LOG_TASK_NAME}_${EVAL_MODEL_TAG}_seed${EVAL_SEED}${ROLLOUT_SEED_SUFFIX}_n${EVAL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

echo "Evaluating ${EVAL_DESCRIPTION}: ${MODEL_PATH}"
echo "Task=${TASK}, samples=${EVAL_SIZE}, eval_batch_size=${EVAL_BATCH_SIZE}, max_steps=${EVAL_MAX_STEPS}, GPU=${GPU_ID:-0}"
echo "Log=${LOG_FILE}"

"${EVAL_RUNNER}" "${TASK}" \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "trainer.experiment_name=${EVAL_KIND}_${EVAL_MODEL_TAG}_${TASK}_seed${EVAL_SEED}_n${EVAL_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    "actor_rollout_ref.rollout.val_kwargs.do_sample=${EVAL_DO_SAMPLE}" \
    "actor_rollout_ref.rollout.val_kwargs.temperature=${EVAL_TEMPERATURE}" \
    "env.seed=${EVAL_SEED}" \
    "${DATA_OVERRIDES[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"

echo
echo "Evaluation complete. Final metric line:"
grep "step:0" "${LOG_FILE}" | tail -n 1 || true

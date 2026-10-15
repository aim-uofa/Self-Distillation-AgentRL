#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(realpath -m -- "${PROJECT_ROOT:-${REPO_ROOT}/..}")"
export PROJECT_ROOT
# shellcheck source=conda.sh
source "${SCRIPT_DIR}/conda.sh"

TASK="${TASK:-${1:-}}"
MODEL_KIND="${MODEL_KIND:-pretrained}"
MAX_PASS_K="${MAX_PASS_K:-${MAX_PASS_NUM:-}}"
TASK_SEED="${TASK_SEED:-0}"
TEMPERATURE="${TEMPERATURE:-}"
EVAL_SEED_MODE="${EVAL_SEED_MODE:-per_task_turn}"
EVAL_SEED_MODES="${EVAL_SEED_MODES:-${EVAL_SEED_MODE}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROLLOUT_SEEDS="${ROLLOUT_SEEDS:-}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"
ATTEMPT_MAX_RETRIES="${ATTEMPT_MAX_RETRIES:-4}"
ATTEMPT_RETRY_DELAY_SECONDS="${ATTEMPT_RETRY_DELAY_SECONDS:-15}"

usage() {
    cat >&2 <<'EOF'
Usage:
  TASK=<alfworld|webshop|search|scienceworld> MODEL_KIND=<pretrained|grpo> \
  MAX_PASS_K=<K> TEMPERATURE=<temperature> [MODEL_PATH=...] \
  [GRPO_MODEL_PATH=...] bash launchers/eval_pass_at_k.sh

Required:
  MAX_PASS_K       Report pass@1 through pass@K.
  TEMPERATURE      Explicit validation sampling temperature, for example 1.0.

Optional:
  TASK_SEED=0              Fixed seed selecting the evaluation task set.
  ROLLOUT_SEEDS=0,1,2,3    Sampling seeds; defaults to 0..MAX_PASS_K-1.
  GPU_IDS=0,1,2,3          Run seed attempts in parallel, one process per GPU.
  MODEL_TAG=<name>          Safe output-directory label.
  PASS_AT_K_OUTPUT_DIR=...  Exact output directory.
  RUN_ID=<name>             Reusable run id for interrupted-run continuation.
  EVAL_SEED_MODE=per_task_turn  Default sampling protocol; legacy is also supported.
  EVAL_SEED_MODES=legacy,per_task_turn  Evaluate both protocols separately.
  ATTEMPT_MAX_RETRIES=4     Retries after a failed attempt (5 total runs).
  ATTEMPT_RETRY_DELAY_SECONDS=15  Delay before each retry.
EOF
}

if [[ -z "${TASK}" || -z "${MAX_PASS_K}" || -z "${TEMPERATURE}" ]]; then
    usage
    exit 2
fi
if [[ ! "${TASK}" =~ ^(alfworld|webshop|search|scienceworld)$ ]]; then
    echo "Unsupported TASK=${TASK}" >&2
    exit 2
fi
# Resolve the same paper defaults before data preparation and resume checks.
if [[ "${TASK}" == "alfworld" && "${PAPER_CONFIG:-0}" == "1" ]]; then
    export ALFWORLD_STRICT_EVAL="${ALFWORLD_STRICT_EVAL:-1}"
    export EVAL_SIZE="${EVAL_SIZE:-140}"
fi
# New benchmark only: allow explicit runtime overrides on a shared GPU without
# changing the established argument handling of the other benchmarks.
SCIENCEWORLD_EXTRA_OVERRIDES=()
if [[ "${TASK}" == "scienceworld" ]]; then
    if [[ "${1:-}" == "scienceworld" ]]; then shift; fi
    SCIENCEWORLD_EXTRA_OVERRIDES=("$@")
fi
if [[ ! "${MODEL_KIND}" =~ ^(pretrained|grpo)$ ]]; then
    echo "MODEL_KIND must be pretrained or grpo, got: ${MODEL_KIND}" >&2
    exit 2
fi
if [[ ! "${MAX_PASS_K}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_PASS_K must be a positive integer, got: ${MAX_PASS_K}" >&2
    exit 2
fi
if [[ ! "${TASK_SEED}" =~ ^[0-9]+$ ]]; then
    echo "TASK_SEED must be a non-negative integer, got: ${TASK_SEED}" >&2
    exit 2
fi
if [[ ! "${ATTEMPT_MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "ATTEMPT_MAX_RETRIES must be a non-negative integer, got: ${ATTEMPT_MAX_RETRIES}" >&2
    exit 2
fi
if [[ ! "${ATTEMPT_RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "ATTEMPT_RETRY_DELAY_SECONDS must be a non-negative integer, got: ${ATTEMPT_RETRY_DELAY_SECONDS}" >&2
    exit 2
fi
if [[ ! "${EVAL_SEED_MODE}" =~ ^(legacy|per_task_turn)$ ]]; then
    echo "Unsupported EVAL_SEED_MODE=${EVAL_SEED_MODE}" >&2
    exit 2
fi
if [[ ! "${EVAL_SEED_MODES}" =~ ^(legacy|per_task_turn)(,(legacy|per_task_turn))*$ ]]; then
    echo "EVAL_SEED_MODES must contain legacy and/or per_task_turn, got: ${EVAL_SEED_MODES}" >&2
    exit 2
fi
IFS=',' read -r -a SEED_MODE_LIST <<< "${EVAL_SEED_MODES}"
declare -A SEEN_MODES=()
for mode in "${SEED_MODE_LIST[@]}"; do
    if [[ -n "${SEEN_MODES[${mode}]:-}" ]]; then
        echo "Duplicate EVAL_SEED_MODES value: ${mode}" >&2
        exit 2
    fi
    SEEN_MODES["${mode}"]=1
done

if [[ -z "${ROLLOUT_SEEDS}" ]]; then
    SEEDS=()
    for ((index = 0; index < MAX_PASS_K; index++)); do
        SEEDS+=("${index}")
    done
else
    IFS=',' read -r -a SEEDS <<< "${ROLLOUT_SEEDS}"
fi

declare -A SEEN_SEEDS=()
for seed in "${SEEDS[@]}"; do
    if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
        echo "Every ROLLOUT_SEEDS value must be a non-negative integer, got: ${seed}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_SEEDS[${seed}]:-}" ]]; then
        echo "Duplicate rollout seed: ${seed}" >&2
        exit 2
    fi
    SEEN_SEEDS["${seed}"]=1
done
if (( ${#SEEDS[@]} < MAX_PASS_K )); then
    echo "Need at least MAX_PASS_K=${MAX_PASS_K} rollout seeds, got ${#SEEDS[@]}." >&2
    exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if (( ${#GPU_LIST[@]} == 0 )); then
    echo "GPU_IDS must contain at least one GPU index." >&2
    exit 2
fi
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_LIST[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Every GPU_IDS value must be a non-negative integer, got: ${gpu_id}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
        echo "Duplicate GPU index: ${gpu_id}" >&2
        exit 2
    fi
    SEEN_GPUS["${gpu_id}"]=1
done

if [[ "${MODEL_KIND}" == "grpo" ]]; then
    GRPO_MODEL_PATH="${GRPO_MODEL_PATH:-${MODEL_PATH:-}}"
    if [[ -z "${GRPO_MODEL_PATH}" || ! -f "${GRPO_MODEL_PATH}/config.json" ]]; then
        echo "GRPO_MODEL_PATH must point to a merged Hugging Face checkpoint." >&2
        echo "Merge a raw global_step_N checkpoint with launchers/merge_grpo_checkpoint.sh first." >&2
        exit 1
    fi
    GRPO_MODEL_PATH="$(realpath -e -- "${GRPO_MODEL_PATH}")"
    MODEL_NAME="$(basename -- "${GRPO_MODEL_PATH}")"
else
    MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3.5-2B}"
    if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
        echo "MODEL_PATH does not contain config.json: ${MODEL_PATH}" >&2
        exit 1
    fi
    MODEL_PATH="$(realpath -e -- "${MODEL_PATH}")"
    MODEL_NAME="$(basename -- "${MODEL_PATH}")"
    export MODEL_PATH
fi

MODEL_NAME="${MODEL_NAME//[^a-zA-Z0-9_.-]/_}"
MODEL_TAG="${MODEL_TAG:-${MODEL_NAME}_${MODEL_KIND}}"
if [[ ! "${MODEL_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "MODEL_TAG may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi
CONDA_ENV="${CONDA_ENV:-agentrl}"
agentrl_activate_conda

if (( ${#SEED_MODE_LIST[@]} > 1 )); then
    if [[ -n "${PASS_AT_K_OUTPUT_DIR:-}" ]]; then
        echo "Set separate PASS_AT_K_OUTPUT_DIR values when running multiple seed modes." >&2
        exit 2
    fi
    for mode in "${SEED_MODE_LIST[@]}"; do
        TASK="${TASK}" MODEL_KIND="${MODEL_KIND}" MODEL_TAG="${MODEL_TAG}" \
        MODEL_PATH="${MODEL_PATH:-}" GRPO_MODEL_PATH="${GRPO_MODEL_PATH:-}" \
        MAX_PASS_K="${MAX_PASS_K}" TASK_SEED="${TASK_SEED}" TEMPERATURE="${TEMPERATURE}" \
        RUN_ID="${RUN_ID}" ROLLOUT_SEEDS="${ROLLOUT_SEEDS}" GPU_IDS="${GPU_IDS}" \
        EVAL_SEED_MODE="${mode}" EVAL_SEED_MODES="${mode}" \
        bash "${SCRIPT_DIR}/eval_pass_at_k.sh" "$@"
    done
    exit 0
fi
EVAL_SEED_MODE="${SEED_MODE_LIST[0]}"
if [[ "${EVAL_SEED_MODE}" == "per_task_turn" ]]; then
    SEED_DERIVATION_VERSION=1
    if ! python3 -c 'import sys; assert float(sys.argv[1]) > 0' "${TEMPERATURE}" 2>/dev/null; then
        echo "per_task_turn requires TEMPERATURE > 0." >&2
        exit 2
    fi
else
    SEED_DERIVATION_VERSION=0
fi

OUTPUT_DIR="${PASS_AT_K_OUTPUT_DIR:-${PROJECT_ROOT}/logs/pass_at_k/${TASK}/${MODEL_TAG}/task_seed_${TASK_SEED}/${RUN_ID}/seed_${EVAL_SEED_MODE}_v${SEED_DERIVATION_VERSION}}"
mkdir -p "${OUTPUT_DIR}"

# ALFWorld and WebShop use a small placeholder parquet to size their vectorized
# environments. Prepare it once before parallel workers start, then prevent each
# worker from concurrently rewriting the shared file.
if [[ "${TASK}" == "alfworld" || "${TASK}" == "webshop" ]]; then
    AGENT_EVAL_SIZE="${EVAL_SIZE:-128}"
    AGENT_EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${AGENT_EVAL_SIZE}}"
    if [[ ! "${AGENT_EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "EVAL_BATCH_SIZE must be a positive integer for ${TASK}." >&2
        exit 2
    fi

    DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
    export DATA_ROOT
    (
        cd "${REPO_ROOT}"
        python -m examples.data_preprocess.prepare \
            --mode text \
            --local_dir "${DATA_ROOT}/verl-agent" \
            --train_data_size 1 \
            --val_data_size "${AGENT_EVAL_BATCH_SIZE}"
    )
    export SKIP_AGENT_DATA_PREP=1
fi

# Search workers must share one dataset identity, including across resume.
# Keep this setup separate so ALFWorld/WebShop behavior is unchanged.
if [[ "${TASK}" == "search" ]]; then
    DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
    export DATA_ROOT
    SEARCH_TEST_DATA="$(python "${SCRIPT_DIR}/prepare_search_eval_subset.py" \
        --data-root "${DATA_ROOT}" --eval-size "${EVAL_SIZE:-all}" \
        --eval-seed "${TASK_SEED}" --prepare-train-subset --run-dir "${OUTPUT_DIR}")"
fi

# ScienceWorld uses a pinned dataset; validate it before starting any workers.
if [[ "${TASK}" == "scienceworld" ]]; then
    export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
    python "${SCRIPT_DIR}/prepare_scienceworld.py" --data-root "${DATA_ROOT}" \
        --eval-size "${EVAL_SIZE:-200}" >/dev/null
fi

echo "Pass@k evaluation"
echo "  task=${TASK}"
echo "  model_kind=${MODEL_KIND}"
echo "  model_tag=${MODEL_TAG}"
echo "  task_seed=${TASK_SEED}"
echo "  rollout_seeds=${SEEDS[*]}"
echo "  gpu_ids=${GPU_LIST[*]}"
echo "  temperature=${TEMPERATURE}"
echo "  eval_seed_mode=${EVAL_SEED_MODE} (v${SEED_DERIVATION_VERSION})"
echo "  output=${OUTPUT_DIR}"
if [[ "${TASK}" == "search" ]]; then
    echo "  search_eval_data=${SEARCH_TEST_DATA}"
fi

EXPECTED_EVAL_SIZE="${EVAL_SIZE:-128}"
if [[ "${TASK}" == "search" ]]; then
    EXPECTED_EVAL_SIZE="${EVAL_SIZE:-all}"
fi
if [[ "${TASK}" == "scienceworld" ]]; then
    EXPECTED_EVAL_SIZE="${EVAL_SIZE:-200}"
    if [[ "${EXPECTED_EVAL_SIZE}" == "all" ]]; then EXPECTED_EVAL_SIZE=200; fi
fi
EXPECTED_VAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${EXPECTED_EVAL_SIZE}}"
if [[ "${TASK}" == "scienceworld" ]]; then
    EXPECTED_VAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
fi
if [[ "${TASK}" == "search" && "${EXPECTED_EVAL_SIZE}" == "all" && -z "${EVAL_BATCH_SIZE:-}" ]]; then
    EXPECTED_VAL_BATCH_SIZE=512
fi
EXPECTED_ALFWORLD_STRICT_EVAL=0
if [[ "${TASK}" == "alfworld" && "${ALFWORLD_STRICT_EVAL:-0}" =~ ^(1|true|yes)$ ]]; then
    EXPECTED_ALFWORLD_STRICT_EVAL=1
fi

run_attempt() {
    local attempt_index="$1"
    local rollout_seed="$2"
    local gpu_id="$3"
    local attempt_dir
    local success_marker
    local attempt_ray_tmp
    local retry_index=0

    attempt_dir="${OUTPUT_DIR}/attempt_$(printf '%03d' "${attempt_index}")_seed_${rollout_seed}"
    success_marker="${attempt_dir}/_SUCCESS"
    mkdir -p "${attempt_dir}"

    if [[ -f "${success_marker}" ]]; then
        if [[ "${TASK}" == "search" ]]; then
            python "${SCRIPT_DIR}/prepare_search_eval_subset.py" --check-attempt "${attempt_dir}" || return 1
        fi
        if [[ "${TASK}" == "scienceworld" ]]; then
            python "${SCRIPT_DIR}/prepare_scienceworld.py" --data-root "${DATA_ROOT}" \
                --eval-size "${EXPECTED_EVAL_SIZE}" --check-attempt "${attempt_dir}" || return 1
        fi
        python3 - "${attempt_dir}/manifest.json" "${TASK}" "${MODEL_KIND}" \
            "${MODEL_TAG}" "${TASK_SEED}" "${rollout_seed}" "${attempt_index}" \
            "${TEMPERATURE}" "${EVAL_SEED_MODE}" "${SEED_DERIVATION_VERSION}" \
            "${MODEL_PATH:-${GRPO_MODEL_PATH}}" "${EXPECTED_VAL_BATCH_SIZE}" \
            "${EVAL_MAX_STEPS:-}" "${EXPECTED_ALFWORLD_STRICT_EVAL}" "${EXPECTED_EVAL_SIZE}" <<'PY'
import json
import os
import sys

(path, task, model_kind, model_tag, task_seed, rollout_seed, attempt_index,
 temperature, mode, version, model_path, val_batch_size, max_env_steps,
 strict_eval, eval_size) = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    manifest = json.load(handle)
expected = {
    "benchmark": task,
    "model_kind": model_kind,
    "model_tag": model_tag,
    "task_seed": int(task_seed),
    "rollout_seed": int(rollout_seed),
    "attempt_index": int(attempt_index),
    "eval_seed_mode": mode,
    "seed_derivation_version": int(version),
    "val_batch_size": int(val_batch_size),
    "do_sample": True,
    "max_env_steps": int(max_env_steps or {"alfworld": 50, "webshop": 15, "search": 4, "scienceworld": 20}[task]),
}
if task == "alfworld":
    expected["alfworld_strict_eval"] = bool(int(strict_eval))
if task == "scienceworld":
    expected.update({
        "max_prompt_length": int(os.environ.get("MAX_PROMPT_LENGTH", 4096)),
        "max_response_length": int(os.environ.get("EVAL_MAX_RESPONSE_LENGTH", 512)),
        "scienceworld_reward_mode": os.environ.get("SCIENCEWORLD_REWARD_MODE", "score"),
    })
for key, value in expected.items():
    actual = manifest.get(
        key,
        "legacy" if key == "eval_seed_mode" else
        0 if key == "seed_derivation_version" else
        False if key == "alfworld_strict_eval" else None,
    )
    if actual != value:
        raise SystemExit(f"Completed attempt mismatch for {key}: {actual!r} != {value!r} ({path})")
if float(manifest.get("temperature", -1)) != float(temperature):
    raise SystemExit(f"Completed attempt temperature mismatch: {path}")
if os.path.realpath(str(manifest.get("model_path"))) != os.path.realpath(model_path):
    raise SystemExit(f"Completed attempt model_path mismatch: {path}")
if not manifest.get("completed") or int(manifest.get("record_count", 0)) < 1:
    raise SystemExit(f"Completed attempt manifest is incomplete: {path}")
if eval_size != "all" and int(manifest["record_count"]) != int(eval_size if task in {"search", "scienceworld"} else val_batch_size):
    raise SystemExit(f"Completed attempt task count mismatch: {path}")
PY
        echo "Skipping completed attempt ${attempt_index}, rollout_seed=${rollout_seed}"
        return 0
    fi

    echo "Running attempt $((attempt_index + 1))/${#SEEDS[@]} with rollout_seed=${rollout_seed} on GPU ${gpu_id}"
    local gdn_prefill_backend=triton
    if [[ "${TASK}" == "scienceworld" ]] && ! CUDA_VISIBLE_DEVICES="${gpu_id}" python -c \
        'import inspect; from vllm.engine.arg_utils import EngineArgs; assert "gdn_prefill_backend" in inspect.signature(EngineArgs).parameters' \
        >/dev/null 2>&1; then
        # vLLM 0.11 predates GDN. The rollout worker drops null engine kwargs;
        # newer vLLM versions keep their existing Triton setting.
        gdn_prefill_backend=null
    fi
    common_overrides=(
        "+actor_rollout_ref.rollout.seed=${rollout_seed}"
        "+actor_rollout_ref.rollout.eval_seed_mode=${EVAL_SEED_MODE}"
        "+actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
        "actor_rollout_ref.rollout.enable_chunked_prefill=True"
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=${gdn_prefill_backend}"
        "trainer.validation_data_dir=${attempt_dir}"
        "+trainer.validation_attempt_index=${attempt_index}"
        "+trainer.validation_benchmark=${TASK}"
        "+trainer.validation_model_kind=${MODEL_KIND}"
        "+trainer.validation_model_tag=${MODEL_TAG}"
    )
    if [[ "${TASK}" == "scienceworld" ]]; then
        common_overrides+=("${SCIENCEWORLD_EXTRA_OVERRIDES[@]}")
    fi

    if [[ "${MODEL_KIND}" == "pretrained" ]]; then
        test_command=("${SCRIPT_DIR}/test_pretrained_${TASK}.sh" "${common_overrides[@]}")
    else
        test_command=("${SCRIPT_DIR}/test_grpo_${TASK}.sh" "${GRPO_MODEL_PATH}" "${common_overrides[@]}")
    fi

    while true; do
        # Ray appends a long session/sockets suffix; keep this base path short
        # and use a fresh one for each retry.
        attempt_ray_tmp="${RAY_TMPDIR_ROOT:-/tmp}/pk_${UID}_${$}_${attempt_index}_${retry_index}"
        if GPU_ID="${gpu_id}" \
           RAY_TMPDIR="${attempt_ray_tmp}" \
           EVAL_SEED="${TASK_SEED}" \
           EVAL_ROLLOUT_SEED="${rollout_seed}" \
           EVAL_DO_SAMPLE=true \
           EVAL_TEMPERATURE="${TEMPERATURE}" \
           EVAL_MODEL_TAG="${MODEL_TAG}" \
           "${test_command[@]}" 2>&1 | tee -a "${attempt_dir}/console.log"; then
            if [[ "${TASK}" == "search" ]]; then
                python "${SCRIPT_DIR}/prepare_search_eval_subset.py" --check-attempt "${attempt_dir}" || return 1
            fi
            if [[ "${TASK}" == "scienceworld" ]]; then
                python "${SCRIPT_DIR}/prepare_scienceworld.py" --data-root "${DATA_ROOT}" \
                    --eval-size "${EXPECTED_EVAL_SIZE}" --check-attempt "${attempt_dir}" || return 1
            fi
            touch "${success_marker}"
            return 0
        fi

        if (( retry_index >= ATTEMPT_MAX_RETRIES )); then
            echo "Attempt ${attempt_index} failed after $((retry_index + 1)) runs; no _SUCCESS marker was written." >&2
            return 1
        fi
        retry_index=$((retry_index + 1))
        echo "Attempt ${attempt_index} failed; retry ${retry_index}/${ATTEMPT_MAX_RETRIES} starts in ${ATTEMPT_RETRY_DELAY_SECONDS}s." >&2
        sleep "${ATTEMPT_RETRY_DELAY_SECONDS}"
    done
}

parallelism="${#GPU_LIST[@]}"
for ((batch_start = 0; batch_start < ${#SEEDS[@]}; batch_start += parallelism)); do
    pids=()
    for ((offset = 0; offset < parallelism; offset++)); do
        attempt_index=$((batch_start + offset))
        if (( attempt_index >= ${#SEEDS[@]} )); then
            break
        fi
        run_attempt "${attempt_index}" "${SEEDS[${attempt_index}]}" "${GPU_LIST[${offset}]}" &
        pids+=("$!")
    done

    batch_failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            batch_failed=1
        fi
    done
    if (( batch_failed )); then
        echo "At least one parallel attempt failed; pass@k was not computed." >&2
        exit 1
    fi
done

python3 "${SCRIPT_DIR}/compute_pass_at_k.py" \
    --input-dir "${OUTPUT_DIR}" \
    --max-pass-k "${MAX_PASS_K}" \
    --output-dir "${OUTPUT_DIR}"

echo "Pass@k evaluation complete: ${OUTPUT_DIR}"

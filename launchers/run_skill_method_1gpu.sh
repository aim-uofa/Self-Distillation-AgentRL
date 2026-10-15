#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

METHOD="${1:-}"
TASK="${2:-}"
if [[ -z "${METHOD}" || -z "${TASK}" ]]; then
    echo "Usage: $0 <grpo|agentopsd|adrs|sdar|icsd|pcsd|ocsd|ovcsd|grpo_opsd|skillsd|rlsd|skill_prompt> <alfworld|webshop|search|scienceworld> [Hydra overrides ...]" >&2
    exit 2
fi
shift 2

METHOD_OVERRIDES=()
case "${METHOD}" in
    grpo)
        TRAINER_MODULE=verl.trainer.main_ppo
        METHOD_DISPLAY=GRPO
        ;;
    agentopsd)
        TRAINER_MODULE=verl.trainer.main_agentopsd
        METHOD_DISPLAY=AgentOPSD
        METHOD_OVERRIDES+=(
            "+algorithm.agentopsd.v0_prior=${AGENTOPSD_V0_PRIOR:-0.5}"
            "+algorithm.agentopsd.granularity=${AGENTOPSD_GRANULARITY:-turn}"
            "+algorithm.agentopsd.belief_mult=true"
            "+algorithm.agentopsd.mult_lambda=${AGENTOPSD_MULT_LAMBDA:-0.5}"
            "+algorithm.agentopsd.mult_band=${AGENTOPSD_MULT_BAND:-0.2}"
            "+algorithm.agentopsd.belief_decay_gamma=${AGENTOPSD_GAMMA:-0.95}"
            "+algorithm.agentopsd.signed=true"
            "+algorithm.agentopsd.warmdown_steps=${AGENTOPSD_WARMDOWN_STEPS:--1}"
            "actor_rollout_ref.actor.clip_ratio_high=${AGENTOPSD_CLIP_HIGH:-0.24}"
        )
        ;;
    adrs)
        TRAINER_MODULE=verl.trainer.main_adrs
        METHOD_DISPLAY=ADRS
        case "${TASK}" in
            alfworld)
                DEFAULT_ADRS_ETA=0.1
                DEFAULT_ADRS_NORMALIZE=global
                DEFAULT_ADRS_GATE_NORM=true
                DEFAULT_ADRS_ENTROPY=0.01
                DEFAULT_ADRS_CLIP_HIGH=0.28
                ;;
            webshop)
                DEFAULT_ADRS_ETA=0.02
                DEFAULT_ADRS_NORMALIZE=per_seq
                DEFAULT_ADRS_GATE_NORM=false
                DEFAULT_ADRS_ENTROPY=0.001
                DEFAULT_ADRS_CLIP_HIGH=0.2
                ;;
            search|scienceworld)
                DEFAULT_ADRS_ETA=0.05
                DEFAULT_ADRS_NORMALIZE=global
                DEFAULT_ADRS_GATE_NORM=true
                DEFAULT_ADRS_ENTROPY=0.001
                DEFAULT_ADRS_CLIP_HIGH=0.2
                ;;
            *) echo "Unsupported task: ${TASK}" >&2; exit 2 ;;
        esac
        METHOD_OVERRIDES+=(
            "+algorithm.adrs.eta=${ADRS_ETA:-${DEFAULT_ADRS_ETA}}"
            "+algorithm.adrs.tva_level=${ADRS_TVA_LEVEL:-L2}"
            "+algorithm.adrs.tva_temperature=${ADRS_TVA_TEMPERATURE:-1.0}"
            "+algorithm.adrs.tva_tau=${ADRS_TVA_TAU:-2.0}"
            "+algorithm.adrs.baseline_mode=${ADRS_BASELINE_MODE:-step}"
            "+algorithm.adrs.normalize_mode=${ADRS_NORMALIZE_MODE:-${DEFAULT_ADRS_NORMALIZE}}"
            "+algorithm.adrs.tva_gate_norm=${ADRS_TVA_GATE_NORM:-${DEFAULT_ADRS_GATE_NORM}}"
            "+algorithm.adrs.wn_warmup_steps=${ADRS_WARMUP_STEPS:-0}"
            "+algorithm.adrs.wn_active_steps=${ADRS_ACTIVE_STEPS:--1}"
            "+algorithm.adrs.wn_clip=${ADRS_WN_CLIP:-0.2}"
            "+algorithm.adrs.wn_sign=${ADRS_WN_SIGN:-1.0}"
            "+algorithm.adrs.wn_adaptive_clip=${ADRS_WN_ADAPTIVE_CLIP:-false}"
            "+algorithm.adrs.wn_mode=${ADRS_WN_MODE:-additive}"
            "+algorithm.adrs.wn_neg_reset=${ADRS_WN_NEG_RESET:-false}"
            "+algorithm.adrs.kl_boost_mult=${ADRS_KL_BOOST_MULT:-1.0}"
            "+algorithm.adrs.ent_coeff_active=${ADRS_ENT_COEFF_ACTIVE:-0.0}"
            "actor_rollout_ref.actor.entropy_coeff=${ADRS_ENTROPY_COEF:-${DEFAULT_ADRS_ENTROPY}}"
            "actor_rollout_ref.actor.clip_ratio_high=${ADRS_CLIP_HIGH:-${DEFAULT_ADRS_CLIP_HIGH}}"
        )
        ;;
    ocsd)
        TRAINER_MODULE=verl.trainer.main_ocsd
        METHOD_DISPLAY=OCSD
        METHOD_OVERRIDES+=(
            "+algorithm.oc_opsd.beta=${OCSD_BETA:-0.5}"
            "+algorithm.oc_opsd.eps_A=${OCSD_ADVANTAGE_EPSILON:-1e-6}"
            "+algorithm.oc_opsd.K_future=${OCSD_FUTURE_HORIZON:-2}"
            "+algorithm.oc_opsd.top_step_ratio=${OCSD_TOP_STEP_RATIO:-0.2}"
            "+algorithm.oc_opsd.env_name=${TASK}"
        )
        ;;
    ovcsd)
        TRAINER_MODULE=verl.trainer.main_sdar
        METHOD_DISPLAY=OVCSD
        METHOD_OVERRIDES+=(
            "+algorithm.ovcsd.enable=true"
            "+algorithm.ovcsd.teacher_attempt_limit=${OVCSD_TEACHER_ATTEMPT_LIMIT:-2}"
            "+algorithm.ovcsd.suffix_top_k=${OVCSD_SUFFIX_TOP_K:-32}"
            "+algorithm.ovcsd.suffix_kl_coef=${OVCSD_SUFFIX_KL_COEF:-0.1}"
        )
        ;;
    sdar)
        TRAINER_MODULE=verl.trainer.main_sdar
        METHOD_DISPLAY=SDAR
        METHOD_OVERRIDES+=(
            "+algorithm.sdar.sdar_coef=${SDAR_COEF:-0.01}"
            "+algorithm.sdar.gate_beta=${GATE_BETA:-5.0}"
        )
        ;;
    icsd)
        TRAINER_MODULE=verl.trainer.main_icsd
        METHOD_DISPLAY=ICSD
        METHOD_OVERRIDES+=(
            "+algorithm.icsd.coef=${ICSD_COEF:-0.01}"
            "+algorithm.icsd.trust_split=${ICSD_TRUST_SPLIT:-0.4}"
            "+algorithm.icsd.influence_split=${ICSD_INFLUENCE_SPLIT:-0.5}"
            "+algorithm.icsd.scale_floor=${ICSD_SCALE_FLOOR:-1e-4}"
            "+algorithm.icsd.min_group_tokens=${ICSD_MIN_GROUP_TOKENS:-8}"
        )
        ;;
    pcsd)
        TRAINER_MODULE=verl.trainer.main_pcsd
        METHOD_DISPLAY=PCSD
        METHOD_OVERRIDES+=(
            "+algorithm.pcsd.coef=${PCSD_COEF:-0.01}"
            "+algorithm.pcsd.n_min=${PCSD_N_MIN:-1}"
            "+algorithm.pcsd.n_max=${PCSD_N_MAX:-8}"
            "+algorithm.pcsd.decay=${PCSD_DECAY:-0.8}"
            "+algorithm.pcsd.variance_low=${PCSD_VAR_LOW:-0.05}"
            "+algorithm.pcsd.variance_high=${PCSD_VAR_HIGH:-0.5}"
            "+algorithm.pcsd.trend_gamma=${PCSD_TREND_GAMMA:-0.3}"
            "+algorithm.pcsd.gate_beta=${PCSD_GATE_BETA:-5.0}"
        )
        ;;
    grpo_opsd)
        TRAINER_MODULE=verl.trainer.main_sdar
        METHOD_DISPLAY=GRPO+OPSD
        METHOD_OVERRIDES+=(
            "+algorithm.sdar.sdar_coef=${SDAR_COEF:-0.01}"
            "+algorithm.sdar.gate_beta=${GATE_BETA:-0.0}"
        )
        ;;
    skillsd)
        TRAINER_MODULE=verl.trainer.main_skillsd
        METHOD_DISPLAY=Skill-SD
        METHOD_OVERRIDES+=(
            "+algorithm.skillsd.sdl_lambda=${SDL_LAMBDA:-0.001}"
            "+algorithm.skillsd.warmdown_steps=${SDL_WARMDOWN_STEPS:--1}"
        )
        ;;
    rlsd)
        TRAINER_MODULE=verl.trainer.main_rlsd
        METHOD_DISPLAY=RLSD
        METHOD_OVERRIDES+=(
            "+algorithm.rlsd.rlsd_lambda=${RLSD_LAMBDA:-0.5}"
            "+algorithm.rlsd.warmdown_steps=${RLSD_WARMDOWN_STEPS:-100}"
            "+algorithm.rlsd.clip_eps=${RLSD_CLIP_EPS:-0.2}"
        )
        ;;
    skill_prompt)
        # Evaluation-only compatibility path; no skill-conditioned training.
        TRAINER_MODULE=verl.trainer.main_skill_prompt
        METHOD_DISPLAY=Skill-Prompt
        ;;
    *)
        echo "Unsupported method: ${METHOD}" >&2
        echo "Expected one of: grpo, agentopsd, adrs, sdar, icsd, pcsd, ocsd, ovcsd, grpo_opsd, skillsd, rlsd, skill_prompt (evaluation only)" >&2
        exit 2
        ;;
esac

if [[ "${TRAINER_MODULE}" == "verl.trainer.main_sdar" ]]; then
    # Display-only metadata for the shared entry point; never selects an algorithm.
    METHOD_OVERRIDES+=("+trainer.method_display_name=${METHOD_DISPLAY}")
fi

PAPER_CONFIG="${PAPER_CONFIG:-0}"
if [[ "${PAPER_CONFIG}" != "0" && "${PAPER_CONFIG}" != "1" ]]; then
    echo "PAPER_CONFIG must be 0 or 1, got: ${PAPER_CONFIG}" >&2
    exit 2
fi

DEFAULT_GPU_MEMORY_UTILIZATION=0.30
if [[ "${PAPER_CONFIG}" == "1" ]]; then
    SELECTED_MODEL_SIZE="${MODEL_SIZE:-$(basename -- "${MODEL_PATH%/}")}"
    case "${SELECTED_MODEL_SIZE,,}" in
        9b|qwen3.5-9b)
            DEFAULT_GPU_MEMORY_UTILIZATION=0.65
            ;;
    esac
fi
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${DEFAULT_GPU_MEMORY_UTILIZATION}}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-1}"
MAX_ACTOR_CKPTS="${MAX_ACTOR_CKPTS:-2}"
RESUME_MODE="${RESUME_MODE:-disable}"
SKILL_ALL="${SKILL_ALL:-false}"
if [[ "${METHOD}" == "ovcsd" ]]; then
    DEFAULT_RAY_NUM_CPUS=64
else
    DEFAULT_RAY_NUM_CPUS=32
fi
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${DEFAULT_RAY_NUM_CPUS}}"

TASK_OVERRIDES=()
case "${TASK}" in
    alfworld)
        # This study evaluates every valid_seen game exactly once in paper mode.
        if [[ "${PAPER_CONFIG}" == "1" ]]; then
            export ALFWORLD_STRICT_EVAL="${ALFWORLD_STRICT_EVAL:-1}"
        fi
        TRAIN_DATA="${DATA_ROOT}/verl-agent/text/train.parquet"
        VAL_DATA="${DATA_ROOT}/verl-agent/text/test.parquet"
        DEFAULT_SKILLS_DIR=skills/alfworld
        DEFAULT_MAX_PROMPT_LENGTH=2048
        DEFAULT_MAX_RESPONSE_LENGTH=512
        DEFAULT_MAX_ENV_STEPS=50
        DEFAULT_KL_COEF=0.01
        DEFAULT_INVALID_ACTION_COEF=0.1
        DEFAULT_VAL_TEMPERATURE=0.4
        DEFAULT_VAL_DO_SAMPLE=true
        DEFAULT_PAPER_TRAIN_SIZE=16
        DEFAULT_PAPER_VAL_SIZE=140
        DEFAULT_PAPER_GROUP_SIZE=8
        DEFAULT_PAPER_PPO_MINI_BATCH_SIZE=256
        DEFAULT_PAPER_TEST_FREQ=5
        TASK_OVERRIDES+=(
            "data.truncation=error"
            "env.env_name=alfworld/AlfredTWEnv"
            "env.resources_per_worker.num_cpus=0.1"
        )
        ;;
    webshop)
        TRAIN_DATA="${DATA_ROOT}/verl-agent/text/train.parquet"
        VAL_DATA="${DATA_ROOT}/verl-agent/text/test.parquet"
        DEFAULT_SKILLS_DIR=skills/webshop
        DEFAULT_MAX_PROMPT_LENGTH=4096
        DEFAULT_MAX_RESPONSE_LENGTH=512
        DEFAULT_MAX_ENV_STEPS=15
        DEFAULT_KL_COEF=0.01
        DEFAULT_INVALID_ACTION_COEF=0.1
        DEFAULT_VAL_TEMPERATURE=0.4
        DEFAULT_VAL_DO_SAMPLE=true
        DEFAULT_PAPER_TRAIN_SIZE=16
        DEFAULT_PAPER_VAL_SIZE=128
        DEFAULT_PAPER_GROUP_SIZE=8
        DEFAULT_PAPER_PPO_MINI_BATCH_SIZE=64
        DEFAULT_PAPER_TEST_FREQ=5
        TASK_OVERRIDES+=(
            "data.truncation=error"
            "env.env_name=Webshop"
            "env.webshop.use_small=${WEBSHOP_USE_SMALL:-true}"
            "env.resources_per_worker.num_cpus=0.1"
        )
        ;;
    scienceworld)
        TRAIN_DATA="${DATA_ROOT}/scienceworld_agentgym/train_disjoint.parquet"
        VAL_DATA="$(python "${SCRIPT_DIR}/prepare_scienceworld.py" --data-root "${DATA_ROOT}")"
        DEFAULT_SKILLS_DIR=skills/scienceworld
        DEFAULT_MAX_PROMPT_LENGTH=4096
        DEFAULT_MAX_RESPONSE_LENGTH=512
        DEFAULT_MAX_ENV_STEPS=20
        DEFAULT_KL_COEF=0.01
        DEFAULT_INVALID_ACTION_COEF=0.1
        DEFAULT_VAL_TEMPERATURE=0.4
        DEFAULT_VAL_DO_SAMPLE=true
        DEFAULT_PAPER_TRAIN_SIZE=16
        DEFAULT_PAPER_VAL_SIZE=32
        DEFAULT_PAPER_GROUP_SIZE=8
        DEFAULT_PAPER_PPO_MINI_BATCH_SIZE=64
        DEFAULT_PAPER_TEST_FREQ=5
        TASK_OVERRIDES+=(
            "data.truncation=left"
            "env.env_name=scienceworld"
            "env.resources_per_worker.num_cpus=0.1"
            "+env.scienceworld.reward_mode=${SCIENCEWORLD_REWARD_MODE:-score}"
            "+env.scienceworld.java_options=${SCIENCEWORLD_JAVA_OPTIONS:--Xmx512m}"
        )
        ;;
    search)
        TRAIN_DATA="${DATA_ROOT}/searchR1_processed_direct/train.parquet"
        VAL_DATA="$(python "${SCRIPT_DIR}/prepare_search_eval_subset.py" --data-root "${DATA_ROOT}")"
        DEFAULT_SKILLS_DIR=skills/search
        DEFAULT_MAX_PROMPT_LENGTH=4096
        DEFAULT_MAX_RESPONSE_LENGTH=512
        DEFAULT_MAX_ENV_STEPS=4
        DEFAULT_KL_COEF=0.001
        DEFAULT_INVALID_ACTION_COEF=0.01
        DEFAULT_VAL_TEMPERATURE=0.0
        DEFAULT_VAL_DO_SAMPLE=false
        DEFAULT_PAPER_TRAIN_SIZE=64
        DEFAULT_PAPER_VAL_SIZE=512
        DEFAULT_PAPER_GROUP_SIZE=8
        DEFAULT_PAPER_PPO_MINI_BATCH_SIZE=256
        DEFAULT_PAPER_TEST_FREQ=150
        TASK_OVERRIDES+=(
            "data.truncation=left"
            "env.env_name=search"
            "env.history_length=4"
            "env.search.search_url=${SEARCH_URL:-http://127.0.0.1:${PORT:-19080}/retrieve}"
        )
        ;;
    *)
        echo "Unsupported task: ${TASK}" >&2
        exit 2
        ;;
esac

if [[ "${METHOD}" == "ocsd" && "${TASK}" == "search" ]]; then
    # Match the released OCSD Search-QA recipe.
    METHOD_OVERRIDES+=("actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1")
fi

SKILLS_DIR="${SKILLS_DIR:-${DEFAULT_SKILLS_DIR}}"
case "${METHOD}" in
    agentopsd)
        METHOD_OVERRIDES+=(
            "+algorithm.agentopsd.skills_dir=${SKILLS_DIR}"
            "+algorithm.agentopsd.skill_all=${SKILL_ALL}"
        )
        ;;
    adrs)
        METHOD_OVERRIDES+=(
            "+algorithm.adrs.skills_dir=${SKILLS_DIR}"
            "+algorithm.adrs.skill_all=${SKILL_ALL}"
        )
        ;;
    ovcsd)
        METHOD_OVERRIDES+=(
            "+algorithm.ovcsd.skills_dir=${SKILLS_DIR}"
            "+algorithm.ovcsd.skill_all=${SKILL_ALL}"
        )
        ;;
    sdar|grpo_opsd)
        METHOD_OVERRIDES+=(
            "+algorithm.sdar.skills_dir=${SKILLS_DIR}"
            "+algorithm.sdar.skill_all=${SKILL_ALL}"
        )
        ;;
    icsd)
        METHOD_OVERRIDES+=(
            "+algorithm.icsd.skills_dir=${SKILLS_DIR}"
            "+algorithm.icsd.skill_all=${SKILL_ALL}"
        )
        ;;
    pcsd)
        METHOD_OVERRIDES+=(
            "+algorithm.pcsd.skills_dir=${SKILLS_DIR}"
            "+algorithm.pcsd.skill_all=${SKILL_ALL}"
        )
        ;;
    skill_prompt)
        # Keep the existing Skill-Prompt Hydra keys for command compatibility.
        # This branch runs an evaluation-only module, never Skill-GRPO training.
        METHOD_OVERRIDES+=(
            "+algorithm.skill_grpo.skills_dir=${SKILLS_DIR}"
            "+algorithm.skill_grpo.skill_all=${SKILL_ALL}"
        )
        ;;
    rlsd)
        METHOD_OVERRIDES+=(
            "+algorithm.rlsd.skills_dir=${SKILLS_DIR}"
            "+algorithm.rlsd.skill_all=${SKILL_ALL}"
        )
        ;;
    skillsd)
        METHOD_OVERRIDES+=(
            "+algorithm.skillsd.skills_dir=${SKILLS_DIR}"
            "+algorithm.skillsd.skill_all=${SKILL_ALL}"
        )
        ;;
esac

MODEL_NAME="$(basename -- "${MODEL_PATH%/}")"
MODEL_NAME="${MODEL_NAME//[^a-zA-Z0-9_.-]/_}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${METHOD}_${MODEL_NAME}_${TASK}_1gpu}"

if [[ "${PAPER_CONFIG}" == "1" ]]; then
    if [[ ( "${METHOD}" == "ocsd" || "${METHOD}" == "adrs" ) && "${TASK}" == "search" ]]; then
        # OCSD reports 0.001 for Search-QA; keep this override isolated from
        # the existing methods' PAPER_CONFIG behavior.
        SELECTED_DEFAULT_KL_COEF=0.001
    else
        SELECTED_DEFAULT_KL_COEF=0.01
    fi
    TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-${DEFAULT_PAPER_TRAIN_SIZE}}"
    VAL_DATA_SIZE="${VAL_DATA_SIZE:-${DEFAULT_PAPER_VAL_SIZE}}"
    GROUP_SIZE="${GROUP_SIZE:-${DEFAULT_PAPER_GROUP_SIZE}}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${DEFAULT_PAPER_PPO_MINI_BATCH_SIZE}}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
    TEST_FREQ="${TEST_FREQ:-${DEFAULT_PAPER_TEST_FREQ}}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
else
    SELECTED_DEFAULT_KL_COEF="${DEFAULT_KL_COEF}"
    TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-4}"
    VAL_DATA_SIZE="${VAL_DATA_SIZE:-4}"
    GROUP_SIZE="${GROUP_SIZE:-2}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$((TRAIN_DATA_SIZE * GROUP_SIZE))}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
    TEST_FREQ="${TEST_FREQ:--1}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
fi

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-${DEFAULT_MAX_PROMPT_LENGTH}}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${DEFAULT_MAX_RESPONSE_LENGTH}}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-${DEFAULT_MAX_ENV_STEPS}}"
KL_COEF="${KL_COEF:-${SELECTED_DEFAULT_KL_COEF}}"
INVALID_ACTION_COEF="${INVALID_ACTION_COEF:-${DEFAULT_INVALID_ACTION_COEF}}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-${DEFAULT_VAL_TEMPERATURE}}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-${DEFAULT_VAL_DO_SAMPLE}}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${TOTAL_TRAINING_STEPS}}"
SAVE_FREQ="${SAVE_FREQ:-${TOTAL_TRAINING_STEPS}}"

if [[ ( "${TASK}" == "alfworld" || "${TASK}" == "webshop" ) && "${SKIP_AGENT_DATA_PREP:-0}" != "1" ]]; then
    RUN_DATA_ROOT="$(mktemp -d "${RAY_TMPDIR}/agent_data_${TASK}_XXXXXX")"
    mkdir -p "${RUN_DATA_ROOT}/text"
    python -m examples.data_preprocess.prepare \
        --mode text \
        --local_dir "${RUN_DATA_ROOT}" \
        --train_data_size "${TRAIN_DATA_SIZE}" \
        --val_data_size "${VAL_DATA_SIZE}"
    TRAIN_DATA="${RUN_DATA_ROOT}/text/train.parquet"
    VAL_DATA="${RUN_DATA_ROOT}/text/test.parquet"
fi

for required_path in "${TRAIN_DATA}" "${VAL_DATA}"; do
    if [[ ! -s "${required_path}" ]]; then
        echo "Missing data file: ${required_path}" >&2
        if [[ "${TASK}" == "search" ]]; then
            echo "Run: bash launchers/prepare_search.sh qa" >&2
        fi
        exit 1
    fi
done

if [[ "${METHOD}" != "grpo" && "${METHOD}" != "ocsd" && ! -d "${SKILLS_DIR}" ]]; then
    echo "Skills directory not found: ${SKILLS_DIR}" >&2
    exit 1
fi

if [[ "${TASK}" == "search" && "${SKIP_RETRIEVER_CHECK:-0}" != "1" ]]; then
    SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:${PORT:-19080}/retrieve}"
    if ! curl -fsS --max-time 30 \
        -H "Content-Type: application/json" \
        -d '{"query":"What is retrieval?","topk":1}' \
        "${SEARCH_URL}" >/dev/null; then
        echo "Search retriever is not ready at ${SEARCH_URL}" >&2
        echo "Run in another terminal: bash launchers/start_search_retriever.sh" >&2
        exit 1
    fi
fi

CHECKPOINT_TASK_NAME="${TASK}"
if [[ "${TASK}" == "search" ]]; then
    CHECKPOINT_TASK_NAME=searchQA
fi
EXPERIMENT_DIR="${PROJECT_ROOT}/checkpoints/${METHOD}/${CHECKPOINT_TASK_NAME}/${EXPERIMENT_NAME}"
LOG_DIR="${PROJECT_ROOT}/logs/${METHOD}_train"
mkdir -p "${EXPERIMENT_DIR}" "${LOG_DIR}"
LOG_TASK_NAME="${TASK}"
if [[ "${TASK}" == "search" ]]; then
    LOG_TASK_NAME=searchQA
fi
LOG_FILE="${TRAIN_LOG_FILE:-${LOG_DIR}/${LOG_TASK_NAME}_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log}"

echo "${METHOD_DISPLAY} training: task=${TASK}, model=${MODEL_PATH}, GPU=${GPU_ID:-0}"
echo "paper_config=${PAPER_CONFIG}, train=${TRAIN_DATA_SIZE}, val_batch=${VAL_DATA_SIZE}, group=${GROUP_SIZE}, steps=${TOTAL_TRAINING_STEPS}"
if [[ "${METHOD}" == "ocsd" ]]; then
    echo "ocsd uses rollout future evidence and does not load a skill bank"
elif [[ "${METHOD}" != "grpo" ]]; then
    echo "skills_dir=${SKILLS_DIR}, skill_all=${SKILL_ALL}"
fi
echo "checkpoint_dir=${EXPERIMENT_DIR}"
echo "log=${LOG_FILE}"

python -m "${TRAINER_MODULE}" \
    algorithm.adv_estimator=grpo \
    "data.train_files=${TRAIN_DATA}" \
    "data.val_files=${VAL_DATA}" \
    "data.train_batch_size=${TRAIN_DATA_SIZE}" \
    "data.val_batch_size=${VAL_DATA_SIZE}" \
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}" \
    "data.max_response_length=${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    "actor_rollout_ref.actor.kl_loss_coef=${KL_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}" \
    "actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    "actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_ACTION_COEF}" \
    algorithm.use_kl_in_reward=False \
    env.seed=0 \
    "env.max_steps=${MAX_ENV_STEPS}" \
    "env.rollout.n=${GROUP_SIZE}" \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    "trainer.project_name=${METHOD}_local_1gpu" \
    "trainer.experiment_name=${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPTS}" \
    "trainer.test_freq=${TEST_FREQ}" \
    "trainer.total_epochs=${TOTAL_EPOCHS}" \
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}" \
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}" \
    "trainer.resume_mode=${RESUME_MODE}" \
    "trainer.default_local_dir=${EXPERIMENT_DIR}" \
    "ray_init.num_cpus=${RAY_NUM_CPUS}" \
    "${METHOD_OVERRIDES[@]}" \
    "${TASK_OVERRIDES[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"

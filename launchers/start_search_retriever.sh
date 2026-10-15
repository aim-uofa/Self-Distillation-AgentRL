#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
REQUIRE_MODEL=0 source "${SCRIPT_DIR}/common.sh"

RETRIEVAL_DIR="${DATA_ROOT}/searchR1"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-${PROJECT_ROOT}/models/e5-base-v2}"
PORT="${PORT:-19080}"

for required_path in \
    "${RETRIEVAL_DIR}/e5_Flat.index" \
    "${RETRIEVAL_DIR}/wiki-18.jsonl" \
    "${RETRIEVER_MODEL}/config.json"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing ${required_path}" >&2
        echo "Run: bash launchers/prepare_search.sh retrieval" >&2
        exit 1
    fi
done

EXTRA_ARGS=()
if [[ "${FAISS_GPU:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--faiss_gpu)
fi

exec python examples/search/retriever/retrieval_server.py \
    --index_path "${RETRIEVAL_DIR}/e5_Flat.index" \
    --corpus_path "${RETRIEVAL_DIR}/wiki-18.jsonl" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "${RETRIEVER_MODEL}" \
    --port "${PORT}" \
    "${EXTRA_ARGS[@]}"

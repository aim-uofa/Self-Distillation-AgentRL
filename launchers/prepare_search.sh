#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
REQUIRE_MODEL=0 source "${SCRIPT_DIR}/common.sh"

MODE="${1:-qa}"
SEARCH_QA_DIR="${DATA_ROOT}/searchR1_processed_direct"
RETRIEVAL_DIR="${DATA_ROOT}/searchR1"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-${PROJECT_ROOT}/models/e5-base-v2}"

if [[ "${MODE}" != "qa" && "${MODE}" != "retrieval" && "${MODE}" != "all" ]]; then
    echo "Usage: $0 [qa|retrieval|all]" >&2
    exit 2
fi

if [[ "${MODE}" == "qa" || "${MODE}" == "all" ]]; then
    if [[ -s "${SEARCH_QA_DIR}/train.parquet" && -s "${SEARCH_QA_DIR}/test.parquet" ]]; then
        echo "Search-QA train/test Parquet files already exist; skipping download."
    else
        python examples/data_preprocess/preprocess_search_r1_dataset.py \
            --local_dir "${SEARCH_QA_DIR}"
    fi
fi

if [[ "${MODE}" == "retrieval" || "${MODE}" == "all" ]]; then
    mkdir -p "${RETRIEVAL_DIR}" "${RETRIEVER_MODEL}"
    echo "Downloading the Search-R1 retrieval assets (~65 GB compressed)..."
    python examples/search/searchr1_download.py --local_dir "${RETRIEVAL_DIR}"

    if [[ ! -s "${RETRIEVAL_DIR}/e5_Flat.index" ]]; then
        cat \
            "${RETRIEVAL_DIR}/part_aa" \
            "${RETRIEVAL_DIR}/part_ab" \
            > "${RETRIEVAL_DIR}/e5_Flat.index.part"
        mv \
            "${RETRIEVAL_DIR}/e5_Flat.index.part" \
            "${RETRIEVAL_DIR}/e5_Flat.index"
    fi
    if [[ ! -s "${RETRIEVAL_DIR}/wiki-18.jsonl" ]]; then
        gzip -dc \
            "${RETRIEVAL_DIR}/wiki-18.jsonl.gz" \
            > "${RETRIEVAL_DIR}/wiki-18.jsonl.part"
        mv \
            "${RETRIEVAL_DIR}/wiki-18.jsonl.part" \
            "${RETRIEVAL_DIR}/wiki-18.jsonl"
    fi

    python - "${RETRIEVER_MODEL}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download('intfloat/e5-base-v2', local_dir=sys.argv[1])
PY
fi

echo "Search-QA preparation completed for mode=${MODE}."

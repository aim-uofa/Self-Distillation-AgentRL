#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
REQUIRE_MODEL=0 source "${SCRIPT_DIR}/common.sh"

DATASET_SIZE="${1:-small}"
WEBSHOP_ROOT="${REPO_ROOT}/agent_system/environments/env_package/webshop/webshop"
WEBSHOP_DATA="${WEBSHOP_ROOT}/data"

if [[ "${DATASET_SIZE}" != "small" && "${DATASET_SIZE}" != "all" ]]; then
    echo "Usage: $0 [small|all]" >&2
    exit 2
fi

mkdir -p "${WEBSHOP_DATA}"

HF_DATASET_REPO="${HF_DATASET_REPO:-HongbangYuan/webshop}"

download_webshop_file() {
    local filename="$1"
    local file_id="$2"
    local output_path="${WEBSHOP_DATA}/${filename}"
    local temporary_path="${output_path}.part"

    if [[ ! -s "${output_path}" ]]; then
        echo "Downloading ${filename} from Hugging Face mirror ${HF_DATASET_REPO}..."
        if curl -fL --retry 3 --retry-delay 2 \
            "https://huggingface.co/datasets/${HF_DATASET_REPO}/resolve/main/${filename}?download=true" \
            -o "${temporary_path}"; then
            mv "${temporary_path}" "${output_path}"
        else
            rm -f "${temporary_path}"
            echo "Hugging Face mirror failed; trying the original Google Drive file..."
            gdown "https://drive.google.com/uc?id=${file_id}" -O "${output_path}"
        fi
    fi
}

download_webshop_file \
    "items_shuffle_1000.json" \
    "1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib"
download_webshop_file \
    "items_ins_v2_1000.json" \
    "1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu"
download_webshop_file \
    "items_human_ins.json" \
    "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O"

if [[ "${DATASET_SIZE}" == "all" ]]; then
    download_webshop_file \
        "items_shuffle.json" \
        "1A2whVgOO0euk5O13n2iYDM0bQRkkRduB"
    download_webshop_file \
        "items_ins_v2.json" \
        "1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi"
fi

if ! python -c "import spacy; spacy.load('en_core_web_sm')" >/dev/null 2>&1; then
    python -m spacy download en_core_web_sm
fi

cd "${WEBSHOP_ROOT}/search_engine"
mkdir -p resources resources_100 resources_1k resources_100k indexes
python convert_product_file_format.py
source ./run_indexing.sh

echo "WebShop ${DATASET_SIZE} data and Lucene indexes are ready."

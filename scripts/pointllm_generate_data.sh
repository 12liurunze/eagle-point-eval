#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EAGLE_EYE_ROOT="${REPO_ROOT}/EAGLE_EYE"

if [[ -d /root/autodl-tmp ]]; then
  DEFAULT_POINTLLM_REPO="/root/autodl-tmp/pointLLM"
  DEFAULT_BASE_MODEL="/root/autodl-tmp/point7B_v1.1"
  DEFAULT_POINT_CLOUD_DATA="/root/autodl-tmp/pointLLM/data/objaverse_data"
  DEFAULT_ANNOTATION="/root/autodl-tmp/pointLLM/data/anno_data/PointLLM_complex_instruction_70K.json"
  DEFAULT_OUTPUT_DIR="/root/autodl-tmp/pointllm_eagle_data"
else
  DEFAULT_POINTLLM_REPO="/c/Users/lrz/PointLLM"
  DEFAULT_BASE_MODEL="/f/download/point7B"
  DEFAULT_POINT_CLOUD_DATA="/f/download/8192_npy"
  DEFAULT_ANNOTATION="/f/download/PointLLM_complex_instruction_70K.json"
  DEFAULT_OUTPUT_DIR="/f/download/pointllm_eagle_data"
fi

POINTLLM_REPO="${POINTLLM_REPO:-${DEFAULT_POINTLLM_REPO}}"
BASE_MODEL="${BASE_MODEL:-${DEFAULT_BASE_MODEL}}"
POINT_CLOUD_DATA="${POINT_CLOUD_DATA:-${DEFAULT_POINT_CLOUD_DATA}}"
ANNOTATION="${ANNOTATION:-${DEFAULT_ANNOTATION}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

cd "${EAGLE_EYE_ROOT}"
export PYTHONPATH="${POINTLLM_REPO}:${EAGLE_EYE_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA_ARGS=()
if [[ "${FORCE_SINGLE_POINT_PROJ:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force-single-point-proj)
fi

"${PYTHON_BIN}" -m eagle_eye.ge_data.get_data_all_pointllm \
  --base-model-path "${BASE_MODEL}" \
  --pointllm-repo-path "${POINTLLM_REPO}" \
  --data-path "${POINT_CLOUD_DATA}" \
  --anno-path "${ANNOTATION}" \
  --outdir "${OUTPUT_DIR}" \
  --index "${INDEX:-0}" \
  --start "${START:-0}" \
  --end "${END:-0}" \
  --conversation-types "${CONVERSATION_TYPES:-single_round,multi_round,detailed_description}" \
  --point-backbone-config-name "${POINT_BACKBONE_CONFIG_NAME:-PointTransformer_8192point_2layer}" \
  "${EXTRA_ARGS[@]}" \
  --torch-dtype "${TORCH_DTYPE:-float32}"

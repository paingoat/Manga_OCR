#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_DIR="${INPUT_DIR:-input/bubble}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
PADDLEOCR_DIR="${PADDLEOCR_DIR:-PaddleOCR}"
CONDA_ENV="${CONDA_ENV:-pp_ocr_jap_infer}"
MODEL="${MODEL:-crnn}"
CONFIG="${CONFIG:-}"

if [[ -z "${CONFIG}" ]]; then
  case "${MODEL}" in
    crnn)
      CONFIG="configs/infer.crnn.yaml"
      ;;
    svtr)
      CONFIG="configs/infer.svtr.yaml"
      ;;
    *)
      echo "Unsupported MODEL='${MODEL}'. Expected 'crnn' or 'svtr'." >&2
      exit 2
      ;;
  esac
fi

cd "${REPO_ROOT}"

conda run -n "${CONDA_ENV}" python -m app.infer \
  --config "${CONFIG}" \
  --input "${INPUT_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --paddleocr-dir "${PADDLEOCR_DIR}" \
  "$@"


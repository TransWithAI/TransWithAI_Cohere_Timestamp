#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_FLAVOR=cuda122 BUILD_DIR="${ROOT_DIR}/build/crispasr-cuda122" "${ROOT_DIR}/scripts/build-crispasr-cuda.sh" "$@"

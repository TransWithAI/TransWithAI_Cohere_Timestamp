#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_FLAVOR="${CUDA_FLAVOR:-cuda128}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/crispasr-${CUDA_FLAVOR}}"

default_cuda_architectures() {
  case "${CUDA_FLAVOR}" in
    cuda118)
      printf '75-real;80-real;86-real;89-real;89-virtual\n'
      ;;
    cuda122)
      printf '75-real;80-real;86-real;89-real;90-real;90-virtual\n'
      ;;
    cuda128)
      printf '75-real;80-real;86-real;89-real;90-real;120-real;120-virtual\n'
      ;;
    *)
      printf '75-real;80-real;86-real;89-real;89-virtual\n'
      ;;
  esac
}

CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-$(default_cuda_architectures)}"

if [ "${1:-}" = "--print-cuda-architectures" ]; then
  printf '%s\n' "${CUDA_ARCHITECTURES}"
  exit 0
fi

cap_jobs() {
  local requested="${JOBS:-}"
  if [ -z "${requested}" ]; then
    requested="$(nproc 2>/dev/null || echo 4)"
  fi
  case "${requested}" in
    ''|*[!0-9]*) requested=4 ;;
  esac
  if [ "${requested}" -gt 16 ]; then
    printf '16\n'
  elif [ "${requested}" -lt 1 ]; then
    printf '1\n'
  else
    printf '%s\n' "${requested}"
  fi
}

JOBS="$(cap_jobs)"

cmake -S "${ROOT_DIR}/third_party/CrispASR" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCRISPASR_BUILD_TESTS=OFF \
  -DCRISPASR_BUILD_EXAMPLES=ON \
  -DGGML_CUDA=ON \
  -DCRISPASR_WITH_ESPEAK_NG=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" >&2
cmake --build "${BUILD_DIR}" --config Release --target crispasr-cli --parallel "${JOBS}" >&2
test -x "${BUILD_DIR}/bin/crispasr"
printf '%s\n' "${BUILD_DIR}/bin/crispasr"

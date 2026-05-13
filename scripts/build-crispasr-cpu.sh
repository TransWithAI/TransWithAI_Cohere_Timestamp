#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/crispasr-cpu"

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
  -DCRISPASR_WITH_ESPEAK_NG=OFF >&2
cmake --build "${BUILD_DIR}" --config Release --target crispasr-cli --parallel "${JOBS}" >&2
test -x "${BUILD_DIR}/bin/crispasr"
printf '%s\n' "${BUILD_DIR}/bin/crispasr"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="cpu"
DOWNLOAD_MODELS=1
RUN_TESTS=1

usage() {
  cat <<'USAGE'
Usage: scripts/setup.sh [cpu|cuda|cuda118|cuda122|cuda128|vulkan] [--skip-models] [--skip-tests]

Builds the local CrispASR binary, downloads pinned Cohere/Qwen3/Whisper VAD models,
and runs a wrapper smoke check. Default backend is cpu.

Examples:
  scripts/setup.sh cpu
  scripts/setup.sh cuda128
  scripts/setup.sh cuda122
  scripts/setup.sh cuda118
  scripts/setup.sh vulkan --skip-models
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    cpu|cuda|cuda118|cuda122|cuda128|vulkan)
      BACKEND="$1"
      ;;
    --skip-models)
      DOWNLOAD_MODELS=0
      ;;
    --skip-tests)
      RUN_TESTS=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown setup argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

case "${BACKEND}" in
  cpu)
    BUILD_SCRIPT="${ROOT_DIR}/scripts/build-crispasr-cpu.sh"
    DEVICE="cpu"
    ;;
  cuda|cuda128)
    BUILD_SCRIPT="${ROOT_DIR}/scripts/build-crispasr-cuda128.sh"
    DEVICE="cuda"
    ;;
  cuda122)
    BUILD_SCRIPT="${ROOT_DIR}/scripts/build-crispasr-cuda122.sh"
    DEVICE="cuda"
    ;;
  cuda118)
    BUILD_SCRIPT="${ROOT_DIR}/scripts/build-crispasr-cuda118.sh"
    DEVICE="cuda"
    ;;
  vulkan)
    BUILD_SCRIPT="${ROOT_DIR}/scripts/build-crispasr-vulkan.sh"
    DEVICE="amd"
    ;;
esac

cd "${ROOT_DIR}"

if command -v git >/dev/null 2>&1; then
  git submodule update --init --recursive third_party/CrispASR
fi

CRISPASR_BIN="$(${BUILD_SCRIPT})"

if [ "${DOWNLOAD_MODELS}" -eq 1 ]; then
  python download_models.py
fi

if [ "${RUN_TESTS}" -eq 1 ]; then
  PYTHONPATH="${ROOT_DIR}/src" python -m compileall -q infer.py download_models.py src tests
  PYTHONPATH="${ROOT_DIR}/src" python -m pytest
fi

SMOKE_DIR="${TMPDIR:-/tmp}/transwithai-cohere-timestamp-smoke"
mkdir -p "${SMOKE_DIR}"
: > "${SMOKE_DIR}/clip.mp3"
python infer.py --dry_run --device "${DEVICE}" --crispasr_path "${CRISPASR_BIN}" "${SMOKE_DIR}/clip.mp3"

printf '\nSetup complete. CrispASR binary: %s\n' "${CRISPASR_BIN}"

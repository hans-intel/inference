#!/bin/bash
# =============================================================================
# Build the vLLM CPU Docker image for x86_64 (linux/amd64)
#
# Source: https://github.com/tianmu-li/vllm/tree/tianmu/switch_omp
# Dockerfile: docker/Dockerfile.cpu
#
# Usage:
#   ./build_vllm_cpu_x86.sh              # build with defaults (AMX enabled)
#   DISABLE_AMX=1 ./build_vllm_cpu_x86.sh  # build without AMX (e.g. pre-SPR)
#   DISABLE_AVX512=1 ./build_vllm_cpu_x86.sh  # build AVX2-only
#
# The resulting image is tagged: vllm-cpu-x86:latest
#
# Build arguments forwarded to Dockerfile.cpu:
#   PYTHON_VERSION        Python version to use (default: 3.12)
#   VLLM_CPU_DISABLE_AVX512  Disable AVX-512 (default: 0)
#   VLLM_CPU_AMXBF16      Enable AMX-BF16 (default: 1 — GNR/SPR)
#   VLLM_CPU_AVX512BF16   Enable AVX512-BF16 (default: 0 — auto-detect at runtime)
#   VLLM_CPU_AVX512VNNI   Enable AVX512-VNNI (default: 0 — auto-detect at runtime)
#   MAX_JOBS              Parallel compile jobs (default: nproc)
#
# Requirements:
#   - docker buildx (Docker >= 19.03)
#   - git
#   - Internet access to github.com
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/tianmu-li/vllm.git"
BRANCH="tianmu/switch_omp"
IMAGE_TAG="${IMAGE_TAG:-vllm-cpu-x86:latest}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
DISABLE_AVX512="${DISABLE_AVX512:-0}"
# AMX-BF16: enabled by default — present on Granite Rapids / Sapphire Rapids.
# Set DISABLE_AMX=1 to build for older Xeon (Ice Lake, Cascade Lake, etc.)
DISABLE_AMX="${DISABLE_AMX:-0}"
if [[ "${DISABLE_AMX}" == "1" ]]; then
    VLLM_CPU_AMXBF16=0
else
    VLLM_CPU_AMXBF16=1
fi
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

BUILD_DIR="$(mktemp -d /tmp/vllm-build.XXXXXX)"
trap 'echo "Cleaning up ${BUILD_DIR}..."; rm -rf "${BUILD_DIR}"' EXIT

echo "=== Cloning ${REPO_URL} (branch: ${BRANCH}) into ${BUILD_DIR} ==="
git clone --depth=1 --branch "${BRANCH}" "${REPO_URL}" "${BUILD_DIR}"

echo ""
echo "=== Build configuration ==="
echo "  Image tag        : ${IMAGE_TAG}"
echo "  Python version   : ${PYTHON_VERSION}"
echo "  Disable AVX-512  : ${DISABLE_AVX512}"
echo "  AMX-BF16         : ${VLLM_CPU_AMXBF16}"
echo "  Parallel jobs    : ${MAX_JOBS}"
echo ""

# Ensure buildx builder exists and supports linux/amd64
if ! docker buildx inspect vllm-x86-builder &>/dev/null; then
    echo "=== Creating buildx builder 'vllm-x86-builder' ==="
    docker buildx create --name vllm-x86-builder --use
else
    docker buildx use vllm-x86-builder
fi

echo "=== Building Docker image: ${IMAGE_TAG} ==="
docker buildx build \
    --platform linux/amd64 \
    --file "${BUILD_DIR}/docker/Dockerfile.cpu" \
    --target vllm-openai \
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
    --build-arg "VLLM_CPU_DISABLE_AVX512=${DISABLE_AVX512}" \
    --build-arg "VLLM_CPU_AMXBF16=${VLLM_CPU_AMXBF16}" \
    --build-arg "max_jobs=${MAX_JOBS}" \
    --tag "${IMAGE_TAG}" \
    --load \
    "${BUILD_DIR}"

echo ""
echo "=== Build complete ==="
echo "Image: ${IMAGE_TAG}"
echo ""
echo "Run the container:"
echo "  DOCKER_IMAGE=${IMAGE_TAG} MODEL_DIR=/path/to/models ./run_container.sh"

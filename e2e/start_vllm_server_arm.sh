#!/bin/bash
# =============================================================================
# Start a single vLLM server for ARM aarch64 (r8g.8xlarge — 32 vCPUs, NUMA 0)
#
# Uses the vllm-cpu-arm64 container image, which preloads LLVM libOMP via
# LD_PRELOAD for correct GNU-compatible threading on Neoverse-V2 (no Intel OMP).
#
# Usage (inside the container, from /workspace):
#   ./start_vllm_server_arm.sh               # foreground
#   ./start_vllm_server_arm.sh --background  # background, waits until ready
#   ./start_vllm_server_arm.sh --stop        # stop a running background server
#
# Environment overrides:
#   MODEL_PATH     Model directory   (default: /data/Meta-Llama-3.1-8B-Instruct)
#   VLLM_PORT      Server port       (default: 8120)
#   MAX_MODEL_LEN  Context length    (default: 8192)
#   KV_CACHE_GB    KV cache in GiB   (default: 60  – safe for 247 GB instance)
#   DTYPE          Compute dtype     (default: bfloat16, Neoverse-V2 SVE BF16)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/data/Meta-Llama-3.1-8B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8120}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
KV_CACHE_GB="${KV_CACHE_GB:-60}"
DTYPE="${DTYPE:-bfloat16}"

BACKGROUND=0
if [[ "${1:-}" == "--stop" ]]; then
    PID_FILE="${SCRIPT_DIR}/vllm_arm.pid"
    if [ -f "${PID_FILE}" ]; then
        echo "Stopping vLLM server (pid=$(cat ${PID_FILE}))..."
        kill "$(cat ${PID_FILE})" 2>/dev/null || true
        rm -f "${PID_FILE}"
    fi
    fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
    exit 0
fi
if [[ "${1:-}" == "--background" ]]; then
    BACKGROUND=1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model path not found: ${MODEL_PATH}"
    echo "Set MODEL_PATH env var and retry."
    exit 1
fi

# Kill any stale instance on this port
fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true

echo "========================================================"
echo " Starting vLLM (ARM / CPU)  dp=1"
echo "  Model  : ${MODEL_PATH}"
echo "  Port   : ${VLLM_PORT}"
echo "  dtype  : ${DTYPE}"
echo "  kv_cache_space: ${KV_CACHE_GB} GiB"
echo "  context: ${MAX_MODEL_LEN} tokens"
echo "  OMP    : LLVM libgomp via LD_PRELOAD (set in container image)"
echo "========================================================"

VLLM_CMD=(
    python -m vllm.entrypoints.openai.api_server
        --model "${MODEL_PATH}"
        --served-model-name meta-llama/Llama-3.1-8B-Instruct
        --port "${VLLM_PORT}"
        --host 0.0.0.0
        --dtype "${DTYPE}"
        --tensor-parallel-size 1
        --max-model-len "${MAX_MODEL_LEN}"
        --max-num-seqs 128
        --max-num-batched-tokens 4096
        --no-enable-prefix-caching
        --no-enable-expert-parallel
        --trust-remote-code
        --trust-request-chat-template
)

export VLLM_CPU_KVCACHE_SPACE="${KV_CACHE_GB}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ "${BACKGROUND}" -eq 1 ]; then
    LOG="${SCRIPT_DIR}/vllm_server_arm.log"
    "${VLLM_CMD[@]}" >"${LOG}" 2>&1 &
    VLLM_PID=$!
    echo "${VLLM_PID}" > "${SCRIPT_DIR}/vllm_arm.pid"
    echo "vLLM started in background (pid=${VLLM_PID})  log: ${LOG}"

    # Wait until the server is ready (up to 5 minutes)
    echo -n "Waiting for vLLM to be ready on port ${VLLM_PORT} ..."
    for _i in $(seq 1 60); do
        sleep 5
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            echo " ready!"
            exit 0
        fi
        echo -n "."
        # Check if still alive
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo ""
            echo "ERROR: vLLM process died. Check ${LOG}"
            exit 1
        fi
    done
    echo ""
    echo "ERROR: vLLM did not become ready within 5 minutes."
    echo "Check ${LOG}"
    exit 1
else
    "${VLLM_CMD[@]}"
fi

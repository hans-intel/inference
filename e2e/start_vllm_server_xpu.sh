#!/bin/bash
# =============================================================================
# Start vLLM OpenAI-compatible server on XPU:0 (dp=1)
#
# Usage (inside the container, from /workspace):
#   ./start_vllm_server_xpu.sh                    # foreground
#   ./start_vllm_server_xpu.sh --background       # background, writes vllm_server.pid
#
# Environment overrides:
#   MODEL_PATH       Path to the model weights   (default below)
#   VLLM_PORT        Port to serve on            (default: 8123)
#   MAX_MODEL_LEN    Max context window tokens    (default: 3072)
#   DTYPE            Compute dtype               (default: float16)
# =============================================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/data/Llama-3.1-8B-Instruct-autoround-w4g-1-iters512-xpu/}"
VLLM_PORT="${VLLM_PORT:-8123}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
DTYPE="${DTYPE:-float16}"

# Parse flags
BACKGROUND=0
for arg in "$@"; do
    case "${arg}" in
        --background) BACKGROUND=1 ;;
    esac
done

# ── XPU performance environment variables ───────────────────────────────────
export ZE_AFFINITY_MASK=0                    # pin to XPU:0
export VLLM_USE_TRITON_XPU_ATTN=1
export VLLM_XPU_USE_W4A8=1
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "============================================="
echo " vLLM XPU Server Configuration"
echo "  Model         : ${MODEL_PATH}"
echo "  Port          : ${VLLM_PORT}"
echo "  Device        : XPU:0  (ZE_AFFINITY_MASK=0)"
echo "  dtype         : ${DTYPE}"
echo "  Max Model Len : ${MAX_MODEL_LEN}"
echo "============================================="

# ── Kill any existing vLLM server on this port ──────────────────────────────
EXISTING_PID_FILE="$(dirname "${BASH_SOURCE[0]}")/vllm_server.pid"
if [ -f "${EXISTING_PID_FILE}" ]; then
    OLD_PID=$(cat "${EXISTING_PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "Stopping existing vLLM server (PID ${OLD_PID})..."
        kill "${OLD_PID}" 2>/dev/null || true
        sleep 2
    fi
    rm -f "${EXISTING_PID_FILE}"
fi
fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true

# ── Verify model path exists ────────────────────────────────────────────────
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model path not found: ${MODEL_PATH}"
    echo "Set MODEL_PATH to a valid model directory and retry."
    exit 1
fi

# ── Build vLLM serve command ────────────────────────────────────────────────
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --port ${VLLM_PORT} \
    --host 0.0.0.0 \
    --dtype ${DTYPE} \
    --tensor-parallel-size 1 \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs 384 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.9 \
    --kv-cache-dtype fp8_e5m2 \
    --async-scheduling \
    --no-enable-expert-parallel \
    --no-enable-prefix-caching \
    --trust-request-chat-template \
    --served-model-name meta-llama/Llama-3.1-8B-Instruct \
    --trust-remote-code"

# ── Launch ──────────────────────────────────────────────────────────────────
if [ "${BACKGROUND}" -eq 1 ]; then
    LOG_FILE="vllm_server.log"
    echo "Starting vLLM server in background. Logs: ${LOG_FILE}"
    nohup bash -c "${VLLM_CMD}" > "${LOG_FILE}" 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: ${VLLM_PID}"
    echo "${VLLM_PID}" > vllm_server.pid

    echo "Waiting for vLLM server to become ready on port ${VLLM_PORT}..."
    for i in $(seq 1 60); do
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            break
        fi
        if [ "$i" -eq 60 ]; then
            echo "WARNING: vLLM server did not respond within 5 minutes."
            echo "Check ${LOG_FILE} for errors."
        fi
        sleep 5
    done
else
    echo "Starting vLLM server in foreground (Ctrl+C to stop)..."
    eval "${VLLM_CMD}"
fi
#!/bin/bash
# =============================================================================
# Start 3 vLLM CPU instances, each pinned to a dedicated SNC-3 NUMA node
# (GNR-AP socket 1 has 3 NUMA nodes × 43 cores each with HT off).
#
# Core / memory layout (socket 1 — nodes 3-5):
#   Instance 0: cores 128-170  mem node 3  port 8120
#   Instance 1: cores 171-213  mem node 4  port 8121
#   Instance 2: cores 214-255  mem node 5  port 8122
#
# Usage (inside the container, from /workspace):
#   ./start_vllm_servers_dp3_cpu.sh           # start all 3 instances
#   ./start_vllm_servers_dp3_cpu.sh --stop    # stop all instances
#
# PIDs are written to vllm_dp3.pids (one per line)
#
# Environment overrides:
#   MODEL_PATH     Path to model weights         (default below)
#   MAX_MODEL_LEN  Max context window             (default: 10240)
#   DTYPE          Compute dtype                  (default: bfloat16)
#   BASE_PORT      First server port              (default: 8120)
#   KV_CACHE_GB    KV cache GiB per instance      (default: 40)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
DTYPE="${DTYPE:-bfloat16}"
BASE_PORT="${BASE_PORT:-8120}"
KV_CACHE_GB="${KV_CACHE_GB:-40}"
PID_FILE="${SCRIPT_DIR}/vllm_dp3.pids"

# Fixed NUMA layout for GNR-AP SNC-3 (socket 1 — nodes 3-5)
INSTANCE_CORES=("128-170" "171-213" "214-255")
INSTANCE_MEM=(3 4 5)
DP=3

# ── Stop all instances ───────────────────────────────────────────────────────
stop_all() {
    if [ -f "${PID_FILE}" ]; then
        echo "Stopping vLLM dp=${DP} instances..."
        while IFS= read -r pid; do
            kill "${pid}" 2>/dev/null || true
        done < "${PID_FILE}"
        rm -f "${PID_FILE}"
    fi
    for i in $(seq 0 $((DP - 1))); do
        fuser -k "$((BASE_PORT + i))/tcp" 2>/dev/null || true
    done
}

if [[ "${1:-}" == "--stop" ]]; then
    stop_all
    exit 0
fi

# ── Pre-flight checks ────────────────────────────────────────────────────────
if ! command -v numactl &>/dev/null; then
    echo "ERROR: numactl is required for core+memory pinning but was not found."
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model path not found: ${MODEL_PATH}"
    echo "Set MODEL_PATH to a valid model directory and retry."
    exit 1
fi

# Kill any previous instances on these ports
stop_all

echo "============================================="
echo " vLLM Data-Parallel Server (dp=3, SNC-3 pinned)"
echo "  Model         : ${MODEL_PATH}"
echo "  dtype         : ${DTYPE}"
echo "  Max Model Len : ${MAX_MODEL_LEN}"
echo "  Ports         : ${BASE_PORT} → $((BASE_PORT + DP - 1))"
echo "  KV Cache/Inst : ${KV_CACHE_GB} GiB  (total $((DP * KV_CACHE_GB)) GiB)"
  echo "  Instance 0    : cores 128-170  mem node 3   port ${BASE_PORT}"
  echo "  Instance 1    : cores 171-213  mem node 4   port $((BASE_PORT + 1))"
  echo "  Instance 2    : cores 214-255  mem node 5   port $((BASE_PORT + 2))"
echo "============================================="

# ── Launch all 3 instances ───────────────────────────────────────────────────
> "${PID_FILE}"   # truncate / create

# Ensure XPU is NOT visible (CPU-only mode)
unset ZE_AFFINITY_MASK 2>/dev/null || true

for i in 0 1 2; do
    PORT=$((BASE_PORT + i))
    CORES="${INSTANCE_CORES[$i]}"
    MEM="${INSTANCE_MEM[$i]}"
    LOG="${SCRIPT_DIR}/vllm_server_${i}.log"

    echo "  [${i}] port=${PORT}  cores=${CORES}  mem=${MEM}  kv_cache=${KV_CACHE_GB}GiB  log=$(basename "${LOG}")"

    VLLM_CPU_KVCACHE_SPACE="${KV_CACHE_GB}" \
    VLLM_CPU_NUM_OF_RESERVED_CPU=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    nohup numactl -C "${CORES}" -m "${MEM}" \
        python -m vllm.entrypoints.openai.api_server \
            --model "${MODEL_PATH}" \
            --port "${PORT}" \
            --host 0.0.0.0 \
            --dtype "${DTYPE}" \
            --tensor-parallel-size 1 \
            --max-model-len "${MAX_MODEL_LEN}" \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 1536 \
            --no-enable-expert-parallel \
            --no-enable-prefix-caching \
            --trust-request-chat-template \
            --trust-remote-code \
        > "${LOG}" 2>&1 &
    echo "$!" >> "${PID_FILE}"
done

echo ""
echo "Waiting for all ${DP} instances to become ready (up to 5 min)..."
ALL_READY=0
for attempt in $(seq 1 60); do
    ALL_READY=1
    for i in $(seq 0 $((DP - 1))); do
        PORT=$((BASE_PORT + i))
        if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
            ALL_READY=0
            break
        fi
    done
    if [ "${ALL_READY}" -eq 1 ]; then
        echo "All ${DP} vLLM instances are ready!"
        break
    fi
    if [ "${attempt}" -eq 60 ]; then
        echo "WARNING: Not all vLLM instances responded within 5 minutes."
        echo "Check vllm_server_*.log for errors."
    fi
    sleep 5
done

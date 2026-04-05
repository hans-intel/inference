#!/bin/bash
# =============================================================================
# Start N vLLM instances for CPU data-parallel inference (default dp=4)
#
# Core layout (10 cores per instance, 40 cores total, all in NUMA node 3):
#   Node 3 = cores 128-170 (43 cores).  4×10=40 used; 168-170 left as OS buffer.
#   Instance 0: cores 128-137  port 8120  mem node 3
#   Instance 1: cores 138-147  port 8121  mem node 3
#   Instance 2: cores 148-157  port 8122  mem node 3
#   Instance 3: cores 158-167  port 8123  mem node 3
#   RAG workload: cores 171-255  mem nodes 4,5  (see run_e2e_with_llm_cpu.sh)
#
# Usage (inside the container):
#   ./start_vllm_servers_dp4.sh               # start all instances
#   ./start_vllm_servers_dp4.sh --stop        # stop all instances
#
# PIDs are written to vllm_dp4.pids (one per line)
#
# Environment overrides:
#   MODEL_PATH         Path to model weights         (default below)
#   MAX_MODEL_LEN      Max context window             (default: 10240)
#   DTYPE              Compute dtype                  (default: bfloat16)
#   BASE_PORT          First server port              (default: 8120)
#   DP                 Number of instances            (default: 4)
#   CORES_PER_INST     CPU cores per instance         (default: 10, so 4×10=40 → cores 0-39)
#   NUMA_NODE          NUMA memory node for vLLM      (default: 0 — all instances stay in node 0)
#   KV_CACHE_GB        KV cache GiB per instance      (default: 40, i.e. 4×40=160 GiB total)
#                      MUST be set explicitly — vLLM will otherwise try to use all free RAM
#                      and get OOM-killed (signal 9) when multiple instances start in parallel.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
DTYPE="${DTYPE:-bfloat16}"
BASE_PORT="${BASE_PORT:-8120}"
DP="${DP:-4}"
CORES_PER_INST="${CORES_PER_INST:-7}"  # 4 × 7 = 28 cores (128-155), all in NUMA node 3
NUMA_NODE="${NUMA_NODE:-3}"
CORE_BASE="${CORE_BASE:-128}"  # first core on socket 1, node 3
KV_CACHE_GB="${KV_CACHE_GB:-40}"        # GiB of KV cache per instance (4×40=160 GiB total)
PID_FILE="${SCRIPT_DIR}/vllm_dp4.pids"

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
        PORT=$((BASE_PORT + i))
        fuser -k "${PORT}/tcp" 2>/dev/null || true
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
echo " vLLM Data-Parallel Server (dp=${DP})"
echo "  Model         : ${MODEL_PATH}"
echo "  dtype         : ${DTYPE}"
echo "  Max Model Len : ${MAX_MODEL_LEN}"
echo "  Base Port     : ${BASE_PORT} → $((BASE_PORT + DP - 1))"
  echo "  Cores/Instance: ${CORES_PER_INST}  (${DP}×${CORES_PER_INST}=$((DP * CORES_PER_INST)) cores, ${CORE_BASE}-$((CORE_BASE + DP * CORES_PER_INST - 1)))"
echo "  KV Cache/Inst : ${KV_CACHE_GB} GiB  (total $((DP * KV_CACHE_GB)) GiB)"
  echo "  Memory Node   : node ${NUMA_NODE}  (all instances NUMA-local)"
echo "============================================="

# ── Launch all DP instances ──────────────────────────────────────────────────
> "${PID_FILE}"   # truncate / create

# Ensure XPU is NOT visible to any instance (CPU-only mode)
unset ZE_AFFINITY_MASK 2>/dev/null || true

for i in $(seq 0 $((DP - 1))); do
    PORT=$((BASE_PORT + i))
    CORE_START=$((CORE_BASE + i * CORES_PER_INST))
    CORE_END=$((CORE_START + CORES_PER_INST - 1))
    LOG="${SCRIPT_DIR}/vllm_server_${i}.log"

    echo "  [${i}] port=${PORT}  cores=${CORE_START}-${CORE_END}  kv_cache=${KV_CACHE_GB}GiB  log=$(basename "${LOG}")"

    VLLM_CPU_KVCACHE_SPACE="${KV_CACHE_GB}" \
    VLLM_CPU_NUM_OF_RESERVED_CPU=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    nohup numactl -C "${CORE_START}-${CORE_END}" -m "${NUMA_NODE}" \
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

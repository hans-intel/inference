#!/bin/bash
# =============================================================================
# End-to-end RAG + LLM pipeline runner  [CPU, dp=3 vLLM, 2-phase execution]
#
# Two-phase design — retrieval and LLM generation use separate core sets:
#
#   Phase 1 — Retrieval (all 128 cores on socket 1, no vLLM running):
#     numactl -C 128-255 -m 3,4,5
#     Ingestion → embedding → FAISS → reranking for 256×10 = 2560 queries
#     Results saved to retrieval_cache.json and script exits.
#
#   Phase 2 — vLLM + LLM generation:
#     3 vLLM servers launched, each NUMA-pinned to one SNC-3 tile (socket 1):
#       Instance 0: cores 128-170  mem 3   port 8120
#       Instance 1: cores 171-213  mem 4   port 8121
#       Instance 2: cores 214-255  mem 5   port 8122
#     LLM client loads retrieval_cache.json → sends 2560 LLM requests
#     round-robin across 3 servers (≈853 requests/server).
#
# Usage (inside the container, from /workspace):
#   ./run_e2e_dp3_cpu.sh             # full run (phase 1 + phase 2)
#   ./run_e2e_dp3_cpu.sh --phase1    # retrieval only, save cache; no vLLM
#   ./run_e2e_dp3_cpu.sh --phase2    # LLM only from existing retrieval_cache.json
#
# Environment overrides:
#   DB               Vector DB path          (default: vector_256_cpu.db)
#   INGEST_PATH      Path to ingest data     (default: passages_256/doc_html_len256.json)
#   REPEAT           Dataset repeat factor   (default: 10  → 256×10=2560 queries)
#   LLM_BATCH        Total LLM concurrency   (default: 256 = ~85/server × 3)
#   FAISS_BATCH      FAISS indexing batch    (default: 256)
#   KV_CACHE_GB      KV cache GiB/instance   (default: 40)
#   CACHE_FILE       Retrieval cache JSON    (default: retrieval_cache.json)
#   RETRIEVER_MODEL  Embedding model         (default: intfloat/e5-base-v2)
#   RERANKER_MODEL   Reranker model          (default: colbert-ir/colbertv2.0)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DB="${DB:-vector_256_hnsw_cpu.db}"
INGEST_PATH="${INGEST_PATH:-${SCRIPT_DIR}/passages_256/doc_html_len256.json}"
REPEAT="${REPEAT:-10}"                  # 256 × 10 = 2560 queries
LLM_BATCH="${LLM_BATCH:-256}"           # total concurrent LLM requests
FAISS_BATCH="${FAISS_BATCH:-256}"
CACHE_FILE="${CACHE_FILE:-${SCRIPT_DIR}/retrieval_cache.json}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RERANKER_MODEL="${RERANKER_MODEL:-colbert-ir/colbertv2.0}"

DP=3
BASE_PORT=8120
# Build comma-separated URL list for all 3 servers
LLM_URLS=""
for _i in 0 1 2; do
    _PORT=$((BASE_PORT + _i))
    LLM_URLS="${LLM_URLS:+${LLM_URLS},}http://127.0.0.1:${_PORT}/v1/chat/completions"
done

VLLM_DP3_PID_FILE="${SCRIPT_DIR}/vllm_dp3.pids"

# ── Parse flags ───────────────────────────────────────────────────────────────
PHASE1_ONLY=0
PHASE2_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --phase1) PHASE1_ONLY=1 ;;
        --phase2) PHASE2_ONLY=1 ;;
    esac
done

# ── Cleanup handler ───────────────────────────────────────────────────────────
cleanup() {
    if [ -f "${VLLM_DP3_PID_FILE}" ]; then
        echo ""
        echo "Stopping vLLM dp=3 instances..."
        while IFS= read -r _pid; do
            kill "${_pid}" 2>/dev/null || true
        done < "${VLLM_DP3_PID_FILE}"
        rm -f "${VLLM_DP3_PID_FILE}"
    fi
    for _i in 0 1 2; do
        fuser -k "$((BASE_PORT + _i))/tcp" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Ingestion + embedding + FAISS + reranking on ALL 128 cores
# (vLLM is NOT running, so all cores are available)
# ─────────────────────────────────────────────────────────────────────────────
if [ "${PHASE2_ONLY}" -eq 0 ]; then
    if [ -f "${CACHE_FILE}" ] && [ "${PHASE1_ONLY}" -eq 0 ]; then
        echo "Found existing retrieval cache: ${CACHE_FILE}"
        echo "Skipping Phase 1 (delete the file to re-run retrieval)."
    else
        echo "========================================================"
        echo " Phase 1: Retrieval — all 128 cores (no vLLM)"
        echo "  Cores   : 128-255  (numactl -C 128-255 -m 3,4,5)"
        echo "  Queries : 256 × ${REPEAT} = $((256 * REPEAT))"
        echo "  Embed bs: 128  |  FAISS bs: ${FAISS_BATCH}  |  Rerank bs: 256"
        echo "  Cache   : ${CACHE_FILE}"
        echo "========================================================"

        # Decide data source: existing DB or fresh ingest
        DB_FILE="${DB}"
        [[ "${DB_FILE}" != *.db ]] && DB_FILE="${DB_FILE}.db"
        if [ -f "${DB_FILE}" ]; then
            DATA_ARGS=(--database "${DB}")
            echo "Using existing vector DB: ${DB_FILE}"
        else
            DATA_ARGS=(--ingest "${INGEST_PATH}")
            echo "Ingesting from: ${INGEST_PATH}"
        fi

        # OMP/KMP env vars are intentionally NOT set here.
        # Each embedding worker (VectorDBInstance) sets its own OMP_NUM_THREADS
        # and KMP_AFFINITY in _worker_main BEFORE importing torch/faiss, which
        # is the only way to guarantee per-worker OMP thread-pool isolation.
        # Setting them here would be inherited by all worker processes and would
        # override the per-worker settings with the wrong (parent-level) values.
        numactl -C 128-255 -m "3,4,5" \
        python -u "${SCRIPT_DIR}/single_shot_retrieval.py" \
            --retrieval_method vector \
            --device cpu \
            "${DATA_ARGS[@]}" \
            --benchmark \
            --retriever_model "${RETRIEVER_MODEL}" \
            --reranker_model "${RERANKER_MODEL}" \
            --model_dtype bfloat16 \
            --embedding_batch_size 512 \
            --num_embedding_devices 3 \
            --reranker_batch_size 256 \
            --faiss_indexing_batch_size "${FAISS_BATCH}" \
            --vector_index_method hnsw \
            --dataset data/frames_dataset_256.tsv \
            --repeat "${REPEAT}" \
            --retrieval-cache-out "${CACHE_FILE}" \
            --eval \
            |& tee e2e_phase1_retrieval_cpu.log

        echo ""
        echo "Phase 1 complete. Cache saved to: ${CACHE_FILE}"
    fi

    if [ "${PHASE1_ONLY}" -eq 1 ]; then
        echo "Phase 1 only — done."
        exit 0
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Start 3 vLLM servers (NUMA pinned), then run LLM generation
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "${CACHE_FILE}" ]; then
    echo "ERROR: Retrieval cache not found: ${CACHE_FILE}"
    echo "Run Phase 1 first:  ./run_e2e_dp3_cpu.sh --phase1"
    exit 1
fi

echo ""
echo "========================================================"
echo " Phase 2: vLLM + LLM generation"
echo "  Server 0: cores 128-170  mem 3   port 8120"
echo "  Server 1: cores 171-213  mem 4   port 8121"
echo "  Server 2: cores 214-255  mem 5   port 8122"
echo "  LLM batch: ${LLM_BATCH} total (~$((LLM_BATCH / 3))/server)"
echo "  Queries  : $(python3 -c "import json; d=json.load(open('${CACHE_FILE}')); print(len(d))" 2>/dev/null || echo "?")"
echo "========================================================"

bash "${SCRIPT_DIR}/start_vllm_servers_dp3_cpu.sh"

# Double-check all 3 instances are ready
ALL_READY=1
for _i in 0 1 2; do
    _PORT=$((BASE_PORT + _i))
    if ! curl -sf "http://127.0.0.1:${_PORT}/v1/models" > /dev/null 2>&1; then
        echo "ERROR: vLLM instance ${_i} (port ${_PORT}) is not responding."
        echo "Check vllm_server_${_i}.log for details."
        ALL_READY=0
    fi
done
if [ "${ALL_READY}" -ne 1 ]; then
    exit 1
fi
echo "All 3 vLLM instances ready. URLs: ${LLM_URLS}"
echo ""

# LLM generation: client is I/O-bound (HTTP requests), run on unrestricted cores
python -u "${SCRIPT_DIR}/single_shot_retrieval.py" \
    --retrieval_method vector \
    --database "${DB}" \
    --retriever_model "${RETRIEVER_MODEL}" \
    --reranker_model "${RERANKER_MODEL}" \
    --retrieval-cache-in "${CACHE_FILE}" \
    --llm_service_url "${LLM_URLS}" \
    --generate-answer \
    --save-results \
    --benchmark \
    --device cpu \
    --model_dtype bfloat16 \
    --dataset data/frames_dataset_256.tsv \
    --llm_batch_size "${LLM_BATCH}" \
    --eval \
    |& tee e2e_phase2_llm_cpu.log

echo ""
echo "=== Done ==="
echo "Phase 1 log : e2e_phase1_retrieval_cpu.log"
echo "Phase 2 log : e2e_phase2_llm_cpu.log"
echo "Results     : result_single_shot.json"

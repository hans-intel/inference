#!/bin/bash
# =============================================================================
# End-to-end RAG + LLM pipeline  [ARM aarch64, r8g.8xlarge, dp=1, 32 vCPUs]
#
# Two-phase design:
#
#   Phase 1 — Retrieval (all 32 cores, no vLLM):
#     Ingestion → embedding → FAISS → reranking for REPEAT×256 queries
#     Results saved to retrieval_cache.json
#
#   Phase 2 — vLLM + LLM generation:
#     1 vLLM server on all 32 cores (port 8120)
#     Loads retrieval_cache.json → sends LLM_BATCH concurrent requests
#
# Usage (inside the container, from /workspace):
#   ./run_e2e_dp1_cpu.sh             # phase 1 + phase 2
#   ./run_e2e_dp1_cpu.sh --phase1    # retrieval only, save cache; no vLLM
#   ./run_e2e_dp1_cpu.sh --phase2    # LLM only from existing retrieval_cache.json
#
# Environment overrides:
#   DB               Vector DB path             (default: vector_256_hnsw_arm.db)
#   INGEST_PATH      Passages JSON              (default: passages_256/doc_html_len256.json)
#   MAX_PASSAGES     Ingest only first N docs   (default: 5000)
#   REPEAT           Dataset repeat factor      (default: 1 → 256 queries)
#   LLM_BATCH        Concurrent LLM requests   (default: 32)
#   FAISS_BATCH      FAISS indexing batch       (default: 256)
#   KV_CACHE_GB      vLLM KV cache in GiB      (default: 60)
#   CACHE_FILE       Retrieval cache JSON       (default: retrieval_cache_arm.json)
#   MODEL_PATH       LLM model directory        (default: /data/Meta-Llama-3.1-8B-Instruct)
#   RETRIEVER_MODEL  Embedding model            (default: intfloat/e5-base-v2)
#   RERANKER_MODEL   Reranker model             (default: colbert-ir/colbertv2.0)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DB="${DB:-vector_256_hnsw_arm.db}"
INGEST_PATH="${INGEST_PATH:-${SCRIPT_DIR}/passages_256/doc_html_len256.json}"
REPEAT="${REPEAT:-1}"                   # 1 × 256 = 256 queries
MAX_PASSAGES="${MAX_PASSAGES:-5000}"    # ingest only the first N passages
LLM_BATCH="${LLM_BATCH:-32}"            # concurrent LLM requests to single server
FAISS_BATCH="${FAISS_BATCH:-256}"
CACHE_FILE="${CACHE_FILE:-${SCRIPT_DIR}/retrieval_cache_arm.json}"
MODEL_PATH="${MODEL_PATH:-/data/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RERANKER_MODEL="${RERANKER_MODEL:-colbert-ir/colbertv2.0}"
KV_CACHE_GB="${KV_CACHE_GB:-60}"

LLM_URL="http://127.0.0.1:8120/v1/chat/completions"
VLLM_PID_FILE="${SCRIPT_DIR}/vllm_arm.pid"

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
    if [ -f "${VLLM_PID_FILE}" ]; then
        echo ""
        echo "Stopping vLLM server..."
        kill "$(cat ${VLLM_PID_FILE})" 2>/dev/null || true
        rm -f "${VLLM_PID_FILE}"
    fi
    fuser -k "8120/tcp" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Ingestion + embedding + FAISS + reranking  (all 32 cores)
# ─────────────────────────────────────────────────────────────────────────────
if [ "${PHASE2_ONLY}" -eq 0 ]; then
    if [ -f "${CACHE_FILE}" ] && [ "${PHASE1_ONLY}" -eq 0 ]; then
        echo "Found existing retrieval cache: ${CACHE_FILE}"
        echo "Skipping Phase 1 (delete the file to re-run retrieval)."
    else
        echo "========================================================"
        echo " Phase 1: Retrieval — 32 cores (no vLLM)"

        echo "  Passages: ${MAX_PASSAGES} (capped)"
        echo "  Queries : 256 × ${REPEAT} = $((256 * REPEAT))"
        echo "  Embed bs: 32  |  FAISS bs: ${FAISS_BATCH}  |  Rerank bs: 32"
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

        # On ARM (Neoverse-V2 / Graviton4):
        #   - Single NUMA node, single socket — no binding needed
        #   - LLVM libOMP is set via LD_PRELOAD in the container image
        python -u "${SCRIPT_DIR}/single_shot_retrieval.py" \
            --retrieval_method vector \
            --device cpu \
            "${DATA_ARGS[@]}" \
            --benchmark \
            --retriever_model "${RETRIEVER_MODEL}" \
            --reranker_model "${RERANKER_MODEL}" \
            --model_dtype bfloat16 \
            --embedding_batch_size 32 \
            --num_embedding_devices 1 \
            --reranker_batch_size 32 \
            --max_passages "${MAX_PASSAGES}" \
            --faiss_indexing_batch_size "${FAISS_BATCH}" \
            --vector_index_method hnsw \
            --dataset data/frames_dataset_256.tsv \
            --repeat "${REPEAT}" \
            --retrieval-cache-out "${CACHE_FILE}" \
            --eval \
            |& tee e2e_phase1_retrieval_arm.log

        echo ""
        echo "Phase 1 complete. Cache saved to: ${CACHE_FILE}"
    fi

    if [ "${PHASE1_ONLY}" -eq 1 ]; then
        echo "Phase 1 only — done."
        exit 0
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Start vLLM server (dp=1, all 32 cores), then run LLM generation
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "${CACHE_FILE}" ]; then
    echo "ERROR: Retrieval cache not found: ${CACHE_FILE}"
    echo "Run Phase 1 first:  ./run_e2e_dp1_cpu.sh --phase1"
    exit 1
fi

echo ""
echo "========================================================"
echo " Phase 2: vLLM + LLM generation  (dp=1)"
echo "  Model     : ${MODEL_PATH}"
  echo "  Server    : port 8120  (all 32 cores)"
echo "  LLM batch : ${LLM_BATCH}"
echo "  Queries   : $(python3 -c "import json; d=json.load(open('${CACHE_FILE}')); print(len(d))" 2>/dev/null || echo "?")"
echo "========================================================"

# Start single vLLM server, wait until ready
MODEL_PATH="${MODEL_PATH}" KV_CACHE_GB="${KV_CACHE_GB}" \
    bash "${SCRIPT_DIR}/start_vllm_server_arm.sh" --background

# Verify it's up
if ! curl -sf "http://127.0.0.1:8120/v1/models" >/dev/null 2>&1; then
    echo "ERROR: vLLM server is not responding on port 8120."
    echo "Check vllm_server_arm.log for details."
    exit 1
fi
echo "vLLM server ready. URL: ${LLM_URL}"
echo ""

# LLM generation — client is I/O bound, runs unrestricted
python -u "${SCRIPT_DIR}/single_shot_retrieval.py" \
    --retrieval_method vector \
    --database "${DB}" \
    --retriever_model "${RETRIEVER_MODEL}" \
    --reranker_model "${RERANKER_MODEL}" \
    --retrieval-cache-in "${CACHE_FILE}" \
    --llm_service_url "${LLM_URL}" \
    --generate-answer \
    --save-results \
    --benchmark \
    --device cpu \
    --model_dtype bfloat16 \
    --dataset data/frames_dataset_256.tsv \
    --llm_batch_size "${LLM_BATCH}" \
    --eval \
    |& tee e2e_phase2_llm_arm.log

echo ""
echo "=== Done ==="
echo "Phase 1 log : e2e_phase1_retrieval_arm.log"
echo "Phase 2 log : e2e_phase2_llm_arm.log"
echo "Results     : result_single_shot.json"

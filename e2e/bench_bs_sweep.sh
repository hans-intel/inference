#!/bin/bash
# =============================================================================
# Batch-size sweep benchmark for ARM r8g.8xlarge
#
# Stage A — Ingestion embedding BS sweep (4 runs, fresh ingest each time):
#   --embedding_batch_size in [16, 32, 64, 128]
#   20K passages, 256 queries (REPEAT=1)
#   Measures: ingestion throughput, ingestion embedding latency
#
# Stage B — Query BS sweep (4 × 3 = 12 runs, pre-built DB):
#   --query_embedding_batch_size in [16, 32, 64, 128]
#   --reranker_batch_size       in [16, 32, 64]
#   20K passages DB (built with embed_bs=128 in Stage A), 2560 queries (REPEAT=10)
#   Measures: per-query embedding, vector search, reranking latency
#
# Usage (inside the container, from /workspace):
#   ./bench_bs_sweep.sh          # run all stages + print table
#   ./bench_bs_sweep.sh --parse  # only (re)parse existing logs + print table
#
# Logs land in:  ./sweep_logs/
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INGEST_PATH="${INGEST_PATH:-${SCRIPT_DIR}/passages_256/doc_html_len256.json}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RERANKER_MODEL="${RERANKER_MODEL:-colbert-ir/colbertv2.0}"
MAX_PASSAGES=20000
LOG_DIR="${SCRIPT_DIR}/sweep_logs"
CANONICAL_DB="sweep_canonical_embs128"   # built in Stage A with BS=128

mkdir -p "${LOG_DIR}"

# ── Embedding BS sweep values ─────────────────────────────────────────────────
EMBED_BS_VALUES=(16 32 64 128)
Q_EMBED_BS_VALUES=(16 32 64 128)
RERANK_BS_VALUES=(16 32 64)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run single_shot_retrieval.py and tee to a log file
# Usage: run_retrieval <log_file> <extra args...>
# ─────────────────────────────────────────────────────────────────────────────
run_retrieval() {
    local log="$1"; shift
    python -u "${SCRIPT_DIR}/single_shot_retrieval.py" \
        --retrieval_method vector \
        --device cpu \
        --benchmark \
        --retriever_model "${RETRIEVER_MODEL}" \
        --reranker_model  "${RERANKER_MODEL}" \
        --model_dtype bfloat16 \
        --num_embedding_devices 1 \
        --vector_index_method hnsw \
        --dataset data/frames_dataset_256.tsv \
        "$@" \
        |& tee "${log}"
}

# =============================================================================
# Decide whether to run or just parse
# =============================================================================
PARSE_ONLY=0
for arg in "$@"; do
    [[ "${arg}" == "--parse" ]] && PARSE_ONLY=1
done

if [ "${PARSE_ONLY}" -eq 0 ]; then

# =============================================================================
# Stage A — Ingestion embedding BS sweep
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Stage A: Ingestion embedding BS sweep (${MAX_PASSAGES} passages, 256 queries) ║"
echo "╚══════════════════════════════════════════════════════════════╝"

for embs in "${EMBED_BS_VALUES[@]}"; do
    DB_NAME="sweep_ingest_embs${embs}"
    LOG="${LOG_DIR}/stageA_embs${embs}.log"

    # Remove any existing DB so we always re-ingest (both named and default)
    rm -f "${DB_NAME}.db" vector.db
    rm -rf "${SCRIPT_DIR}/${DB_NAME}_data" "${SCRIPT_DIR}/vector_data"

    echo ""
    echo "─── Stage A | embed_bs=${embs} → DB=${DB_NAME}.db ───"
    run_retrieval "${LOG}" \
        --ingest "${INGEST_PATH}" \
        --database "${DB_NAME}" \
        --max_passages "${MAX_PASSAGES}" \
        --embedding_batch_size "${embs}" \
        --query_embedding_batch_size "${embs}" \
        --reranker_batch_size 32 \
        --faiss_indexing_batch_size 0 \
        --repeat 1 \
        --eval

    echo "  → ${LOG}"
done

# =============================================================================
# Stage B — Query BS sweep  (load canonical DB, REPEAT=10 → 2560 queries)
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Stage B: Query BS sweep (load DB, REPEAT=10 → 2560 queries)  ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Canonical DB is the one built with embed_bs=128 in Stage A
CANONICAL_DB_PATH="${CANONICAL_DB}.db"
if [ ! -f "${CANONICAL_DB_PATH}" ]; then
    # Rename the BS=128 DB built above to the canonical name
    cp "sweep_ingest_embs128.db" "${CANONICAL_DB_PATH}" 2>/dev/null || true
    cp -r "${SCRIPT_DIR}/sweep_ingest_embs128_data" \
         "${SCRIPT_DIR}/${CANONICAL_DB}_data" 2>/dev/null || true
fi

for qbs in "${Q_EMBED_BS_VALUES[@]}"; do
    for rbs in "${RERANK_BS_VALUES[@]}"; do
        LOG="${LOG_DIR}/stageB_qbs${qbs}_rbs${rbs}.log"
        echo ""
        echo "─── Stage B | query_embed_bs=${qbs}  rerank_bs=${rbs} ───"
        run_retrieval "${LOG}" \
            --database "${CANONICAL_DB}" \
            --query_embedding_batch_size "${qbs}" \
            --embedding_batch_size 128 \
            --reranker_batch_size "${rbs}" \
            --faiss_indexing_batch_size 0 \
            --repeat 10 \
            --eval
        echo "  → ${LOG}"
    done
done

fi  # end PARSE_ONLY gate

# =============================================================================
# Stage C — Parse logs and print tables
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Results                                                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

python3 - <<'PYEOF'
import os, re, glob

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_logs")

def parse_log(path):
    """Extract key metrics from a single log file."""
    metrics = {}
    try:
        text = open(path).read()
    except FileNotFoundError:
        return metrics

    # Ingestion: "Embedding generation: 20,000 sequences in 24.17s"
    m = re.search(r"Embedding generation:\s*[\d,]+\s*sequences in\s*([\d.]+)s", text)
    if m:
        metrics["ingest_embed_s"] = float(m.group(1))

    # Ingestion throughput: "Ingestion of 20000 passages took 24.64 seconds. 202.92 docs/sec"
    m = re.search(r"Ingestion of \d+ passages took ([\d.]+) seconds\.\s*([\d.]+) docs/sec", text)
    if m:
        metrics["ingest_total_s"] = float(m.group(1))
        metrics["ingest_docs_per_s"] = float(m.group(2))

    # Retrieval timings block — lines like:
    #   "   query_embedding [CPU only]:"
    #   "      calls=256  avg=12.34ms  min=...  max=...  total=3.162s"
    for comp in ("query_embedding", "vector_search", "reranking"):
        # Match the component header then peek at the next non-empty line
        pattern = rf"{re.escape(comp)}.*?\n\s+(calls=\d+\s+avg=([\d.]+)ms.*?total=([\d.]+)s|(\d+\.\d+) ms)"
        m = re.search(pattern, text)
        if m:
            if m.group(2):  # multi-call: avg + total
                metrics[f"{comp}_avg_ms"] = float(m.group(2))
                metrics[f"{comp}_total_s"] = float(m.group(3))
            elif m.group(4):  # single call
                metrics[f"{comp}_avg_ms"] = float(m.group(4))

    return metrics

# ── Table A: Ingestion embedding BS ──────────────────────────────────────────
embed_bs_values = [16, 32, 64, 128]
print("\n┌── Table A: Ingestion Embedding Batch Size ──────────────────────────────┐")
print(f"  {'embed_bs':>9} │ {'embed_time(s)':>13} │ {'ingest_total(s)':>15} │ {'docs/sec':>10}")
print(f"  {'─'*9}─┼─{'─'*13}─┼─{'─'*15}─┼─{'─'*10}")
for bs in embed_bs_values:
    log = os.path.join(LOG_DIR, f"stageA_embs{bs}.log")
    m = parse_log(log)
    et  = f"{m['ingest_embed_s']:.2f}"    if 'ingest_embed_s'    in m else "—"
    it  = f"{m['ingest_total_s']:.2f}"    if 'ingest_total_s'    in m else "—"
    dps = f"{m['ingest_docs_per_s']:.1f}" if 'ingest_docs_per_s' in m else "—"
    print(f"  {bs:>9} │ {et:>13} │ {it:>15} │ {dps:>10}")
print(f"└{'─'*77}┘")

# ── Table B: Query BS sweep ───────────────────────────────────────────────────
q_bs_values = [16, 32, 64, 128]
r_bs_values = [16, 32, 64]
print("\n┌── Table B: Query Embedding × Reranking Batch Size (2560 queries) ────────────────────────────────────────────────┐")
hdr = f"  {'q_emb_bs':>9} │ {'rerank_bs':>9} │ {'q_embed_avg(ms)':>16} │ {'q_embed_tot(s)':>14} │ {'vec_srch_avg(ms)':>17} │ {'rerank_avg(ms)':>15} │ {'rerank_tot(s)':>13}"
print(hdr)
sep = f"  {'─'*9}─┼─{'─'*9}─┼─{'─'*16}─┼─{'─'*14}─┼─{'─'*17}─┼─{'─'*15}─┼─{'─'*13}"
print(sep)
for qbs in q_bs_values:
    for rbs in r_bs_values:
        log = os.path.join(LOG_DIR, f"stageB_qbs{qbs}_rbs{rbs}.log")
        m = parse_log(log)
        qe_avg  = f"{m['query_embedding_avg_ms']:.2f}"   if 'query_embedding_avg_ms'  in m else "—"
        qe_tot  = f"{m['query_embedding_total_s']:.2f}"  if 'query_embedding_total_s' in m else "—"
        vs_avg  = f"{m['vector_search_avg_ms']:.2f}"     if 'vector_search_avg_ms'    in m else "—"
        rk_avg  = f"{m['reranking_avg_ms']:.2f}"         if 'reranking_avg_ms'        in m else "—"
        rk_tot  = f"{m['reranking_total_s']:.2f}"        if 'reranking_total_s'       in m else "—"
        print(f"  {qbs:>9} │ {rbs:>9} │ {qe_avg:>16} │ {qe_tot:>14} │ {vs_avg:>17} │ {rk_avg:>15} │ {rk_tot:>13}")
    print(sep)
print(f"└{'─'*107}┘")
PYEOF

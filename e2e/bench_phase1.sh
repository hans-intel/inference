#!/bin/bash
# =============================================================================
# bench_phase1.sh — Run run_e2e_dp3_cpu.sh --phase1 N times and report
#                   the median total latency for each retrieval component.
#
# Usage:
#   ./bench_phase1.sh [N]        # default N=5
#
# Output:
#   Per-run logs saved to bench_phase1_run_<i>.log
#   Median summary printed to stdout and bench_phase1_summary.txt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:-5}"
SUMMARY="${SCRIPT_DIR}/bench_phase1_summary.txt"

echo "Running run_e2e_dp3_cpu.sh --phase1  x${N} (+ 1 warmup) ..."
echo ""

# ---- Warmup run: primes oneDNN AMX JIT kernels for all components ----
# Results are discarded; this prevents first-run JIT compilation penalty
# from inflating the first measurement run.
echo "=== WARMUP RUN (discarded) ==="
bash "${SCRIPT_DIR}/run_e2e_dp3_cpu.sh" --phase1 2>&1 > "${SCRIPT_DIR}/bench_phase1_warmup.log" \
    && echo "Warmup complete." \
    || { echo "WARNING: warmup run failed — check bench_phase1_warmup.log"; }
echo ""

# Arrays to collect total-seconds values per component across runs
declare -a DB_DESER=()
declare -a QUERY_EMB=()
declare -a VEC_SEARCH=()
declare -a RERANK=()

for i in $(seq 1 "$N"); do
    LOG="${SCRIPT_DIR}/bench_phase1_run_${i}.log"
    echo "--- Run ${i}/${N} ---"

    # Forward all env overrides passed to this script
    bash "${SCRIPT_DIR}/run_e2e_dp3_cpu.sh" --phase1 2>&1 | tee "$LOG"

    # ---- parse timings from the log ----

    # db_deserialize: line is "      1873.46 ms"  (just a bare number + " ms")
    db_ms=$(grep -A1 'db_deserialize' "$LOG" | grep -oE '[0-9]+\.[0-9]+[[:space:]]*ms' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    db_s=$(awk "BEGIN{printf \"%.3f\", ${db_ms}/1000}")

    # query_embedding / vector_search / reranking: "total=<X>s"
    qe_s=$(grep 'query_embedding' "$LOG" -A2 | grep -oE 'total=[0-9]+\.[0-9]+s' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    vs_s=$(grep 'vector_search'   "$LOG" -A2 | grep -oE 'total=[0-9]+\.[0-9]+s' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    rk_s=$(grep 'reranking'       "$LOG" -A2 | grep -oE 'total=[0-9]+\.[0-9]+s' | grep -oE '[0-9]+\.[0-9]+' | head -1)

    DB_DESER+=("$db_s")
    QUERY_EMB+=("$qe_s")
    VEC_SEARCH+=("$vs_s")
    RERANK+=("$rk_s")

    echo "  db_deserialize : ${db_s}s"
    echo "  query_embedding: ${qe_s}s"
    echo "  vector_search  : ${vs_s}s"
    echo "  reranking      : ${rk_s}s"
    echo ""
done

# ---- compute median of an array of floats using Python ----
median() {
    python3 -c "
import sys, statistics
vals = [float(x) for x in sys.argv[1:]]
print(f'{statistics.median(vals):.3f}')
" "$@"
}

MED_DB=$(median "${DB_DESER[@]}")
MED_QE=$(median "${QUERY_EMB[@]}")
MED_VS=$(median "${VEC_SEARCH[@]}")
MED_RK=$(median "${RERANK[@]}")

{
echo "============================================================"
echo "  bench_phase1.sh  —  Median over ${N} runs"
echo "============================================================"
echo ""
printf "  %-22s  %s\n" "Component" "Median (s)"
printf "  %-22s  %s\n" "---------" "----------"
printf "  %-22s  %s\n" "db_deserialize"  "${MED_DB}s"
printf "  %-22s  %s\n" "query_embedding" "${MED_QE}s"
printf "  %-22s  %s\n" "vector_search"   "${MED_VS}s"
printf "  %-22s  %s\n" "reranking"       "${MED_RK}s"
echo ""
echo "  Raw values per run:"
printf "  %-22s  %s\n" "db_deserialize"  "${DB_DESER[*]}"
printf "  %-22s  %s\n" "query_embedding" "${QUERY_EMB[*]}"
printf "  %-22s  %s\n" "vector_search"   "${VEC_SEARCH[*]}"
printf "  %-22s  %s\n" "reranking"       "${RERANK[*]}"
echo ""
echo "  Per-run logs: bench_phase1_run_1.log ... bench_phase1_run_${N}.log"
echo "  Warmup log : bench_phase1_warmup.log"
echo "============================================================"
} | tee "$SUMMARY"

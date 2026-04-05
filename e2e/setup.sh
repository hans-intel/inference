#!/bin/bash

pip install -r requirements.txt
apt-get update
apt-get install -y --no-install-recommends \
    wkhtmltopdf
export ZE_AFFINITY_MASK=0
# DP=1 CORES_PER_INST=42  ./run_e2e_with_llm_cpu.sh --eval &> cpu_intel_em_128_rerank_256.log 
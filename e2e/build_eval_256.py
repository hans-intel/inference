#!/usr/bin/env python3
"""
Build a self-contained 256-query benchmark subset.

Selects 256 queries whose required Wikipedia documents are ALL present
in the full passages corpus, then:
  1. Writes data/frames_dataset_256.tsv  — the filtered dataset
  2. Writes passages_256/doc_html_len256.json — passages for those docs only
  3. Reports coverage statistics

Usage (from /workspace or /data2/.../e2e):
    python3 build_eval_256.py
"""

import json
import os
import sys
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATASET_IN  = SCRIPT_DIR / "data" / "frames_dataset.tsv"
PASSAGES_IN = SCRIPT_DIR / "passages" / "doc_html_len256.json"
DATASET_OUT = SCRIPT_DIR / "data" / "frames_dataset_256.tsv"
PASSAGES_DIR = SCRIPT_DIR / "passages_256"
PASSAGES_OUT = PASSAGES_DIR / "doc_html_len256.json"
N_QUERIES = 256

def main():
    # ── Load full passages ────────────────────────────────────────────────
    print(f"Loading passages from {PASSAGES_IN} ...")
    with open(PASSAGES_IN) as f:
        full_data = json.load(f)

    passages = full_data["passages"]
    base_dir = str(PASSAGES_DIR)

    # Map original_url → list of passage dicts (multiple passages per doc)
    url_to_passages: dict[str, list] = {}
    for p in passages:
        url = p.get("original_url", "")
        if url:
            url_to_passages.setdefault(url, []).append(p)

    available_urls = set(url_to_passages.keys())
    print(f"  Full corpus: {len(passages):,} passages across {len(available_urls):,} documents")

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"\nLoading dataset from {DATASET_IN} ...")
    df = pd.read_csv(DATASET_IN, sep="\t")
    print(f"  Total queries: {len(df)}")

    link_cols = [c for c in df.columns
                 if c.startswith("wikipedia_link_") and c != "wikipedia_link_11+"]

    df["_required_links"] = df[link_cols].apply(
        lambda row: [v for v in row if pd.notna(v) and str(v).startswith("http")],
        axis=1
    )
    df["_n_required"] = df["_required_links"].apply(len)
    df["_n_found"] = df["_required_links"].apply(
        lambda links: sum(1 for l in links if l in available_urls)
    )
    df["_coverage"] = df.apply(
        lambda r: r["_n_found"] / r["_n_required"] if r["_n_required"] > 0 else 0.0,
        axis=1
    )

    # Select N_QUERIES with best coverage (all 100% first, then by n_required)
    eligible = df[df["_n_required"] > 0].copy()
    selected = (eligible
                .sort_values(["_coverage", "_n_found"], ascending=False)
                .head(N_QUERIES))

    print(f"\nSelected {len(selected)} queries:")
    print(f"  Full coverage (100%) : {(selected['_coverage'] == 1.0).sum()}")
    print(f"  Avg coverage         : {selected['_coverage'].mean():.2%}")
    print(f"  Avg required links   : {selected['_n_required'].mean():.1f}")

    # ── Save filtered dataset ─────────────────────────────────────────────
    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    clean = selected.drop(columns=["_required_links", "_n_required", "_n_found", "_coverage"])
    clean.to_csv(DATASET_OUT, sep="\t", index=False)
    print(f"\nDataset written → {DATASET_OUT}")

    # ── Build passages subset ─────────────────────────────────────────────
    all_required_urls: set[str] = set()
    for links in selected["_required_links"]:
        all_required_urls.update(links)

    # Keep all passages for the required documents
    subset_passages = []
    for url in sorted(all_required_urls):
        for p in url_to_passages.get(url, []):
            subset_passages.append(p)

    # Reassign sequential indices
    for i, p in enumerate(subset_passages):
        p = dict(p)
        p["index"] = i
        subset_passages[i] = p

    PASSAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_data = {"base_dir": str(PASSAGES_DIR), "passages": subset_passages}
    with open(PASSAGES_OUT, "w") as f:
        json.dump(out_data, f, indent=2)

    unique_docs = len({p["original_url"] for p in subset_passages})
    print(f"Passages written → {PASSAGES_OUT}")
    print(f"  {len(subset_passages):,} passages across {unique_docs} documents")
    print(f"  (was {len(passages):,} passages / {len(available_urls):,} docs in full corpus)")

    print("\n========================================================")
    print(" BUILD COMPLETE — Use these paths in your run scripts:")
    print(f"  --dataset  {DATASET_OUT}")
    print(f"  --ingest   {PASSAGES_OUT}")
    print("========================================================")


if __name__ == "__main__":
    main()

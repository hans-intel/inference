#!/usr/bin/env python3
"""
Debug Vector DB Retrieval - Distance Analysis and Visualization

This script measures and plots the distance distribution of retrieved neighbors
from the vector database to help debug retrieval quality.

Usage:
  python debug_vector_distances.py --database DB/vector_hnsw_len2048_ov32_word.db \
                                   --dataset data/frames_dataset.tsv \
                                   --num_queries 10 \
                                   --top_k 100
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from retrieve import VectorDB
from utils import set_deterministic_seeds


def plot_distance_distribution(all_distances, all_queries, output_path="distance_distribution.png"):
    """
    Plot distance distributions for all queries.
    
    Args:
        all_distances: List of (query_idx, distances) tuples
        all_queries: List of query strings
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Individual query distance curves
    ax1 = axes[0, 0]
    for query_idx, distances in all_distances:
        ranks = np.arange(1, len(distances) + 1)
        ax1.plot(ranks, distances, alpha=0.6, linewidth=1, label=f"Q{query_idx}")
    
    ax1.set_xlabel("Rank (k)")
    ax1.set_ylabel("L2 Distance")
    ax1.set_title("Distance vs Rank for Each Query")
    ax1.grid(True, alpha=0.3)
    if len(all_distances) <= 20:
        ax1.legend(fontsize=8, ncol=2)
    
    # Plot 2: Aggregated statistics (mean, median, percentiles)
    ax2 = axes[0, 1]
    
    # Collect all distances by rank
    max_k = max(len(distances) for _, distances in all_distances)
    distances_by_rank = [[] for _ in range(max_k)]
    
    for _, distances in all_distances:
        for rank, dist in enumerate(distances):
            distances_by_rank[rank].append(dist)
    
    ranks = np.arange(1, max_k + 1)
    means = [np.mean(dists) for dists in distances_by_rank]
    medians = [np.median(dists) for dists in distances_by_rank]
    p25 = [np.percentile(dists, 25) for dists in distances_by_rank]
    p75 = [np.percentile(dists, 75) for dists in distances_by_rank]
    
    ax2.plot(ranks, means, label='Mean', linewidth=2, color='blue')
    ax2.plot(ranks, medians, label='Median', linewidth=2, color='green')
    ax2.fill_between(ranks, p25, p75, alpha=0.3, color='blue', label='25-75 percentile')
    
    ax2.set_xlabel("Rank (k)")
    ax2.set_ylabel("L2 Distance")
    ax2.set_title("Aggregated Distance Statistics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distance distribution histogram (all queries combined)
    ax3 = axes[1, 0]
    
    all_dists_flat = []
    for _, distances in all_distances:
        all_dists_flat.extend(distances)
    
    ax3.hist(all_dists_flat, bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(all_dists_flat), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(all_dists_flat):.2f}')
    ax3.axvline(np.median(all_dists_flat), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(all_dists_flat):.2f}')
    
    ax3.set_xlabel("L2 Distance")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Overall Distance Distribution (All Queries)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distance gap analysis (distance difference between consecutive ranks)
    ax4 = axes[1, 1]
    
    gaps_by_rank = [[] for _ in range(max_k - 1)]
    for _, distances in all_distances:
        for rank in range(len(distances) - 1):
            gap = distances[rank + 1] - distances[rank]
            gaps_by_rank[rank].append(gap)
    
    gap_ranks = np.arange(1, max_k)
    gap_means = [np.mean(gaps) for gaps in gaps_by_rank]
    gap_medians = [np.median(gaps) for gaps in gaps_by_rank]
    
    ax4.plot(gap_ranks, gap_means, label='Mean Gap', linewidth=2, color='blue')
    ax4.plot(gap_ranks, gap_medians, label='Median Gap', linewidth=2, color='green')
    
    ax4.set_xlabel("Rank (k)")
    ax4.set_ylabel("Distance Gap (Δ)")
    ax4.set_title("Distance Gap Between Consecutive Ranks")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    plt.close()


def analyze_distances(db, db_file_path, dataset_path, num_queries=10, top_k=100, output_dir="debug", seed=42):
    """
    Analyze distance distributions for queries from the dataset.
    
    Args:
        db: VectorDB instance
        db_file_path: Path to database file (for display)
        dataset_path: Path to dataset TSV file
        num_queries: Number of queries to analyze
        top_k: Number of neighbors to retrieve
        output_dir: Directory to save outputs
        seed: Random seed for reproducibility
    """
    set_deterministic_seeds(seed)
    
    # Load dataset
    df = pd.read_csv(dataset_path, sep='\t')
    df = df.head(num_queries)
    
    print(f"\n{'='*80}")
    print(f"VECTOR DB DISTANCE ANALYSIS")
    print(f"{'='*80}")
    print(f"Database file: {db_file_path}")
    print(f"Queries: {len(df)}")
    print(f"Top-k: {top_k}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_distances = []
    all_queries = []
    stats_records = []
    
    for idx, row in df.iterrows():
        query = row['Prompt']
        all_queries.append(query)
        
        print(f"\n[Query {idx}] {query[:80]}{'...' if len(query) > 80 else ''}")
        
        # Retrieve with scores (distances)
        results_with_scores = db.lookup_with_scores(query, k=top_k)
        
        # Extract distances (convert back from similarity scores)
        # lookup_with_scores returns (doc, -distance), so we negate back to get L2 distances
        distances = [-score for doc, score in results_with_scores]
        
        # Statistics
        min_dist = min(distances)
        max_dist = max(distances)
        mean_dist = np.mean(distances)
        median_dist = np.median(distances)
        std_dist = np.std(distances)
        
        print(f"  Distance stats:")
        print(f"    Min:    {min_dist:.4f}")
        print(f"    Max:    {max_dist:.4f}")
        print(f"    Mean:   {mean_dist:.4f}")
        print(f"    Median: {median_dist:.4f}")
        print(f"    Std:    {std_dist:.4f}")
        print(f"    Range:  {max_dist - min_dist:.4f}")
        
        # Store for plotting
        all_distances.append((idx, distances))
        
        # Store for CSV
        stats_records.append({
            'query_idx': idx,
            'query': query[:100],
            'min_distance': min_dist,
            'max_distance': max_dist,
            'mean_distance': mean_dist,
            'median_distance': median_dist,
            'std_distance': std_dist,
            'distance_range': max_dist - min_dist,
            'num_retrieved': len(distances)
        })
        
        # Print top-5 and bottom-5 distances
        print(f"  Top-5 closest (rank 1-5):")
        for rank in range(min(5, len(distances))):
            doc, _ = results_with_scores[rank]
            url = doc.metadata.get('original_url', 'Unknown')
            print(f"    [{rank+1}] dist={distances[rank]:.4f} | {url}")
        
        if len(distances) > 5:
            print(f"  Bottom-5 farthest (rank {len(distances)-4}-{len(distances)}):")
            for rank in range(max(5, len(distances) - 5), len(distances)):
                doc, _ = results_with_scores[rank]
                url = doc.metadata.get('original_url', 'Unknown')
                print(f"    [{rank+1}] dist={distances[rank]:.4f} | {url}")
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_records)
    stats_csv_path = output_path / "distance_statistics.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"\n✓ Statistics saved to: {stats_csv_path}")
    
    # Plot distance distributions
    plot_path = output_path / "distance_distribution.png"
    plot_distance_distribution(all_distances, all_queries, str(plot_path))
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS (across {len(all_distances)} queries)")
    print(f"{'='*80}")
    
    all_dists = [dist for _, dists in all_distances for dist in dists]
    print(f"Overall distance statistics:")
    print(f"  Min:    {np.min(all_dists):.4f}")
    print(f"  Max:    {np.max(all_dists):.4f}")
    print(f"  Mean:   {np.mean(all_dists):.4f}")
    print(f"  Median: {np.median(all_dists):.4f}")
    print(f"  Std:    {np.std(all_dists):.4f}")
    print(f"  P25:    {np.percentile(all_dists, 25):.4f}")
    print(f"  P75:    {np.percentile(all_dists, 75):.4f}")
    print(f"  P95:    {np.percentile(all_dists, 95):.4f}")
    print(f"  P99:    {np.percentile(all_dists, 99):.4f}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Debug vector DB retrieval by analyzing distance distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--database', required=True,
                       help='Path to vector database file (e.g., DB/vector_hnsw_len2048_ov32_word.db)')
    parser.add_argument('--dataset', default='data/frames_dataset.tsv',
                       help='Path to dataset TSV file')
    parser.add_argument('--num_queries', type=int, default=10,
                       help='Number of queries to analyze')
    parser.add_argument('--top_k', type=int, default=100,
                       help='Number of neighbors to retrieve per query')
    parser.add_argument('--output_dir', default='debug',
                       help='Directory to save output files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', default='cpu',
                       help='Device for embeddings (cpu/cuda)')
    parser.add_argument('--retriever_model', default='intfloat/e5-base-v2',
                       help='Embedding model for vector retrieval')
    
    args = parser.parse_args()
    
    # Normalize database path
    db_file_path = args.database if args.database.endswith('.db') else f"{args.database}.db"
    db_base_name = args.database.replace('.db', '') if args.database.endswith('.db') else args.database
    
    # Load vector database
    print(f"Loading vector database from {db_file_path}...")
    db = VectorDB(retriever_model=args.retriever_model, device=args.device, database=db_base_name)
    db.from_serialized(db_file_path)
    print(f"✓ Loaded {len(db._doc_list)} documents")
    
    # Run analysis
    analyze_distances(
        db=db,
        db_file_path=db_file_path,
        dataset_path=args.dataset,
        num_queries=args.num_queries,
        top_k=args.top_k,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

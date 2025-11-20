"""
Retrieval Evaluation Metrics Module

This module provides comprehensive retrieval evaluation metrics including:
- Precision@k, Recall@k, F1@k
- Mean Average Precision (MAP)
- Comprehensive retrieval metrics
- Detailed dataset analysis by reasoning type and answer link count

Designed for reuse across different retrieval systems including multi-hop QA.
"""

import concurrent.futures
import contextlib
import io
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

import pandas as pd

from utils import filter_dataset_by_difficulty, print_evaluation_summary


def calculate_retrieval_metrics(expected_urls: List[str], retrieved_urls: List[str], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """
    Calculate comprehensive retrieval metrics.
    
    Args:
        expected_urls: List of expected/ground truth URLs
        retrieved_urls: List of retrieved URLs in ranking order
        k_values: List of k values for Precision@k, Recall@k, F1@k
        
    Returns:
        Dictionary containing all calculated metrics
    """
    expected_set = set(url for url in expected_urls if url and url.strip())
    
    # Handle edge cases
    if not expected_set:
        return {f'precision@{k}': 1.0 if len(retrieved_urls) == 0 else 0.0 for k in k_values} | \
               {f'recall@{k}': 1.0 for k in k_values} | \
               {f'f1@{k}': 1.0 if len(retrieved_urls) == 0 else 0.0 for k in k_values} | \
               {'average_precision': 1.0 if len(retrieved_urls) == 0 else 0.0}
    
    metrics = {}
    
    # Calculate metrics for different k values, including @N (actual retrieved count)
    num_retrieved = len(retrieved_urls)
    num_expected = len(expected_set)
    k_values_with_n = k_values + [num_retrieved]  # Add N to k_values
    
    for k in k_values_with_n:
        # Determine the label (use 'N' for the actual retrieved count)
        k_label = 'N' if k == num_retrieved else str(k)
        
        # Get top k documents
        top_k = retrieved_urls[:k]
        top_k_set = set(top_k)
        relevant_retrieved = len(expected_set.intersection(top_k_set))
        
        # Precision@k: fraction of retrieved documents that are relevant
        precision_k = relevant_retrieved / k if k > 0 else 0.0
        metrics[f'precision@{k_label}'] = precision_k
        
        # Recall@k: fraction of relevant documents that are retrieved
        recall_k = relevant_retrieved / num_expected if num_expected > 0 else 0.0
        metrics[f'recall@{k_label}'] = recall_k
        
        # F1@k: harmonic mean of precision and recall
        if precision_k + recall_k > 0:
            f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
        else:
            f1_k = 0.0
        metrics[f'f1@{k_label}'] = f1_k
    
    # Mean Average Precision (MAP) - considers ranking order
    ap_sum = 0.0
    relevant_found = 0
    
    for i, url in enumerate(retrieved_urls):
        if url in expected_set:
            relevant_found += 1
            precision_at_i = relevant_found / (i + 1)
            ap_sum += precision_at_i
    
    average_precision = ap_sum / len(expected_set) if len(expected_set) > 0 else 0.0
    metrics['average_precision'] = average_precision
    
    return metrics


def evaluate_retrieval_query(rag_db, query: str, expected_urls: List[str], 
                         top_k_retriever: int = 50, top_k_reranking: int = 10,
                         verbose: bool = True, no_rerank: bool = False,
                         retrieval_strategy: str = "fixed_k", print_results: bool = False,
                         return_results: bool = False, **strategy_params) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[Any]]]:
    """
    Evaluate a single retrieval query and return comprehensive retrieval metrics.
    
    Args:
        rag_db: RAG database instance
        query: Query string
        expected_urls: List of expected URLs
        top_k_retriever: Number of documents to retrieve initially
        top_k_reranking: Number of documents after reranking
        verbose: Whether to print detailed results
        no_rerank: Skip reranking step for fair comparison between retrieval methods
        retrieval_strategy: Strategy for retrieval ("fixed_k", "top_p", "relative")
        **strategy_params: Parameters for adaptive retrieval strategies
        
    Returns:
        Dictionary containing all metrics. When return_results=True, returns a tuple of (metrics_dict, retrieved_results).
    """
    import time
    
    # Step 1: Time the initial retrieval
    retrieval_start = time.perf_counter()
    if retrieval_strategy == "fixed_k":
        results = rag_db.lookup(query, k=top_k_retriever)
    else:
        from retrieve.filter import filter
        max_results = strategy_params.pop("max_results", 20)
        results = filter(rag_db, query, method=retrieval_strategy, 
                       max_results=max_results, **strategy_params)
    retrieval_time = time.perf_counter() - retrieval_start
    
    # Step 2: Apply reranking if enabled and reranker is available  
    reranking_time = 0.0
    if not no_rerank and hasattr(rag_db, '_reranker_model') and rag_db._reranker_model is not None:
        # Safety check: If no results retrieved, skip reranking
        if not results:
            if verbose:
                print(f"Warning: No documents retrieved for query: {query[:50]}")
        else:
            reranking_start = time.perf_counter()
            # Extract text content for reranking (rerank expects strings)
            passages = [result.page_content for result in results]
            scored_passages = rag_db.rerank(query, passages)
            
            # Reconstruct document objects with reranked order
            # scored_passages is [(text, score), ...] ordered by score
            reranked_results = []
            for text, score in scored_passages:
                # Find the original document object for this text
                for doc in results:
                    if doc.page_content == text:
                        reranked_results.append(doc)
                        break
            
            # Apply top_k_reranking limit AFTER reranking
            # For adaptive strategies (top_p, relative, etc.), respect the number of documents
            # selected by the strategy, only limit for fixed_k
            if retrieval_strategy == "fixed_k":
                results = reranked_results[:top_k_reranking]
            else:
                # For adaptive strategies, keep all documents selected by the strategy
                results = reranked_results
            reranking_time = time.perf_counter() - reranking_start
    
    # Extract URLs from results in order (maintaining ranking)
    retrieved_urls = []
    for result in results:
        if 'original_url' in result.metadata and result.metadata['original_url']:
            retrieved_urls.append(result.metadata['original_url'])

    # Deduplicate URLs preserving first appearance order (for accurate MAP calculation)
    deduplicated_urls = list(dict.fromkeys(retrieved_urls))  # Preserves order, removes duplicates
    
    # Calculate comprehensive metrics using deduplicated URLs (accurate MAP)
    expected_set = set(url for url in expected_urls if url and url.strip())
    metrics = calculate_retrieval_metrics(list(expected_set), deduplicated_urls)
    
    # Track both passages and unique documents
    num_passages = len(results)
    num_unique_docs = len(deduplicated_urls)
    
    if verbose:
        print(f"Query: {query:50}")
        matches = len(expected_set.intersection(set(deduplicated_urls)))
        print(f"Expected ({len(expected_set)}): {sorted(list(expected_set)[:3])}{'...' if len(expected_set) > 3 else ''}")
        print(f"Retrieved ({num_passages} passages, {num_unique_docs} unique docs): {deduplicated_urls[:3]}{'...' if num_unique_docs > 3 else ''}")
        print(f"Matches: {matches}")

        metric_categories = [
            ("P", "precision"),
            ("R", "recall"),
            ("F1", "f1")
        ]

        for label, metric_prefix in metric_categories:
            parts = [f"{label}@N: {metrics[f'{metric_prefix}@N']:.3f}"]
            for k in [3, 5, 10]:
                key = f"{metric_prefix}@{k}"
                if key in metrics:
                    parts.append(f"{label}@{k}: {metrics[key]:.3f}")
            print(", ".join(parts))

        print(f"MAP: {metrics['average_precision']:.3f}")
        print("-" * 80)
    
    # Print detailed results for single query mode
    if print_results:
        print(f"\n{retrieval_strategy.upper()} lookup took time. {len(results)} results found:")
        
        # Display which PDFs the passages are from
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.metadata}")
            print("-" * 50)

        # Show reranked results if reranker is available and reranking was used
        if not no_rerank and rag_db._reranker_model is not None:
            print(f"\nReranking to top-{top_k_reranking}")
            print(f"Reranking results:")

            for i, result in enumerate(results, 1):
                print(f"{i}. {result.metadata}")
                print(f"   {result.page_content[:200]}...")
                print("-" * 50)
        else:
            if no_rerank:
                print("No reranker used (--no-rerank specified)")
            else:
                print("No reranker available - showing retrieval results only")
    
    # Calculate retrieval performance metrics
    total_time = retrieval_time + reranking_time
    docs_per_second = len(results) / total_time if total_time > 0 else 0
    
    # Add retrieval performance to metrics
    retrieval_metrics = {
        'retrieval_time': retrieval_time,
        'reranking_time': reranking_time, 
        'total_retrieval_time': total_time,
        'retrieved_passages_count': len(results),
        'retrieved_docs_count': num_unique_docs,
        'docs_per_second': docs_per_second
    }
    
    # Print retrieval performance if in benchmark mode and single query mode (not evaluation)
    if hasattr(rag_db, '_benchmark') and rag_db._benchmark and print_results:
        print(f"\n🔍 RETRIEVAL PERFORMANCE METRICS")
        print("=" * 50)
        print(f"📊 Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        print(f"⏱️  Retrieval time: {retrieval_time*1000:.2f}ms")
        if reranking_time > 0:
            print(f"🔄 Reranking time: {reranking_time*1000:.2f}ms")
        print(f"🕐 Total time: {total_time*1000:.2f}ms")
        print(f"📦 Documents retrieved: {len(results)}")
        print(f"🚀 Retrieval speed: {docs_per_second:.1f} docs/sec")
        print(f"💾 Time per query: {total_time:.4f}s")
        print()
    
    # Return metrics dict with retrieval performance
    merged_metrics = {**metrics, **retrieval_metrics}
    if return_results:
        return merged_metrics, results
    return merged_metrics


def run_evaluation(rag_db, dataset_path: str,
                   top_k_retriever: int = 50, top_k_reranking: int = 10,
                   max_queries: Optional[int] = None, no_rerank: bool = False,
                   retrieval_strategy: str = "fixed_k", detailed_analysis: bool = False,
                   difficulty: int = 0, collect_results: bool = False,
                   result_handler: Optional[Callable[[str, List[Any], Dict[str, Any]], Optional[Any]]] = None,
                   llm_batch_size: int = 64,
                   **strategy_params) -> Union[Dict[str, float], Tuple[Dict[str, float], List[Dict[str, Any]]]]:
    """
    Run comprehensive evaluation on a dataset with detailed metrics reporting.

    Args:
        rag_db: RAG database instance
        dataset_path: Path to the dataset TSV file
        top_k_retriever: Number of documents to retrieve initially
        top_k_reranking: Number of documents after reranking
        max_queries: Maximum number of queries to evaluate (None = all)
        no_rerank: Skip reranking step for fair comparison between retrieval methods
        retrieval_strategy: Strategy for retrieval ("fixed_k", "top_p", "relative")
        detailed_analysis: If True, print detailed breakdown by reasoning types and link counts
        difficulty: Minimum number of answer links required (0 = no filtering)
        collect_results: If True, also collect retrieval outputs for each query
        result_handler: Optional callback invoked per query with (prompt, retrieved_docs, metrics)
        llm_batch_size: Maximum number of questions to process in parallel
        **strategy_params: Parameters for adaptive retrieval strategies

    Returns:
        Dictionary of averaged metrics across all queries. When collect_results=True, returns a tuple of (metrics_dict, collected_results).
    """
    df = pd.read_csv(dataset_path, sep='\t')
    df = filter_dataset_by_difficulty(df, difficulty)

    if isinstance(max_queries, int) and max_queries > 0:
        df = df.head(max_queries).reset_index(drop=True)
    else:
        max_queries = len(df)
        df = df.reset_index(drop=True)

    print(f"\nRunning evaluation on {max_queries} queries from dataset")

    records = []
    for idx, row in df.iterrows():
        expected_urls = []
        for col in df.columns:
            if col.startswith('wikipedia_link_') and pd.notna(row[col]):
                expected_urls.append(row[col].strip())
        if expected_urls:
            records.append({
                "prompt": row['Prompt'],
                "expected_urls": expected_urls,
                "row_index": idx,
            })

    total_records = len(records)
    if total_records == 0:
        print("No valid queries found!")
        if collect_results:
            return {}, []
        return {}

    llm_batch_size = max(1, int(llm_batch_size or 1))
    max_workers = min(llm_batch_size, total_records)

    total_metrics: Dict[str, float] = {}
    retrieval_times: List[float] = []
    reranking_times: List[float] = []
    total_times: List[float] = []
    docs_per_sec_list: List[float] = []
    collected_queries: Optional[List[Dict[str, Any]]] = [] if collect_results else None
    analysis_entries: List[Dict[str, Any]] = []
    valid_queries = 0

    need_results = collect_results or (result_handler is not None)

    def _summarize_docs_for_print(retrieved_docs: List[Any]) -> Dict[str, Any]:
        doc_entries = []
        seen_urls = set()
        for doc in retrieved_docs:
            url = None
            content = ""
            if hasattr(doc, 'metadata'):
                url = doc.metadata.get('original_url') or doc.metadata.get('source')
                content = doc.page_content
            elif isinstance(doc, dict):
                url = doc.get('url')
                content = doc.get('content', "")
            if url and url in seen_urls:
                continue
            doc_entries.append({
                "url": url,
                "content": (content or "")[:2000]
            })
            if url:
                seen_urls.add(url)
        return {"docs": doc_entries}

    def _process_record(record_index: int, record: Dict[str, Any]) -> Dict[str, Any]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            metrics_output = evaluate_retrieval_query(
                rag_db,
                record['prompt'],
                record['expected_urls'],
                top_k_retriever,
                top_k_reranking,
                verbose=True,
                no_rerank=no_rerank,
                retrieval_strategy=retrieval_strategy,
                return_results=need_results,
                **strategy_params,
            )
        if need_results:
            metrics, retrieved_docs = metrics_output
        else:
            metrics = metrics_output
            retrieved_docs = []

        collected_entry = None
        if collect_results and retrieved_docs:
            summary = _summarize_docs_for_print(retrieved_docs)
            summary["prompt"] = record['prompt']
            collected_entry = summary

        handler_output = None
        if result_handler:
            handler_output = result_handler(record['prompt'], retrieved_docs, metrics)

        return {
            "metrics": metrics,
            "log": buffer.getvalue(),
            "collected": collected_entry,
            "handler_output": handler_output,
        }

    results: List[Optional[Dict[str, Any]]] = [None] * total_records
    next_to_emit = 0

    def _flush_ready_outputs():
        nonlocal next_to_emit, valid_queries
        while next_to_emit < total_records and results[next_to_emit] is not None:
            record = records[next_to_emit]
            result = results[next_to_emit]
            question_label = f"[Q{next_to_emit + 1}/{total_records}]"
            prompt_preview = record['prompt'].replace('\n', ' ')
            if len(prompt_preview) > 120:
                prompt_preview = prompt_preview[:117] + '...'
            print(f"\n{question_label} {prompt_preview}")

            log_text = result['log'] or ""
            if log_text:
                print(log_text, end="" if log_text.endswith("\n") else "\n")

            handler_text = result.get('handler_output')
            if handler_text:
                print(handler_text, end="" if handler_text.endswith("\n") else "\n")

            metrics = result['metrics']
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    total_metrics[metric_name] = total_metrics.get(metric_name, 0.0) + float(value)

            if 'retrieval_time' in metrics:
                retrieval_times.append(metrics['retrieval_time'])
                reranking_times.append(metrics.get('reranking_time', 0.0))
                total_times.append(metrics.get('total_retrieval_time', 0.0))
                docs_per_sec_list.append(metrics.get('docs_per_second', 0.0))

            if collect_results and result['collected'] is not None:
                collected_queries.append(result['collected'])

            if detailed_analysis:
                analysis_entries.append({
                    "metrics": metrics,
                    "row_index": record['row_index'],
                })

            valid_queries += 1
            next_to_emit += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_process_record, idx, record): idx for idx, record in enumerate(records)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()
            _flush_ready_outputs()

    if valid_queries > 0:
        avg_metrics = {name: total / valid_queries for name, total in total_metrics.items()}
        results_title = "OVERALL EVALUATION RESULTS" if detailed_analysis else "EVALUATION RESULTS"

        print_evaluation_summary(
            avg_metrics,
            valid_queries,
            title=results_title,
        )

        if retrieval_times and hasattr(rag_db, '_benchmark') and rag_db._benchmark:
            import numpy as np

            print("")
            print(f"🔍 RETRIEVAL PERFORMANCE STATISTICS:")
            print(f"  Retrieval Time (ms):")
            print(f"    Average:                  {np.mean(retrieval_times) * 1000:.2f}ms")
            print(f"    P50 (Median):             {np.percentile(retrieval_times, 50) * 1000:.2f}ms")
            print(f"    P99:                      {np.percentile(retrieval_times, 99) * 1000:.2f}ms")

            if any(t > 0 for t in reranking_times):
                print(f"  Reranking Time (ms):")
                print(f"    Average:                  {np.mean(reranking_times) * 1000:.2f}ms")
                print(f"    P50 (Median):             {np.percentile(reranking_times, 50) * 1000:.2f}ms")
                print(f"    P99:                      {np.percentile(reranking_times, 99) * 1000:.2f}ms")

            print(f"  Total Query Time (ms):")
            print(f"    Average:                  {np.mean(total_times) * 1000:.2f}ms")
            print(f"    P50 (Median):             {np.percentile(total_times, 50) * 1000:.2f}ms")
            print(f"    P99:                      {np.percentile(total_times, 99) * 1000:.2f}ms")

            print(f"  Retrieval Throughput (docs/sec):")
            print(f"    Average:                  {np.mean(docs_per_sec_list):.1f} docs/sec")
            print(f"    P50 (Median):             {np.percentile(docs_per_sec_list, 50):.1f} docs/sec")
            print(f"    P99:                      {np.percentile(docs_per_sec_list, 99):.1f} docs/sec")

        if detailed_analysis:
            _print_detailed_analysis(df, analysis_entries, valid_queries)

        if collect_results:
            return avg_metrics, collected_queries or []
        return avg_metrics

    print("No valid queries found!")
    if collect_results:
        return {}, []
    return {}


def _print_detailed_analysis(df: pd.DataFrame, analysis_entries: List[Dict[str, Any]],
                            valid_queries: int) -> None:
    """
    Print detailed dataset analysis broken down by reasoning types and answer link counts.
    (Internal helper function for run_evaluation)
    
    Args:
        df: DataFrame with dataset (must have 'reasoning_types' column)
        all_query_metrics: List of metrics dictionaries for each query
        valid_queries: Number of valid queries processed
    """
    if valid_queries == 0:
        return
    
    print("\n" + "="*80)
    print("DETAILED DATASET ANALYSIS")
    print("="*80)
    
    # Prepare data - match metrics with reasoning types and link counts
    analysis_data = []
    for entry in analysis_entries:
        row_idx = entry.get('row_index')
        if row_idx is None or row_idx >= len(df):
            continue
        row = df.iloc[row_idx]
        metrics = entry.get('metrics', {})
        reasoning_types = row.get('reasoning_types', 'Unknown')

        num_links = sum(
            1 for col in df.columns if col.startswith('wikipedia_link_') and pd.notna(row[col])
        )

        analysis_data.append({
            'reasoning_types': reasoning_types,
            'num_links': num_links,
            'metrics': metrics
        })
    
    # === ANALYSIS 1: By Reasoning Classification ===
    print("\n" + "-"*80)
    print("ANALYSIS BY REASONING CLASSIFICATION")
    print("-"*80)
    
    # Group by reasoning types
    reasoning_groups = defaultdict(list)
    for data in analysis_data:
        reasoning_groups[data['reasoning_types']].append(data['metrics'])
    
    # Calculate averages for each reasoning type
    reasoning_results = []
    for reasoning_type, metrics_list in reasoning_groups.items():
        if not metrics_list:
            continue
        
        avg_metrics = {}
        for key in ['precision@N', 'recall@N', 'f1@N', 'average_precision']:
            values = [m.get(key, 0.0) for m in metrics_list]
            avg_metrics[key] = sum(values) / len(values)
        
        reasoning_results.append({
            'type': reasoning_type,
            'count': len(metrics_list),
            **avg_metrics
        })
    
    # Sort by count (most common first)
    reasoning_results.sort(key=lambda x: x['count'], reverse=True)
    
    # Print top reasoning types
    print(f"\nTop reasoning type combinations:")
    print(f"{'Reasoning Type':<50} {'Count':>6} {'P@N':>6} {'R@N':>6} {'F1@N':>6} {'MAP':>6}")
    print("-"*80)
    
    for i, result in enumerate(reasoning_results):
        rt = result['type'][:48] if len(result['type']) > 48 else result['type']
        print(f"{rt:<50} {result['count']:6d} "
              f"{result['precision@N']:6.3f} {result['recall@N']:6.3f} "
              f"{result['f1@N']:6.3f} {result['average_precision']:6.3f}")
    
    # === ANALYSIS 2: By Individual Reasoning Tags ===
    print(f"\n" + "-"*80)
    print("ANALYSIS BY INDIVIDUAL REASONING TAGS")
    print("-"*80)
    
    # Parse reasoning tags (split by |)
    tag_groups = defaultdict(list)
    for data in analysis_data:
        tags = [tag.strip() for tag in data['reasoning_types'].split('|')]
        for tag in tags:
            tag_groups[tag].append(data['metrics'])
    
    tag_results = []
    for tag, metrics_list in tag_groups.items():
        if not metrics_list:
            continue
        
        avg_metrics = {}
        for key in ['precision@N', 'recall@N', 'f1@N', 'average_precision']:
            values = [m.get(key, 0.0) for m in metrics_list]
            avg_metrics[key] = sum(values) / len(values)
        
        tag_results.append({
            'tag': tag,
            'count': len(metrics_list),
            'percentage': len(metrics_list) / valid_queries * 100,
            **avg_metrics
        })
    
    # Sort by count
    tag_results.sort(key=lambda x: x['count'], reverse=True)
    
    print(f"\nPerformance by reasoning tag:")
    print(f"{'Tag':<30} {'Count':>6} {'%':>6} {'P@N':>6} {'R@N':>6} {'F1@N':>6} {'MAP':>6}")
    print("-"*80)
    
    for result in tag_results:
        tag = result['tag'][:28] if len(result['tag']) > 28 else result['tag']
        print(f"{tag:<30} {result['count']:6d} {result['percentage']:5.1f}% "
              f"{result['precision@N']:6.3f} {result['recall@N']:6.3f} "
              f"{result['f1@N']:6.3f} {result['average_precision']:6.3f}")
    
    # === ANALYSIS 3: By Number of Answer Links ===
    print(f"\n" + "-"*80)
    print("ANALYSIS BY NUMBER OF ANSWER LINKS (Multi-hop Analysis)")
    print("-"*80)
    
    # Group by number of links
    link_groups = defaultdict(list)
    for data in analysis_data:
        link_groups[data['num_links']].append(data['metrics'])
    
    link_results = []
    for num_links, metrics_list in link_groups.items():
        if not metrics_list:
            continue
        
        avg_metrics = {}
        for key in ['precision@N', 'recall@N', 'f1@N', 'average_precision']:
            values = [m.get(key, 0.0) for m in metrics_list]
            avg_metrics[key] = sum(values) / len(values)
        
        # Classify complexity
        if num_links <= 2:
            complexity = "Simple"
        elif num_links <= 4:
            complexity = "Multi-hop"
        else:
            complexity = "Complex"
        
        link_results.append({
            'num_links': num_links,
            'complexity': complexity,
            'count': len(metrics_list),
            'percentage': len(metrics_list) / valid_queries * 100,
            **avg_metrics
        })
    
    # Sort by number of links
    link_results.sort(key=lambda x: x['num_links'])
    
    print(f"\nPerformance by number of Wikipedia links (reasoning hops):")
    print(f"{'Links':>5} {'Complexity':<12} {'Count':>6} {'%':>6} {'P@N':>6} {'R@N':>6} {'F1@N':>6} {'MAP':>6}")
    print("-"*80)
    
    for result in link_results:
        print(f"{result['num_links']:5d} {result['complexity']:<12} "
              f"{result['count']:6d} {result['percentage']:5.1f}% "
              f"{result['precision@N']:6.3f} {result['recall@N']:6.3f} "
              f"{result['f1@N']:6.3f} {result['average_precision']:6.3f}")
    
    # Summary by complexity category
    print(f"\n" + "-"*80)
    print("SUMMARY BY COMPLEXITY LEVEL")
    print("-"*80)
    
    complexity_groups = defaultdict(list)
    for result in link_results:
        for _ in range(result['count']):
            # Get original metrics for this group
            complexity_groups[result['complexity']].append({
                'precision@N': result['precision@N'],
                'recall@N': result['recall@N'],
                'f1@N': result['f1@N'],
                'average_precision': result['average_precision']
            })
    
    # Calculate totals
    complexity_summary = []
    for complexity in ["Simple", "Multi-hop", "Complex"]:
        if complexity not in complexity_groups:
            continue
        
        metrics_list = complexity_groups[complexity]
        count = len(metrics_list)
        
        # Recalculate from link_results
        matching_results = [r for r in link_results if r['complexity'] == complexity]
        total_count = sum(r['count'] for r in matching_results)
        
        # Weighted average
        weighted_metrics = {}
        for key in ['precision@N', 'recall@N', 'f1@N', 'average_precision']:
            weighted_sum = sum(r[key] * r['count'] for r in matching_results)
            weighted_metrics[key] = weighted_sum / total_count if total_count > 0 else 0.0
        
        complexity_summary.append({
            'complexity': complexity,
            'count': total_count,
            'percentage': total_count / valid_queries * 100,
            **weighted_metrics
        })
    
    print(f"\n{'Complexity':<12} {'Count':>6} {'%':>6} {'P@N':>6} {'R@N':>6} {'F1@N':>6} {'MAP':>6}")
    print("-"*80)
    
    for result in complexity_summary:
        print(f"{result['complexity']:<12} {result['count']:6d} {result['percentage']:5.1f}% "
              f"{result['precision@N']:6.3f} {result['recall@N']:6.3f} "
              f"{result['f1@N']:6.3f} {result['average_precision']:6.3f}")
    
    # === ANALYSIS 4: Correlation Between Complexity and Reasoning Types ===
    print(f"\n" + "-"*80)
    print("CORRELATION: COMPLEXITY vs REASONING TYPES")
    print("-"*80)
    
    # Build correlation matrix: complexity level x reasoning tags
    complexity_reasoning_data = defaultdict(lambda: defaultdict(list))
    
    for data in analysis_data:
        # Determine complexity
        num_links = data['num_links']
        if num_links <= 2:
            complexity = "Simple"
        elif num_links <= 4:
            complexity = "Multi-hop"
        else:
            complexity = "Complex"
        
        # Extract individual reasoning tags
        tags = [tag.strip() for tag in data['reasoning_types'].split('|')]
        for tag in tags:
            complexity_reasoning_data[complexity][tag].append(data['metrics'])
    
    # Calculate statistics for each complexity-reasoning combination
    print(f"\n1. REASONING TAG DISTRIBUTION BY COMPLEXITY:")
    print(f"{'Reasoning Tag':<30} {'Simple':>10} {'Multi-hop':>10} {'Complex':>10} {'Total':>10}")
    print("-"*80)
    
    # Get all unique tags
    all_tags = set()
    for complexity_data in complexity_reasoning_data.values():
        all_tags.update(complexity_data.keys())
    
    tag_distribution = {}
    for tag in sorted(all_tags):
        simple_count = len(complexity_reasoning_data.get('Simple', {}).get(tag, []))
        multihop_count = len(complexity_reasoning_data.get('Multi-hop', {}).get(tag, []))
        complex_count = len(complexity_reasoning_data.get('Complex', {}).get(tag, []))
        total = simple_count + multihop_count + complex_count
        
        tag_distribution[tag] = {
            'simple': simple_count,
            'multihop': multihop_count,
            'complex': complex_count,
            'total': total
        }
        
        tag_display = tag[:28] if len(tag) > 28 else tag
        print(f"{tag_display:<30} {simple_count:10d} {multihop_count:10d} {complex_count:10d} {total:10d}")
    
    # Calculate percentage distribution
    print(f"\n2. REASONING TAG PERCENTAGE BY COMPLEXITY:")
    print(f"{'Reasoning Tag':<30} {'Simple %':>10} {'Multi %':>10} {'Complex %':>10}")
    print("-"*80)
    
    for tag in sorted(all_tags):
        dist = tag_distribution[tag]
        total = dist['total']
        if total > 0:
            simple_pct = (dist['simple'] / total) * 100
            multihop_pct = (dist['multihop'] / total) * 100
            complex_pct = (dist['complex'] / total) * 100
            
            tag_display = tag[:28] if len(tag) > 28 else tag
            print(f"{tag_display:<30} {simple_pct:9.1f}% {multihop_pct:9.1f}% {complex_pct:9.1f}%")
    
    # Calculate average number of links per reasoning tag
    print(f"\n3. AVERAGE COMPLEXITY (# LINKS) BY REASONING TAG:")
    print(f"{'Reasoning Tag':<30} {'Avg Links':>10} {'Count':>10}")
    print("-"*80)
    
    tag_link_stats = defaultdict(list)
    for data in analysis_data:
        tags = [tag.strip() for tag in data['reasoning_types'].split('|')]
        for tag in tags:
            tag_link_stats[tag].append(data['num_links'])
    
    tag_avg_links = []
    for tag in sorted(all_tags):
        links = tag_link_stats[tag]
        if links:
            avg_links = sum(links) / len(links)
            tag_avg_links.append((tag, avg_links, len(links)))
    
    # Sort by average links (descending)
    tag_avg_links.sort(key=lambda x: x[1], reverse=True)
    
    for tag, avg_links, count in tag_avg_links:
        tag_display = tag[:28] if len(tag) > 28 else tag
        print(f"{tag_display:<30} {avg_links:10.2f} {count:10d}")
    
    # Performance by complexity x reasoning tag (for top tags only)
    print(f"\n4. PERFORMANCE BY COMPLEXITY x TOP REASONING TAGS:")
    print("-"*80)
    
    # Get top 5 most common tags
    top_tags = sorted(tag_distribution.items(), key=lambda x: x[1]['total'], reverse=True)[:5]
    
    for tag, _ in top_tags:
        print(f"\n{tag}:")
        print(f"{'Complexity':<12} {'Count':>6} {'P@N':>6} {'R@N':>6} {'F1@N':>6} {'MAP':>6}")
        print("-"*70)
        
        for complexity in ["Simple", "Multi-hop", "Complex"]:
            metrics_list = complexity_reasoning_data.get(complexity, {}).get(tag, [])
            if metrics_list:
                count = len(metrics_list)
                avg_p = sum(m.get('precision@N', 0.0) for m in metrics_list) / count
                avg_r = sum(m.get('recall@N', 0.0) for m in metrics_list) / count
                avg_f1 = sum(m.get('f1@N', 0.0) for m in metrics_list) / count
                avg_map = sum(m.get('average_precision', 0.0) for m in metrics_list) / count
                
                print(f"{complexity:<12} {count:6d} {avg_p:6.3f} {avg_r:6.3f} {avg_f1:6.3f} {avg_map:6.3f}")
    
    print("="*80)

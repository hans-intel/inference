"""
Multi-shot Retrieval System

This module implements multi-shot retrieval with query decomposition:
1. Takes a complex query
2. Uses LLM to rewrite/decompose into multiple sub-queries (max k=3)
3. Retrieves documents for each sub-query
4. Optionally reranks combined results
5. Evaluates performance

Architecture:
    Prompt → Query Rewriter (LLM) → k Sub-queries → Retrieval → Reranking → Evaluation
"""

import argparse
import json
import time
import os
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
import pandas as pd

# Set no_proxy to bypass proxy for localhost/127.0.0.1
original_no_proxy = os.environ.get('no_proxy', '')
os.environ['no_proxy'] = '127.0.0.1,localhost,' + original_no_proxy
os.environ['NO_PROXY'] = '127.0.0.1,localhost,' + original_no_proxy

from retrieve import VectorDB, BM25DB
from evaluation import evaluate_retrieval_query, run_evaluation
from utils import (
    set_deterministic_seeds,
    filter_dataset_by_difficulty,
    setup_llm_config,
    get_device_config,
    serialize_cli_args,
    is_token_limit_error,
)
from params import add_all_args
import requests

# Prompts
QUERY_REWRITER_PROMPT = """You are helping answer this multi-hop question by finding the right information step by step.

=== STRATEGY GUIDE ===

When you look at a multi-hop question, think about it like solving a puzzle:

1. IDENTIFY THE BUILDING BLOCKS
   • What are the key entities mentioned? (names, places, dates, positions)
   • What relationships connect them? ("mother of", "birthplace of", "authored by")
   • What's the dependency chain? (find X first, then use X to find Y)

2. CHECK WHAT YOU ALREADY HAVE
   • Read through the documents carefully - the answer might already be there
   • Look for specific names, dates, and facts that match the question
   • If you have enough information to answer, provide the answer now

3. IDENTIFY THE NEXT MISSING PIECE
   • What is the ONE specific fact you need next?
   • Which entity or relationship are you targeting?
   • Can this fact unlock other dependent facts?

4. CRAFT A PRECISE SEARCH
   Strategy selection:
   ✓ For specific people/places/things: Use exact names + the attribute you need
     Example: "James A. Garfield mother maiden name"
   ✓ For positions in lists: Get the full ordered list first
     Example: "list of US presidents chronological order 1-20"
   ✓ For verification: Search the main Wikipedia page title
     Example: "Abraham Lincoln assassination"
   ✓ For connections: Combine both entities with their relationship
     Example: "Charlotte Bronte Jane Eyre publication year"

5. LEARN AND ADAPT
   Study your search history above:
   ✓ If a similar query already failed, try a completely different angle
   ✓ If searches are too broad, add more specific constraints
   ✓ If stuck on one path, pivot to find a different required fact first
   ✓ Use exact Wikipedia page titles when you know the entity name

CRITICAL SUCCESS PATTERNS:
• Use specific entity names, not generic descriptions
• Search for ONE clear fact per query
• When you have facts, combine them to find the answer in documents
• Try different but related search terms if first approach finds nothing new
• Check if current documents already contain the answer before requesting more searches

RESPONSE FORMAT (JSON ONLY):
{{
    "answer": "<final answer if you have enough information, otherwise empty string>",
    "queries": ["<precise query 1>", "<precise query 2>", ...],
    "feedback": "<what specific fact you're targeting and why>",
    "reasoning": "<what you have, what's missing, how your new queries differ from previous attempts>"
}}
Return exactly this structure and nothing else.

QUESTION: {question}

=== SEARCH HISTORY (Chronological) ===
{history_chronological}

=== CURRENT DOCUMENTS ===
{context}

"""


def _uniform_clip_texts(texts: List[str], total_limit: int) -> List[str]:
    if total_limit is None or total_limit <= 0 or not texts:
        return list(texts)
    count = len(texts)
    if count == 0:
        return []
    per_doc, remainder = divmod(total_limit, count)
    if per_doc <= 0 and remainder == 0:
        return [""] * count
    clipped: List[str] = []
    for idx, text in enumerate(texts):
        extra = 1 if idx < remainder else 0
        limit = per_doc + extra
        clipped.append(text[:max(limit, 0)])
    return clipped



def _format_documents_for_prompt(documents: List[Dict[str, Any]], total_char_limit: int) -> str:
    if not documents:
        return "(No new passages retrieved yet)"

    texts = [doc.get("content", "") or "" for doc in documents]
    clipped_texts = _uniform_clip_texts(texts, total_char_limit)
    parts = ["CURRENT PASSAGES:"]

    for idx, (doc, snippet) in enumerate(zip(documents, clipped_texts), 1):
        url = doc.get("url") or doc.get("metadata", {}).get("base_filename", "Unknown source")
        passage_index = doc.get("metadata", {}).get("index", -1)
        header = f"[P{idx}] URL: {url}"
        if passage_index is not None:
            header += f" | Passage #{passage_index}"
        parts.append(header)
        parts.append(snippet.strip())
        parts.append("")

    return "\n".join(parts).strip()


def _format_history_entries(history_entries: List[Dict[str, Any]]) -> str:
    if not history_entries:
        return "(No previous iterations yet)"

    lines: List[str] = []
    for entry in history_entries:
        iteration = entry.get("iteration")
        queries = entry.get("queries", [])
        documents = entry.get("documents", [])
        feedback = entry.get("feedback") or ""

        lines.append(f"Iteration {iteration}:")
        if queries:
            for idx, query in enumerate(queries, 1):
                lines.append(f"  Query {idx}: {query}")
        else:
            lines.append("  Queries: (none)")

        if documents:
            lines.append("  Passages: " + "; ".join(documents))
        else:
            lines.append("  Passages: (none)")

        if feedback:
            lines.append(f"  Feedback: {feedback}")

        lines.append("")

    return "\n".join(lines).strip()


def query_rewriter(question: str,
                   documents: List[Dict[str, Any]],
                   history_entries: Optional[List[Dict[str, Any]]] = None,
                   max_queries: int = 3,
                   reasoning_effort: str = "medium",
                   llm_config: Optional[Dict[str, Any]] = None,
                   context_char_limit: int = 0) -> Dict[str, Any]:
    """
    Evaluates documents AND generates new queries in one LLM call.

    Args:
        question: The user's original question
        new_documents: List of NEW document texts to evaluate
        kept_documents: List of KEPT document texts (already marked relevant)
        max_queries: Maximum number of new queries to generate
        reasoning_effort: LLM reasoning level
    documents: Passages retrieved in the most recent iteration
    history_entries: Chronological list of previous iterations containing queries, docs, and feedback
        context_char_limit: Maximum characters to include per document snippet in the prompt

    Returns:
        Dict with:
        - 'queries': list of new query strings (empty if answer provided)
        - 'feedback': short description of remaining gaps
        - 'answer': final answer if sufficient, otherwise empty string
        - 'reasoning': optional short justification string
    """
    history_entries = history_entries or []
    base_char_limit = context_char_limit if context_char_limit and context_char_limit > 0 else sum(len(doc.get("content", "")) for doc in documents)
    attempt_limit = base_char_limit
    min_limit = max(512, attempt_limit // 4) if attempt_limit else 512
    retry_factor = 0.6
    max_attempts = 4

    if llm_config:
        output_token_limit = llm_config.get("output_token_limit")
        model_name = llm_config["model_name"]
        service_url = llm_config["service_url"]

    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        context = _format_documents_for_prompt(documents, attempt_limit)
        history_text = _format_history_entries(history_entries)

        if documents:
            print(f"    Evaluating {len(documents)} passage(s) from previous iteration")
        else:
            print("    No passages to evaluate yet; generating initial queries")

        prompt = QUERY_REWRITER_PROMPT.format(
            question=question,
            context=context,
            history_chronological=history_text,
        )

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research expert helping solve multi-hop questions. "
                        "Your strengths: (1) finding precise information quickly, "
                        "(2) learning from search results to refine your approach, "
                        "(3) using exact entity names and relationships, "
                        "(4) knowing when you have enough information to answer. "
                        "Focus on what works: specific queries, clear targets, adaptive strategy."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": output_token_limit,
        }

        try:
            response = requests.post(service_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            message = result["choices"][0]["message"]
            llm_output = message.get("content")

            if llm_output is None or not llm_output.strip():
                print("    Warning: LLM returned empty content")
                return {
                    "answer": "",
                    "queries": [],
                    "feedback": "LLM returned empty response",
                    "reasoning": "",
                }

            llm_output = llm_output.strip()

            if llm_output.startswith("```json"):
                llm_output = llm_output.replace("```json", "").replace("```", "").strip()
            elif llm_output.startswith("```"):
                llm_output = llm_output.split("```", 2)[1].strip()
                if llm_output.startswith("json"):
                    llm_output = llm_output[4:].strip()

            result_data = json.loads(llm_output)

            if "answer" not in result_data:
                result_data["answer"] = ""
            if "queries" not in result_data or not isinstance(result_data["queries"], list):
                result_data["queries"] = []
            if "feedback" not in result_data:
                result_data["feedback"] = ""
            if "reasoning" not in result_data:
                result_data["reasoning"] = ""

            return result_data

        except requests.exceptions.HTTPError as http_err:
            last_error = http_err
            if is_token_limit_error(http_err.response, str(http_err)) and attempt < max_attempts - 1:
                new_limit = int(attempt_limit * retry_factor)
                attempt_limit = max(min_limit, new_limit)
                continue
            return {
                "answer": "",
                "queries": [question],
                "feedback": f"API error: {http_err}",
                "reasoning": "",
            }
        except requests.exceptions.RequestException as req_err:
            last_error = req_err
            return {
                "answer": "",
                "queries": [question],
                "feedback": f"API error: {req_err}",
                "reasoning": "",
            }
        except json.JSONDecodeError as json_err:
            last_error = json_err
            print(f"Error parsing LLM output: {json_err}")
            print(f"LLM output: {llm_output[:200] if 'llm_output' in locals() else ''}")
            return {
                "answer": "",
                "queries": [question],
                "feedback": f"JSON parse error: {json_err}",
                "reasoning": "",
            }
        except Exception as exc:
            last_error = exc
            import traceback
            traceback.print_exc()
            return {
                "answer": "",
                "queries": [question],
                "feedback": f"Unexpected error: {exc}",
                "reasoning": "",
            }

    if last_error:
        raise last_error

    return {
        "answer": "",
        "queries": [question],
        "feedback": "Failed to generate response",
        "reasoning": "",
    }


def multi_shot_retrieval(rag_db, original_query: str, expected_urls: List[str],
                         expected_answer: str = "",
                         max_sub_queries: int = 3,
                         top_k_retriever: int = 10,
                         top_k_reranking: int = 10,
                         max_iterations: int = 10,
                         no_rerank: bool = False,
                         retrieval_strategy: str = "fixed_k",
                         verbose: bool = True,
                         reasoning_effort: str = "medium",
                         llm_config: Optional[Dict[str, Any]] = None,
                         context_char_limit: int = 0,
                         **strategy_params) -> Dict[str, Any]:
    """
    Multi-shot retrieval with iterative query refinement and document evaluation.
    
    Algorithm:
    1. Generate initial search queries based on the original question
    2. Retrieve documents for each query
    3. Evaluate documents and check if sufficient to answer
    4. If not sufficient: generate new queries based on what's missing, go to step 2
    5. Repeat until sufficient or max_iterations reached
    
    Args:
        rag_db: RAG database instance
        original_query: Original user question
        expected_urls: Expected ground truth URLs for evaluation
        max_sub_queries: Maximum number of sub-queries per iteration
        top_k_retriever: Number of documents to retrieve per sub-query
        top_k_reranking: Final number of documents to return
        max_iterations: Maximum number of retrieval iterations (default: 10)
        no_rerank: Skip reranking step
        retrieval_strategy: Strategy for retrieval
        verbose: Print detailed information
        reasoning_effort: LLM reasoning level
    context_char_limit: Maximum characters to keep per document snippet used for LLM context
        **strategy_params: Additional parameters for retrieval strategy
        
    Returns:
        Dictionary containing evaluation metrics and iteration statistics
    """
    
    start_time = time.perf_counter()
    
    kept_docs: List[Dict[str, Any]] = []  # Passages already reviewed by the LLM
    pending_docs: List[Dict[str, Any]] = []  # Passages to review in the next iteration
    iteration_history: List[Dict[str, Any]] = []
    iteration_times = []
    total_query_count = 0
    seen_passages: Set[Tuple[str, Any]] = set()
    
    sufficient = False
    iteration = 0
    final_answer = ""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"MULTI-SHOT RETRIEVAL")
        print(f"{'='*80}")
        print(f"Original Query: {original_query}")
        print(f"Max iterations: {max_iterations}")
        print(f"Max sub-queries per iteration: {max_sub_queries}")
        print(f"{'='*80}\n")
    
    while not sufficient and iteration < max_iterations:
        iteration += 1
        iteration_start = time.perf_counter()
        
        if verbose:
            print(f"\n{'─'*80}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print(f"{'─'*80}")
        
        # Step 1: Evaluate most recent passages and decide next searches
        if verbose:
            print(f"\n  Evaluating documents and generating queries...")

        result = query_rewriter(
            original_query,
            documents=pending_docs,
            history_entries=iteration_history,
            max_queries=max_sub_queries,
            reasoning_effort=reasoning_effort,
            llm_config=llm_config,
            context_char_limit=context_char_limit
        )
        
        # Check if we have an answer (sufficient)
        answer_text = result.get("answer", "") or ""
        sufficient = bool(answer_text.strip())
        sub_queries = result.get("queries", [])
        current_feedback = result.get("feedback", "")
        final_answer = answer_text
        reasoning_steps = result.get("reasoning", "")
        total_query_count += len(sub_queries)
        
        if verbose:
            print(f"    Sufficient: {'yes' if sufficient else 'no'}")
            print(f"    Reviewed passages so far: {len(kept_docs)}")
            if pending_docs:
                print(f"    Pending passages: {len(pending_docs)}")
            if reasoning_steps:
                print(f"    Reasoning: {reasoning_steps[:300]}...")
            if not sufficient:
                print(f"    Feedback: {current_feedback}")
                print(f"    Generated {len(sub_queries)} new queries")
        
        # Move previously evaluated passages into the kept set
        if pending_docs:
            kept_docs.extend(pending_docs)
            pending_docs = []
            if verbose:
                print(f"    Stored passages reviewed so far: {len(kept_docs)}")
        
        # If sufficient, we're done
        if sufficient:
            if verbose:
                print(f"\n  ✓ Sufficient information found!")
                if final_answer:
                    print(f"  Answer: {final_answer[:200]}...")
            iteration_times.append(time.perf_counter() - iteration_start)
            break
        
        # If no queries generated, break
        if not sub_queries:
            if verbose:
                print(f"\n  ⚠ No new queries generated, stopping")
            iteration_times.append(time.perf_counter() - iteration_start)
            break
        
        if verbose and sub_queries:
            print(f"\n  New queries:")
            for i, q in enumerate(sub_queries, 1):
                print(f"    {i}. {q}")
        
        # Step 2: Retrieve for each sub-query and track results
        num_sub_queries = len(sub_queries)
        docs_per_subquery = max(1, top_k_retriever)
        iteration_doc_descriptions: List[str] = []
        target_docs_per_subquery = top_k_retriever
        retrieved_next_iter: List[Dict[str, Any]] = []
        
        for i, sub_query in enumerate(sub_queries, 1):
            if verbose:
                print(f"\n  Retrieving for query {i}: {sub_query[:60]}...")
            
            # Retrieve
            if retrieval_strategy == "fixed_k":
                results = rag_db.lookup(sub_query, k=docs_per_subquery)
            else:
                from retrieve.filter import filter
                original_max_results = strategy_params.get("max_results", 20)
                #adjusted_max_results = max(1, original_max_results // num_sub_queries)
                adjusted_max_results = max(1, original_max_results)
                strategy_params_copy = strategy_params.copy()
                strategy_params_copy["max_results"] = adjusted_max_results
                results = filter(rag_db, sub_query, method=retrieval_strategy, **strategy_params_copy)
            
            # Apply per-subquery reranking if enabled
            if not no_rerank and len(results) > target_docs_per_subquery:
                if verbose:
                    print(f"    Reranking {len(results)} docs for this subquery to top {target_docs_per_subquery}...")
                
                # Extract contents for reranking
                contents = [r.page_content for r in results]
                scored_passages = rag_db.rerank(sub_query, contents)
                
                # Reorder results by reranking scores and take top-k
                reranked_indices = [i for i, _ in sorted(enumerate(scored_passages), 
                                                         key=lambda x: x[1][1], reverse=True)]
                results = [results[idx] for idx in reranked_indices[:target_docs_per_subquery]]
                
                if verbose:
                    print(f"    After reranking: keeping top {len(results)} docs")
            elif len(results) > target_docs_per_subquery:
                # No reranking, just limit to target
                results = results[:target_docs_per_subquery]
            
            new_passage_count = 0
            for result in results:
                metadata = result.metadata or {}
                url = metadata.get('original_url') or metadata.get('base_filename')
                passage_index = metadata.get('index')
                passage_key = (url, passage_index)

                if passage_key in seen_passages:
                    continue
                seen_passages.add(passage_key)

                doc_record = {
                    "url": url,
                    "content": result.page_content,
                    "metadata": metadata,
                }
                retrieved_next_iter.append(doc_record)

                descriptor = f"{url or 'Unknown'} (idx {passage_index})"
                iteration_doc_descriptions.append(descriptor)
                new_passage_count += 1

            if verbose:
                print(f"    Retrieved {len(results)} docs, {new_passage_count} new unique passages")

        pending_docs = retrieved_next_iter

        # Update iteration history for future LLM calls
        iteration_history.append({
            "iteration": iteration,
            "queries": sub_queries,
            "documents": iteration_doc_descriptions,
            "feedback": (current_feedback or "").strip()
        })

        if verbose:
            if iteration_doc_descriptions:
                print("  New passages:")
                for desc in iteration_doc_descriptions:
                    print(f"      - {desc}")
            print(f"  Total reviewed passages: {len(kept_docs)} | Pending: {len(pending_docs)}")
        
        iteration_time = time.perf_counter() - iteration_start
        iteration_times.append(iteration_time)
        
        if iteration >= max_iterations:
            if verbose:
                print(f"\n  ⚠ Maximum iterations reached")
            break
    
    # Final processing
    total_time = time.perf_counter() - start_time
    
    # Extract URLs from kept_docs
    retrieved_urls: List[str] = []
    seen_urls: Set[str] = set()
    for doc in kept_docs:
        url = doc.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            retrieved_urls.append(url)
    
    # Limit to top_k_reranking (reranking already done per-subquery)
    retrieved_urls = retrieved_urls[:top_k_reranking]
    
    # Calculate metrics
    from evaluation import calculate_retrieval_metrics
    expected_set = set(url for url in expected_urls if url and url.strip())
    metrics = calculate_retrieval_metrics(list(expected_set), retrieved_urls)
    
    # Add iteration statistics
    metrics.update({
        'total_time': total_time,
        'num_iterations': iteration,
        'total_queries': total_query_count,
        'final_docs_count': len(retrieved_urls),
        'sufficient': sufficient,
        'avg_iteration_time': sum(iteration_times) / len(iteration_times) if iteration_times else 0
    })
    metrics['retrieved_urls'] = retrieved_urls
    cleaned_answer = final_answer.strip() if isinstance(final_answer, str) else str(final_answer)
    metrics['final_answer'] = cleaned_answer if cleaned_answer else "Unknown"
    
    # Print final results
    if verbose:
        print(f"\n{'='*80}")
        print(f"MULTI-SHOT RETRIEVAL RESULTS")
        print(f"{'='*80}")
        print(f"Original Query: {original_query[:100]}...")
        print(f"Iterations: {iteration}")
    print(f"Total queries issued: {total_query_count}")
    print(f"Sufficient: {'Yes' if sufficient else 'No'}")
    if final_answer:
        print(f"LLM Answer: {final_answer}")
    if expected_answer:
        print(f"Expected Answer: {expected_answer}")
    print(f"Expected ({len(expected_set)}): {sorted(list(expected_set)[:5])}{'...' if len(expected_set) > 5 else ''}")
    print(f"Retrieved ({len(retrieved_urls)} unique docs): {retrieved_urls[:5]}{'...' if len(retrieved_urls) > 5 else ''}")
    matches = len(expected_set.intersection(set(retrieved_urls)))
    print(f"Matches: {matches}")
    print(f"\nMetrics:")
    print(f"  P@N: {metrics.get('precision@N', 0.0):.3f}")
    print(f"  R@N: {metrics.get('recall@N', 0.0):.3f}")
    print(f"  F1@N: {metrics.get('f1@N', 0.0):.3f}")
    print(f"  MAP: {metrics.get('average_precision', 0.0):.3f}")
    print(f"\nTiming:")
    print(f"  Avg per iteration: {metrics['avg_iteration_time']*1000:.1f}ms")
    print(f"  Total: {total_time*1000:.1f}ms")
    print(f"{'='*80}\n")
    
    return metrics


def run_multi_shot_evaluation(
    rag_db,
    dataset_path: str,
    max_sub_queries: int = 5,
    top_k_retriever: int = 10,
    top_k_reranking: int = 10,
    max_queries: Optional[int] = None,
    no_rerank: bool = False,
    retrieval_strategy: str = "fixed_k",
    reasoning_effort: str = "medium",
    detailed_analysis: bool = False,
    difficulty: int = 0,
    max_iterations: int = 5,
    llm_config: Optional[Dict[str, Any]] = None,
    context_char_limit: int = 0,
    result_handler: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    **strategy_params,
) -> Dict[str, float]:
    """
    Run multi-shot evaluation on a dataset.
    
    Args:
        rag_db: RAG database instance
        dataset_path: Path to dataset TSV file
        max_sub_queries: Maximum number of sub-queries per query
        top_k_retriever: Number of documents to retrieve per sub-query
        top_k_reranking: Number of documents after final reranking
        max_queries: Maximum number of queries to evaluate
        no_rerank: Skip reranking step
        retrieval_strategy: Strategy for retrieval
        reasoning_effort: LLM reasoning level
        detailed_analysis: Enable detailed complexity-based analysis
        difficulty: Minimum number of answer links required (0 = no filtering)
        max_iterations: Maximum iterations for iterative retrieval (default: 10)
    context_char_limit: Maximum characters to keep per document snippet used for LLM context
        **strategy_params: Additional parameters for retrieval strategy
        
        Optional result_handler: callable receiving (prompt, metrics) per query

    Returns:
        Dictionary of averaged metrics
    """
    
    df = pd.read_csv(dataset_path, sep='\t')
    
    # Filter by difficulty if specified
    df = filter_dataset_by_difficulty(df, difficulty)
    
    if isinstance(max_queries, int) and max_queries > 0:
        df = df.head(max_queries)
    else:
        max_queries = len(df)
    
    print(f"\n{'='*80}")
    print(f"MULTI-SHOT EVALUATION")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_path}")
    print(f"Queries: {max_queries}")
    print(f"Max sub-queries: {max_sub_queries}")
    print(f"Retrieval strategy: {retrieval_strategy}")
    print(f"LLM reasoning effort: {reasoning_effort}")
    print(f"Detailed analysis: {detailed_analysis}")
    print(f"Max iterations: {max_iterations}")
    if difficulty > 0:
        print(f"Difficulty filter: >= {difficulty} answer links")
    print(f"{'='*80}\n")
    
    total_metrics = {}
    valid_queries = 0
    all_query_metrics = []  # For detailed analysis
    
    for idx, row in df.iterrows():
        print(f"\n[Query {idx+1}/{max_queries}]")
        
        # Extract expected URLs
        expected_urls = []
        for col in df.columns:
            if col.startswith('wikipedia_link_') and pd.notna(row[col]):
                expected_urls.append(row[col].strip())
        
        # Extract expected answer
        expected_answer = row.get('Answer', '').strip() if 'Answer' in row and pd.notna(row.get('Answer')) else ""
        
        if expected_urls:
            # Multi-shot retrieval with iterative refinement
            metrics = multi_shot_retrieval(
                rag_db, row['Prompt'], expected_urls,
                expected_answer=expected_answer,
                max_sub_queries=max_sub_queries,
                top_k_retriever=top_k_retriever,
                top_k_reranking=top_k_reranking,
                max_iterations=max_iterations,
                no_rerank=no_rerank,
                retrieval_strategy=retrieval_strategy,
                verbose=True,
                reasoning_effort=reasoning_effort,
                llm_config=llm_config,
                context_char_limit=context_char_limit,
                **strategy_params
            )
            if result_handler:
                result_handler(row['Prompt'], metrics)
            
            # Accumulate metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    total_metrics[metric_name] = total_metrics.get(metric_name, 0.0) + float(value)
            
            valid_queries += 1
            
            # Collect metrics for detailed analysis
            if detailed_analysis:
                all_query_metrics.append(metrics.copy())
    
    if valid_queries > 0:
        # Calculate averages
        avg_metrics = {name: total / valid_queries for name, total in total_metrics.items()}
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"MULTI-SHOT EVALUATION SUMMARY ({valid_queries} queries)")
        print(f"{'='*80}")
        print(f"\nPRECISION METRICS:")
        print(f"  Precision@N:                {avg_metrics.get('precision@N', 0.0):.3f}")
        print(f"\nRECALL METRICS:")
        print(f"  Recall@N:                   {avg_metrics.get('recall@N', 0.0):.3f}")
        print(f"\nF1 METRICS:")
        print(f"  F1@N:                       {avg_metrics.get('f1@N', 0.0):.3f}")
        print(f"\nRANKING METRICS:")
        print(f"  Mean Average Precision:     {avg_metrics.get('average_precision', 0.0):.3f}")
        print(f"\nRETRIEVAL STATISTICS:")
        print(f"  Avg Sub-queries:            {avg_metrics.get('num_sub_queries', 0.0):.1f}")
        print(f"  Avg Passages Retrieved:     {avg_metrics.get('retrieval_time', 0.0):.1f}")
        print(f"  Avg Unique Docs (N):        {avg_metrics.get('retrieved_docs_count', 0.0):.1f}")
        print(f"\nTIMING:")
        print(f"  Avg Decomposition Time:     {avg_metrics.get('decomposition_time', 0.0)*1000:.1f}ms")
        print(f"  Avg Retrieval Time:         {avg_metrics.get('retrieval_time', 0.0)*1000:.1f}ms")
        if avg_metrics.get('reranking_time', 0.0) > 0:
            print(f"  Avg Reranking Time:         {avg_metrics.get('reranking_time', 0.0)*1000:.1f}ms")
        print(f"  Avg Total Time:             {avg_metrics.get('total_time', 0.0)*1000:.1f}ms")
        print(f"{'='*80}\n")
        
        # Print detailed analysis if requested
        if detailed_analysis and all_query_metrics:
            from evaluation import _print_detailed_analysis
            _print_detailed_analysis(df, all_query_metrics, valid_queries)
        
        return avg_metrics
    else:
        print("No valid queries found!")
        return {}


if __name__ == "__main__":
    args = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                   description="Multi-shot retrieval with query decomposition")
    
    # Add all standard parameters
    add_all_args(args)
    
    # Add multi-shot specific parameters
    args.add_argument('--max-sub-queries', type=int, default=5,
                     help='Maximum number of sub-queries to generate (default: 5)')
    args.add_argument('--reasoning', type=str, default='medium',
                     choices=['low', 'medium', 'high'],
                     help='LLM reasoning level for query decomposition (default: medium)')
    args.add_argument('--max-iterations', type=int, default=5,
                     help='Maximum number of retrieval iterations (default: 5)')
    
    # Special handling for --eval argument
    for action in args._actions:
        if '--eval' in action.option_strings:
            action.type = lambda x: int(x) if x.isdigit() else True
            action.const = True
            break
    
    args = args.parse_args()
    
    # Set deterministic seeds
    set_deterministic_seeds(args.seed)

    context_token_limit = args.llm_context_token_limit 
    chars_per_token = args.llm_chars_per_token 
    context_char_limit = int(context_token_limit * chars_per_token)
    
    # Setup LLM configuration with auto-detection
    llm_config = setup_llm_config(args)
    if llm_config:
        context_token_limit = llm_config.get("context_token_limit", context_token_limit)
        chars_per_token = llm_config.get("chars_per_token", chars_per_token)
        context_char_limit = llm_config.get("context_char_limit", context_char_limit)
    print(f"LLM Config: {llm_config}")
    print(f"Context limits -> tokens: {context_token_limit}, chars: {context_char_limit}, chars/token: {chars_per_token}")
    
    # Setup device-specific environment
    device_config = get_device_config()
    print(f"Device Config: {device_config}")
    
    # Initialize database
    if args.retrieval_method == "bm25":
        db_class = BM25DB
    else:
        db_class = VectorDB
    
    if args.database is None:
        args.database = db_class.get_default_db_name()
    
    db_file_path = args.database if args.database.endswith('.db') else f"{args.database}.db"
    db_base_name = args.database.replace('.db', '') if args.database.endswith('.db') else args.database
    
    rag_db = db_class(
        retriever_model=args.retriever_model, 
        reranker_model=args.reranker_model, 
        device=args.device,
        k1=args.bm25_k1, b=args.bm25_b, method=args.bm25_method, 
        database=db_base_name,
        delta=args.bm25_delta, backend=args.bm25_backend, 
        stopwords=args.bm25_stopwords,
        show_progress=args.bm25_show_progress, stemmer=args.bm25_stemmer,
        vector_index_method=args.vector_index_method, 
        ivf_nprobe=args.ivf_nprobe,
        load_embeddings=args.load_embeddings, 
        num_embedding_devices=args.num_embedding_devices,
        benchmark=args.benchmark
    )
    
    # Load database
    if os.path.exists(db_file_path):
        print(f"Loading existing database from {db_file_path}")
        rag_db.from_serialized(db_file_path)
    else:
        raise ValueError(f"Database not found: {db_file_path}. Please create it first using single_shot_retrieval.py")
    
    # Build strategy parameters
    strategy_params = {"max_results": args.max_results}
    if args.retrieval_strategy == "top_p":
        strategy_params["p"] = args.top_p
    elif args.retrieval_strategy == "relative":
        strategy_params["ratio"] = args.relative_ratio

    # Run evaluation or single query
    if args.eval:
        max_queries = args.eval if isinstance(args.eval, int) and not isinstance(args.eval, bool) and args.eval > 0 else None
        
        answer_records: List[Dict[str, Any]] = []

        def handle_result(prompt: str, metrics: Dict[str, Any]) -> None:
            raw_answer = metrics.get("final_answer", "")
            if not isinstance(raw_answer, str):
                raw_answer = str(raw_answer)
            answer_text = raw_answer.strip() or "Unknown"
            if args.generate_answer:
                print(f"LLM Answer: {answer_text}")
            if args.save_results:
                record = {
                    "prompt": prompt,
                    "retrieved_urls": metrics.get("retrieved_urls", []),
                    "llm_answer": answer_text,
                }
                answer_records.append(record)

        metrics = run_multi_shot_evaluation(
            rag_db, args.dataset,
            max_sub_queries=args.max_sub_queries,
            top_k_retriever=args.top_k_retriever,
            top_k_reranking=args.top_k_reranking,
            max_queries=max_queries,
            no_rerank=args.no_rerank,
            retrieval_strategy=args.retrieval_strategy,
            reasoning_effort=args.reasoning,
            detailed_analysis=True,  # Enable detailed complexity analysis
            difficulty=args.difficulty,
            max_iterations=args.max_iterations,
            llm_config=llm_config,
            context_char_limit=context_char_limit,
            result_handler=handle_result if (args.generate_answer or args.save_results) else None,
            **strategy_params
        )
        
        # Save results
        results_data = {
            "multi_shot": True,
            "max_sub_queries": args.max_sub_queries,
            "reasoning_effort": args.reasoning,
            "metrics": metrics
        }
        
        with open("multi_shot_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to multi_shot_results.json")

        if args.save_results:
            with open("result_multi_shot.json", "w") as f:
                json.dump({
                    "params": serialize_cli_args(args),
                    "results": answer_records,
                }, f, indent=2)
            print("Compatible results saved to result_multi_shot.json")
        
    else:
        # Single query multi-shot retrieval
        if not args.query:
            args.query = "Who won the French Open Mens Singles tournament the year that New York City FC won their first MLS Cup title?"
        
        print(f"\nRunning multi-shot retrieval for single query...")
        multi_shot_retrieval(
            rag_db, args.query, expected_urls=[],
            max_sub_queries=args.max_sub_queries,
            top_k_retriever=args.top_k_retriever,
            top_k_reranking=args.top_k_reranking,
            no_rerank=args.no_rerank,
            retrieval_strategy=args.retrieval_strategy,
            verbose=True,
            reasoning_effort=args.reasoning,
            llm_config=llm_config,
            context_char_limit=context_char_limit,
            **strategy_params
        )

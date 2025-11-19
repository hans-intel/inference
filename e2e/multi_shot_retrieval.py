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
import re
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
from llm_answer import (
    convert_results_to_entries,
    generate_answer_from_entries,
    uniform_clip_texts,
)
from prompt import QUERY_REWRITER_PROMPT
import requests


QUESTION_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "of", "to", "for", "and", "or",
    "if", "my", "your", "their", "our", "is", "are", "was", "were", "be",
    "been", "have", "had", "has", "do", "does", "did", "with", "from", "by",
    "who", "what", "when", "where", "why", "how", "which", "that", "this",
    "these", "those", "same", "first", "second", "third", "fourth", "fifth",
    "last", "latest", "year", "years", "as", "than", "into", "about",
    "over", "under", "after", "before"
}


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [token.lower() for token in re.findall(r"[A-Za-z0-9']+", text)]


def _split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    return [s.strip() for s in sentences if s.strip()]


def _format_documents_for_prompt(
    documents: List[Dict[str, Any]],
    total_char_limit: Optional[int],
) -> str:
    if not documents:
        return "(No documents yet)"

    normalized_snippets: List[str] = []
    sources: List[str] = []

    for doc in documents:
        metadata = doc.get("metadata") or {}
        source = (
            doc.get("url")
            or metadata.get("original_url")
            or metadata.get("source")
            or metadata.get("base_filename")
            or "Unknown source"
        )
        text = doc.get("content") or doc.get("raw_content") or ""
        text = re.sub(r"\s+", " ", text).strip()
        normalized_snippets.append(text)
        sources.append(source)

    clip_limit = total_char_limit if total_char_limit and total_char_limit > 0 else None
    clipped_snippets = uniform_clip_texts(normalized_snippets, clip_limit)

    formatted_parts: List[str] = []
    for idx, snippet in enumerate(clipped_snippets, 1):
        source = sources[idx - 1] if idx - 1 < len(sources) else "Unknown source"
        body = snippet.strip() if snippet else "(No content)"
        formatted_parts.append(f"[P{idx}] Source: {source}\n{body}")

    return "\n\n".join(formatted_parts)


def _build_evidence_clips(
    question: str,
    documents: List[Dict[str, Any]],
    max_sentences: int = 12,
    per_doc_limit: int = 2,
    total_char_limit: int = 4096,
) -> str:
    if not documents:
        return "(No evidence clips yet)"

    question_tokens = [t for t in _tokenize(question) if t not in QUESTION_STOPWORDS]
    if not question_tokens:
        question_tokens = _tokenize(question)

    scored_sentences: List[Tuple[float, str, str]] = []
    for idx, doc in enumerate(documents, 1):
        label = f"P{idx}"
        sentences = _split_into_sentences(doc.get("content", "") or "")
        if not sentences:
            continue
        doc_scores: List[Tuple[float, str]] = []
        for sentence in sentences:
            tokens = _tokenize(sentence)
            if not tokens:
                continue
            overlap = sum(1 for token in tokens if token in question_tokens)
            if overlap == 0:
                continue
            score = overlap / len(tokens)
            doc_scores.append((score, sentence))
        doc_scores.sort(reverse=True, key=lambda x: x[0])
        for score, sentence in doc_scores[:per_doc_limit]:
            scored_sentences.append((score, label, sentence))

    if not scored_sentences:
        return "(No evidence clips yet)"

    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    selected = scored_sentences[:max_sentences]

    lines: List[str] = []
    total_chars = 0
    for _, label, sentence in selected:
        snippet = f"{label}: {sentence.strip()}"
        projected = total_chars + len(snippet) + 1
        if total_char_limit and projected > total_char_limit:
            break
        lines.append(snippet)
        total_chars = projected

    if not lines:
        return "(No evidence clips yet)"
    return "\n".join(lines)

def _answer_doc_limit(top_k_retriever: Optional[int], top_k_reranking: Optional[int]) -> Optional[int]:
    if isinstance(top_k_reranking, int) and top_k_reranking > 0:
        return top_k_reranking
    if isinstance(top_k_retriever, int) and top_k_retriever > 0:
        return top_k_retriever
    return None


def _attempt_single_shot_answer(
    question: str,
    documents: List[Dict[str, Any]],
    llm_config: Optional[Dict[str, Any]],
    context_char_limit: int,
    doc_limit: Optional[int] = None,
) -> str:
    if not documents or not llm_config:
        return ""
    doc_entries = convert_results_to_entries(
        documents,
        limit=doc_limit,
        context_char_limit=context_char_limit,
    )
    base_limit = llm_config.get("context_char_limit", context_char_limit)
    return generate_answer_from_entries(question, doc_entries, llm_config, base_limit)


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


def _retrieve_documents_for_queries(
    sub_queries: List[str],
    rag_db,
    top_k_retriever: int,
    no_rerank: bool,
    retrieval_strategy: str,
    strategy_params: Optional[Dict[str, Any]],
    seen_passages: Set[Tuple[str, Any]],
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Run retrieval for the provided queries and return unique passages plus descriptions."""
    if not sub_queries:
        return [], []

    docs_per_subquery = max(1, top_k_retriever)
    target_docs_per_subquery = top_k_retriever
    iteration_doc_descriptions: List[str] = []
    retrieved_docs: List[Dict[str, Any]] = []

    for i, sub_query in enumerate(sub_queries, 1):
        if verbose:
            print(f"\n  Retrieving for query {i}: {sub_query[:60]}...")

        if retrieval_strategy == "fixed_k":
            results = rag_db.lookup(sub_query, k=docs_per_subquery)
        else:
            from retrieve.filter import filter as adaptive_filter

            params_copy = dict(strategy_params or {})
            original_max_results = params_copy.get("max_results", 20)
            params_copy["max_results"] = max(1, original_max_results)
            results = adaptive_filter(
                rag_db,
                sub_query,
                method=retrieval_strategy,
                **params_copy,
            )

        if not no_rerank and results:
            if verbose:
                print(
                    f"    Reranking {len(results)} docs for this subquery to top {target_docs_per_subquery}..."
                )

            contents = [r.page_content for r in results]
            scored_passages = rag_db.rerank(sub_query, contents)
            reranked_results: List[Dict[str, Any]] = []
            used_indices: Set[int] = set()
            for passage, _ in scored_passages:
                for idx, doc in enumerate(results):
                    if idx in used_indices:
                        continue
                    if doc.page_content == passage:
                        reranked_results.append(doc)
                        used_indices.add(idx)
                        break

            if reranked_results:
                results = reranked_results

            if verbose:
                print(f"    After reranking: keeping top {len(results)} docs")

        if len(results) > target_docs_per_subquery:
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
            retrieved_docs.append(doc_record)

            descriptor = f"{url or 'Unknown'} (idx {passage_index})"
            iteration_doc_descriptions.append(descriptor)
            new_passage_count += 1

        if verbose:
            print(f"    Retrieved {len(results)} docs, {new_passage_count} new unique passages")

    return retrieved_docs, iteration_doc_descriptions


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
        evidence_clips = _build_evidence_clips(
            question,
            documents,
            total_char_limit=attempt_limit if attempt_limit else 4096,
        )

        if documents:
            print(f"    Evaluating {len(documents)} passage(s) from previous iteration")
        else:
            print("    No passages to evaluate yet; generating initial queries")

        prompt = QUERY_REWRITER_PROMPT.format(
            question=question,
            context=context,
            history_chronological=history_text,
            evidence_clips=evidence_clips,
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
    
    original_query_clean = (original_query or "").strip()
    pending_single_shot = False
    answer_doc_limit = _answer_doc_limit(top_k_retriever, top_k_reranking)
    if original_query_clean:
        baseline_docs, initial_doc_descriptions = _retrieve_documents_for_queries(
            [original_query_clean],
            rag_db,
            top_k_retriever,
            no_rerank,
            retrieval_strategy,
            strategy_params,
            seen_passages,
            verbose,
        )
        pending_docs.extend(baseline_docs)
        total_query_count += 1

        history_feedback = "Initial retrieval using original question"
        iteration_history.append({
            "iteration": 0,
            "queries": [original_query_clean],
            "documents": initial_doc_descriptions,
            "feedback": history_feedback,
        })

        pending_single_shot = bool(pending_docs and llm_config)

    while not sufficient and (iteration < max_iterations or pending_single_shot):
        if pending_single_shot:
            if verbose:
                print(f"\n{'─'*80}")
                print("BASELINE SINGLE-SHOT EVALUATION")
                print(f"{'─'*80}")

            try:
                baseline_doc_limit = answer_doc_limit or len(pending_docs)
                if baseline_doc_limit:
                    baseline_doc_limit = min(baseline_doc_limit, len(pending_docs))
                else:
                    baseline_doc_limit = len(pending_docs)
                if verbose:
                    print(f"    Using top {baseline_doc_limit} document(s) for baseline answer")
                    for idx, doc in enumerate(pending_docs[:baseline_doc_limit], 1):
                        metadata = doc.get("metadata") or {}
                        url = doc.get("url") or metadata.get("original_url") or metadata.get("source") or metadata.get("base_filename")
                        passage_index = metadata.get("index")
                        print(f"      [{idx}] URL: {url} | Passage #{passage_index}")
                baseline_answer = _attempt_single_shot_answer(
                    original_query,
                    pending_docs,
                    llm_config,
                    context_char_limit,
                    doc_limit=baseline_doc_limit,
                )
                if verbose:
                    preview = baseline_answer.strip() if isinstance(baseline_answer, str) else str(baseline_answer)
                    print(f"    Baseline answer preview: {preview[:200] if preview else '(empty)'}")
            except Exception as exc:
                baseline_answer = ""
                if verbose:
                    print(f"    Baseline answer attempt failed: {exc}")

            note = "Single-shot answer: " + (baseline_answer.strip() or "(empty)")
            if iteration_history:
                existing_feedback = iteration_history[-1].get("feedback", "")
                iteration_history[-1]["feedback"] = (
                    f"{existing_feedback} | {note}" if existing_feedback else note
                )

            pending_single_shot = False

            baseline_clean = ""
            if isinstance(baseline_answer, str):
                baseline_clean = baseline_answer.strip()
            elif baseline_answer is not None:
                baseline_clean = str(baseline_answer).strip()
            normalized_baseline = baseline_clean.lower()

            if baseline_clean and normalized_baseline != "unknown":
                final_answer = baseline_clean
                sufficient = True
                kept_docs.extend(pending_docs)
                pending_docs = []
                if verbose:
                    print("    Single-shot answer deemed sufficient. Skipping iterative refinement.")
                break

            if verbose:
                print("    Single-shot attempt insufficient; starting iterative refinement.")
            continue

        if iteration >= max_iterations:
            break

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
        retrieved_next_iter, iteration_doc_descriptions = _retrieve_documents_for_queries(
            sub_queries,
            rag_db,
            top_k_retriever,
            no_rerank,
            retrieval_strategy,
            strategy_params,
            seen_passages,
            verbose,
        )
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
    if pending_docs:
        kept_docs.extend(pending_docs)
        pending_docs = []

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

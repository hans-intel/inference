import argparse
import json
import time
import os
from retrieve import VectorDB, BM25DB
from evaluation import evaluate_retrieval_query, run_evaluation
from utils import (
    setup,
    serialize_cli_args,
    ensure_unique_path,
)
from llm_answer import (
    convert_results_to_entries,
    extract_unique_urls,
    generate_answer_from_entries,
)

# Taken below from frames: https://huggingface.co/datasets/google/frames-benchmark
DEFAULT_QUERY = "Who won the French Open Mens Singles tournament the year that New York City FC won their first MLS Cup title?"


def _answer_doc_limit(args) -> int:
    if getattr(args, "top_k_reranking", None):
        return args.top_k_reranking
    return args.top_k_retriever

if __name__ == "__main__":
    args = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    
    args, llm_config, device_config, rag_db, retrieval_config = setup(args)

    # Run evaluation or single query lookup
    if args.eval:
        max_queries = args.eval if isinstance(args.eval, int) and not isinstance(args.eval, bool) and args.eval > 0 else None
        
        # Build retrieval_config with correct parameter names for filter function
        retrieval_config = {"max_results": args.max_results}
        if args.retrieval_strategy == "top_p":
            retrieval_config["p"] = args.top_p
        elif args.retrieval_strategy == "relative":
            retrieval_config["ratio"] = args.relative_ratio
        
        answer_records = []
        answer_doc_limit = _answer_doc_limit(args)

        def handle_result(prompt, retrieved_docs, metrics):
            urls = extract_unique_urls(retrieved_docs)
            answer_text = None
            if args.generate_answer:
                doc_entries = convert_results_to_entries(
                    retrieved_docs,
                    limit=answer_doc_limit,
                    full_doc=args.full_doc_context,
                    base_dir=doc_base_dir,
                    context_char_limit=context_char_limit
                )
                base_limit = llm_config.get("context_char_limit", context_char_limit)
                answer_text = generate_answer_from_entries(prompt, doc_entries, llm_config, base_limit)
                print(f"LLM Answer: {answer_text}")
            if args.save_results:
                record = {
                    "prompt": prompt,
                    "retrieved_urls": urls
                }
                if answer_text is not None:
                    record["llm_answer"] = answer_text
                answer_records.append(record)

        metrics = run_evaluation(
            rag_db,
            args.dataset,
            top_k_retriever=args.top_k_retriever,
            top_k_reranking=args.top_k_reranking,
            max_queries=max_queries,
            no_rerank=args.no_rerank,
            retrieval_strategy=args.retrieval_strategy,
            detailed_analysis=args.detailed_analysis,
            difficulty=args.difficulty,
            result_handler=handle_result if (args.generate_answer or args.save_results) else None,
            retrieval_config=retrieval_config,
        )
        
        # Save results for optimization
        results_data = {
            "accuracy": metrics.get('legacy_score', 0.0),  # Backward compatibility
            "metrics": metrics
        }
        
        with open("results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        if args.save_results:
            result_path = ensure_unique_path("result_single_shot.json")
            with open(result_path, "w") as f:
                json.dump({
                    "params": serialize_cli_args(args),
                    "results": answer_records
                }, f, indent=2)
            print(f"Compatible results saved to {result_path}")
        exit(0)  # Exit after evaluation
    else:
        # Single query lookup - reuse evaluation code for consistency
        
        retrieval_config = {}
        if args.retrieval_strategy == "top_p":
            retrieval_config["p"] = args.top_p
        elif args.retrieval_strategy == "relative":
            retrieval_config["ratio"] = args.relative_ratio
        
        # Time the retrieval
        tic = time.time()
        need_results = args.generate_answer or args.save_results
        eval_output = evaluate_retrieval_query(
            rag_db,
            args.query,
            expected_urls=[],
            top_k_retriever=args.top_k_retriever,
            top_k_reranking=args.top_k_reranking,
            verbose=False,
            no_rerank=getattr(args, 'no_rerank', False),
            retrieval_strategy=args.retrieval_strategy,
            print_results=True,
            return_results=need_results,
            max_results=args.max_results,
            retrieval_config=retrieval_config,
        )
        if need_results:
            _, retrieved_docs = eval_output
        else:
            retrieved_docs = []

        answer_value = None
        answer_doc_limit = _answer_doc_limit(args)
        if args.generate_answer:
            doc_entries = convert_results_to_entries(
                retrieved_docs,
                limit=answer_doc_limit,
                full_doc=args.full_doc_context,
                base_dir=doc_base_dir,
                context_char_limit=context_char_limit
            )
            base_limit = llm_config.get("context_char_limit", context_char_limit)
            answer_value = generate_answer_from_entries(args.query, doc_entries, llm_config, base_limit)
            print(f"LLM Answer: {answer_value}")

        if args.save_results:
            record = {
                "prompt": args.query,
                "retrieved_urls": extract_unique_urls(retrieved_docs)
            }
            if answer_value is not None:
                record["llm_answer"] = answer_value
            with open("result_single_shot.json", "w") as f:
                json.dump({
                    "params": serialize_cli_args(args),
                    "results": [record]
                }, f, indent=2)
        toc = time.time()
        
        print(f"\nLookup took {toc - tic:.3f} seconds")

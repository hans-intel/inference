import argparse
import json
import time
import os
from retrieve import VectorDB, BM25DB
from evaluation import evaluate_retrieval_query, run_evaluation
from utils import set_deterministic_seeds, setup_llm_config, serialize_cli_args
from params import add_all_args
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
    
    # Add all parameters from centralized definitions
    # This includes: Common, General, BM25, Vector, Strategy, and Reranking parameters
    add_all_args(args)
    
    # Special handling for --eval argument (needs custom type)
    # Override the default eval argument with custom type
    for action in args._actions:
        if '--eval' in action.option_strings:
            action.type = lambda x: int(x) if x.isdigit() else True
            action.const = True
            break

    args = args.parse_args()

    context_token_limit = args.llm_context_token_limit 
    chars_per_token = args.llm_chars_per_token 
    context_char_limit = int(context_token_limit * chars_per_token)

    # Set deterministic seeds for reproducible results
    set_deterministic_seeds(args.seed)
    llm_config = setup_llm_config(args) if args.generate_answer else None
    if llm_config:
        context_char_limit = llm_config.get("context_char_limit", context_char_limit)
        context_token_limit = llm_config.get("context_token_limit", context_token_limit)
        chars_per_token = llm_config.get("chars_per_token", chars_per_token)
    doc_base_dir = args.base_doc_dir

    # Initialize the appropriate database class
    if args.retrieval_method == "bm25":
        db_class = BM25DB
    else:
        db_class = VectorDB

    # Set default database path based on database class if not provided
    if args.database is None:
        args.database = db_class.get_default_db_name()
    
    # Normalize database path: ensure .db extension for file operations
    db_file_path = args.database if args.database.endswith('.db') else f"{args.database}.db"
    db_base_name = args.database.replace('.db', '') if args.database.endswith('.db') else args.database

    # Create database instance (pass base name without .db)
    rag_db = db_class(retriever_model=args.retriever_model, reranker_model=args.reranker_model, device=args.device, 
                        k1=args.bm25_k1, b=args.bm25_b, method=args.bm25_method, database=db_base_name,
                        delta=args.bm25_delta, backend=args.bm25_backend, stopwords=args.bm25_stopwords, 
                        show_progress=args.bm25_show_progress, stemmer=args.bm25_stemmer, 
                        vector_index_method=args.vector_index_method, ivf_nprobe=args.ivf_nprobe,
                        load_embeddings=args.load_embeddings, num_embedding_devices=args.num_embedding_devices,
                        benchmark=args.benchmark)

    if os.path.exists(db_file_path):
        # Load existing database
        print(f"Loading existing database from {db_file_path}")
        rag_db.from_serialized(db_file_path)
    else:
        if not args.ingest:
            raise ValueError("Either --database (existing) or --ingest (to create new) must be provided")
        
        # Ingest from file or folder
        tic = time.time()
        rag_db.ingest_from_path(args.ingest, num_threads=args.threads)
        
        # Get number of passages for timing calculation
        num_passages = len(rag_db._doc_list)  # This should be available after ingestion
        toc = time.time()
        ingestion_speed = num_passages/(toc-tic)
        print(f"Ingestion of {num_passages} passages took {toc - tic:.2f} seconds. {ingestion_speed:.2f} docs/sec")
        
        # Save the database (unless --no-save is specified)
        if not args.no_save:
            print(f"Saving database to {db_file_path}")
            rag_db.serialize(db_file_path)
        else:
            print("Skipping database save (--no-save specified)")

    # Run evaluation or single query lookup
    if args.eval:
        max_queries = args.eval if isinstance(args.eval, int) and not isinstance(args.eval, bool) and args.eval > 0 else None
        
        # Build strategy_params with correct parameter names for filter function
        strategy_params = {"max_results": args.max_results}
        if args.retrieval_strategy == "top_p":
            strategy_params["p"] = args.top_p
        elif args.retrieval_strategy == "relative":
            strategy_params["ratio"] = args.relative_ratio
        
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
            detailed_analysis=True,
            difficulty=args.difficulty,
            result_handler=handle_result if (args.generate_answer or args.save_results) else None,
            **strategy_params
        )
        
        # Save results for optimization
        results_data = {
            "accuracy": metrics.get('legacy_score', 0.0),  # Backward compatibility
            "metrics": metrics
        }
        
        with open("results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        if args.save_results:
            with open("result_single_shot.json", "w") as f:
                json.dump({
                    "params": serialize_cli_args(args),
                    "results": answer_records
                }, f, indent=2)
        exit(0)  # Exit after evaluation
    else:
        # Single query lookup - reuse evaluation code for consistency
        
        strategy_params = {}
        if args.retrieval_strategy == "top_p":
            strategy_params["p"] = args.top_p
        elif args.retrieval_strategy == "relative":
            strategy_params["ratio"] = args.relative_ratio
        
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
            **strategy_params
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

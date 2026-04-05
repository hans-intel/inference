import argparse
import json
import time
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from functools import lru_cache
# retrieve (VectorDB/BM25DB) is imported lazily inside __main__ — AFTER starting
# NUMA-pinned worker instances.  This ensures OMP does not initialise in the main
# process (via 'import faiss' inside vectordb.py) before the workers set their own
# affinity and OMP env vars.  Each spawned worker imports vectordb_instance.py
# (no OMP-linked libs), sets KMP_AFFINITY=disabled + sched_setaffinity, then
# imports faiss/torch — so OMP initialises correctly in every worker.
from retrieve.vectordb_instance import VectorDBInstance, FAISSIndexInstance, RerankInstance, get_numa_layout
from evaluation import evaluate_retrieval_query, run_evaluation
from utils import set_deterministic_seeds, setup_llm_config
from params import add_all_args

# Taken below from frames: https://huggingface.co/datasets/google/frames-benchmark
DEFAULT_QUERY = "Who won the French Open Mens Singles tournament the year that New York City FC won their first MLS Cup title?"
MAX_PASSAGE_PREVIEW = 4096
FULL_DOC_MAX_CHARS = 39000


def _get_metadata(doc):
    if hasattr(doc, "metadata"):
        return doc.metadata or {}
    if isinstance(doc, dict):
        return doc
    return {}


def _serialize_params(args):
    params = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            params[key] = value
        else:
            params[key] = str(value)
    return params


@lru_cache(maxsize=256)
def _read_text(path: str) -> str:
    path_obj = Path(path)
    if not path_obj.exists():
        return ""
    return path_obj.read_text(encoding="utf-8", errors="ignore")


def _load_document_text(metadata, base_dir=None, default_base_dir="doc_html", max_chars=FULL_DOC_MAX_CHARS):
    target_dir = base_dir or default_base_dir
    base_filename = metadata.get("base_filename")
    if not base_filename:
        return "", None

    base_path = Path(target_dir)
    candidates = [
        base_path / f"{base_filename}.txt",
        base_path / f"{base_filename}.html",
        base_path / f"{base_filename}.htm"
    ]

    for candidate in candidates:
        candidate_path = str(candidate)
        content = _read_text(candidate_path)
        if content:
            return content[:max_chars], candidate_path
    return "", None


def _convert_results_to_entries(results, limit=5, full_doc=False, base_dir=None, default_base_dir="doc_html"):
    entries = []
    seen_ids = set()
    count = 0
    for doc in results:
        metadata = _get_metadata(doc)
        url = metadata.get("original_url") or metadata.get("source")
        doc_id = url or metadata.get("base_filename")
        if doc_id and doc_id in seen_ids:
            continue
        if full_doc:
            content, source_path = _load_document_text(
                metadata,
                base_dir=base_dir,
                default_base_dir=default_base_dir
            )
            if not content:
                content = getattr(doc, "page_content", metadata.get("content", ""))[:MAX_PASSAGE_PREVIEW]
        else:
            content = getattr(doc, "page_content", metadata.get("content", ""))[:MAX_PASSAGE_PREVIEW]
            source_path = None
        entry = {"url": url, "content": content}
        if source_path:
            entry["source_path"] = source_path
        entries.append(entry)
        if doc_id:
            seen_ids.add(doc_id)
        count += 1
        if limit and limit > 0 and count >= limit:
            break
    return entries


def _extract_unique_urls(results):
    urls = []
    seen = set()
    for doc in results:
        metadata = _get_metadata(doc)
        url = metadata.get("original_url") or metadata.get("source")
        if url and url not in seen:
            urls.append(url)
            seen.add(url)
    return urls


def _generate_llm_answer(query, doc_entries, llm_config):
    context_parts = []
    for idx, doc in enumerate(doc_entries, 1):
        source = doc.get("url") or "Unknown source"
        snippet = doc.get("content", "").strip()
        context_parts.append(f"[{idx}] Source: {source}\n{snippet}")
    evidence_block = "\n\n".join(context_parts) if context_parts else "No supporting documents were retrieved."
    user_prompt = (
        "Answer the question using only the provided evidence."
        " Respond with a single word or short phrase, or 'Unknown' if the evidence is insufficient.\n\n"
        f"Question:\n{query}\n\nEvidence:\n{evidence_block}"
    )
    max_tokens = llm_config["max_tokens"]
    if isinstance(max_tokens, int):
        max_tokens = min(max_tokens, 256)
    else:
        max_tokens = 256
    payload = {
        "model": llm_config["model_name"],
        "messages": [
            {
                "role": "system",
                "content": "You are a concise retrieval QA assistant who trusts the supplied context."
            },
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    response = requests.post(llm_config["service_url"], json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    isl = usage.get("prompt_tokens", 0)
    osl = usage.get("completion_tokens", 0)
    return answer, isl, osl



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
    # Parse comma-separated service URLs (dp=N support: distribute requests round-robin)
    _raw_urls = args.llm_service_url or ""
    _all_service_urls = [u.strip() for u in _raw_urls.split(",") if u.strip()]
    if len(_all_service_urls) > 1:
        # Use only the first URL for model/token auto-detection in setup_llm_config
        args.llm_service_url = _all_service_urls[0]

    # Set deterministic seeds for reproducible results
    set_deterministic_seeds(args.seed)
    llm_config = setup_llm_config(args) if args.generate_answer else None
    # Attach full URL list so batch dispatch can load-balance across all instances
    if llm_config is not None and _all_service_urls:
        llm_config["service_urls"] = _all_service_urls
    doc_base_dir = args.base_doc_dir

    # Compute cache flags early — needed before starting instances
    retrieval_cache_in  = getattr(args, 'retrieval_cache_in',  None)
    retrieval_cache_out = getattr(args, 'retrieval_cache_out', None)

    # Resolve 'auto' device using ONLY filesystem checks — no torch/faiss import.
    # torch/faiss must not be imported before workers are spawned so each worker
    # process starts with a clean OMP state and can set its own KMP_AFFINITY.
    if args.device == 'auto':
        if os.path.exists('/dev/hl0'):           # Intel Gaudi (HPU)
            args.device = 'hpu'
        elif os.path.exists('/dev/nvidia0'):     # NVIDIA GPU
            args.device = 'cuda'
        elif os.path.exists('/dev/accel/accel0') or os.path.exists('/dev/dri/renderD128'):
            args.device = 'xpu'                  # Intel GPU / XPU
        else:
            args.device = 'cpu'
    print(f"[startup] device={args.device}  num_embedding_devices={args.num_embedding_devices}",
          flush=True)

    # ── Start NUMA-pinned embedding instances BEFORE importing faiss/torch ────
    # Worker processes are spawned here with a fresh Python interpreter.
    # vectordb_instance.py has no OMP-linked imports, so the worker sets
    # sched_setaffinity + KMP_AFFINITY=disabled + OMP_NUM_THREADS BEFORE
    # importing faiss/torch — guaranteeing OMP initialises with the right settings.
    # The parent (this process) only imports faiss/torch below, after all workers
    # have been spawned, so parent OMP state cannot bleed into them.
    _embedding_instances = []
    if (not retrieval_cache_in
            and args.ingest
            and args.device == 'cpu'
            and args.num_embedding_devices > 1):
        _layout = get_numa_layout(args.num_embedding_devices)
        if _layout:
            _encode_kw = {
                'normalize_embeddings': True,
                'batch_size': args.embedding_batch_size,
            }
            print(f"Starting {len(_layout)} NUMA-pinned embedding instances ...")
            for _i, (_s, _e, _node) in enumerate(_layout):
                _inst = VectorDBInstance(
                    instance_id=_i,
                    cpu_start=_s,
                    cpu_end=_e,
                    mem_node=_node,
                    model_name=args.retriever_model,
                    model_dtype=getattr(args, 'model_dtype', 'bfloat16'),
                    encode_kwargs=_encode_kw,
                )
                _embedding_instances.append(_inst)
            for _inst in _embedding_instances:
                _inst.start()   # spawns process; blocks until model loaded + READY
            print(f"All {len(_embedding_instances)} instances ready.")
        else:
            print(f"No NUMA layout for {args.num_embedding_devices} instances "
                  f"\u2014 falling back to single-instance embedding")

    # ── Start NUMA-pinned FAISS indexing instances (same layout as embedding) ──
    # Workers set their own CPU affinity + OMP env vars BEFORE importing faiss,
    # so each builds its partial sub-index with the correct per-NUMA-node thread count.
    # FAISS indexing workers: only beneficial for IVF (training is compute-bound).
    # For flat: add() is pure memcpy — 128-core serial is already optimal.
    # For hnsw: workers build sub-indexes then serialize ~220MB each back via IPC,
    #   we reconstruct vectors and rebuild the full index anyway — net negative vs
    #   a direct 128-OMP-thread add() in the main process (~4-5s vs 16s).
    _faiss_indexing_instances = []
    _idx_method = getattr(args, 'vector_index_method', 'flat')
    if (not retrieval_cache_in
            and args.ingest
            and args.device == 'cpu'
            and args.num_embedding_devices > 1
            and _embedding_instances           # only if embedding instances started
            and _idx_method == 'ivf'):         # workers only help for IVF training
        _layout = get_numa_layout(args.num_embedding_devices)
        if _layout:
            # Collect index_params from CLI args for non-flat methods
            _idx_params = {}
            if _idx_method == 'hnsw':
                _idx_params = {
                    'M': 32, 'efConstruction': 200, 'efSearch': 100,
                    'add_batch_size': args.faiss_indexing_batch_size,
                }
            elif _idx_method == 'ivf':
                _idx_params = {'nlist': 100, 'nprobe': getattr(args, 'ivf_nprobe', 10)}
            print(f"Starting {len(_layout)} NUMA-pinned FAISS indexing instances "
                  f"(method={_idx_method}) ...")
            for _i, (_s, _e, _node) in enumerate(_layout):
                _finst = FAISSIndexInstance(
                    instance_id=_i,
                    cpu_start=_s,
                    cpu_end=_e,
                    mem_node=_node,
                    method=_idx_method,
                    index_params=_idx_params,
                )
                _faiss_indexing_instances.append(_finst)
            for _finst in _faiss_indexing_instances:
                _finst.start()   # spawns process; blocks until READY
            print(f"All {len(_faiss_indexing_instances)} FAISS indexing instances ready.")

    # ── Start NUMA-pinned reranking instances (same NUMA layout as embedding) ──
    # The cross-encoder runs on CPU only; 3 × 43-core workers run in parallel,
    # each scoring their query slice independently → ~3× speedup.
    # Only start when: not using retrieval cache, reranker is configured,
    # and multiple embedding devices (i.e. NUMA layout) are enabled.
    _reranking_instances = []
    if (args.device == 'cpu'
            and getattr(args, 'reranker_model', None)
            and args.num_embedding_devices > 1
            and _embedding_instances):
        _layout = get_numa_layout(args.num_embedding_devices)
        if _layout:
            _reranker_dtype = getattr(args, 'model_dtype', 'bfloat16')
            _reranker_bs    = getattr(args, 'reranker_batch_size', 256)
            print(f"Starting {len(_layout)} NUMA-pinned reranking instances ...")
            for _i, (_s, _e, _node) in enumerate(_layout):
                _rinst = RerankInstance(
                    instance_id=_i,
                    cpu_start=_s,
                    cpu_end=_e,
                    mem_node=_node,
                    model_name=args.reranker_model,
                    model_dtype=_reranker_dtype,
                    batch_size=_reranker_bs,
                )
                _reranking_instances.append(_rinst)
            for _rinst in _reranking_instances:
                _rinst.start()   # spawns process; blocks until model loaded + READY
            print(f"All {len(_reranking_instances)} reranking instances ready.")
        else:
            print(f"No NUMA layout for {args.num_embedding_devices} instances "
                  f"— falling back to single-instance reranking")

    # ── Import retrieve NOW (faiss/torch OMP init happens here in main process) ──
    # All worker processes are already spawned and running at this point.
    from retrieve import VectorDB, BM25DB

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
                        embedding_batch_size=args.embedding_batch_size,
                        query_embedding_batch_size=getattr(args, 'query_embedding_batch_size', None),
                        faiss_indexing_batch_size=args.faiss_indexing_batch_size,
                        benchmark=args.benchmark,
                        model_dtype=getattr(args, 'model_dtype', 'bfloat16'),
                        reranker_batch_size=getattr(args, 'reranker_batch_size', 256))

    # Register pre-started NUMA-pinned instances with the db (started before import above)
    if _embedding_instances:
        rag_db.set_embedding_instances(_embedding_instances)
    if _faiss_indexing_instances:
        rag_db.set_indexing_instances(_faiss_indexing_instances)
    if _reranking_instances:
        rag_db.set_reranking_instances(_reranking_instances)

    # Populate per-component config so print_retrieval_timings can show batch sizes alongside latencies
    _llm_dp = len(_all_service_urls) if _all_service_urls else 1
    _llm_total_batch = getattr(args, "llm_batch_size", 128)
    _llm_cfg = {"concurrent_batch": _llm_total_batch}
    if _llm_dp > 1:
        _llm_cfg["dp"] = _llm_dp
    _rerank_cfg = {"k_in": args.top_k_retriever, "k_out": args.top_k_reranking,
                   "batch": getattr(args, 'reranker_batch_size', 256)}
    if _reranking_instances:
        _rerank_cfg["dp"] = len(_reranking_instances)
    rag_db._component_config = {
        "query_embedding": {"model_batch_size": getattr(args, 'query_embedding_batch_size', None) or args.embedding_batch_size},
        "vector_search":   {"k": args.top_k_retriever},
        "reranking":       _rerank_cfg,
        "llm_generation":  _llm_cfg,
    }

    # Skip DB load/ingest when loading pre-computed retrieval results from cache
    if not retrieval_cache_in:
        if os.path.exists(db_file_path):
            # Load existing database
            print(f"Loading existing database from {db_file_path}")
            rag_db.from_serialized(db_file_path)
        else:
            if not args.ingest:
                raise ValueError("Either --database (existing) or --ingest (to create new) must be provided")
            
            # Ingest from file or folder
            # (embedding instances already started + registered with rag_db above)
            tic = time.time()
            max_passages = getattr(args, 'max_passages', None)
            rag_db.ingest_from_path(args.ingest, num_threads=args.threads,
                                    max_passages=max_passages)
            
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

            # Stop embedding instances (model no longer needed after ingest)
            for _inst in _embedding_instances:
                _inst.stop()
            # Stop FAISS indexing instances
            for _finst in _faiss_indexing_instances:
                _finst.stop()

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
        # Collect (prompt, doc_entries, urls) for batched LLM calls after retrieval
        pending_llm = []  # list of (prompt, doc_entries, urls)

        if retrieval_cache_in:
            # ── Phase 2: skip retrieval, load pre-computed results from cache ──
            print(f"Loading retrieval cache from: {retrieval_cache_in}")
            with open(retrieval_cache_in) as _f:
                _cache = json.load(_f)
            pending_llm = [(_item["prompt"], _item["doc_entries"], _item["urls"])
                           for _item in _cache]
            print(f"Loaded {len(pending_llm)} entries from retrieval cache.")
            metrics = {}
        else:
            # ── Phase 1 (or normal mode): run retrieval + precomputed reranking ─
            def handle_result(prompt, retrieved_docs, metrics):
                urls = _extract_unique_urls(retrieved_docs)
                if args.generate_answer or retrieval_cache_out:
                    # Always collect doc_entries when saving a retrieval cache
                    doc_entries = _convert_results_to_entries(
                        retrieved_docs,
                        limit=5,
                        full_doc=args.full_doc_context,
                        base_dir=doc_base_dir
                    )
                    pending_llm.append((prompt, doc_entries, urls))
                elif args.save_results:
                    answer_records.append({"prompt": prompt, "retrieved_urls": urls})

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
                repeat=getattr(args, 'repeat', 1),
                result_handler=handle_result if (args.generate_answer or args.save_results or retrieval_cache_out) else None,
                **strategy_params
            )

            # ── Phase-1 exit: save retrieval cache and stop before starting LLM ─
            if retrieval_cache_out:
                print(f"Saving retrieval cache ({len(pending_llm)} entries) to: {retrieval_cache_out}")
                _cache_data = [
                    {"prompt": p, "doc_entries": d, "urls": u}
                    for p, d, u in pending_llm
                ]
                with open(retrieval_cache_out, 'w') as _f:
                    json.dump(_cache_data, _f)
                print("Retrieval-only phase complete. "
                      "Start vLLM servers, then re-run with --retrieval-cache-in.")
                if args.benchmark:
                    rag_db.print_retrieval_timings()
                exit(0)
        
        # Batch LLM calls concurrently for peak vLLM throughput
        if args.generate_answer and pending_llm:
            llm_batch_size = getattr(args, 'llm_batch_size', 128)
            _svc_urls = (llm_config or {}).get("service_urls") or [llm_config["service_url"]]
            _dp = len(_svc_urls)
            _per_srv = llm_batch_size // _dp if _dp > 1 else llm_batch_size
            if _dp > 1:
                print(f"\nSending {len(pending_llm)} LLM requests "
                      f"(total_batch={llm_batch_size}, dp={_dp}, {_per_srv}/server)...")
            else:
                print(f"\nSending {len(pending_llm)} LLM requests (batch_size={llm_batch_size})...")
            llm_start = time.time()

            def _call_llm(item_and_idx):
                item, global_idx = item_and_idx
                url = _svc_urls[global_idx % len(_svc_urls)]
                cfg = {**llm_config, "service_url": url}
                prompt, doc_entries, urls = item
                answer, isl, osl = _generate_llm_answer(prompt, doc_entries, cfg)
                return prompt, urls, answer, isl, osl

            results_map = {}
            # Process in chunks of llm_batch_size; distribute round-robin across dp instances
            for chunk_start in range(0, len(pending_llm), llm_batch_size):
                chunk = pending_llm[chunk_start:chunk_start + llm_batch_size]
                with ThreadPoolExecutor(max_workers=len(chunk)) as executor:
                    futures = {executor.submit(_call_llm, (item, chunk_start + i)): chunk_start + i
                               for i, item in enumerate(chunk)}
                    for future in as_completed(futures):
                        orig_idx = futures[future]
                        prompt, urls, answer, isl, osl = future.result()
                        results_map[orig_idx] = (prompt, urls, answer, isl, osl)
                        print(f"  [{orig_idx+1}/{len(pending_llm)}] LLM Answer: {answer}")

            # Reassemble in original order and record timings
            total_llm_time = time.time() - llm_start
            avg_ms = total_llm_time / len(pending_llm) * 1000
            print(f"LLM batch complete: {len(pending_llm)} queries in {total_llm_time:.2f}s "
                  f"(avg {avg_ms:.1f}ms/query)")
            # Record aggregate timing in rag_db for the benchmark report
            rag_db._retrieval_timings.setdefault("llm_generation", []).append(total_llm_time / len(pending_llm))

            total_isl = sum(results_map[i][3] for i in range(len(pending_llm)))
            total_osl = sum(results_map[i][4] for i in range(len(pending_llm)))
            n = len(pending_llm)
            print(f"  Avg ISL (input tokens) : {total_isl/n:.1f}")
            print(f"  Avg OSL (output tokens): {total_osl/n:.1f}")
            rag_db._llm_token_stats = {"avg_isl": total_isl/n, "avg_osl": total_osl/n}

            for i in range(len(pending_llm)):
                prompt, urls, answer, _, _ = results_map[i]
                record = {"prompt": prompt, "retrieved_urls": urls, "llm_answer": answer}
                answer_records.append(record)

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
                    "params": _serialize_params(args),
                    "results": answer_records
                }, f, indent=2)

        if args.benchmark:
            rag_db.print_retrieval_timings()
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
        if args.generate_answer:
            doc_entries = _convert_results_to_entries(
                retrieved_docs,
                limit=5,
                full_doc=args.full_doc_context,
                base_dir=doc_base_dir
            )
            # GPU (XPU): NO — remote HTTP call to LLM service
            def _single_llm_call():
                answer, isl, osl = _generate_llm_answer(args.query, doc_entries, llm_config)
                rag_db._llm_token_stats = {"avg_isl": float(isl), "avg_osl": float(osl)}
                return answer
            answer_value = rag_db._time_op("llm_generation", _single_llm_call)
            print(f"LLM Answer: {answer_value}")

        if args.save_results:
            record = {
                "prompt": args.query,
                "retrieved_urls": _extract_unique_urls(retrieved_docs)
            }
            if answer_value is not None:
                record["llm_answer"] = answer_value
            with open("result_single_shot.json", "w") as f:
                json.dump({
                    "params": _serialize_params(args),
                    "results": [record]
                }, f, indent=2)
        toc = time.time()

        print(f"\nLookup took {toc - tic:.3f} seconds")
        if args.benchmark:
            rag_db.print_retrieval_timings()

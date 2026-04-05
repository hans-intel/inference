import abc
import os
from typing import List, Dict, Any

class RagDB(abc.ABC):
    """Base class for retrieval-augmented generation databases."""
    
    def __init__(self, reranker_model: str = None, device: str = "auto", benchmark: bool = False, model_dtype: str = "bfloat16", reranker_batch_size: int = 256):
        self._reranker_model_name = reranker_model
        self._device = self._determine_device(device)
        self._model_dtype = model_dtype
        self._reranker_batch_size = reranker_batch_size
        self._reranker_model = None
        self._reranker_tokenizer = None
        # Persistent NUMA-pinned reranking worker instances (set via set_reranking_instances)
        self._reranking_instances = None
        self._benchmark = benchmark
        self._monitor = None
        # Dict of {component_name: [latency_seconds, ...]} for query-time components.
        # Single-shot values (serialize/deserialize) are stored as a one-element list.
        self._retrieval_timings: Dict[str, list] = {}
        # LLM token stats: populated by single_shot_retrieval.py after LLM calls
        self._llm_token_stats: Dict[str, float] = {}
        # Per-component config (batch sizes, k values); populated by caller after construction
        self._component_config: Dict[str, Any] = {}
        
        # Initialize monitoring if benchmark mode enabled
        if self._benchmark:
            from ingestion_monitor import IngestionMonitor
            self._monitor = IngestionMonitor()
        
        # Initialize reranker if specified
        if self._reranker_model_name:
            self._init_reranker()
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use."""
        import torch

        if device == "auto":
            if getattr(torch, 'hpu', None) and torch.hpu.is_available():
                print("Using HPU device")
                return "hpu"
            elif torch.cuda.is_available():
                print("Using CUDA device")
                return "cuda"
            elif getattr(torch, 'xpu', None) and torch.xpu.is_available():
                print("Using XPU device")
                return "xpu"
            else:
                print("Using CPU device")
                return "cpu"
        else:
            return device
    
    @staticmethod
    def get_data_dir(db_name: str) -> str:
        """Get data directory based on database name."""
        from pathlib import Path
        base_name = Path(db_name).stem  # Remove .db extension if present
        return f"{base_name}_data"
    
    @staticmethod
    def get_db_path(db_name: str) -> str:
        """Get database file path based on database name."""
        from pathlib import Path
        base_name = Path(db_name).stem  # Remove .db extension if present
        return f"{base_name}.db"

    def _init_reranker(self):
        """Initialize the reranker model."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        dtype = getattr(torch, self._model_dtype, torch.float32)
        self._reranker_model = AutoModelForSequenceClassification.from_pretrained(
            self._reranker_model_name, dtype=dtype)
        self._reranker_tokenizer = AutoTokenizer.from_pretrained(self._reranker_model_name)
        device = self._device
        if device in ("hpu", "auto"):
            print(f"Falling back to CPU for reranker (device='{device}' not supported).")
            device = "cpu"
        self._reranker_model = self._reranker_model.to(device)
        self._reranker_model.eval()
        actual_dtype = next(self._reranker_model.parameters()).dtype
        actual_device = next(self._reranker_model.parameters()).device
        print(f"  Reranker  : {self._reranker_model_name}")
        print(f"             dtype={actual_dtype}  device={actual_device}")
    
    def _track_component(self, name: str, total_chars: int, item_count: int, func, 
                        is_pipeline_input: bool = False, is_pipeline_output: bool = False):
        """Execute function with optional component tracking.
        
        Args:
            name: Component name
            total_chars: Input size in bytes
            item_count: Number of items processed
            func: Function to execute
            is_pipeline_input: Mark as pipeline input for aggregation
            is_pipeline_output: Mark as pipeline output for aggregation
        """
        if self._benchmark and self._monitor:
            with self._monitor.track_component(name, input_size_bytes=total_chars, 
                                             items_count=item_count, text_only=True,
                                             is_pipeline_input=is_pipeline_input,
                                             is_pipeline_output=is_pipeline_output) as ctx:
                result = func()
                ctx.add_text_bytes(total_chars)
                return result
        else:
            return func()
    
    def _time_op(self, name: str, func):
        """Time a single operation and accumulate into _retrieval_timings when benchmark=True.
        
        Always executes func() and returns its result. When benchmark is enabled,
        records wall-clock duration under the given name.
        """
        if not self._benchmark:
            return func()
        import time
        t0 = time.perf_counter()
        result = func()
        duration = time.perf_counter() - t0
        if name not in self._retrieval_timings:
            self._retrieval_timings[name] = []
        self._retrieval_timings[name].append(duration)
        return result

    def _time_op_batch(self, name: str, func, n: int):
        """Time a batch operation and record per-item latency (total/n) n times.

        This preserves the avg/min/max semantics of _retrieval_timings so that
        print_retrieval_timings reports meaningful per-query averages even when
        the underlying calls are batched.
        """
        if not self._benchmark:
            return func()
        import time
        t0 = time.perf_counter()
        result = func()
        total = time.perf_counter() - t0
        per_item = total / max(n, 1)
        timings = self._retrieval_timings.setdefault(name, [])
        timings.extend([per_item] * n)
        return result

    def print_retrieval_timings(self):
        """Print a summary of all retrieval-phase component timings."""
        if not self._retrieval_timings:
            return
        import os as _os
        device_label = self._device.upper()
        # Annotate which components are XPU-accelerated
        xpu_components = {"query_embedding", "reranking"}
        remote_components = {"llm_generation"}
        xpu_available = False
        try:
            import torch
            xpu_available = torch.xpu.is_available()
        except Exception:
            pass

        print(f"\n⏱️  RETRIEVAL COMPONENT TIMINGS  [device={device_label}]")
        print("=" * 62)
        all_names = [
            "db_deserialize",
            "db_serialize",
            "query_embedding",
            "vector_search",
            "reranking",
            "llm_generation",
        ]
        # Print in pipeline order; include any unexpected names at the end
        ordered = [n for n in all_names if n in self._retrieval_timings]
        ordered += [n for n in self._retrieval_timings if n not in all_names]
        for name in ordered:
            times = self._retrieval_timings[name]
            count = len(times)
            total = sum(times)
            avg = total / count
            mn = min(times)
            mx = max(times)
            gpu_tag = ""
            if name in xpu_components:
                gpu_tag = " [XPU ✅]" if xpu_available else " [CPU only - XPU not available]"
            elif name in remote_components:
                gpu_tag = " [remote vLLM server]"
            else:
                gpu_tag = " [CPU only]"
            print(f"   {name}{gpu_tag}:")
            if count == 1:
                print(f"      {total*1000:.2f} ms")
                if name == "llm_generation" and self._llm_token_stats:
                    print(f"      ISL (input tokens) : {self._llm_token_stats.get('avg_isl', 0):.0f}")
                    print(f"      OSL (output tokens): {self._llm_token_stats.get('avg_osl', 0):.0f}")
            else:
                print(f"      calls={count:,}  avg={avg*1000:.2f}ms  "
                      f"min={mn*1000:.2f}ms  max={mx*1000:.2f}ms  total={total:.3f}s")
                if name == "llm_generation" and self._llm_token_stats:
                    print(f"      avg ISL (input tokens) : {self._llm_token_stats.get('avg_isl', 0):.1f}")
                    print(f"      avg OSL (output tokens): {self._llm_token_stats.get('avg_osl', 0):.1f}")
            # Print per-component config (batch sizes, k values) if available
            cfg = self._component_config.get(name, {})
            if cfg:
                cfg_parts = []
                if "model_batch_size" in cfg:
                    cfg_parts.append(f"embed_batch={cfg['model_batch_size']}")
                if "k" in cfg:
                    cfg_parts.append(f"k={cfg['k']}")
                if "k_in" in cfg and "k_out" in cfg:
                    cfg_parts.append(f"k_in={cfg['k_in']}  k_out={cfg['k_out']}")
                if "batch" in cfg:
                    _rdp = cfg.get('dp', 1)
                    if _rdp > 1:
                        cfg_parts.append(f"rerank_batch={cfg['batch']}  dp={_rdp}")
                    else:
                        cfg_parts.append(f"rerank_batch={cfg['batch']}")
                if "concurrent_batch" in cfg:
                    cb = cfg['concurrent_batch']
                    dp = cfg.get('dp', 1)
                    if dp > 1:
                        cfg_parts.append(f"concurrent_batch={cb}  dp={dp}  per_server={cb // dp}")
                    else:
                        cfg_parts.append(f"concurrent_batch={cb}")
                if cfg_parts:
                    print(f"      [{', '.join(cfg_parts)}]")
        print()

    def _start_ingestion_timer(self):
        """Start the ingestion timer. Works for both benchmark and non-benchmark modes."""
        import time
        if self._benchmark and self._monitor:
            self._monitor.start_ingestion()
        return time.perf_counter()
    
    def _report_performance(self, ingestion_start_time: float, item_count: int, total_chars: int, db_type: str):
        """Report performance metrics with optional detailed breakdown.
        
        Args:
            ingestion_start_time: Start time from _start_ingestion_timer() (used only in non-benchmark mode)
            item_count: Number of items processed
            total_chars: Total characters processed
            db_type: Database type string for display
        """
        import time
        
        if self._benchmark and self._monitor:
            with self._monitor.track_ingestion() as ingestion_ctx:
                ingestion_ctx.set_item_count(item_count)
            print(f"\n=== {db_type} Performance ===")
            self._monitor.print_summary()
        else:
            end_time = time.perf_counter()
            duration = end_time - ingestion_start_time
            docs_per_sec = item_count / duration if duration > 0 else 0
            chars_per_sec = total_chars / duration if duration > 0 else 0
            print(f"{db_type} ingestion: {item_count} docs, {total_chars:,} chars in {duration:.2f}s")
            print(f"  Performance: {docs_per_sec:.1f} docs/sec, {chars_per_sec/1024:.1f} KB/sec")
    
    @abc.abstractmethod
    def ingest(self, passages: List[str], metadatas: List[Dict[str, Any]]):
        """Ingest passages and their metadata into the database."""
        pass
    
    @abc.abstractmethod
    def lookup(self, query: str, k: int) -> List[Any]:
        """Retrieve top-k relevant passages for a query."""
        pass
    
    @abc.abstractmethod
    def serialize(self, path: str):
        """Serialize the database to disk."""
        pass
    
    @abc.abstractmethod
    def from_serialized(self, path: str):
        """Load the database from disk."""
        pass
    
    def ingest_from_folder(self, folder_path: str, **kwargs):
        """Ingest data from a folder. Default implementation raises NotImplementedError."""
        raise NotImplementedError(f"Folder ingestion not supported for {self.__class__.__name__}")
    
    def ingest_from_file(self, file_path: str, **kwargs):
        """Ingest data from a JSON file. Default implementation for JSON files."""
        import json

        with open(file_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        passage_data = payload.get('passages', [])

        max_passages = kwargs.pop('max_passages', None)
        if max_passages is not None:
            passage_data = passage_data[:max_passages]
            print(f"  (limited to first {max_passages} passages via --max_passages)")

        doc_list = []
        passage_metadata = []
        for entry in passage_data:
            doc_list.append(entry['passage'])
            passage_metadata.append({k: v for k, v in entry.items() if k != 'passage'})

        print(f"Ingesting {len(doc_list)} passages from JSON file {file_path}")
        return self.ingest(doc_list, passage_metadata, passages_path=file_path, **kwargs)

    def ingest_from_path(self, source_path: str, **kwargs):
        """Handle both file and folder ingestion.
        
        Default implementation that delegates to appropriate methods:
        - Folders: calls ingest_from_folder() (may raise NotImplementedError if not overridden)
        - Files: calls ingest_from_file() (default JSON implementation)
        """
        from pathlib import Path
        
        source_path = Path(source_path)
        
        if source_path.is_dir():
            print(f"Ingesting documents from folder {source_path}")
            return self.ingest_from_folder(source_path, **kwargs)
        elif source_path.is_file():
            return self.ingest_from_file(source_path, **kwargs)
        else:
            raise ValueError(f"Source path {source_path} is neither a file nor a directory")

    def rerank(self, query: str, passages: List[str]):
        """Rerank passages using the reranker model.

        GPU (XPU): YES — cross-encoder inference runs on self._device.
        FAISS search is CPU-only for Intel XPU; this is the GPU-acceleratable part.

        When NUMA-pinned reranking instances are registered, dispatches to them
        even for single-query calls (instance 0 handles it; others stay idle).
        """
        # Dispatch to instances when available (works even for single-query calls)
        if self._reranking_instances:
            results = self._rerank_with_instances([query], [passages],
                                                  self._reranker_batch_size)
            return results[0]

        if self._reranker_model is None:
            # If no reranker, return passages with dummy scores
            return [(p, 0.0) for p in passages]
        
        import torch
        
        pairs = [[query, passage] for passage in passages]

        def _run():
            with torch.no_grad():
                inputs = self._reranker_tokenizer(pairs, padding=True, return_tensors='pt',
                                                truncation=True, max_length=512)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                scores = self._reranker_model(**inputs).logits.view(-1).float()
            scored = list(zip(passages, scores.cpu().tolist()))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored

        return self._time_op("reranking", _run)
    
    def rerank_batch(self, queries: List[str], passages_list: List[List[str]], batch_size: int = 256):
        """Batch rerank for multiple (query, passages) pairs.

        Instead of N separate rerank() calls (one per query), collects all
        (query, passage) pairs across all queries into a flat pool, processes
        them in batches of batch_size forward passes, then reassembles
        per-query scored lists.

        For 256 queries × 10 passages = 2560 pairs @ batch_size=256:
          → 10 forward passes instead of 256.

        When NUMA-pinned reranking instances are registered via
        set_reranking_instances(), the queries are split across instances and
        processed in parallel, giving ~N× speedup on multi-NUMA hardware.

        Returns:
            List[List[Tuple[str, float]]] — one scored+sorted list per query,
            same format as rerank().
        """
        if self._reranker_model is None and not self._reranking_instances:
            return [[(p, 0.0) for p in passages] for passages in passages_list]

        # Dispatch to NUMA-pinned parallel workers when available
        if self._reranking_instances:
            return self._rerank_with_instances(queries, passages_list, batch_size)

        import torch

        # Build flat list of (query, passage) pairs, tracking per-query offsets
        flat_pairs = []
        offsets = []  # (start_idx, length) per query
        for query, passages in zip(queries, passages_list):
            offsets.append((len(flat_pairs), len(passages)))
            for passage in passages:
                flat_pairs.append([query, passage])

        n_queries = len(queries)
        n_pairs = len(flat_pairs)

        def _run_batch():
            all_scores = []
            for start in range(0, n_pairs, batch_size):
                chunk = flat_pairs[start:start + batch_size]
                with torch.no_grad():
                    inputs = self._reranker_tokenizer(
                        chunk, padding=True, return_tensors='pt',
                        truncation=True, max_length=512
                    )
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    chunk_scores = self._reranker_model(**inputs).logits.view(-1).float()
                all_scores.extend(chunk_scores.cpu().tolist())
            return all_scores

        all_scores = self._time_op_batch("reranking", _run_batch, n_queries)

        # Reassemble per-query scored+sorted lists
        results = []
        for (start, length), passages in zip(offsets, passages_list):
            query_scores = all_scores[start:start + length]
            scored = list(zip(passages, query_scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            results.append(scored)
        return results

    def set_reranking_instances(self, instances: list):
        """Register pre-started RerankInstance workers for parallel reranking.

        When set, rerank_batch() will distribute queries across these workers
        instead of running all forward passes in the main process.
        Each worker must already be started (inst.start() called) before
        passing them here.
        """
        self._reranking_instances = instances if instances else None
        if instances:
            print(f"Using {len(instances)} pre-started NUMA-pinned reranking instance(s)",
                  flush=True)

    def _rerank_with_instances(self, queries: List[str], passages_list: List[List[str]],
                               batch_size: int) -> List[List]:
        """Distribute reranking across pre-started RerankInstance workers.

        Splits queries into N equal chunks (one per worker), sends them all
        asynchronously, then collects results in query order.
        """
        import time as _time
        instances = self._reranking_instances
        n = len(instances)
        n_queries = len(queries)
        chunk_size = (n_queries + n - 1) // n

        t0 = _time.perf_counter()

        # Send all chunks to their respective workers (non-blocking)
        actual_chunks = 0
        for i, inst in enumerate(instances):
            s = i * chunk_size
            e = min(s + chunk_size, n_queries)
            if s >= n_queries:
                break
            inst.rerank_async(i, queries[s:e], passages_list[s:e])
            print(f"  [rerank_instance {i}] sent {e - s} queries", flush=True)
            actual_chunks += 1

        # Collect results — each call blocks on that specific worker finishing
        collected = {}
        for i in range(actual_chunks):
            chunk_id, chunk_results = instances[i].get_result()
            collected[chunk_id] = chunk_results
            print(f"  [rerank_instance {chunk_id}] result received: "
                  f"{len(chunk_results)} scored lists", flush=True)

        # Reassemble in query order
        all_results = []
        for i in range(actual_chunks):
            all_results.extend(collected[i])

        # Record timing using _time_op_batch semantics (per-query latency × n_queries)
        total = _time.perf_counter() - t0
        if self._benchmark:
            per_item = total / max(n_queries, 1)
            timings = self._retrieval_timings.setdefault("reranking", [])
            timings.extend([per_item] * n_queries)

        return all_results

    def lookup_with_rerank(self, query: str, k: int, rerank_k: int = None) -> List[Any]:
        """Retrieve and rerank passages."""
        if rerank_k is None:
            rerank_k = k
            
        # Get initial results
        results = self.lookup(query, k=rerank_k)
        
        # If no reranker or fewer results than requested, return as-is
        if self._reranker_model is None or len(results) <= k:
            return results[:k]
        
        # Extract passages for reranking
        passages = [result.page_content for result in results]
        
        # Rerank
        reranked_passages = self.rerank(query, passages)
        
        # Map back to original results and return top-k
        reranked_results = []
        for passage, score in reranked_passages[:k]:
            for result in results:
                if result.page_content == passage:
                    reranked_results.append(result)
                    break
        
        return reranked_results

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device

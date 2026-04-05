import os
import faiss
import torch
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .ragdb import RagDB
import time

# ── CPU vendor / NUMA helpers (module-level, CPU-only) ──────────────────────
def _get_cpu_vendor() -> str:
    """Return 'intel', 'amd', or 'unknown' by reading /proc/cpuinfo."""
    try:
        with open('/proc/cpuinfo') as _f:
            for _line in _f:
                if _line.startswith('vendor_id'):
                    _v = _line.split(':', 1)[1].strip().lower()
                    if 'intel' in _v:
                        return 'intel'
                    if 'amd' in _v:
                        return 'amd'
                    break
    except Exception:
        pass
    return 'unknown'


def _get_numa_node_for_core(core_id: int) -> int:
    """Return NUMA node for a CPU core by reading /sys (Linux sysfs)."""
    import glob, re as _re
    for _path in glob.glob(f'/sys/devices/system/cpu/cpu{core_id}/node*'):
        _m = _re.search(r'node(\d+)$', _path)
        if _m:
            return int(_m.group(1))
    return 0


def _bind_memory_to_node(node: int) -> None:
    """Soft-bind this process's memory allocations to a NUMA node via libnuma.

    Uses numa_set_preferred() which is a *preference* (not strict binding) so
    remote allocations still succeed if the local node is full.  Silently
    ignored if libnuma is unavailable.
    """
    try:
        import ctypes
        _libnuma = ctypes.CDLL('libnuma.so.1', use_errno=True)
        _libnuma.numa_set_preferred(ctypes.c_int(node))
    except Exception:
        pass  # libnuma not available — CPU affinity still steers first-touch
# ─────────────────────────────────────────────────────────────────────────────


# Worker function for parallel embedding generation (must be at module level for multiprocessing)
def _parallel_embed_worker(device_id, chunk_idx, chunk_file, result_queue, model_name, encode_kwargs, base_device, model_dtype="bfloat16", cpu_cores=None, mem_node=None):
    """Worker function to generate embeddings on a specific device.

    chunk_file: path to a temp pickle file containing the List[str] passages for
    this worker.  Passing a file path instead of the list itself keeps the
    Process args tiny (a single string) so p.start() returns immediately for
    all workers regardless of dataset size or system load.
    """
    try:
        import os
        import time as _wtime

        # ── CPU NUMA pinning ─────────────────────────────────────────────────
        # CRITICAL: sched_setaffinity + OMP settings must happen BEFORE importing
        # torch/oneDNN.  With spawn the child starts fresh, so OMP has NOT been
        # initialised yet at this point — setting the env var here takes effect.
        if base_device == 'cpu' and cpu_cores:
            try:
                os.sched_setaffinity(0, cpu_cores)
                _actual = sorted(os.sched_getaffinity(0))
                _ok = (_actual[0] == cpu_cores[0] and _actual[-1] == cpu_cores[-1])
            except Exception as _e:
                print(f"[worker {device_id}] sched_setaffinity warning: {_e}")
                _ok = False
            n_threads = len(cpu_cores)
            # NOTE: OMP/KMP env vars are intentionally NOT set here.
            # This code path (_parallel_embed_worker) is disabled — all CPU
            # multi-instance embedding goes through VectorDBInstance._worker_main
            # in vectordb_instance.py, which sets OMP/KMP before any heavy import.
            if mem_node is not None:
                _bind_memory_to_node(mem_node)
            print(f"[worker {device_id}] affinity={'OK' if _ok else 'WARN'} "
                  f"cores={min(cpu_cores)}-{max(cpu_cores)} "
                  f"actual={_actual[0] if _actual else '?'}-{_actual[-1] if _actual else '?'} "
                  f"n_threads={n_threads} mem_node={mem_node}", flush=True)
        # ────────────────────────────────────────────────────────────────────

        # Heavy imports happen AFTER affinity + OMP env vars are set so that
        # oneDNN / MKL initialise their thread pools with the correct count.
        import torch
        from langchain_huggingface import HuggingFaceEmbeddings

        # For CPU: reinforce thread count after torch init (belt-and-suspenders)
        if base_device == 'cpu' and cpu_cores:
            torch.set_num_threads(n_threads)
            torch.set_num_interop_threads(1)  # no inter-op parallelism needed
            print(f"[worker {device_id}] torch threads: intra={torch.get_num_threads()} "
                  f"interop={torch.get_num_interop_threads()}", flush=True)

        # Format device string: CPU doesn't use indices, others do
        if base_device == 'cpu':
            device = 'cpu'
        elif base_device == 'hpu':
            device = 'hpu'
            import habana_frameworks.torch.core as htcore
        else:
            device = f'{base_device}:{device_id}'

        import torch as _torch
        _dtype = getattr(_torch, model_dtype, _torch.float32)
        model_kwargs = {'device': device}

        print(f"[worker {device_id}] loading model {model_name} ...", flush=True)
        _t0 = _wtime.perf_counter()
        # Load model once for this device
        embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        embedder._client.to(_dtype)  # SentenceTransformer doesn't accept dtype in __init__

        # Load passages from temp file (avoids large pickle-via-pipe in p.start())
        import pickle as _pkl
        with open(chunk_file, 'rb') as _cf:
            chunk = _pkl.load(_cf)
        print(f"[worker {device_id}] model loaded in {_wtime.perf_counter()-_t0:.1f}s  "
              f"→ starting inference on {len(chunk)} passages", flush=True)

        _t1 = _wtime.perf_counter()
        embeddings = embedder.embed_documents(chunk)
        import numpy as _np
        _arr = _np.array(embeddings, dtype=_np.float32)
        print(f"[worker {device_id}] inference done: {len(chunk)} seqs in "
              f"{_wtime.perf_counter()-_t1:.1f}s  → sending {_arr.nbytes/1e6:.0f} MB via queue",
              flush=True)
        result_queue.put((chunk_idx, _arr))
        
    except Exception as e:
        print(f"\u274c Error on worker {device_id} ({base_device}): {e}", flush=True)
        import traceback
        traceback.print_exc()
        result_queue.put((chunk_idx, None))

class VectorDB(RagDB):
    @classmethod
    def get_default_db_name(cls) -> str:
        """Get the default database filename for VectorDB."""
        return "vector.db"
    
    def __init__(self,
            retriever_model: str = None,
            reranker_model: str = None,
            device: str = "auto",
            vector_index_method: str = "hnsw",
            ivf_nprobe: int = 10,
            load_embeddings: bool = True,
            num_embedding_devices: int = 4,
            embedding_batch_size: int = 128,
            query_embedding_batch_size: int = None,
            faiss_indexing_batch_size: int = 256,
            benchmark: bool = False,
            model_dtype: str = "bfloat16",
            reranker_batch_size: int = 256,
            **kwargs
        ):
        super().__init__(reranker_model, device, benchmark, model_dtype, reranker_batch_size)
        self._retriever_model_name = retriever_model
        self._reranker_model_name = reranker_model
        self._vector_index_method = vector_index_method
        self._ivf_nprobe = ivf_nprobe
        self._load_embeddings = load_embeddings

        if self._device == "hpu":
            try:
                import habana_frameworks.torch.core as htcore
                os.environ["PT_HPU_LAZY_MODE"] = "1"
            except ImportError:
                print("Warning: HPU device requested but habana_frameworks not found. Falling back to CPU.")
                self._device = "cpu"

        # Auto-detect number of embedding devices.
        # XPU path is completely unchanged — only the cpu branch is affected.
        if self._device == "cpu":
            # Detect CPU vendor once and cache so _embed_documents_parallel can use it.
            self._cpu_vendor = _get_cpu_vendor()
            # ── TEMPORARILY force single-instance to verify all cores are used ──
            self._num_embedding_devices = 1
            if self._num_embedding_devices > 1:
                print(f"CPU parallel embedding: {self._num_embedding_devices} NUMA-pinned "
                      f"workers ({self._cpu_vendor.upper()})")
        elif self._device == "xpu" and hasattr(torch, 'xpu') and torch.xpu.is_available():
            available_xpu = torch.xpu.device_count()
            # num_embedding_devices==4 is the CLI default; treat it as "auto" for XPU
            # so both GPUs are used without requiring an extra flag.
            # If the user consciously passed a value >1, respect it as a cap.
            self._num_embedding_devices = (
                available_xpu if num_embedding_devices == 4
                else min(num_embedding_devices, available_xpu)
            )
            if self._num_embedding_devices > 1:
                print(f"XPU auto-detect: {available_xpu} device(s) found, "
                      f"using {self._num_embedding_devices} for embedding generation")
        else:
            self._num_embedding_devices = num_embedding_devices

        self._embedding_batch_size = embedding_batch_size
        # If not specified, query embedding uses same batch size as ingestion embedding
        self._query_embedding_batch_size = query_embedding_batch_size if query_embedding_batch_size is not None else embedding_batch_size
        # 0 means auto-compute in ingest() based on dataset size
        self._faiss_indexing_batch_size = faiss_indexing_batch_size
        # Persistent NUMA-pinned worker instances (set via set_embedding_instances)
        self._embedding_instances = None
        # Persistent NUMA-pinned FAISS indexing instances (set via set_indexing_instances)
        self._indexing_instances = None

        # Initialize embedding model with device configuration
        _dtype = getattr(torch, self._model_dtype, torch.float32)
        model_kwargs = {'device': self._device}
        encode_kwargs = {'normalize_embeddings': True, 'batch_size': self._embedding_batch_size}
        
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self._retriever_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self._embedding_model._client.to(_dtype)  # SentenceTransformer doesn't accept dtype in __init__
        _inner = self._embedding_model._client  # SentenceTransformer object
        _first_param = next(_inner.parameters())
        print(f"  Embedding : {self._retriever_model_name}")
        print(f"             dtype={_first_param.dtype}  device={_first_param.device}")
        self._embedding_dimension = len(self._embedding_model.embed_query("hello world"))
        
        # Check the dtype of the embedding without using numpy
        test_embedding_raw = self._embedding_model.embed_query("test")
        
        # Calculate dtype and itemsize from Python native list
        if isinstance(test_embedding_raw, list) and len(test_embedding_raw) > 0:
            test_element = test_embedding_raw[0]
            embedding_dtype = type(test_element)
            # sentence-transformers returns Python floats, but the underlying
            # tensor is float32 = 4 bytes. __sizeof__() gives 24 (CPython object
            # overhead) which is wrong for memory/throughput calculations.
            self._embedding_bytes_per_element = 4  # float32
        else:
            raise ValueError("Embedding query did not return a valid list of floats.")

        if self._benchmark:
            per_embedding_bytes = self._embedding_dimension * self._embedding_bytes_per_element
            print(f"   Embedding element type: {embedding_dtype} (stored as float32 = 4 bytes)")
            print(f"   Embedding dimension   : {self._embedding_dimension}")
            print(f"   Per-embedding size    : {per_embedding_bytes:,} bytes "
                  f"({per_embedding_bytes/1024:.1f} KB)")
            print(f"   Embedding inference batch size: {self._embedding_batch_size}")

        # The index defines the algorithm used for the similarity search
        # Support multiple vector index types (currently FAISS-based)
        self._index = self._create_vector_index(self._vector_index_method, self._embedding_dimension)

        # The docstore is used to store the documents and their metadata
        self._docstore = InMemoryDocstore()

        self._vector_store = FAISS(
            embedding_function=self._embedding_model,
            index=self._index,
            docstore=self._docstore,
            index_to_docstore_id={}, # This will be populated as documents are added
        )
        
        # Keep track of ingested documents for consistency with BM25DB
        self._doc_list = []
    
    def _create_vector_index(self, method: str, dimension: int):
        """Create a vector index based on the specified method.
        
        Currently uses FAISS backend, but abstracted to allow future support
        for other vector databases (e.g., Milvus, Qdrant, Weaviate).
        
        Args:
            method: Index method - 'flat', 'hnsw', or 'ivf'
            dimension: Embedding dimension
            
        Returns:
            Vector index object (FAISS index)
            
        Index Method Details:
        
        1. FLAT (IndexFlatL2):
           - Exact brute-force search using L2 distance
           - Pros: Perfect accuracy, simple
           - Cons: O(N) search time, slow for large datasets
           - Best for: Small datasets (<10K), when accuracy is critical
        
        2. HNSW (Hierarchical Navigable Small World):
           - Graph-based approximate nearest neighbor search
           - Pros: Very fast search O(log N), excellent recall, no training needed
           - Cons: Higher memory usage (stores graph), slower indexing
           - Best for: Most use cases, default choice
        
        3. IVF (Inverted File):
           - Clustering-based approximate search
           - Parameters:
             * nlist: number of clusters (auto-adjusted to ~2*sqrt(N))
             * nprobe: clusters to search per query (default: 10)
               - nprobe=1: fastest but lowest accuracy (~80-90%)
               - nprobe=10: good balance (~95-98% accuracy)
               - nprobe=50: high accuracy (~99%) but slower
           - Pros: Memory efficient, good for large datasets, faster than flat
           - Cons: Requires training, slightly lower recall than HNSW
           - Best for: Very large datasets (>1M), when memory is limited
        """
        if method == "flat":
            return faiss.IndexFlatL2(dimension)
        elif method == "hnsw":
            # M: number of connections per layer (higher = better recall, more memory)
            # efConstruction: quality of index construction (higher = better quality, slower build)
            M = 32  # Default: 32, good balance
            index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = 200  # 40=min/fast build; 200=production (one-time cost, better graph)
            index.hnsw.efSearch = 100  # 16=min; 100=production (~0.99 recall@10, ~0.15ms/query batched)
            return index
        elif method == "ivf":
            # nlist: number of clusters/cells (sqrt(N) is a good heuristic for N docs)
            nlist = 100  # Will be adjusted based on dataset size during training
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            # Note: IVF index needs training before use (will be done during ingest)
            return index
        else:
            raise ValueError(f"Unknown vector index method: {method}. Choose 'flat', 'hnsw', or 'ivf'.")
    
    def _train_vector_index(self, index, embeddings: np.ndarray):
        """Train IVF index on embeddings if needed.
        
        IVF (Inverted File Index) requires a one-time training phase to:
        1. Cluster the embedding space into nlist regions using k-means
        2. Build an inverted index mapping cluster_id -> vector_ids
        
        After training, search works by:
        1. Finding the nprobe nearest cluster centroids to the query
        2. Searching only within those clusters (much faster than full scan)
        
        Note: In incremental scenarios, this trains on the FIRST batch only.
        Subsequent batches are assigned to existing clusters without retraining.
        For production systems handling continuous data growth, consider:
        - Periodic retraining when dataset size doubles
        - Using all accumulated data for retraining
        - Online clustering algorithms that adapt to new data
        
        Args:
            index: FAISS index (only IVF types need training)
            embeddings: Numpy array of embeddings to train on
        """
        import numpy as np
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Adjust nlist (number of clusters) based on dataset size
        # Rule of thumb: nlist = sqrt(N) to 4*sqrt(N)
        n_samples = len(embeddings)
        optimal_nlist = max(10, min(int(np.sqrt(n_samples) * 2), 1000))
        
        # Update nlist if different from default
        if optimal_nlist != self._index.nlist:
            print(f"Adjusting IVF nlist from {self._index.nlist} to {optimal_nlist} based on {n_samples} samples")
            # Need to recreate index with new nlist
            quantizer = faiss.IndexFlatL2(self._embedding_dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self._embedding_dimension, optimal_nlist)
            # Update vector store's index
            self._vector_store.index = self._index
        
        print(f"Training IVF index on {n_samples} samples...")
        self._index.train(embeddings_array)
        
        # Set nprobe (number of clusters to search) for better accuracy
        self._index.nprobe = self._ivf_nprobe
        print(f"IVF index trained successfully with {self._index.nlist} clusters, nprobe={self._ivf_nprobe}")
        print(f"  → Will search {self._ivf_nprobe} clusters per query (~{100*self._ivf_nprobe/self._index.nlist:.1f}% of clusters)")
    
    def _get_embeddings_cache_path(self, passages_path: str) -> str:
        """Get the cache path for embeddings based on passages file path."""
        from pathlib import Path
        passages_path = Path(passages_path)
        # Replace extension with .emb.pkl
        cache_path = passages_path.with_suffix('.emb.pkl')
        return str(cache_path)
    
    def _save_embeddings_cache(self, embeddings: list, passages_path: str):
        """Save embeddings to a pickle file for reuse."""
        import os, pickle
        from pathlib import Path
        
        cache_path = self._get_embeddings_cache_path(passages_path)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        
        if os.path.exists(cache_path):
            print(f"Embeddings cache exists: {cache_path}")
            return

        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"💾 Saved embeddings cache to {cache_path}")
    
    def _load_embeddings_cache(self, passages_path: str) -> list:
        """Load embeddings from cache if available."""
        import pickle
        from pathlib import Path
        
        cache_path = self._get_embeddings_cache_path(passages_path)
        if not Path(cache_path).exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"✓ Loaded embeddings from cache: {cache_path}")
            return embeddings
        except Exception as e:
            print(f"⚠️  Failed to load embeddings cache: {e}")
            return None
    
    def set_embedding_instances(self, instances: list):
        """Register pre-started VectorDBInstance workers for embedding.

        When set, ingest() will distribute passages across these workers
        instead of using single-process embedding or spawning new processes.
        Each worker must already be started (inst.start() called) before
        passing them here.
        """
        self._embedding_instances = instances
        print(f"Using {len(instances)} pre-started NUMA-pinned embedding instance(s)",
              flush=True)

    def _embed_with_instances(self, passages: List[str]) -> list:
        """Distribute passages across pre-started VectorDBInstance workers.

        Splits passages into N equal chunks (one per worker), sends them all
        asynchronously, then collects results in chunk order.
        """
        instances = self._embedding_instances
        n = len(instances)
        chunk_size = (len(passages) + n - 1) // n
        chunks = [passages[i:i + chunk_size] for i in range(0, len(passages), chunk_size)]

        # Send all chunks to their respective workers (non-blocking)
        for i, chunk in enumerate(chunks):
            instances[i].embed_async(i, chunk)
            print(f"  [instance {i}] sent {len(chunk)} passages", flush=True)

        # Collect results — each call blocks on that specific worker finishing
        # Results may arrive out of chunk order, so key by chunk_id
        results = {}
        for inst in instances[:len(chunks)]:
            chunk_id, arr = inst.get_result()
            results[chunk_id] = arr
            print(f"  [instance {chunk_id}] result received: {arr.shape}", flush=True)

        # Reassemble in chunk order as a single numpy array.
        # Returning numpy avoids the expensive list[list[float]] → np.array() conversion
        # in _index_with_instances (185k × 768 Python floats ≈ 8-9s overhead).
        import numpy as _np
        return _np.vstack([results[i] for i in range(len(chunks))])

    def set_indexing_instances(self, instances: list):
        """Register pre-started FAISSIndexInstance workers for parallel FAISS indexing.

        When set, ingest() will distribute embeddings + passages across these workers
        to build partial FAISS sub-indexes in parallel, then merge them into the main
        vector store.  Each worker must already be started (inst.start() called).
        """
        self._indexing_instances = instances
        print(f"Using {len(instances)} pre-started NUMA-pinned FAISS indexing instance(s)",
              flush=True)

    def _index_with_instances(self, passages: List[str], metadatas: List[dict],
                              embeddings: list) -> None:
        """Distribute FAISS indexing across pre-started FAISSIndexInstance workers.

        Splits passages+embeddings into N equal chunks, sends them asynchronously,
        then merges the partial FAISS sub-indexes and docstores into self._vector_store.
        """
        import faiss
        import numpy as np
        from langchain_community.docstore.in_memory import InMemoryDocstore

        instances = self._indexing_instances
        n = len(instances)

        # Accept either numpy array (from _embed_with_instances) or list[list[float]]
        # (from single-device path or cache).  Avoid re-converting a numpy array — that
        # would cost ~8-9s for 185k × 768 Python floats and is the main source of
        # hidden overhead in the parallel indexing path.
        if isinstance(embeddings, np.ndarray):
            emb_array = embeddings if embeddings.dtype == np.float32 else embeddings.astype(np.float32)
        else:
            emb_array = np.array(embeddings, dtype=np.float32)

        # Partition into N equal chunks
        chunk_size = (len(passages) + n - 1) // n
        actual_chunks = []
        for i, inst in enumerate(instances):
            s = i * chunk_size
            e = min(s + chunk_size, len(passages))
            if s >= len(passages):
                break
            p_chunk = passages[s:e]
            m_chunk = metadatas[s:e] if metadatas else [{}] * (e - s)
            e_chunk = emb_array[s:e]
            inst.index_async(i, p_chunk, m_chunk, e_chunk)
            actual_chunks.append((i, len(p_chunk)))
            print(f"  [faiss_instance {i}] sent {len(p_chunk)} passages for indexing",
                  flush=True)

        # Collect results — each call blocks on its own worker's output queue
        results = {}
        for i, inst in enumerate(instances[:len(actual_chunks)]):
            chunk_id, faiss_bytes, docs, idx_to_docstore = inst.get_result()
            results[chunk_id] = (faiss_bytes, docs, idx_to_docstore)
            print(f"  [faiss_instance {chunk_id}] result received: {len(docs)} docs",
                  flush=True)

        # Assemble merged index + docstores from worker results.
        # Strategy per index type:
        #   flat  → merge_from() (pure memcpy, supported, serializable)
        #   hnsw  → reconstruct_n() all vectors from each sub-index, add() into one
        #           fresh IndexHNSWFlat.  IndexShards is NOT serializable by faiss.write_index.
        #           The vectors are already stored in each HNSW sub-index so reconstruct_n
        #           is O(N) and cheap; only vector storage is copied, not the graph.
        #   ivf   → same reconstruct_n approach (IndexIVFFLat supports it)

        merged_docstore        = {}
        merged_idx_to_docstore = {}
        id_offset              = 0

        # Deserialize all sub-indexes and collect their vectors
        sub_indexes   = []
        sub_all_vecs  = []
        for chunk_id in range(len(actual_chunks)):
            faiss_bytes, docs, local_idx_to_docstore = results[chunk_id]
            sub_idx = faiss.deserialize_index(np.frombuffer(faiss_bytes, dtype=np.uint8))
            n_vecs  = sub_idx.ntotal
            sub_indexes.append((sub_idx, n_vecs))

            if self._vector_index_method != 'flat':
                # Extract stored vectors for re-adding to the merged index
                vecs = np.zeros((n_vecs, emb_array.shape[1]), dtype=np.float32)
                sub_idx.reconstruct_n(0, n_vecs, vecs)
                sub_all_vecs.append(vecs)

            for local_id, uuid_str in local_idx_to_docstore.items():
                merged_idx_to_docstore[id_offset + local_id] = uuid_str
            merged_docstore.update(docs)
            id_offset += n_vecs

        total_vecs = id_offset
        dim        = emb_array.shape[1]

        if self._vector_index_method == 'flat':
            # flat: merge_from() is supported and serializable
            merged_idx = None
            for sub_idx, _ in sub_indexes:
                if merged_idx is None:
                    merged_idx = sub_idx
                else:
                    merged_idx.merge_from(sub_idx)
        else:
            # hnsw / ivf: build one fresh serializable index, bulk-add all vectors
            if self._vector_index_method == 'hnsw':
                M = self._indexing_instances[0].index_params.get('M', 32) if hasattr(self._indexing_instances[0], 'index_params') else 32
                merged_idx = faiss.IndexHNSWFlat(dim, M)
                merged_idx.hnsw.efConstruction = 200
                merged_idx.hnsw.efSearch       = 100
            else:  # ivf
                nlist     = 100
                quantizer = faiss.IndexFlatL2(dim)
                merged_idx = faiss.IndexIVFFlat(quantizer, dim, nlist)

            all_vecs = np.vstack(sub_all_vecs)  # (total_vecs, dim)
            print(f"  Rebuilding merged {self._vector_index_method.upper()} index from "
                  f"{total_vecs} vectors ...", flush=True)
            if self._vector_index_method == 'ivf':
                merged_idx.train(all_vecs)
            import time as _t; _rb0 = _t.perf_counter()
            merged_idx.add(all_vecs)
            print(f"  Merged index built in {_t.perf_counter()-_rb0:.1f}s  "
                  f"ntotal={merged_idx.ntotal}", flush=True)

        # Splice merged results back into the LangChain FAISS wrapper
        self._vector_store.index               = merged_idx
        self._vector_store.docstore            = InMemoryDocstore(merged_docstore)
        self._vector_store.index_to_docstore_id = merged_idx_to_docstore

        # Update _doc_list used for passage-count reporting
        self._doc_list.extend(passages)

        print(f"✓ Merged {id_offset} vectors from {len(actual_chunks)} FAISS indexing instances",
              flush=True)

    def _embed_documents_parallel(self, passages: List[str]) -> list:
        """Generate embeddings using multiple devices in parallel.
        
        Uses the device type from --device option and spawns multiple workers.
        
        Args:
            passages: List of text passages to embed
            
        Returns:
            List of embeddings (one per passage)
        """
        import torch
        import multiprocessing as mp
        
        # Use the device type already configured via --device option
        base_device = self._device  # e.g., 'xpu', 'cuda', 'cpu', 'hpu'
        
        # Determine number of available devices based on device type
        if base_device == 'cpu':
            # For CPU, use requested number as process count
            num_devices = self._num_embedding_devices
        elif base_device == 'hpu' and hasattr(torch, 'hpu'):
            num_devices = torch.hpu.device_count()
        elif base_device == 'xpu' and hasattr(torch, 'xpu'):
            num_devices = torch.xpu.device_count()
        elif base_device == 'cuda':
            num_devices = torch.cuda.device_count()
        else:
            # Fallback for unknown device types
            num_devices = 1
        print("num_devices requested:", num_devices)
        num_workers = min(self._num_embedding_devices, num_devices, len(passages))
        
        if num_workers <= 1:
            # Fallback to single device
            return self._embedding_model.embed_documents(passages)
        
        # Set spawn method for device compatibility (required for XPU/CUDA)
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, ignore
            pass
        
        # ── For CPU: compute per-worker core ranges based on CPU vendor ────────
        # XPU/CUDA/HPU workers use device IDs, not CPU core pinning.
        worker_cpu_cores = [None] * num_workers
        worker_mem_nodes = [None] * num_workers
        if base_device == 'cpu':
            vendor = getattr(self, '_cpu_vendor', _get_cpu_vendor())
            if vendor == 'intel':
                # Intel GNR-AP SNC-3: fixed per-NUMA-node layouts.
                # sched_getaffinity() is unreliable inside Docker (may return the
                # daemon's narrow default affinity, not the true host topology).
                # Use hardcoded ranges matching the physical NUMA tile boundaries.
                _intel_layouts = {
                    3: [(  0,  42, 0),   # node 0
                        ( 43,  85, 1),   # node 1
                        ( 86, 127, 2)],  # node 2
                    2: [(  0,  63, 0),   # nodes 0+1
                        ( 64, 127, 2)],  # nodes 2+3 (use node 2 for memory)
                    1: [(  0, 127, 0)],  # single worker, full socket
                }
                _layout = _intel_layouts.get(num_workers)
                if _layout is None:
                    # Unexpected worker count: fall back to sched_getaffinity
                    import os as _os
                    all_cores = sorted(_os.sched_getaffinity(0))
                    if len(all_cores) < num_workers:
                        num_workers = max(1, len(all_cores))
                        worker_cpu_cores = [None] * num_workers
                        worker_mem_nodes  = [None] * num_workers
                    q, r = divmod(len(all_cores), num_workers)
                    idx = 0
                    for w in range(num_workers):
                        size = q + (1 if w < r else 0)
                        worker_cpu_cores[w] = all_cores[idx: idx + size]
                        worker_mem_nodes[w] = _get_numa_node_for_core(worker_cpu_cores[w][0])
                        idx += size
                else:
                    for w, (s, e, node) in enumerate(_layout):
                        worker_cpu_cores[w] = list(range(s, e + 1))
                        worker_mem_nodes[w] = node
            elif vendor == 'amd':
                # AMD Turin: fixed 4×32-core layout, all memory on node 0.
                _amd_ranges = [(0, 31), (32, 63), (64, 95), (96, 127)]
                for w in range(min(num_workers, len(_amd_ranges))):
                    s, e = _amd_ranges[w]
                    worker_cpu_cores[w] = list(range(s, e + 1))
                    worker_mem_nodes[w] = 0  # all on node 0 for AMD Turin
            print(f"  CPU NUMA layout ({vendor.upper()}): "
                  f"{[f'{min(c)}-{max(c)}(node{n})' for c, n in zip(worker_cpu_cores, worker_mem_nodes) if c]}")

        print(f"🚀 Parallel embedding on {num_workers} {base_device.upper()} worker(s)...")

        # Split passages into chunks - one chunk per worker
        chunk_size = (len(passages) + num_workers - 1) // num_workers
        chunks = [passages[i:i + chunk_size] for i in range(0, len(passages), chunk_size)]

        print(f"   Split {len(passages)} passages into {len(chunks)} chunks (~{chunk_size} passages/worker)")

        # Write each chunk + worker args to temp files so p.start() is lightweight.
        # CRITICAL: we spawn via retrieve._emb_launcher.bootstrap (not _parallel_embed_worker)
        # so that CPU affinity + OMP env vars are set BEFORE 'import faiss' triggers
        # Intel OMP initialization in the child.  If the child imported vectordb.py
        # first (to resolve _parallel_embed_worker), 'import faiss' at module level
        # would init OMP with the PARENT's KMP_AFFINITY=compact → all workers' threads
        # would fight for core 0 and fall back to 1 thread each.
        import tempfile, pickle as _pkl, atexit
        from retrieve._emb_launcher import bootstrap as _launcher_bootstrap

        chunk_files = []
        worker_arg_files = []
        encode_kwargs = {'normalize_embeddings': True, 'batch_size': self._embedding_batch_size}
        for i, chunk in enumerate(chunks):
            # Chunk text file
            _tf = tempfile.NamedTemporaryFile(delete=False, suffix=f'_emb_chunk{i}.pkl')
            _pkl.dump(chunk, _tf)
            _tf.close()
            chunk_files.append(_tf.name)
            atexit.register(os.unlink, _tf.name)

            # Worker kwargs file (everything _parallel_embed_worker needs except result_queue,
            # which is injected by the launcher since mp.Queue can't be pickled to disk)
            _af = tempfile.NamedTemporaryFile(delete=False, suffix=f'_emb_args{i}.pkl')
            _pkl.dump({
                'device_id': i,
                'chunk_idx': i,
                'chunk_file': _tf.name,
                'model_name': self._retriever_model_name,
                'encode_kwargs': encode_kwargs,
                'base_device': base_device,
                'model_dtype': self._model_dtype,
                'cpu_cores': worker_cpu_cores[i] if i < len(worker_cpu_cores) else None,
                'mem_node': worker_mem_nodes[i] if i < len(worker_mem_nodes) else None,
            }, _af)
            _af.close()
            worker_arg_files.append(_af.name)
            atexit.register(os.unlink, _af.name)

        # Create result queue and spawn all workers via the lightweight launcher.
        result_queue = mp.Queue()
        processes = []

        for device_id in range(num_workers):
            if device_id < len(chunks):
                # Patch the result_queue into the args file (Queue can't be pickled to disk)
                # Instead, pass it directly as a launcher that injects it.
                _cores_csv = ','.join(str(c) for c in worker_cpu_cores[device_id]) if worker_cpu_cores[device_id] else ''
                _mem = worker_mem_nodes[device_id] if worker_mem_nodes[device_id] is not None else -1

                # Re-write args file with result_queue=None; worker receives queue via launcher
                p = mp.Process(target=_launcher_bootstrap,
                           args=(result_queue, _cores_csv, _mem, worker_arg_files[device_id]))
                p.start()
                processes.append(p)
                print(f"  [worker {device_id}] spawned (pid={p.pid})", flush=True)
        
        # Collect results
        results = {}
        for _ in range(min(num_workers, len(chunks))):
            chunk_idx, embeddings = result_queue.get()
            if embeddings is not None:
                # Workers return numpy arrays; convert to list[list[float]] for
                # langchain FAISS add_embeddings compatibility.
                results[chunk_idx] = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Combine results in order
        all_embeddings = []
        for i in range(len(chunks)):
            if i in results:
                all_embeddings.extend(results[i])
        
        print(f"✓ Generated {len(all_embeddings)} embeddings across {num_workers} devices")
        
        return all_embeddings
    
    def _calculate_index_output_size(self):
        """Calculate the size of VectorDB output data (db file - metadata).
        
        Returns the total size in bytes of the serialized database file,
        excluding configuration metadata overhead.
        
        The .db file contains:
        - FAISS index (vectors)
        - Passages (docstore)
        - Metadata (small overhead)
        
        We estimate metadata size and subtract it from total file size.
        """
        from pathlib import Path
        
        # VectorDB uses serialize path, not _database_name like BM25
        if not hasattr(self, '_serialize_path') or not self._serialize_path:
            return 0
        
        db_path = Path(self._serialize_path)
        if not db_path.exists():
            return 0
        
        total_file_size = db_path.stat().st_size
        return total_file_size
    
    def ingest(self, passages: List[str], metadatas: List[dict], **kwargs):
        """Ingest passages with performance monitoring."""
        # Handle BM25-specific parameters gracefully
        if 'num_threads' in kwargs:
            print(f"Warning: num_threads parameter is not used in VectorDB, ignoring")
        
        # Extract passages source path for embeddings caching
        passages_path = kwargs.get('passages_path', None)
        
        # Start timing (works for both benchmark and non-benchmark modes)
        ingestion_start = self._start_ingestion_timer()
        
        total_chars = sum(len(passage) for passage in passages)
        
        # Handle embeddings: try to load from cache or generate new ones
        embeddings = None

        if self._load_embeddings and passages_path:
            embeddings = self._load_embeddings_cache(passages_path)

        # Generate embeddings if not cached
        if embeddings is None:
            import time as _time
            _embed_start = _time.perf_counter()
            if self._embedding_instances:
                # Use pre-started NUMA-pinned worker instances (best for CPU multi-core).
                # Each instance was spawned with NO OMP-linked imports so it set its own
                # affinity + OMP env vars before importing faiss/torch — guaranteed all-core use.
                embeddings = self._track_component("embedding_generation", total_chars, len(passages),
                                                  lambda: self._embed_with_instances(passages),
                                                  is_pipeline_input=True)
            # elif self._num_embedding_devices > 1:
            #     # DISABLED: _embed_documents_parallel spawns workers that inherit the parent's
            #     # OMP initialization (triggered by 'import faiss' at the top of vectordb.py)
            #     # so KMP_AFFINITY/OMP_NUM_THREADS set inside the worker are ignored —
            #     # each worker falls back to 1 thread.  Use set_embedding_instances() instead.
            #     embeddings = self._track_component("embedding_generation", total_chars, len(passages),
            #                                       lambda: self._embed_documents_parallel(passages),
            #                                       is_pipeline_input=True)
            else:
                # Single device embedding generation
                embeddings = self._track_component("embedding_generation", total_chars, len(passages), 
                                                  lambda: self._embedding_model.embed_documents(passages),
                                                  is_pipeline_input=True)
            _embed_elapsed = _time.perf_counter() - _embed_start
            _emb_per_sec = len(passages) / _embed_elapsed if _embed_elapsed > 0 else 0
            _per_emb_bytes = self._embedding_dimension * self._embedding_bytes_per_element
            _total_emb_mb = (len(passages) * _per_emb_bytes) / (1024 * 1024)
            _avg_seq_chars = total_chars // len(passages) if passages else 0
            print(f"Embedding generation: {len(passages):,} sequences in {_embed_elapsed:.2f}s")
            print(f"  Throughput   : {_emb_per_sec:,.0f} seq/sec")
            print(f"  Batch size   : {self._embedding_batch_size}")
            print(f"  Sequence size: ~{_avg_seq_chars} chars/seq (avg)")
            print(f"  Vector size  : {_per_emb_bytes} bytes  "
                  f"({self._embedding_dimension} dims × float32)")
            print(f"  Total data   : {_total_emb_mb:.1f} MB  "
                  f"({len(passages):,} × {_per_emb_bytes} bytes)")
            print(f"  Device       : {self._device.upper()}  "
                  f"(devices={self._num_embedding_devices})")
        
        # Train IVF index if needed (before adding any embeddings)
        if self._vector_index_method == "ivf" and not self._index.is_trained:
            self._train_vector_index(self._index, embeddings)
        
        # Track total indexing time for component metrics
        import time
        indexing_component_start = time.perf_counter()

        if self._indexing_instances:
            # ── Parallel FAISS indexing via pre-started NUMA-pinned workers ──────
            # Each worker builds a partial sub-index on its own NUMA node then
            # returns the serialized bytes; we merge them here in the main process.
            self._index_with_instances(passages, metadatas, embeddings)
            track_incremental = False   # no per-batch metrics when using instances
        else:
            # ── Single-process FAISS indexing (original path) ─────────────────
            # Determine FAISS indexing batch size.
            # HNSW/IVF: graph/cluster construction is most efficient as a
            # single bulk add() — all OMP threads work on one call.  Splitting
            # into small batches forces sequential graph-search sweeps through
            # the growing index, multiplying total work by ~N_batches.
            _approx_index = self._vector_index_method in ('hnsw', 'ivf')
            track_incremental = (self._benchmark and self._monitor
                                 and len(passages) >= 500
                                 and not _approx_index)
            if track_incremental:
                if self._faiss_indexing_batch_size > 0:
                    batch_size = min(self._faiss_indexing_batch_size, len(passages))
                else:
                    batch_size = max(1000, len(passages) // 10)  # auto: 10 batches, min 1000
                print(f"🔬 Incremental indexing analysis: {len(passages)} docs in batches of {batch_size}")
            else:
                batch_size = len(passages)  # Single batch
            
            # Process in batches
            for i in range(0, len(passages), batch_size):
                batch_end = min(i + batch_size, len(passages))
                self._ingest_single_batch(passages, metadatas, embeddings, i, batch_end, track_incremental)
        
        indexing_component_end = time.perf_counter()
        indexing_component_duration = indexing_component_end - indexing_component_start
        _idx_label = "parallel" if self._indexing_instances else "serial"
        print(f"Total FAISS indexing time ({_idx_label}): {indexing_component_duration:.2f}s")
        # Create component metrics for the entire indexing operation
        if self._indexing_instances and self._monitor:
            # Parallel indexing: build metrics from wall-clock time of merge + workers
            embedding_bytes = len(passages) * self._embedding_dimension * self._embedding_bytes_per_element
            from ingestion_monitor import ComponentMetrics
            self._monitor.components["faiss_indexing"] = ComponentMetrics(
                name="faiss_indexing",
                duration=indexing_component_duration,
                input_size_bytes=embedding_bytes,
                output_size_bytes=embedding_bytes,
                items_processed=len(passages),
                throughput_mb_per_sec=(embedding_bytes / (1024 * 1024)) / indexing_component_duration if indexing_component_duration > 0 else 0,
                throughput_items_per_sec=len(passages) / indexing_component_duration if indexing_component_duration > 0 else 0,
                is_pipeline_input=False,
                is_pipeline_output=True
            )
        elif not track_incremental:
            # For single batch, component was tracked inside _ingest_single_batch
            pass
        elif self._monitor:
            # For incremental, create component metrics here for the entire operation
            embedding_bytes = len(passages) * self._embedding_dimension * self._embedding_bytes_per_element
            
            from ingestion_monitor import ComponentMetrics
            self._monitor.components["faiss_indexing"] = ComponentMetrics(
                name="faiss_indexing",
                duration=indexing_component_duration,
                input_size_bytes=embedding_bytes,
                output_size_bytes=embedding_bytes,  # Vectors stored in FAISS index
                items_processed=len(passages),
                throughput_mb_per_sec=(embedding_bytes / (1024 * 1024)) / indexing_component_duration if indexing_component_duration > 0 else 0,
                throughput_items_per_sec=len(passages) / indexing_component_duration if indexing_component_duration > 0 else 0,
                is_pipeline_input=False,
                is_pipeline_output=True
            )
        
        # Store ingestion metrics for later reporting
        self._ingestion_start = ingestion_start
        self._ingestion_item_count = len(passages)
        self._ingestion_total_chars = total_chars
        
        # Save embeddings to cache 
        if self._load_embeddings and passages_path:
            self._save_embeddings_cache(embeddings, passages_path)

    def _ingest_single_batch(self, passages: List[str], metadatas: List[dict], embeddings: list,
                            batch_start: int, batch_end: int, track_incremental: bool):
        """Ingest a batch of passages. Can be used for single or incremental indexing.
        
        Args:
            passages: All passages
            metadatas: All metadata
            embeddings: All embeddings
            batch_start: Start index for this batch
            batch_end: End index for this batch (exclusive)
            track_incremental: Whether to track this batch for incremental analysis
        """
        import time
        
        # Extract batch data
        batch_passages = passages[batch_start:batch_end]
        batch_metadatas = metadatas[batch_start:batch_end] if metadatas else [{}] * (batch_end - batch_start)
        batch_embeddings = embeddings[batch_start:batch_end]
        
        # Track DB size before adding (for incremental tracking)
        db_size_before = len(self._doc_list) if track_incremental else 0
        
        # Calculate embedding size for this batch
        batch_embedding_bytes = len(batch_passages) * self._embedding_dimension * self._embedding_bytes_per_element
        
        # Time and execute indexing operation
        indexing_start = time.perf_counter()
        
        if track_incremental:
            # For incremental: just add embeddings without component tracking
            self._vector_store.add_embeddings(
                list(zip(batch_passages, batch_embeddings)), 
                batch_metadatas
            )
        else:
            # For single batch: use component tracking
            self._track_component("faiss_indexing", batch_embedding_bytes, len(batch_passages),
                                 lambda: self._vector_store.add_embeddings(
                                     list(zip(batch_passages, batch_embeddings)), batch_metadatas),
                                 is_pipeline_output=True)
        
        indexing_end = time.perf_counter()
        indexing_time = indexing_end - indexing_start
        _vec_per_sec = len(batch_passages) / indexing_time if indexing_time > 0 else 0
        _vec_bytes = self._embedding_dimension * self._embedding_bytes_per_element
        print(f"Indexed batch {batch_start}-{batch_end} ({len(batch_passages):,} vectors) in {indexing_time:.2f}s")
        print(f"  Throughput  : {_vec_per_sec:,.0f} vectors/sec")
        print(f"  Vector size : {_vec_bytes} bytes  ({self._embedding_dimension} dims × float32)")
        # Update document list
        self._doc_list.extend(batch_passages)
        
        # Track for incremental analysis if requested
        if track_incremental and self._monitor:
            self._monitor.track_incremental_indexing(
                db_size_before=db_size_before,
                batch_size=len(batch_passages),
                indexing_time=indexing_time
            )
    
    def lookup(self, query: str, k: int):
        """Retrieve top-k results.

        When --benchmark is active, query_embedding and vector_search are
        timed separately:
          query_embedding — GPU (XPU): YES, same model as ingestion embedding.
          vector_search   — GPU (XPU): NO, FAISS has no Intel XPU backend.
        """
        # Step 1: embed the query — use _query_embedding_batch_size
        _orig_bs = self._embedding_model.encode_kwargs.get('batch_size')
        self._embedding_model.encode_kwargs['batch_size'] = self._query_embedding_batch_size
        query_vector = self._time_op(
            "query_embedding",
            lambda: self._embedding_model.embed_query(query)
        )
        self._embedding_model.encode_kwargs['batch_size'] = _orig_bs
        # Step 2: FAISS nearest-neighbour search (CPU-only for XPU)
        results = self._time_op(
            "vector_search",
            lambda: self._vector_store.similarity_search_by_vector(query_vector, k=k)
        )
        return results

    def lookup_batch(self, queries: List[str], k: int) -> List[List]:
        """Batch embed + batch FAISS search for a list of queries.

        Embeds all queries in a single model call (the embedding model internally
        processes them in chunks of _embedding_batch_size) then performs one
        vectorised FAISS index.search() call.  This eliminates per-query kernel
        launch overhead for both the embedder and FAISS.

        Timing: _time_op_batch records total/n per slot so that
        print_retrieval_timings reports correct per-query averages.

        Returns:
            List[List[Document]] — one result list per input query.
        """
        n = len(queries)
        if n == 0:
            return []

        # Step 1: embed all queries — use _query_embedding_batch_size (may differ from ingestion BS)
        _orig_bs = self._embedding_model.encode_kwargs.get('batch_size')
        self._embedding_model.encode_kwargs['batch_size'] = self._query_embedding_batch_size
        query_vectors = self._time_op_batch(
            "query_embedding",
            lambda: self._embedding_model.embed_documents(queries),
            n
        )
        self._embedding_model.encode_kwargs['batch_size'] = _orig_bs

        # Step 2: single vectorised FAISS search over all query vectors
        def _faiss_batch():
            vectors_np = np.array(query_vectors, dtype=np.float32)
            _, indices_2d = self._vector_store.index.search(vectors_np, k)
            return indices_2d

        indices_2d = self._time_op_batch("vector_search", _faiss_batch, n)

        # Step 3: map flat FAISS indices → LangChain Document objects
        index_to_id = self._vector_store.index_to_docstore_id
        docstore = self._vector_store.docstore
        all_results = []
        for row_indices in indices_2d:
            docs = []
            for idx in row_indices:
                if idx == -1:
                    continue
                doc_id = index_to_id.get(int(idx))
                if doc_id:
                    doc = docstore.search(doc_id)
                    if doc and not isinstance(doc, str):
                        docs.append(doc)
            all_results.append(docs)
        return all_results

    def lookup_with_scores(self, query: str, k: int):
        """
        Lookup documents with similarity scores.
        Returns list of (document, score) tuples.

        Note: FAISS returns L2 distances (lower is better), but we convert to
        similarity scores (higher is better) for consistency with BM25.

        query_embedding — GPU (XPU): YES.
        vector_search   — GPU (XPU): NO (FAISS CPU-only for Intel XPU).
        """
        _orig_bs2 = self._embedding_model.encode_kwargs.get('batch_size')
        self._embedding_model.encode_kwargs['batch_size'] = self._query_embedding_batch_size
        query_vector = self._time_op(
            "query_embedding",
            lambda: self._embedding_model.embed_query(query)
        )
        self._embedding_model.encode_kwargs['batch_size'] = _orig_bs2
        results_with_scores = self._time_op(
            "vector_search",
            lambda: self._vector_store.similarity_search_with_score_by_vector(query_vector, k=k)
        )
        # FAISS returns L2 distance (lower is better); negate for consistent "higher is better"
        results_with_similarity = [(doc, -distance) for doc, distance in results_with_scores]
        return results_with_similarity

    def rerank(self, query: str, passages: List[str]):
        """Rerank with cross-encoder. GPU (XPU): YES — runs on self._device."""
        assert self._reranker_model is not None, "Reranker model not initialized"
        pairs = [[query, passage] for passage in passages]

        def _run():
            with torch.no_grad():
                inputs = self._reranker_tokenizer(
                    pairs, padding=True, return_tensors='pt',
                    truncation=True, max_length=512
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                scores = self._reranker_model(**inputs).logits.view(-1).float()
            scored = list(zip(passages, scores.cpu().tolist()))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [(p, s) for p, s in scored]

        return self._time_op("reranking", _run)


    def serialize(self, path: str):
        """Serialize FAISS index + docstore to disk.

        GPU (XPU): NO — pure file I/O (serializes the CPU FAISS index).
        """
        # Store path for output size calculation
        self._serialize_path = path
        start_serialization_time = time.time()
        def _write():
            data = self._vector_store.serialize_to_bytes()
            with open(path, "wb") as f:
                f.write(data)

        self._time_op("db_serialize", _write)

        # Report file size
        file_size_mb = os.path.getsize(path) / (1024 ** 2)
        end_serialization_time = time.time()
        print(f"Database saved in {end_serialization_time - start_serialization_time:.2f} seconds")
        print(f"VectorDB saved to {path} ({file_size_mb:.1f} MB)")

        # Update output size after serialization (now file exists)
        if self._benchmark and self._monitor:
            self._monitor.set_output_size_callback("faiss_indexing", self._calculate_index_output_size)
        
        # Report ingestion performance summary
        if self._benchmark and self._monitor and hasattr(self, '_ingestion_start'):
            db_type = "VectorDB (Incremental)" if hasattr(self._monitor, 'indexing_trend') and len(self._monitor.indexing_trend) > 0 else "VectorDB"
            self._report_performance(self._ingestion_start, self._ingestion_item_count,
                                    self._ingestion_total_chars, db_type)

    def from_serialized(self, path: str):
        """Load FAISS index + docstore from disk.

        GPU (XPU): NO — FAISS index stays on CPU. The embedding model (already
        loaded in __init__) does run on XPU for subsequent query embedding calls.
        """
        assert len(self._vector_store.index_to_docstore_id) == 0, "Vector store already has documents"
        file_size_mb = os.path.getsize(path) / (1024 ** 2)

        def _load():
            with open(path, "rb") as f:
                data = f.read()
            return FAISS.deserialize_from_bytes(
                embeddings=self._embedding_model,
                serialized=data,
                allow_dangerous_deserialization=True  # Only deserialize files you trust
            )

        self._vector_store = self._time_op("db_deserialize", _load)
        print(f"VectorDB loaded from {path} ({file_size_mb:.1f} MB)")

        # If it's an IVF index, restore nprobe setting
        if self._vector_index_method == "ivf" and hasattr(self._vector_store.index, 'nprobe'):
            self._vector_store.index.nprobe = self._ivf_nprobe
            print(f"Restored IVF index with nprobe={self._ivf_nprobe}")

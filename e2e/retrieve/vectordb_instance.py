"""VectorDBInstance — NUMA-pinned embedding worker.

CRITICAL DESIGN RULE:
  This module must have NO imports of faiss, torch, numpy, or any library
  that links to OpenMP at module level.

  Why: multiprocessing.spawn resolves the target function (_worker_main) by
  importing THIS module.  If any OMP-linked library is imported here, Intel
  OMP initialises in the child process with the PARENT's env vars (KMP_AFFINITY,
  OMP_NUM_THREADS) before _worker_main can override them — and OMP only reads
  those vars once, at initialization time.

  Correct call sequence:
    1  spawn imports this file          → no OMP init happens
    2  _worker_main() runs
    3  sets os.sched_setaffinity()      → CPU affinity pinned
    4  sets OMP_NUM_THREADS, KMP_AFFINITY env vars
    5  imports torch / faiss / langchain → OMP initialises NOW with correct vals
    6  signals READY via output_queue
    7  inference loop: recv chunk → embed → send result

  The main process calls:
    inst.start()         → spawns process, waits for READY
    inst.embed_async()   → sends (chunk_id, passages) to worker
    inst.get_result()    → blocks until (chunk_id, np.ndarray) arrives
    inst.stop()          → sends None sentinel, joins process
"""

import os
import multiprocessing as mp


# ── Hardcoded NUMA layouts ────────────────────────────────────────────────────
# sched_getaffinity() returns the Docker daemon's narrow default affinity
# inside privileged containers — not the true host topology.  Use hardcoded
# ranges that match the physical NUMA tile boundaries.

# Intel GNR-AP SNC-3: 6 NUMA nodes total (3 per socket).
# Using socket 1 (nodes 3-5)
#   node 3: CPUs 128-170 (43 cores)
#   node 4: CPUs 171-213 (43 cores)
#   node 5: CPUs 214-255 (42 cores)
INTEL_NUMA_LAYOUTS = {
    3: [(128, 170, 3),
        (171, 213, 4),
        (214, 255, 5)],
    2: [(128, 191, 3),
        (192, 255, 5)],
    1: [(128, 255, 3)],
}

# AMD Turin: 4 × 32-core groups, all memory on node 0
AMD_NUMA_LAYOUTS = {
    4: [(  0,  31, 0),
        ( 32,  63, 0),
        ( 64,  95, 0),
        ( 96, 127, 0)],
    3: [(  0,  42, 0),
        ( 43,  85, 0),
        ( 86, 127, 0)],
    2: [(  0,  63, 0),
        ( 64, 127, 0)],
    1: [(  0, 127, 0)],
}


def get_numa_layout(num_instances: int) -> list:
    """Return list of (cpu_start, cpu_end, mem_node) for this machine.

    Returns an empty list if num_instances is not in the known layouts,
    or if the layout's core ranges exceed the CPUs actually available to
    this process (e.g. cloud VMs with 32 vCPUs where the hardcoded ranges
    reference cores 128-255 that don't exist).
    """
    try:
        with open('/proc/cpuinfo') as _f:
            for _line in _f:
                if _line.startswith('vendor_id'):
                    _v = _line.split(':', 1)[1].strip().lower()
                    if 'intel' in _v:
                        return INTEL_NUMA_LAYOUTS.get(num_instances, [])
                    if 'amd' in _v:
                        return AMD_NUMA_LAYOUTS.get(num_instances, [])
                    break
    except Exception:
        pass
    return []


class VectorDBInstance:
    """One NUMA-pinned embedding worker running in its own process.

    Each instance owns an input queue and an output queue.  The worker process
    polls the input queue, embeds passages, and pushes results to the output
    queue.  Because the process is spawned with no OMP-linked imports in this
    module, it can set CPU affinity and OMP env vars before importing torch/
    faiss — guaranteeing that Intel OMP initialises with the correct per-worker
    thread count.

    Usage::

        layout = get_numa_layout(3)          # [(0,42,0), (43,85,1), (86,127,2)]
        instances = []
        for i, (s, e, node) in enumerate(layout):
            inst = VectorDBInstance(i, s, e, node, model_name, model_dtype, encode_kwargs)
            instances.append(inst)

        for inst in instances:
            inst.start()                     # spawns process, blocks until READY

        # Distribute work
        for i, chunk in enumerate(chunks):
            instances[i].embed_async(i, chunk)

        # Collect results (order may differ from send order)
        results = {}
        for inst in instances:
            chunk_id, arr = inst.get_result()
            results[chunk_id] = arr

        for inst in instances:
            inst.stop()
    """

    def __init__(self, instance_id: int, cpu_start: int, cpu_end: int,
                 mem_node: int, model_name: str, model_dtype: str,
                 encode_kwargs: dict):
        self.instance_id   = instance_id
        self.cpu_cores     = list(range(cpu_start, cpu_end + 1))
        self.mem_node      = mem_node
        self.model_name    = model_name
        self.model_dtype   = model_dtype
        self.encode_kwargs = encode_kwargs

        self._input_q  = None
        self._output_q = None
        self._process  = None

    def start(self):
        """Spawn the worker process and block until the model is loaded and ready."""
        # CRITICAL: queues MUST be created from the same context as the Process.
        # mp.Queue() uses the default 'fork' context on Linux; passing a fork-context
        # Queue to a spawn-context Process raises:
        #   RuntimeError: A SemLock created in a fork context is being shared with
        #                 a process in a spawn context.
        # Solution: create both queues and the Process from the same spawn context.
        ctx = mp.get_context('spawn')
        self._input_q  = ctx.Queue()
        self._output_q = ctx.Queue()
        self._process = ctx.Process(
            target=_worker_main,
            args=(
                self.instance_id,
                self.cpu_cores,
                self.mem_node,
                self.model_name,
                self.model_dtype,
                self.encode_kwargs,
                self._input_q,
                self._output_q,
            ),
            daemon=True,
        )
        self._process.start()
        # Block until READY signal (model loaded, worker looping)
        msg = self._output_q.get()
        if msg != 'READY':
            raise RuntimeError(
                f"[instance {self.instance_id}] unexpected startup message: {msg!r}"
            )
        print(f"[instance {self.instance_id}] ready "
              f"(pid={self._process.pid}, "
              f"cores={self.cpu_cores[0]}-{self.cpu_cores[-1]}, "
              f"mem_node={self.mem_node})",
              flush=True)

    def embed_async(self, chunk_id: int, passages: list):
        """Non-blocking: send a chunk of passages to this worker for embedding."""
        self._input_q.put((chunk_id, passages))

    def get_result(self):
        """Blocking: wait for and return (chunk_id, numpy.ndarray) from this worker."""
        return self._output_q.get()

    def stop(self):
        """Send stop sentinel and join the worker process."""
        self._input_q.put(None)
        if self._process is not None:
            self._process.join(timeout=30)
            if self._process.is_alive():
                self._process.terminate()


# ── Worker entry point ────────────────────────────────────────────────────────
# Must be module-level so multiprocessing.spawn can pickle the reference.
# NO heavy imports above this line — they live inside the function body.

def _worker_main(instance_id, cpu_cores, mem_node, model_name, model_dtype,
                 encode_kwargs, input_q, output_q):
    """Runs in a fresh spawned process.

    Execution order is critical:
      affinity → OMP env vars → import torch/faiss → model load → inference loop
    """
    import os as _os
    import time as _time

    # ── 1. CPU affinity ───────────────────────────────────────────────────────
    try:
        _os.sched_setaffinity(0, cpu_cores)
        _actual = sorted(_os.sched_getaffinity(0))
    except Exception as _e:
        print(f"[instance {instance_id}] sched_setaffinity warning: {_e}", flush=True)
        _actual = list(cpu_cores)

    # ── 2. OMP / MKL env vars — BEFORE any import of faiss / torch / numpy ───
    n = len(cpu_cores)
    _os.environ['OMP_NUM_THREADS']      = str(n)
    _os.environ['MKL_NUM_THREADS']      = str(n)
    _os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    _os.environ['KMP_BLOCKTIME']        = '1'

    # Use explicit proclist so Intel OMP knows exactly which physical cores to
    # use for THIS worker.  'disabled' was causing OMP to collapse to 1 thread
    # because without explicit placement guidance it does not spread across the
    # restricted sched_setaffinity mask.
    #
    # granularity=fine  → bind at the logical-processor (thread) level
    # explicit          → use the proclist below for thread-to-core mapping
    # proclist=[a-b]    → cores a through b (matches our sched_setaffinity range)
    # respect           → honour the existing sched_setaffinity process mask
    #
    # Each worker has a non-overlapping core range so there is no collision:
    #   worker 0 → proclist=[0-42]
    #   worker 1 → proclist=[43-85]
    #   worker 2 → proclist=[86-127]
    # compact,1,0 = place threads compactly (like single-instance does).
    # respect      = honour the sched_setaffinity mask set above, so threads
    #                stay within this worker's core range and don't collide
    #                with other workers.  This is exactly what the proven
    #                single-instance case does, but scoped to 43 cores.
    _kmp = 'granularity=fine,compact,1,0,respect'
    _os.environ['KMP_AFFINITY'] = _kmp

    print(
        f"[instance {instance_id}] pid={_os.getpid()}  "
        f"cores={_actual[0]}-{_actual[-1]}  "
        f"OMP_NUM_THREADS={n}  KMP_AFFINITY={_kmp}",
        flush=True,
    )

    # ── 3. Optional NUMA memory binding ──────────────────────────────────────
    if mem_node >= 0:
        try:
            import ctypes as _ct
            _libnuma = _ct.CDLL('libnuma.so.1', use_errno=True)
            _libnuma.numa_set_preferred(_ct.c_int(mem_node))
        except Exception:
            pass  # libnuma unavailable — CPU affinity still steers first-touch

    # ── 4. Heavy imports happen here — OMP initialises with correct env vars ──
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)

    _dtype = getattr(torch, model_dtype, torch.bfloat16)
    print(
        f"[instance {instance_id}] torch.get_num_threads()={torch.get_num_threads()}  "
        f"loading model {model_name} ...",
        flush=True,
    )

    # ── 5. Load model ─────────────────────────────────────────────────────────
    _t0 = _time.perf_counter()
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs=encode_kwargs,
    )
    embedder._client.to(_dtype)
    _first_param = next(embedder._client.parameters())
    print(
        f"[instance {instance_id}] model loaded in {_time.perf_counter() - _t0:.1f}s  "
        f"dtype={_first_param.dtype}  device={_first_param.device}  "
        f"threads={torch.get_num_threads()}",
        flush=True,
    )

    # ── 6. Signal ready ───────────────────────────────────────────────────────
    output_q.put('READY')

    # ── 7. Inference loop ─────────────────────────────────────────────────────
    import numpy as _np
    while True:
        item = input_q.get()
        if item is None:       # stop sentinel from inst.stop()
            break
        chunk_id, passages = item
        _t1 = _time.perf_counter()
        embeddings = embedder.embed_documents(passages)
        arr = _np.array(embeddings, dtype=_np.float32)
        elapsed = _time.perf_counter() - _t1
        print(
            f"[instance {instance_id}] chunk {chunk_id}: "
            f"{len(passages)} passages in {elapsed:.1f}s",
            flush=True,
        )
        output_q.put((chunk_id, arr))


# ─────────────────────────────────────────────────────────────────────────────
# FAISSIndexInstance — NUMA-pinned parallel FAISS sub-index builder
# ─────────────────────────────────────────────────────────────────────────────

class FAISSIndexInstance:
    """One NUMA-pinned FAISS indexing worker running in its own process.

    Receives pre-computed embeddings (as raw bytes) + passages, builds a
    partial FAISS sub-index — same OMP discipline as VectorDBInstance:
    no faiss/numpy imported at module level, so the worker sets CPU affinity
    and OMP env vars BEFORE importing faiss, guaranteeing correct thread counts.

    Usage::

        layout = get_numa_layout(3)
        instances = [
            FAISSIndexInstance(i, s, e, node, method='flat')
            for i, (s, e, node) in enumerate(layout)
        ]
        for inst in instances:
            inst.start()

        # Distribute work (passages and numpy embedding chunks):
        for i, inst in enumerate(instances):
            inst.index_async(i, passages_chunk[i], metadatas_chunk[i], embeddings_chunk[i])

        results = {}
        for inst in instances:
            chunk_id, faiss_bytes, docs, idx_to_docstore = inst.get_result()
            results[chunk_id] = (faiss_bytes, docs, idx_to_docstore)

        for inst in instances:
            inst.stop()
    """

    def __init__(self, instance_id: int, cpu_start: int, cpu_end: int,
                 mem_node: int, method: str = 'flat', index_params: dict = None):
        self.instance_id  = instance_id
        self.cpu_cores    = list(range(cpu_start, cpu_end + 1))
        self.mem_node     = mem_node
        self.method       = method
        self.index_params = index_params or {}

        self._input_q  = None
        self._output_q = None
        self._process  = None

    def start(self):
        """Spawn the worker process and block until it signals READY."""
        ctx = mp.get_context('spawn')
        self._input_q  = ctx.Queue()
        self._output_q = ctx.Queue()
        self._process  = ctx.Process(
            target=_faiss_index_worker,
            args=(
                self.instance_id,
                self.cpu_cores,
                self.mem_node,
                self.method,
                self.index_params,
                self._input_q,
                self._output_q,
            ),
            daemon=True,
        )
        self._process.start()
        msg = self._output_q.get()
        if msg != 'READY':
            raise RuntimeError(
                f"[faiss_instance {self.instance_id}] unexpected startup message: {msg!r}"
            )
        print(f"[faiss_instance {self.instance_id}] ready "
              f"(pid={self._process.pid}, "
              f"cores={self.cpu_cores[0]}-{self.cpu_cores[-1]}, "
              f"mem_node={self.mem_node})",
              flush=True)

    def index_async(self, chunk_id: int, passages: list, metadatas: list,
                    embeddings_np):
        """Non-blocking: send passages + embeddings to this worker for indexing.

        embeddings_np must be a numpy array of shape (n_vecs, dim) float32.
        """
        import numpy as _np
        emb = _np.asarray(embeddings_np, dtype=_np.float32)
        # Send raw bytes to avoid numpy pickle header overhead on large arrays.
        self._input_q.put((chunk_id, passages, metadatas, emb.tobytes(), emb.shape))

    def get_result(self):
        """Blocking: wait for (chunk_id, faiss_bytes, docs, idx_to_docstore)."""
        return self._output_q.get()

    def stop(self):
        """Send stop sentinel and join the worker process."""
        self._input_q.put(None)
        if self._process is not None:
            self._process.join(timeout=30)
            if self._process.is_alive():
                self._process.terminate()


# ── FAISS index worker entry point ────────────────────────────────────────────
# Must be module-level so multiprocessing.spawn can pickle the reference.
# NO heavy imports above this line.

def _faiss_index_worker(instance_id, cpu_cores, mem_node, method, index_params,
                        input_q, output_q):
    """Runs in a fresh spawned process. Builds a partial FAISS sub-index.

    Execution order is critical:
      affinity → OMP env vars → import faiss/numpy → index loop
    """
    import os as _os
    import time as _time

    # ── 1. CPU affinity ───────────────────────────────────────────────────────
    try:
        _os.sched_setaffinity(0, cpu_cores)
        _actual = sorted(_os.sched_getaffinity(0))
    except Exception as _e:
        print(f"[faiss_instance {instance_id}] sched_setaffinity warning: {_e}", flush=True)
        _actual = list(cpu_cores)

    # ── 2. OMP / MKL env vars — BEFORE any import of faiss / numpy ───────────
    n = len(cpu_cores)
    _os.environ['OMP_NUM_THREADS']      = str(n)
    _os.environ['MKL_NUM_THREADS']      = str(n)
    _os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    _os.environ['KMP_BLOCKTIME']        = '1'
    _kmp = 'granularity=fine,compact,1,0,respect'
    _os.environ['KMP_AFFINITY'] = _kmp

    print(
        f"[faiss_instance {instance_id}] pid={_os.getpid()}  "
        f"cores={_actual[0]}-{_actual[-1]}  "
        f"OMP_NUM_THREADS={n}  KMP_AFFINITY={_kmp}",
        flush=True,
    )

    # ── 3. Optional NUMA memory binding ──────────────────────────────────────
    if mem_node >= 0:
        try:
            import ctypes as _ct
            _libnuma = _ct.CDLL('libnuma.so.1', use_errno=True)
            _libnuma.numa_set_preferred(_ct.c_int(mem_node))
        except Exception:
            pass  # libnuma unavailable — CPU affinity steers first-touch allocation

    # ── 4. Heavy imports — OMP initialises with correct env vars ─────────────
    import faiss as _faiss
    import numpy as _np
    import uuid as _uuid
    from langchain_core.documents import Document as _Document

    print(
        f"[faiss_instance {instance_id}] faiss loaded  "
        f"omp_max_threads={_faiss.omp_get_max_threads()}",
        flush=True,
    )

    # ── 5. Signal ready ───────────────────────────────────────────────────────
    output_q.put('READY')

    # ── 6. Index loop ─────────────────────────────────────────────────────────
    while True:
        item = input_q.get()
        if item is None:       # stop sentinel from inst.stop()
            break

        chunk_id, passages, metadatas, emb_bytes, emb_shape = item
        _t0 = _time.perf_counter()

        # Reconstruct numpy array from raw bytes (no pickle header overhead)
        emb_array = _np.frombuffer(emb_bytes, dtype=_np.float32).reshape(emb_shape)
        n_vecs, dim = emb_shape

        # Build raw FAISS sub-index according to requested method
        if method == 'flat':
            raw_idx = _faiss.IndexFlatL2(dim)
        elif method == 'hnsw':
            _M = index_params.get('M', 32)
            raw_idx = _faiss.IndexHNSWFlat(dim, _M)
            raw_idx.hnsw.efConstruction = index_params.get('efConstruction', 40)
            raw_idx.hnsw.efSearch       = index_params.get('efSearch', 16)
        elif method == 'ivf':
            _nlist = index_params.get('nlist', 100)
            _q     = _faiss.IndexFlatL2(dim)
            raw_idx = _faiss.IndexIVFFlat(_q, dim, _nlist)
            raw_idx.train(emb_array)
            raw_idx.nprobe = index_params.get('nprobe', 10)
        else:
            raw_idx = _faiss.IndexFlatL2(dim)

        # Add vectors in batches — for HNSW this improves L3 cache reuse
        # during graph construction; for flat it's a no-op performance-wise.
        _add_bs = index_params.get('add_batch_size', 0)  # 0 = single call
        print(
            f"[faiss_instance {instance_id}] indexing {n_vecs} vectors  "
            f"method={method}  add_batch_size={_add_bs if _add_bs > 0 else 'full'}",
            flush=True,
        )
        if _add_bs > 0 and n_vecs > _add_bs:
            for _b in range(0, n_vecs, _add_bs):
                raw_idx.add(emb_array[_b:_b + _add_bs])
        else:
            raw_idx.add(emb_array)

        # Build docstore + index-to-docstore mappings
        ids = [str(_uuid.uuid4()) for _ in passages]
        idx_to_docstore = {i: uid for i, uid in enumerate(ids)}
        _metas = metadatas if metadatas else [{}] * n_vecs
        docs = {
            uid: _Document(page_content=p, metadata=m if m else {})
            for uid, p, m in zip(ids, passages, _metas)
        }

        # Serialize FAISS index to bytes for IPC
        faiss_bytes = bytes(_faiss.serialize_index(raw_idx))

        elapsed = _time.perf_counter() - _t0
        print(
            f"[faiss_instance {instance_id}] chunk {chunk_id}: "
            f"{n_vecs} vectors indexed in {elapsed:.1f}s",
            flush=True,
        )
        output_q.put((chunk_id, faiss_bytes, docs, idx_to_docstore))


# ─────────────────────────────────────────────────────────────────────────────
# RerankInstance — NUMA-pinned parallel cross-encoder reranking worker
# ─────────────────────────────────────────────────────────────────────────────

class RerankInstance:
    """One NUMA-pinned cross-encoder reranking worker running in its own process.

    Follows the exact same OMP discipline as VectorDBInstance: no torch/
    transformers imported at module level, so the worker sets CPU affinity
    and OMP env vars BEFORE importing torch — guaranteeing that Intel OMP
    initialises with the correct per-NUMA-node thread count.

    Usage::

        layout = get_numa_layout(3)
        instances = [
            RerankInstance(i, s, e, node, model_name, model_dtype='bfloat16')
            for i, (s, e, node) in enumerate(layout)
        ]
        for inst in instances:
            inst.start()                     # spawns process, blocks until READY

        # Distribute queries across instances
        chunk_size = (len(queries) + len(instances) - 1) // len(instances)
        for i, inst in enumerate(instances):
            s = i * chunk_size
            e = min(s + chunk_size, len(queries))
            if s < len(queries):
                inst.rerank_async(i, queries[s:e], passages_list[s:e])

        # Collect results
        results = {}
        for i, inst in enumerate(instances):
            chunk_id, scored_lists = inst.get_result()
            results[chunk_id] = scored_lists

        for inst in instances:
            inst.stop()
    """

    def __init__(self, instance_id: int, cpu_start: int, cpu_end: int,
                 mem_node: int, model_name: str, model_dtype: str = 'bfloat16',
                 batch_size: int = 256):
        self.instance_id = instance_id
        self.cpu_cores   = list(range(cpu_start, cpu_end + 1))
        self.mem_node    = mem_node
        self.model_name  = model_name
        self.model_dtype = model_dtype
        self.batch_size  = batch_size

        self._input_q  = None
        self._output_q = None
        self._process  = None

    def start(self):
        """Spawn the worker process and block until the model is loaded and ready."""
        ctx = mp.get_context('spawn')
        self._input_q  = ctx.Queue()
        self._output_q = ctx.Queue()
        self._process  = ctx.Process(
            target=_rerank_worker,
            args=(
                self.instance_id,
                self.cpu_cores,
                self.mem_node,
                self.model_name,
                self.model_dtype,
                self.batch_size,
                self._input_q,
                self._output_q,
            ),
            daemon=True,
        )
        self._process.start()
        msg = self._output_q.get()
        if msg != 'READY':
            raise RuntimeError(
                f"[rerank_instance {self.instance_id}] unexpected startup message: {msg!r}"
            )
        print(f"[rerank_instance {self.instance_id}] ready "
              f"(pid={self._process.pid}, "
              f"cores={self.cpu_cores[0]}-{self.cpu_cores[-1]}, "
              f"mem_node={self.mem_node})",
              flush=True)

    def rerank_async(self, chunk_id: int, queries: list, passages_list: list):
        """Non-blocking: send a query/passage chunk to this worker for reranking."""
        self._input_q.put((chunk_id, queries, passages_list))

    def get_result(self):
        """Blocking: return (chunk_id, List[List[Tuple[str, float]]]) from this worker."""
        return self._output_q.get()

    def stop(self):
        """Send stop sentinel and join the worker process."""
        self._input_q.put(None)
        if self._process is not None:
            self._process.join(timeout=30)
            if self._process.is_alive():
                self._process.terminate()


# ── Rerank worker entry point ─────────────────────────────────────────────────
# Must be module-level so multiprocessing.spawn can pickle the reference.
# NO heavy imports above this line.

def _rerank_worker(instance_id, cpu_cores, mem_node, model_name, model_dtype,
                   batch_size, input_q, output_q):
    """Runs in a fresh spawned process.  Cross-encoder reranking loop.

    Execution order is critical:
      affinity → OMP env vars → import torch/transformers → model load → loop
    """
    import os as _os
    import time as _time

    # ── 1. CPU affinity ───────────────────────────────────────────────────────
    try:
        _os.sched_setaffinity(0, cpu_cores)
        _actual = sorted(_os.sched_getaffinity(0))
    except Exception as _e:
        print(f"[rerank_instance {instance_id}] sched_setaffinity warning: {_e}", flush=True)
        _actual = list(cpu_cores)

    # ── 2. OMP / MKL env vars — BEFORE any import of torch / transformers ─────
    n = len(cpu_cores)
    _os.environ['OMP_NUM_THREADS']      = str(n)
    _os.environ['MKL_NUM_THREADS']      = str(n)
    _os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    _os.environ['KMP_BLOCKTIME']        = '1'
    _kmp = 'granularity=fine,compact,1,0,respect'
    _os.environ['KMP_AFFINITY'] = _kmp

    print(
        f"[rerank_instance {instance_id}] pid={_os.getpid()}  "
        f"cores={_actual[0]}-{_actual[-1]}  "
        f"OMP_NUM_THREADS={n}  KMP_AFFINITY={_kmp}",
        flush=True,
    )

    # ── 3. Optional NUMA memory binding ──────────────────────────────────────
    if mem_node >= 0:
        try:
            import ctypes as _ct
            _libnuma = _ct.CDLL('libnuma.so.1', use_errno=True)
            _libnuma.numa_set_preferred(_ct.c_int(mem_node))
        except Exception:
            pass  # libnuma unavailable — CPU affinity still steers first-touch

    # ── 4. Heavy imports — OMP initialises with correct env vars ─────────────
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)

    _dtype = getattr(torch, model_dtype, torch.bfloat16)
    print(
        f"[rerank_instance {instance_id}] torch.get_num_threads()={torch.get_num_threads()}  "
        f"loading reranker {model_name} ...",
        flush=True,
    )

    # ── 5. Load model ─────────────────────────────────────────────────────────
    _t0 = _time.perf_counter()
    _model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=_dtype)
    _model = _model.to('cpu')
    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _first_param = next(_model.parameters())
    print(
        f"[rerank_instance {instance_id}] reranker loaded in {_time.perf_counter() - _t0:.1f}s  "
        f"dtype={_first_param.dtype}  device={_first_param.device}  "
        f"threads={torch.get_num_threads()}",
        flush=True,
    )

    # ── 6. Signal ready ───────────────────────────────────────────────────────
    output_q.put('READY')

    # ── 7. Reranking loop ─────────────────────────────────────────────────────
    while True:
        item = input_q.get()
        if item is None:       # stop sentinel from inst.stop()
            break

        chunk_id, queries, passages_list = item
        _t1 = _time.perf_counter()

        # Build flat (query, passage) pairs with per-query offsets
        flat_pairs = []
        offsets = []   # (start_idx, length) per query
        for _query, _passages in zip(queries, passages_list):
            offsets.append((len(flat_pairs), len(_passages)))
            for _p in _passages:
                flat_pairs.append([_query, _p])

        n_pairs = len(flat_pairs)
        all_scores = []
        for _start in range(0, n_pairs, batch_size):
            _chunk = flat_pairs[_start:_start + batch_size]
            with torch.no_grad():
                _inputs = _tokenizer(
                    _chunk, padding=True, return_tensors='pt',
                    truncation=True, max_length=512
                )
                _scores = _model(**_inputs).logits.view(-1).float()
            all_scores.extend(_scores.cpu().tolist())

        # Reassemble per-query scored+sorted lists
        results = []
        for (_off, _len), _passages in zip(offsets, passages_list):
            _scored = list(zip(_passages, all_scores[_off:_off + _len]))
            _scored.sort(key=lambda x: x[1], reverse=True)
            results.append(_scored)

        elapsed = _time.perf_counter() - _t1
        print(
            f"[rerank_instance {instance_id}] chunk {chunk_id}: "
            f"{len(queries)} queries / {n_pairs} pairs in {elapsed:.2f}s",
            flush=True,
        )
        output_q.put((chunk_id, results))

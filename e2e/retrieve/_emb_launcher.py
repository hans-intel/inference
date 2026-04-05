"""Lightweight bootstrap for NUMA-pinned embedding workers.

IMPORTANT: This module must NOT import faiss, torch, numpy, or any library
that links to OpenMP at module level.  multiprocessing.spawn resolves the
target function by importing its module — if this module triggered an OMP
init, the child's env-var overrides would be ignored because Intel OMP
reads KMP_AFFINITY / OMP_NUM_THREADS only once at initialization.

Sequence:
  1. spawn unpickles target → imports this file (no OMP init happens)
  2. bootstrap() sets sched_setaffinity + OMP/KMP env vars
  3. bootstrap() imports retrieve.vectordb → 'import faiss' triggers OMP init
     NOW with the correct per-worker env vars already in place
  4. Calls _parallel_embed_worker() which loads model & runs inference
"""

def bootstrap(result_queue, cpu_cores_csv, mem_node, worker_args_file):
    """Pin CPU affinity, set OMP env vars, then import & call the real worker.

    Args:
        result_queue:  multiprocessing.Queue passed directly (can't be pickled to disk)
        cpu_cores_csv: comma-separated core ids (e.g. "0,1,2,...,42") or empty string
        mem_node:      NUMA memory node (int) or -1 for no binding
        worker_args_file: path to pickle file containing a dict of kwargs
                          for _parallel_embed_worker (result_queue is injected separately)
    """
    import os

    # ── 1. CPU affinity + OMP env vars BEFORE any heavy import ──────────────
    if cpu_cores_csv:
        cores = [int(c) for c in cpu_cores_csv.split(',')]
        try:
            os.sched_setaffinity(0, cores)
        except Exception as _e:
            print(f"[launcher] sched_setaffinity warning: {_e}", flush=True)
        n = len(cores)
        os.environ['OMP_NUM_THREADS']      = str(n)
        os.environ['MKL_NUM_THREADS']      = str(n)
        os.environ['OPENBLAS_NUM_THREADS'] = str(n)
        # MUST be 'disabled' so Intel OMP uses the sched_setaffinity mask
        # instead of trying absolute core indices (which conflict across workers)
        os.environ['KMP_AFFINITY']         = 'disabled'
        os.environ['KMP_BLOCKTIME']        = '1'
        print(f"[launcher] affinity set: {len(cores)} cores "
              f"({cores[0]}-{cores[-1]}) OMP={n} KMP_AFFINITY=disabled",
              flush=True)

    if mem_node >= 0:
        try:
            import ctypes
            _libnuma = ctypes.CDLL('libnuma.so.1', use_errno=True)
            _libnuma.numa_set_preferred(ctypes.c_int(mem_node))
        except Exception:
            pass

    # ── 2. NOW import the real worker (triggers faiss/torch/OMP init) ───────
    import pickle
    with open(worker_args_file, 'rb') as f:
        kwargs = pickle.load(f)

    # Inject the result_queue (can't be in the pickle file - Queue is not file-picklable)
    kwargs['result_queue'] = result_queue

    from retrieve.vectordb import _parallel_embed_worker
    _parallel_embed_worker(**kwargs)

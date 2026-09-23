"""A process pool for independent optimisation jobs, with the failure modes handled.

Every point in the convergence and stage grids is independent, so the work is
embarrassingly parallel. The three things that actually go wrong, all met earlier
in this project, are handled here rather than rediscovered at each call site:

* **The pool may not start at all.** A child that cannot import ``qllm``, or a
  fork-after-threads deadlock, otherwise produces one warning per job and then
  grinds through every one of them singly. A preflight ping proves the pool works
  before any real work is committed, and reports the true cause once.
* **Workers get OS-killed, almost always by the OOM reaper.** A deep circuit's
  autograd graph is large, and N of them at once is N times larger. A killed
  worker's jobs are not lost: they stay pending and are retried with the worker
  count halved, repeatedly, down to running in this process. The pool is fully
  reaped between attempts so a dead worker's memory is reclaimed before the
  retry, instead of the survivors piling onto it.
* **Thread oversubscription.** Torch defaults to all cores per process, so N
  workers each spawn N threads and spend their time context switching. The
  budget is split across the workers.

``forkserver`` is preferred over ``fork`` (which deadlocks after OpenMP has
started threads) and over ``spawn`` (which re-imports everything per worker).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def _ping():
    import qllm  # noqa: F401  -- proves the child can import the package
    return True


def _context():
    for method in ("forkserver", "spawn"):
        try:
            return mp.get_context(method), method
        except ValueError:                                  # pragma: no cover
            continue
    return mp.get_context(), "default"


def run_jobs(fn, jobs, n_jobs=1, desc="job", on_result=None, log=print):
    """Run ``fn(job)`` over ``jobs``, yielding results as they complete.

    ``fn`` must be importable from the worker (a module-level function) and
    ``jobs`` must pickle. Returns the list of results. ``on_result(job, result)``
    is called in THIS process as each completes, which is where a caller should
    write its checkpoint -- workers must never share a file handle.
    """
    jobs = list(jobs)
    if not jobs:
        return []
    n_jobs = int(n_jobs)
    if n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, len(jobs)))
    out = []

    def _emit(job, res):
        out.append(res)
        if on_result:
            on_result(job, res)

    if n_jobs == 1:
        for job in jobs:
            _emit(job, fn(job))
        return out

    ctx, method = _context()
    pending = list(range(len(jobs)))
    workers = n_jobs
    first = True
    while pending:
        n = max(1, min(workers, len(pending)))
        threads = max(1, (os.cpu_count() or 1) // n)
        _limit_env(threads)          # before any worker is forked, not inside it
        kw = {"max_workers": n, "mp_context": ctx}
        try:                       # bound peak memory: recycle workers regularly
            ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1)
            kw["max_tasks_per_child"] = 4 if n > 1 else 1
        except TypeError:                                   # pragma: no cover
            pass
        left = set(pending)
        ex = ProcessPoolExecutor(**kw)
        try:
            if first:
                try:
                    ex.submit(_ping).result(timeout=180)
                except Exception as exc:                    # noqa: BLE001
                    log(f"  [warn] parallel workers could not start "
                        f"({type(exc).__name__}: {str(exc)[:160]}); running "
                        f"{len(pending)} {desc}(s) in this process instead.")
                    ex.shutdown(wait=True, cancel_futures=True)
                    for i in pending:
                        _emit(jobs[i], fn(jobs[i]))
                    return out
                first = False
            log(f"  running {len(pending)} {desc}(s) across {n} process(es), "
                f"{threads} thread(s) each ({method})")
            futs = {ex.submit(fn, _with_threads(jobs[i], threads)): i
                    for i in pending}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    res = fut.result()
                except Exception:              # noqa: BLE001  worker OS-killed
                    continue                   # stays pending; retried smaller
                left.discard(i)
                _emit(jobs[i], res)
        finally:
            # Fully reap before the next, smaller attempt, so a dead worker's
            # memory is reclaimed rather than inherited by the retry.
            ex.shutdown(wait=True, cancel_futures=True)
        pending = sorted(left)
        if not pending:
            break
        if workers <= 1:
            log(f"  [warn] {len(pending)} {desc}(s) still undelivered at one "
                f"worker; running them in this process.")
            for i in pending:
                _emit(jobs[i], fn(jobs[i]))
            break
        workers = max(1, workers // 2)
        log(f"  [warn] {len(pending)} {desc}(s) lost (workers killed, usually "
            f"OOM); retrying with {workers} worker(s).")
    return out


def _with_threads(job, threads):
    """Attach the per-worker torch thread budget to a job payload."""
    if isinstance(job, dict):
        j = dict(job)
        j["_threads"] = threads
        return j
    return job


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _limit_env(threads):
    """Cap every BLAS runtime's thread pool, for children that inherit this env.

    ``torch.set_num_threads`` governs torch's own OpenMP pool and nothing else.
    OpenBLAS -- which numpy and some torch builds sit on -- sizes its pool from
    these variables when the library loads, so setting them after the import has
    no effect. The pool therefore sets them in the PARENT, before any worker is
    forked, so each child picks them up as it imports.

    Without this, ``--jobs 8`` on an 8-core box gives each of 8 workers a torch
    budget of one thread and an OpenBLAS pool of eight: 64 threads competing for
    8 cores. That is the state in which an OpenBLAS built against pthreads inside
    an OpenMP application prints "Detect OpenMP Loop and this application may
    hang" -- and occasionally does.

    An explicit setting from the environment is left alone: the user meant it.
    """
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(max(1, int(threads))))


def apply_threads(job):
    """Called at the top of a worker: honour the thread budget the pool set."""
    n = (job or {}).get("_threads") if isinstance(job, dict) else None
    if n:
        import torch
        torch.set_num_threads(max(1, int(n)))
        _limit_env(n)

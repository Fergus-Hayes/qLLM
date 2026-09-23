"""Parallel-disentangling smoke test (separate file: needs a guarded __main__).

The hybrid sweep runs independent (layer, D, k) disentanglings across processes
with ``--jobs``. That uses the ``forkserver`` start method (never ``fork``, which
deadlocks after torch spins up OpenMP threads), and forkserver re-imports the
entry module -- so this must live behind ``if __name__ == "__main__"`` rather than
in the flat ``smoke_hybrid.py`` script.

A preflight ping job runs first; if the pool can't start the sweep drops to a
single-process fallback, and if a worker dies mid-run the remaining jobs finish
sequentially. Verifies that ``--jobs 2`` produces the same grid as ``--jobs 1``
(bit-identical relative errors are not expected: worker threads differ, so the
non-convex SVD lands FP-different optima) and that ``--max-total-params 0`` caps
each layer at its own ``params_original``.
"""
import os
import shutil
import sys

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

import qllm.benchmark as B  # noqa: E402
import qllm.hybrid_sweep as H  # noqa: E402

SCRATCH = os.environ.get("QLLM_SMOKE_DIR", "/tmp/qllm-smoke-parallel")


def _blas_env(job):
    """Module-level so forkserver can import it: report the child's BLAS caps."""
    from qllm.parallel import apply_threads
    apply_threads(job)
    return {v: os.environ.get(v) for v in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


def _check_thread_caps():
    """Workers must inherit a BLAS thread cap, not just a torch one.

    ``torch.set_num_threads`` governs torch's OpenMP pool and nothing else, while
    OpenBLAS sizes its own pool from the environment AT IMPORT. Set it only inside
    the worker and it is already too late, so ``--jobs 8`` on 8 cores would give
    64 threads on 8 cores -- the oversubscription behind OpenBLAS's "Detect OpenMP
    Loop and this application may hang".
    """
    from qllm.parallel import _THREAD_VARS, _limit_env, run_jobs
    saved = {v: os.environ.get(v) for v in _THREAD_VARS}
    try:
        for v in _THREAD_VARS:
            os.environ.pop(v, None)
        got = run_jobs(_blas_env, [{"i": i} for i in range(4)], 2, "blas-probe",
                       log=lambda _m: None)
        assert got, "no results from the probe pool"
        for g in got:
            assert g["OMP_NUM_THREADS"] and int(g["OMP_NUM_THREADS"]) >= 1, g
            assert g["OPENBLAS_NUM_THREADS"] == g["OMP_NUM_THREADS"], g
        # An explicit setting is the user's, and must survive untouched.
        os.environ["OPENBLAS_NUM_THREADS"] = "7"
        _limit_env(1)
        assert os.environ["OPENBLAS_NUM_THREADS"] == "7", "clobbered a user setting"
        return got[0]
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def _run(njobs, tag):
    shutil.rmtree(SCRATCH + tag, ignore_errors=True)
    cfg = H.HybridConfig(
        model_id="tiny-gpt2", device="cpu", dtype="float32",
        include_pattern=r"h\.0\.attn\.c_proj", exclude_pattern=r"(wte|wpe|lm_head|ln|bias)",
        tensorization="qubit", chi_values=[1, 2, 4, 16, 64], circuit_depths=[1, 4, 16],
        gate_sizes=[2, 3], num_depths=1, disentangle_sweeps=6,
        disentangle_optimizer="explicit", measure_perplexity=False,
        max_total_params=0, n_jobs=njobs, results_dir=SCRATCH + tag, make_plots=False)
    return H._read_rows(H.run_hybrid_sweep(cfg))


def main():
    import torch
    model = GPT2LMHeadModel(GPT2Config(vocab_size=512, n_positions=64, n_embd=48,
                                       n_layer=1, n_head=4))

    class _Tok:
        def decode(self, ids):
            return ""

    B.load_model_and_tokenizer = H.load_model_and_tokenizer = lambda c, d: (model.to(d), _Tok())
    # Warm up torch so OpenMP threads are live -- this is exactly the state that
    # deadlocks a naive 'fork' pool; the forkserver path must survive it.
    for _ in range(3):
        a = torch.randn(256, 256)
        torch.linalg.svd(a @ a)

    caps = _check_thread_caps()
    print(f"  worker BLAS caps inherited: {caps}")

    seq = _run(1, "-seq")
    par = _run(2, "-par")

    def key(r):
        return (r["param_name"], r["method"], r["chi"], r["circuit_depth"], r["gate_size"])

    assert {key(r) for r in seq} == {key(r) for r in par}, "parallel grid != sequential grid"
    assert len(seq) == len(par), f"row count differs: {len(seq)} vs {len(par)}"
    hy = [r for r in par if r["method"] == "hybrid"]
    kept = [r for r in hy if r["relative_error"] not in ("", "nan")]
    skip = [r for r in hy if r["relative_error"] in ("", "nan")]
    assert kept and skip, "cap=0 must keep some and skip others"
    for r in kept:
        assert int(r["total_params"]) <= int(r["params_original"]), \
            f"cap=0 kept a row over params_original: {r['total_params']}"
    for r in skip:
        assert int(r["total_params"]) > int(r["params_original"])
    print(f"\n  --jobs 2 grid == --jobs 1 grid ({len(par)} rows); cap=0 kept {len(kept)}, "
          f"skipped {len(skip)}")

    # Fallback: kill some workers mid-run (simulated OOM) -- the sweep must still
    # complete every point via the sequential fallback.
    os.environ["QLLM_TEST_KILL_JOBS"] = "0,2"
    try:
        fb = _run(2, "-fallback")
    finally:
        del os.environ["QLLM_TEST_KILL_JOBS"]
    assert {key(r) for r in fb} == {key(r) for r in seq}, "fallback lost/added grid points"
    fb_kept = [r for r in fb if r["method"] == "hybrid" and r["relative_error"] not in ("", "nan")]
    assert len(fb_kept) == len(kept), \
        f"fallback did not finish every disentangle: {len(fb_kept)} vs {len(kept)}"
    print(f"  worker-death fallback: all {len(fb)} rows completed ({len(fb_kept)} disentangled)")

    # Preflight failure: the pool can't start at all (every worker dies at bootstrap,
    # exactly the user's Python 3.12 spawn case). The preflight ping must catch it
    # once and complete every point sequentially -- no per-job warning storm.
    os.environ["QLLM_TEST_KILL_PING"] = "1"
    try:
        pf = _run(2, "-preflight")
    finally:
        del os.environ["QLLM_TEST_KILL_PING"]
    assert {key(r) for r in pf} == {key(r) for r in seq}, "preflight fallback lost/added grid points"
    pf_kept = [r for r in pf if r["method"] == "hybrid" and r["relative_error"] not in ("", "nan")]
    assert len(pf_kept) == len(kept), \
        f"preflight fallback did not finish every disentangle: {len(pf_kept)} vs {len(kept)}"
    print(f"  dead-pool preflight fallback: all {len(pf)} rows completed ({len(pf_kept)} disentangled)")
    print("\nPARALLEL SMOKE PASSED")


if __name__ == "__main__":
    main()

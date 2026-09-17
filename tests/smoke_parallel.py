"""Parallel-disentangling smoke test (separate file: needs a guarded __main__).

The hybrid sweep runs independent (layer, D, k) disentanglings across processes
with ``--jobs``. That uses the ``forkserver`` start method (never ``fork``, which
deadlocks after torch spins up OpenMP threads), and forkserver re-imports the
entry module -- so this must live behind ``if __name__ == "__main__"`` rather than
in the flat ``smoke_hybrid.py`` script.

Verifies that ``--jobs 2`` produces the same grid of points as ``--jobs 1``
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
    print("\nPARALLEL SMOKE PASSED")


if __name__ == "__main__":
    main()

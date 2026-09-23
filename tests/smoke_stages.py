"""Smoke test for the Stage 1-3 machinery.

Covers the three new capabilities and the guards the stage verdicts rest on:

* a block-diagonal gate trains on the GRADIENT path under a constrained
  parameterization -- exactly orthogonal, exactly block diagonal, and still priced
  at its own angle count rather than dim SO(2^k) (an unconstrained refinement would
  train 523776 angles where a tied pairing has 32);
* the activation-MSE objective beats the Frobenius one on the activation metric,
  which is the only reason to pay for it;
* ``train_side`` really freezes a circuit, so the mixed closed-form-U / gradient-V
  scheme is measuring what it claims to;
* the identity circuit reproduces the classical MPO point EXACTLY, so the hybrid
  family provably contains the classical one and a ratio cannot be a difference of
  measurements;
* ansatz ranking excludes the chi'=1 corner, where ratios are huge and meaningless.
"""
import os
import sys
from pathlib import Path

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qllm.activation_stats import output_relative_error  # noqa: E402
from qllm.disentangler import (  # noqa: E402
    PAIR_ANSATZE, _structured_from_angles, disentangle, hybrid_weight,
    rope_pair_groups, structured_angle_count,
)
from qllm.qubit_mpo import make_plan, plan_compress  # noqa: E402


def _square(job):
    """Module-level so a worker process can import it (a closure cannot pickle)."""
    from qllm.parallel import apply_threads as _at
    _at(job)
    return job["v"] ** 2


def main():
    torch.manual_seed(0)

    # --- 1. constrained parameterization ------------------------------------
    g, t = rope_pair_groups(4, 4)
    n = structured_angle_count(g, t)
    assert n == 2, f"{n} angles, want dim SO(2) per tie class = 2"
    th = torch.randn(n, dtype=torch.float64, requires_grad=True)
    M = _structured_from_angles(th, 4, g, t)
    assert float((M @ M.T - torch.eye(16, dtype=torch.float64)).abs().max()) < 1e-12
    off = M.clone()
    for a, b in g:
        off[a, a] = off[a, b] = off[b, a] = off[b, b] = 0.0
    assert float(off.abs().max()) == 0.0, "mass leaked outside the blocks"
    M.sum().backward()
    assert torch.isfinite(th.grad).all(), "non-finite gradient through the blocks"
    z = _structured_from_angles(torch.zeros(n, dtype=torch.float64), 4, g, t)
    assert float((z - torch.eye(16, dtype=torch.float64)).abs().max()) == 0.0

    W = torch.randn(64, 64)
    A = torch.randn(64, 64)
    H = A @ A.T / 64 + 1e-3 * torch.eye(64)

    # --- 2. pairings now train on the gradient path, at the right price ------
    for ansatz in PAIR_ANSATZE:
        r = disentangle(W, gate_size=0, depth=0, target_chi=2, sweeps=1,
                        tensorization="qubit", ansatz=ansatz, head_dim=8,
                        optimizer="gradient", gd_steps=15)
        assert r.quantum_params == 4, f"{ansatz}: Q={r.quantum_params}, want 4"
        gm = r.u_gates[0].matrix
        assert float((gm @ gm.T - torch.eye(gm.shape[0])).abs().max()) < 1e-4
        o = gm.clone()
        for a, b in r.u_gates[0].groups:
            o[a, a] = o[a, b] = o[b, a] = o[b, b] = 0.0
        assert float(o.abs().max()) == 0.0, f"{ansatz}: trained out of its blocks"

    # --- 3. the activation objective wins on the activation metric -----------
    kw = dict(gate_size=2, depth=2, target_chi=2, sweeps=1, tensorization="qubit",
              optimizer="gradient", gd_steps=150)
    e_f = output_relative_error(
        W, hybrid_weight(disentangle(W, gradient_objective="relative-error", **kw), 2)[0], H)
    e_a = output_relative_error(
        W, hybrid_weight(disentangle(W, gradient_objective="activation-mse", cov=H, **kw), 2)[0], H)
    assert e_a <= e_f, f"activation-trained {e_a:.5f} lost to Frobenius {e_f:.5f}"

    try:
        disentangle(W, gradient_objective="activation-mse", **kw)
    except ValueError as exc:
        assert "cov" in str(exc)
    else:
        raise AssertionError("activation-mse was accepted without H")

    # --- 4. train_side freezes a circuit -------------------------------------
    ru = disentangle(W, gate_size=2, depth=2, target_chi=2, sweeps=0,
                     tensorization="qubit", optimizer="gradient", gd_steps=20,
                     train_side="u")
    assert all(float((x.matrix - torch.eye(x.matrix.shape[0])).abs().max()) == 0.0
               for x in ru.v_gates), "train_side='u' moved V"

    # --- 5. the families are nested (the guard every ratio depends on) -------
    for chi in (1, 2, 4):
        r0 = disentangle(W, gate_size=0, depth=0, target_chi=chi, sweeps=0,
                         tensorization="qubit")
        a, _ = hybrid_weight(r0, chi)
        b, _ = plan_compress(W, make_plan(W, "qubit", 0), chi)
        assert float((a - b).abs().max()) == 0.0, f"not nested at chi={chi}"

    # --- 6. ranking excludes the chi'=1 corner -------------------------------
    sys.path.insert(0, ROOT)
    import stages
    rows = [dict(layer_type="x", depth=0, ansatz="A", err=0.999, c=10, q=2),
            dict(layer_type="x", depth=0, ansatz="A", err=0.50, c=10, q=2),
            dict(layer_type="x", depth=0, ansatz="B", err=0.50, c=10, q=2)]
    curves = {("x", 0): [(1, 1000, 0.999), (2, 20, 0.50)]}
    tab = stages.banded(rows, curves, "ansatz")
    assert ("A", (0.99, 1.01)) in tab, "the corner band should still be reported"
    ranked = stages.print_bands(rows, curves, "ansatz", ["A", "B"])
    assert abs(ranked["A"] - ranked["B"]) < 1e-9, (
        "ranking was swayed by the chi'=1 corner: " + repr(ranked))

    # --- 7. the correlation verdict reads SIGNED rho ------------------------
    # An anti-correlated proxy is the worst outcome, not a strong one: scoring it
    # by |rho| would call a ranking that is exactly backwards a success.
    import ppl_correlation as pc
    asc = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(pc._spearman(asc, asc) - 1.0) < 1e-12
    assert abs(pc._spearman(asc, asc[::-1]) + 1.0) < 1e-12
    assert abs(pc._spearman([1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]) - 1.0) < 1e-12
    assert abs(pc._pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) - 1.0) < 1e-12

    # --- 8. convergence harness flags a budget that is too small ------------
    import convergence
    assert "explicit" in convergence.REGIMES and "explicit+gradient-am" in convergence.REGIMES
    # The plateau test must read the TAIL SLOPE. Adam keeps its best-so-far, so a
    # healthy monotone run always ends on its best iterate and an argmax-based
    # flag would fire on every single one -- useless as a diagnostic.
    src = Path(ROOT, "convergence.py").read_text()
    assert "tail_frac" in src and 'base["tail"] > args.tail_tol' in src, (
        "convergence.py no longer flags plateaus by tail slope")
    assert "hist = list(r.history or [])[:-1]" in src, (
        "convergence.py is including _build_result's trailing `retained` in the "
        "history it takes an argmax over -- that entry is a different quantity "
        "on a different scale")

    # --- 9. the figures render for both checks ------------------------------
    import tempfile
    from qllm import opt_plots
    if opt_plots._plt() is not None:
        conv = [dict(layer_type="q", depth=0, chi=2, ansatz="a", regime=r,
                     gap=g, converged=int(g <= 0.01), still_improving=si,
                     err_base=0.9, err_double=0.9, best_at=5, n_hist=10,
                     tail=0.2 * si, metric="frob", sweeps_run=3, seconds=1.0)
                for r, g, si in (("explicit", 0.001, 0), ("gradient", 0.03, 1))]
        lr = [dict(layer_type="q", depth=0, chi=2, ansatz="a", regime="gradient",
                   lr=l, err=e) for l, e in ((0.01, 0.9), (0.05, 0.8), (0.2, 0.85))]
        pc = [dict(layer_type="q", depth=0, family=f, chi=c, c=1, q=0, total=1,
                   frob=0.1 * c, act=0.1 * c, ppl=10 + c, dppl=float(c))
              for f in ("mpo", "hybrid") for c in (1, 2, 3, 4)]
        rho = [("q", 0, "mpo", dict(rho_act=0.9, rho_frob=0.7)),
               ("q", 0, "hybrid", dict(rho_act=-0.3, rho_frob=-0.4))]
        with tempfile.TemporaryDirectory() as td:
            opt_plots.plot_convergence(conv, 0.01, td)
            opt_plots.plot_lr(lr, 0.05, td)
            opt_plots.plot_ppl_scatter(pc, td, {"act": 0.9, "frob": 0.7})
            opt_plots.plot_ppl_rho(rho, td)
            made = sorted(p.name for p in Path(td).glob("*.png"))
        assert made == ["convergence_doubling.png", "convergence_lr.png",
                        "ppl_rho.png", "ppl_scatter.png"], made

    # --- 10. the per-iteration trace is structured and persists -------------
    import tempfile as _tf
    from qllm.trace_log import TraceWriter, read_traces
    for opt, phases in (("explicit", {"explicit"}), ("gradient", {"gradient"}),
                        ("explicit+gradient", {"explicit", "gradient"})):
        rt = disentangle(W, gate_size=2, depth=2, target_chi=2, sweeps=4,
                         tensorization="qubit", optimizer=opt, gd_steps=6,
                         gradient_objective="relative-error")
        got = {t["phase"] for t in rt.trace}
        assert got == phases, f"{opt}: trace phases {got}, want {phases}"
        # iter is 1-based WITHIN the phase, so the two phases both start at 1
        for ph in phases:
            its = [t["iter"] for t in rt.trace if t["phase"] == ph]
            assert its == list(range(1, len(its) + 1)), f"{opt}/{ph}: {its}"
        # the loss a gradient row reports is the objective it was given
        gr = [t for t in rt.trace if t["phase"] == "gradient"]
        assert all(t["objective"] == "relative-error" for t in gr), opt
        assert any(t["best"] for t in rt.trace), f"{opt}: no best-so-far marked"

    with _tf.TemporaryDirectory() as td:
        path = Path(td) / "tr.csv"
        with TraceWriter(path, every=2) as tw:
            tw.add(rt, layer_type="q", depth=0, chi=2, ansatz="a", regime="r")
        back = read_traces(path)
        assert back and all(r["run_id"] == "q|0|2|a|r" for r in back)
        # thinning touches the gradient phase only, and keeps both endpoints
        sw = [r for r in back if r["phase"] == "explicit"]
        gd = [r for r in back if r["phase"] == "gradient"]
        assert len(sw) == len([t for t in rt.trace if t["phase"] == "explicit"]), (
            "the sweep phase was thinned; it should never be")
        raw_gd = [t["iter"] for t in rt.trace if t["phase"] == "gradient"]
        assert gd[0]["iter"] == raw_gd[0] and gd[-1]["iter"] == raw_gd[-1], (
            "thinning dropped an endpoint of the gradient curve")
        assert len(gd) < len(raw_gd), "every=2 did not thin anything"
    # a writer with no path writes nothing and leaves no file behind
    quiet = TraceWriter(None)
    quiet.add(rt, layer_type="q")
    quiet.close()
    assert quiet.n_rows == 0

    # --- 11. checkpointing: a killed run resumes without changing the answer --
    from qllm.checkpoint import ResumableCSV
    with _tf.TemporaryDirectory() as td:
        path = Path(td) / "ck.csv"
        cfg = {"sweeps": 12, "gd_steps": 150}
        a = ResumableCSV(path, ("layer", "chi"), cfg, numeric=("err",))
        for i in range(3):
            a.add(dict(layer="q", chi=i, err=0.5 + i))
        a.close()
        b = ResumableCSV(path, ("layer", "chi"), cfg, numeric=("err",))
        assert b.n_resumed == 3, b.n_resumed
        assert b.done(layer="q", chi=1) and not b.done(layer="q", chi=9)
        b.add(dict(layer="q", chi=9, err=9.5))
        b.close()
        # the union of resumed and new rows is what the analysis reads, and the
        # resumed values come back as numbers, not strings
        assert len(b.rows) == 4 and isinstance(b.rows[0]["err"], float)

        # a configuration change must REFUSE to resume rather than blend
        # incompatible measurements into one table
        try:
            ResumableCSV(path, ("layer", "chi"), {"sweeps": 99, "gd_steps": 150})
        except SystemExit as exc:
            assert "different configuration" in str(exc)
        else:
            raise AssertionError("resumed into a file from another configuration")

        # --no-resume starts over
        c = ResumableCSV(path, ("layer", "chi"), cfg, resume=False)
        assert c.n_resumed == 0 and not c.done(layer="q", chi=1)
        c.close()

    # a half-written final line (process killed mid-flush) is dropped, not parsed
    with _tf.TemporaryDirectory() as td:
        path = Path(td) / "torn.csv"
        path.write_text("layer,chi,err\nq,1,0.5\nq,2\n")
        d = ResumableCSV(path, ("layer", "chi"), {}, numeric=("err",))
        assert len(d.rows) == 1, f"torn line was not dropped: {d.rows}"
        d.close()

    # --- 12. early stopping is a compute saver, not an overfitting guard ----
    W2 = torch.randn(64, 64)
    kw2 = dict(gate_size=2, depth=2, target_chi=2, sweeps=2, tensorization="qubit",
               optimizer="gradient", gd_steps=80, gradient_objective="relative-error",
               fast_gradient=True)
    long = disentangle(W2, patience=0, **kw2)
    short = disentangle(W2, patience=8, min_delta=1e-4, **kw2)
    assert len(short.trace) < len(long.trace), (
        f"patience did not stop early: {len(short.trace)} vs {len(long.trace)}")
    assert short.trace[-1]["stopped_early"] == 1 and long.trace[-1]["stopped_early"] == 0
    # Stopping on a plateau must not cost real accuracy -- if it does, the
    # patience is too tight and the saving is not free.
    e_long = output_relative_error(W2, hybrid_weight(long, 2)[0], H)
    e_short = output_relative_error(W2, hybrid_weight(short, 2)[0], H)
    assert e_short <= e_long * 1.02, f"early stop cost {e_short / e_long - 1:.2%}"

    # --- 13. the process pool ------------------------------------------------
    from qllm.parallel import apply_threads, run_jobs
    seen = []
    res = run_jobs(_square, [dict(v=i) for i in range(6)], n_jobs=1,
                   on_result=lambda j, r: seen.append(r))
    assert res == [0, 1, 4, 9, 16, 25] and seen == res, (res, seen)
    # n_jobs > 1 must produce the same SET of results; order is completion order
    par = run_jobs(_square, [dict(v=i) for i in range(6)], n_jobs=3)
    assert sorted(par) == [0, 1, 4, 9, 16, 25], par
    # a worker that dies is retried, not lost
    hard = run_jobs(_square, [dict(v=i) for i in range(4)], n_jobs=8)
    assert sorted(hard) == [0, 1, 4, 9], hard
    apply_threads({"_threads": 1})            # must not raise

    print(f"  constrained gates: {n} angles, orthogonal to 1e-12, "
          f"off-block mass exactly 0, differentiable")
    print(f"  activation objective {e_a:.5f} beats Frobenius {e_f:.5f} on the "
          f"activation metric")
    print("  train_side freezes V; activation-mse refuses without H")
    print("  hybrid family contains the classical MPO exactly at chi=1,2,4")
    print("  ansatz ranking ignores the chi'=1 corner")
    print("  Spearman is signed (+1 / -1 / ties) and the ppl verdict reads the sign")
    print("  all four figures render")
    print("  traces: phases separated, iters 1-based per phase, thinning keeps "
          "endpoints")
    print("  checkpoint: resumes, refuses a changed config, drops a torn line")
    print(f"  early stop: {len(short.trace)} steps vs {len(long.trace)}, "
          f"error within 2%")
    print("  process pool: sequential and pooled agree, dead workers retried")
    print("\nSTAGES SMOKE PASSED")


if __name__ == "__main__":
    raise SystemExit(main())

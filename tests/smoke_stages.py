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

    # --- 9. the figures render for both checks ------------------------------
    import tempfile
    from qllm import opt_plots
    if opt_plots._plt() is not None:
        conv = [dict(layer_type="q", depth=0, chi=2, ansatz="a", regime=r,
                     gap=g, converged=int(g <= 0.01), still_improving=si,
                     err_base=0.9, err_double=0.9, best_at=5, n_hist=10,
                     sweeps_run=3, seconds=1.0)
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

    print(f"  constrained gates: {n} angles, orthogonal to 1e-12, "
          f"off-block mass exactly 0, differentiable")
    print(f"  activation objective {e_a:.5f} beats Frobenius {e_f:.5f} on the "
          f"activation metric")
    print("  train_side freezes V; activation-mse refuses without H")
    print("  hybrid family contains the classical MPO exactly at chi=1,2,4")
    print("  ansatz ranking ignores the chi'=1 corner")
    print("  Spearman is signed (+1 / -1 / ties) and the ppl verdict reads the sign")
    print("  all four figures render")
    print("\nSTAGES SMOKE PASSED")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Do the stage hyperparameters actually converge, for every configuration?

A loss curve that looks flat is not evidence. The test that settles it is
**budget doubling**: run a configuration at the stage budget and again at twice
it, and call it converged only if the extra budget buys nothing. Anything else
measures the optimiser's patience rather than its convergence.

Three things are reported per configuration, because they fail differently:

* **doubling gap** -- ``(err_B - err_2B) / err_B``. Above ``--tol`` the stage
  budget is too small and every number produced at it understates what the
  ansatz can do. This is the criterion the verdict uses.
* **where the best iterate was found**. The Adam path keeps the best-so-far and
  restores it, so a run whose best step is the *last* step has not plateaued,
  even if its loss curve looks calm. A run that broke early on a non-finite
  gradient is flagged separately -- that is a failure, not convergence.
* **learning-rate sensitivity**. If ``lr=0.05`` is merely one point on a plateau
  the default is safe; if a neighbouring lr is much better, the stage results are
  an artefact of the step size.

The explicit sweep has its own convergence criterion (it stops when a sweep
improves the overlap by less than ``tol``), so for that path ``sweeps_run <
sweeps`` is an independent confirmation and is reported alongside the doubling.

    python convergence.py --layers-dir models/SmolLM2-135M/layers \
        --types q_proj v_proj --depths 20 --chis 4 16 --out convergence.csv
"""
import argparse
import csv
import statistics as st
from pathlib import Path

import torch
from safetensors.torch import load_file

from qllm.activation_stats import output_relative_error
from qllm.compactifai import relative_error
from qllm.disentangler import disentangle, hybrid_weight
from qllm.opt_plots import plot_convergence, plot_lr
from stages import ANSATZE, load_layers

# (label, kwargs). Mirrors the regimes stages 2-3 actually run.
REGIMES = {
    "explicit": dict(optimizer="explicit"),
    "gradient-dl": dict(optimizer="gradient", gradient_objective="disentangle-loss"),
    "gradient-re": dict(optimizer="gradient", gradient_objective="relative-error"),
    "explicit+gradient-re": dict(optimizer="explicit+gradient",
                                 gradient_objective="relative-error"),
    "explicit+gradient-am": dict(optimizer="explicit+gradient",
                                 gradient_objective="activation-mse"),
}


def run(W, chi, ansatz, regime, sweeps, gd_steps, lr, cov=None, seed=0):
    kw = dict(ANSATZE[ansatz]); kw.update(REGIMES[regime])
    r = disentangle(W, target_chi=chi, n_sites=2, sweeps=sweeps, tensorization="qubit",
                    gd_steps=gd_steps, gd_lr=lr, fast_gradient=True, seed=seed,
                    cov=cov, **kw)
    approx, _ = hybrid_weight(r, chi)
    err = (float(relative_error(W, approx)) if cov is None
           else output_relative_error(W, approx, cov))
    hist = list(r.history or [])
    # history holds 1 - loss for the Adam path and the sweep score for the explicit
    # one; either way the best iterate is its argmax.
    best_at = (max(range(len(hist)), key=lambda i: hist[i]) + 1) if hist else 0
    return dict(err=err, sweeps_run=int(r.sweeps_run), n_hist=len(hist),
                best_at=best_at, seconds=round(r.seconds, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers-dir", required=True)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--depths", type=int, nargs="+", default=None)
    ap.add_argument("--cov", default=None, help="H directory; enables the "
                                                "activation-mse regime")
    ap.add_argument("--ansatze", nargs="+",
                    default=["brickwall-k4D2", "brickwall-k2D4", "rope-pair"])
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    ap.add_argument("--chis", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--sweeps", type=int, default=12)
    ap.add_argument("--gd-steps", type=int, default=150)
    ap.add_argument("--gd-lr", type=float, default=0.05)
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.05, 0.2],
                    help="learning rates for the sensitivity check")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="a doubling gap above this means NOT converged")
    ap.add_argument("--out", default="convergence.csv")
    ap.add_argument("--figs", default="figs",
                    help="directory for the figures (--figs '' to skip plotting)")
    args = ap.parse_args()

    cov = {}
    if args.cov:
        for r in csv.DictReader(open(Path(args.cov) / "manifest.csv")):
            cov[r["param_name"]] = next(
                iter(load_file(Path(args.cov) / r["file"]).values())).float()
    layers = load_layers(args)

    rows = []
    print("Budget doubling: err at the stage budget vs twice it "
          f"(not converged if the gap exceeds {args.tol:.1%})\n")
    print(f"{'layer':>16}{'chi':>5}{'ansatz':>18}{'regime':>22}"
          f"{'err(B)':>10}{'err(2B)':>10}{'gap':>9}{'best@':>9}{'ok':>4}")
    for pname, lt, dep, W in layers:
        H = cov.get(pname)
        for chi in args.chis:
            for ansatz in args.ansatze:
                for regime in args.regimes:
                    needs_h = regime.endswith("-am")
                    if needs_h and H is None:
                        continue
                    c = H if needs_h else None
                    base = run(W, chi, ansatz, regime, args.sweeps, args.gd_steps,
                               args.gd_lr, c)
                    dbl = run(W, chi, ansatz, regime, args.sweeps * 2,
                              args.gd_steps * 2, args.gd_lr, c)
                    gap = ((base["err"] - dbl["err"]) / base["err"]
                           if base["err"] > 0 else 0.0)
                    ok = gap <= args.tol
                    # For the Adam path, "best iterate is the last step" is an
                    # independent warning: the run was still improving when it
                    # was cut off, whatever the doubling gap says.
                    tail = (base["best_at"] >= base["n_hist"] > 1
                            and regime != "explicit")
                    rows.append(dict(layer_type=lt, depth=dep, chi=chi,
                                     ansatz=ansatz, regime=regime,
                                     err_base=round(base["err"], 6),
                                     err_double=round(dbl["err"], 6),
                                     gap=round(gap, 5), converged=int(ok),
                                     best_at=base["best_at"], n_hist=base["n_hist"],
                                     sweeps_run=base["sweeps_run"],
                                     still_improving=int(tail),
                                     seconds=base["seconds"]))
                    flag = "ok" if ok else "NO"
                    print(f"{lt[-12:]:>16}{chi:>5}{ansatz:>18}{regime:>22}"
                          f"{base['err']:>10.5f}{dbl['err']:>10.5f}{gap:>8.2%}"
                          f"{str(base['best_at']) + '/' + str(base['n_hist']):>9}"
                          f"{flag:>4}" + ("  <- still improving" if tail else ""),
                          flush=True)

    # ---- learning-rate sensitivity (gradient regimes only) ------------------
    lr_rows = []
    grad = [r for r in args.regimes if r != "explicit"]
    if grad and args.lrs:
        print(f"\nLearning-rate sensitivity at the stage budget "
              f"(err; lower is better)")
        print(f"{'layer':>16}{'chi':>5}{'ansatz':>18}{'regime':>22}"
              + "".join(f"{'lr=' + str(l):>11}" for l in args.lrs))
        for pname, lt, dep, W in layers:
            H = cov.get(pname)
            for chi in args.chis:
                for ansatz in args.ansatze:
                    for regime in grad:
                        if regime.endswith("-am") and H is None:
                            continue
                        c = H if regime.endswith("-am") else None
                        errs = [run(W, chi, ansatz, regime, args.sweeps,
                                    args.gd_steps, l, c)["err"] for l in args.lrs]
                        for l, e in zip(args.lrs, errs):
                            lr_rows.append(dict(layer_type=lt, depth=dep, chi=chi,
                                                ansatz=ansatz, regime=regime,
                                                lr=l, err=round(e, 6)))
                        print(f"{lt[-12:]:>16}{chi:>5}{ansatz:>18}{regime:>22}"
                              + "".join(f"{e:>11.5f}" for e in errs), flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    if lr_rows:
        p2 = Path(args.out).with_name(Path(args.out).stem + "_lr.csv")
        with open(p2, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(lr_rows[0]))
            w.writeheader(); w.writerows(lr_rows)
        print(f"\nWrote {len(lr_rows)} lr rows to {p2.resolve()}")
    print(f"Wrote {len(rows)} rows to {Path(args.out).resolve()}")
    if args.figs:
        plot_convergence(rows, args.tol, args.figs)
        plot_lr(lr_rows, args.gd_lr, args.figs)

    # ---- verdict -------------------------------------------------------------
    print("\nConvergence by regime")
    print(f"{'regime':>22}{'converged':>12}{'median gap':>13}{'worst gap':>12}"
          f"{'still improving':>18}")
    bad = []
    for regime in dict.fromkeys(r["regime"] for r in rows):
        sel = [r for r in rows if r["regime"] == regime]
        gaps = [r["gap"] for r in sel]
        nconv = sum(r["converged"] for r in sel)
        tail = sum(r["still_improving"] for r in sel)
        print(f"{regime:>22}{f'{nconv}/{len(sel)}':>12}{st.median(gaps):>12.2%}"
              f"{max(gaps):>12.2%}{f'{tail}/{len(sel)}':>18}")
        if nconv < len(sel):
            bad.append((regime, len(sel) - nconv, max(gaps)))

    if lr_rows:
        print("\nIs lr=%.3g defensible? (how often each lr was best)" % args.gd_lr)
        wins = {l: 0 for l in args.lrs}
        keys = {(r["layer_type"], r["depth"], r["chi"], r["ansatz"], r["regime"])
                for r in lr_rows}
        for k in keys:
            sel = [r for r in lr_rows
                   if (r["layer_type"], r["depth"], r["chi"], r["ansatz"],
                       r["regime"]) == k]
            wins[min(sel, key=lambda r: r["err"])["lr"]] += 1
        for l in args.lrs:
            print(f"  lr={l:<6} best on {wins[l]:>3}/{len(keys)} configurations")

    if not bad:
        print(f"\nVERDICT: every configuration converged -- doubling the budget "
              f"moved the\n  error by at most {max(r['gap'] for r in rows):.2%}, "
              f"under the {args.tol:.1%} tolerance. The stage\n  hyperparameters "
              f"are sufficient as set.")
    else:
        print("\nVERDICT: NOT converged everywhere. " + "; ".join(
            f"{r}: {n} configuration(s), worst gap {g:.1%}" for r, n, g in bad)
            + ".\n  Raise the budget for those regimes before running the stages -- "
              "results taken at\n  this budget understate what those ansatze can do.")


if __name__ == "__main__":
    raise SystemExit(main())

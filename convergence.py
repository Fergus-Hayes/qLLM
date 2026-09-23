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
from qllm.opt_plots import plot_convergence, plot_lr, plot_traces
from qllm.checkpoint import ResumableCSV
from qllm.trace_log import TraceWriter, read_traces
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


def run(W, chi, ansatz, regime, sweeps, gd_steps, lr, cov=None, seed=0,
        writer=None, key=None, patience=0, min_delta=0.0, cov_val=None):
    kw = dict(ANSATZE[ansatz]); kw.update(REGIMES[regime])
    r = disentangle(W, target_chi=chi, n_sites=2, sweeps=sweeps, tensorization="qubit",
                    gd_steps=gd_steps, gd_lr=lr, fast_gradient=True, seed=seed,
                    cov=cov, patience=patience, min_delta=min_delta, **kw)
    approx, _ = hybrid_weight(r, chi)
    err = (float(relative_error(W, approx)) if cov is None
           else output_relative_error(W, approx, cov))
    # history is NOT one homogeneous series. _build_result appends the achieved
    # retained weight as a final element, which is a different quantity from the
    # per-iteration entries (1 - loss for Adam, the sweep score for the explicit
    # path) and lives on its own scale -- taking an argmax over the whole list
    # compares the two and lands on the last element for arbitrary reasons. Drop
    # it, and for a warm-started run look only at the gradient segment, since
    # "was it still improving when cut off" is a question about Adam.
    if writer:
        writer.add(r, **(key or {}))
    hist = list(r.history or [])[:-1]
    if r.optimizer == "explicit+gradient":
        hist = hist[int(r.sweeps_run):]
    best_at = (max(range(len(hist)), key=lambda i: hist[i]) + 1) if hist else 0
    # "Best iterate is the last step" is NOT a useful plateau test: Adam keeps its
    # best-so-far, so a healthy monotone run always ends on its best and the flag
    # fires every time. What separates plateaued from cut-off is the TAIL SLOPE --
    # what fraction of the whole run's improvement arrived in its last tenth.
    tail_frac = 0.0
    if len(hist) >= 10:
        total = hist[-1] - hist[0]
        cut = hist[int(len(hist) * 0.9) - 1]
        if abs(total) > 1e-12:
            tail_frac = max(0.0, (hist[-1] - cut) / total)
    # Held-out H: the ONE place an overfitting worry is real. The activation
    # objective is fitted against an H estimated from finite calibration tokens,
    # so scoring the trained circuit against an H from DIFFERENT tokens turns
    # "are we over-optimising a noisy target" from a worry into a number.
    err_val = float("nan")
    if cov_val is not None:
        err_val = output_relative_error(W, approx, cov_val)
    return dict(err=err, err_val=err_val, sweeps_run=int(r.sweeps_run),
                n_hist=len(hist), best_at=best_at, tail=tail_frac,
                stopped_early=int(r.trace[-1].get("stopped_early", 0))
                if r.trace else 0,
                seconds=round(r.seconds, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers-dir", required=True)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--depths", type=int, nargs="+", default=None)
    ap.add_argument("--cov", default=None,
                    help="directory of captured H matrices. The activation-MSE "
                         "loss tr[D H D^T]/tr[W H W^T] reads the whole input "
                         "distribution out of one stored d_in x d_in matrix, so "
                         "no model is needed here -- but without --cov the "
                         "regimes that use it are skipped, not approximated")
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
    ap.add_argument("--patience", type=int, default=25,
                    help="stop a gradient run after this many steps with no "
                         "relative improvement. This buys COMPUTE: the objective "
                         "is the exact quantity being minimised on one fixed "
                         "matrix, so there is no generalisation gap and more "
                         "steps can only lower or flatten the loss. 0 disables it")
    ap.add_argument("--min-delta", type=float, default=1e-4,
                    help="relative improvement that counts as progress")
    ap.add_argument("--cov-val", default=None,
                    help="a SECOND H, captured from different calibration tokens. "
                         "Activation-MSE runs are scored against it as well, which "
                         "is the only honest test of over-fitting a noisy H")
    ap.add_argument("--tail-tol", type=float, default=0.10,
                    help="flag a run whose last tenth of budget delivered more "
                         "than this fraction of its total improvement")
    ap.add_argument("--trace", default=None,
                    help="write every optimisation's per-iteration trace to this "
                         "CSV (one tidy long-format file)")
    ap.add_argument("--trace-every", type=int, default=1,
                    help="keep every Nth Adam step in the trace (endpoints always "
                         "kept); the sweep phase is never thinned")
    ap.add_argument("--no-resume", action="store_true",
                    help="discard any existing output and start over")
    ap.add_argument("--out", default="convergence.csv")
    ap.add_argument("--figs", default="figs",
                    help="directory for the figures (--figs '' to skip plotting)")
    args = ap.parse_args()

    cov_val = {}
    if args.cov_val:
        for r in csv.DictReader(open(Path(args.cov_val) / "manifest.csv")):
            cov_val[r["param_name"]] = next(
                iter(load_file(Path(args.cov_val) / r["file"]).values())).float()
    cov = {}
    if args.cov:
        for r in csv.DictReader(open(Path(args.cov) / "manifest.csv")):
            cov[r["param_name"]] = next(
                iter(load_file(Path(args.cov) / r["file"]).values())).float()
    layers = load_layers(args)

    cfg = dict(sweeps=args.sweeps, gd_steps=args.gd_steps, gd_lr=args.gd_lr,
               tol=args.tol, layers_dir=args.layers_dir, cov=args.cov,
               patience=args.patience, min_delta=args.min_delta)
    ck = ResumableCSV(args.out, ("layer_type", "depth", "chi", "ansatz", "regime"),
                      cfg, resume=not args.no_resume,
                      numeric=("gap", "err_base", "err_double", "tail"))
    tracer = TraceWriter(args.trace, args.trace_every,
                         append=bool(ck.n_resumed))
    print("Budget doubling: err at the stage budget vs twice it "
          f"(not converged if the gap exceeds {args.tol:.1%})")
    print("'tail' is the share of the run's total improvement that arrived in its "
          "last tenth;\nlarge means it was cut off mid-descent, not left on a "
          "plateau.")
    print("Each regime is scored in ITS OWN objective's metric -- 'act' rows are "
          "the H-weighted\nerror, 'frob' rows the Frobenius one. Compare down a "
          "column only within a metric.\n")
    print(f"{'layer':>16}{'chi':>5}{'ansatz':>18}{'regime':>22}{'metric':>7}"
          f"{'err(B)':>10}{'err(2B)':>10}{'gap':>9}{'tail':>8}{'ok':>6}")
    for pname, lt, dep, W in layers:
        H = cov.get(pname)
        for chi in args.chis:
            for ansatz in args.ansatze:
                for regime in args.regimes:
                    needs_h = regime.endswith("-am")
                    if needs_h and H is None:
                        continue
                    c = H if needs_h else None
                    if ck.done(layer_type=lt, depth=dep, chi=chi, ansatz=ansatz,
                               regime=regime):
                        continue          # already measured by an earlier run
                    base = run(W, chi, ansatz, regime, args.sweeps, args.gd_steps,
                               args.gd_lr, c, patience=args.patience,
                               min_delta=args.min_delta,
                               cov_val=cov_val.get(pname) if needs_h else None,
                               writer=tracer,
                               key=dict(layer_type=lt, depth=dep, chi=chi,
                                        ansatz=ansatz, regime=regime,
                                        lr=args.gd_lr, budget="1x"))
                    # The control runs UNCONSTRAINED. If it early-stopped too it
                    # would halt at the same place and the gap would be a vacuous
                    # 0.00%; running it to the full doubled budget makes the test
                    # ask the right question -- does early stopping cost anything.
                    dbl = run(W, chi, ansatz, regime, args.sweeps * 2,
                              args.gd_steps * 2, args.gd_lr, c, patience=0)
                    gap = ((base["err"] - dbl["err"]) / base["err"]
                           if base["err"] > 0 else 0.0)
                    # A run that recorded no iterations never trained: the
                    # optimiser's finite-gradient guard aborted at step 0 and the
                    # circuit stayed at identity. Both the B and 2B runs fail the
                    # same way, so the gap is a clean 0.00% and it would otherwise
                    # be counted as the best-converged row in the table.
                    dead = (regime != "explicit" and base["n_hist"] == 0)
                    ok = (gap <= args.tol) and not dead
                    # Independent of the doubling gap: if a tenth of the budget
                    # is still delivering a tenth of the total gain, the run was
                    # cut off mid-descent rather than left on a plateau.
                    tail = base["tail"] > args.tail_tol
                    ck.add(dict(layer_type=lt, depth=dep, chi=chi,
                                     ansatz=ansatz, regime=regime,
                                     err_base=round(base["err"], 6),
                                     err_double=round(dbl["err"], 6),
                                     gap=round(gap, 5), converged=int(ok),
                                     best_at=base["best_at"], n_hist=base["n_hist"],
                                     tail=round(base["tail"], 5),
                                     stopped_early=base["stopped_early"],
                                     err_val=(round(base["err_val"], 6)
                                              if base["err_val"] == base["err_val"]
                                              else ""),
                                     sweeps_run=base["sweeps_run"],
                                     metric=("act" if needs_h else "frob"),
                                     never_trained=int(dead),
                                     still_improving=int(tail),
                                     seconds=base["seconds"]))
                    rows = ck.rows
                    flag = "DEAD" if dead else ("ok" if ok else "NO")
                    print(f"{lt[-12:]:>16}{chi:>5}{ansatz:>18}{regime:>22}"
                          f"{('act' if needs_h else 'frob'):>7}"
                          f"{base['err']:>10.5f}{dbl['err']:>10.5f}{gap:>8.2%}"
                          f"{base['tail']:>7.1%}{flag:>6}"
                          + ("  <- NEVER TRAINED (0 iterations recorded)" if dead
                             else "  <- still improving" if tail else ""),
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

    rows = ck.rows
    ck.close()
    if lr_rows:
        p2 = Path(args.out).with_name(Path(args.out).stem + "_lr.csv")
        with open(p2, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(lr_rows[0]))
            w.writeheader(); w.writerows(lr_rows)
        print(f"\nWrote {len(lr_rows)} lr rows to {p2.resolve()}")
    tracer.close()
    if args.figs:
        plot_convergence(rows, args.tol, args.figs)
        plot_lr(lr_rows, args.gd_lr, args.figs)
        if args.trace and Path(args.trace).exists():
            plot_traces(read_traces(args.trace), args.figs)

    # ---- verdict -------------------------------------------------------------
    print("\nConvergence by regime")
    print(f"{'regime':>22}{'converged':>12}{'median gap':>13}{'worst gap':>12}"
          f"{'still improving':>18}{'never trained':>16}")
    # Held-out H: is a long optimisation chasing sampling noise in H?
    vr = [r for r in rows if str(r.get("err_val", "")) not in ("", "nan")]
    if vr:
        print("\nActivation-MSE runs scored against a held-out H "
              "(captured from different tokens)")
        print(f"{'ansatz':>18}{'n':>4}{'train H':>11}{'held-out H':>13}"
              f"{'gap':>9}{'early-stopped':>15}")
        for a in dict.fromkeys(r["ansatz"] for r in vr):
            sel = [r for r in vr if r["ansatz"] == a]
            tr = st.mean([float(r["err_base"]) for r in sel])
            va = st.mean([float(r["err_val"]) for r in sel])
            es = sum(int(r.get("stopped_early", 0)) for r in sel)
            print(f"{a:>18}{len(sel):>4}{tr:>11.5f}{va:>13.5f}"
                  f"{(va - tr) / tr:>8.2%}{f'{es}/{len(sel)}':>15}")
        worst = max((float(r["err_val"]) - float(r["err_base"]))
                    / max(float(r["err_base"]), 1e-12) for r in vr)
        print("  (a large positive gap means the circuit is fitting THIS H's "
              "sampling noise;\n   that is the only sense in which these runs can "
              f"over-fit. Worst here: {worst:+.2%}.)")

    es_rows = [r for r in rows if int(r.get("stopped_early", 0))]
    if es_rows:
        print(f"\n{len(es_rows)}/{len(rows)} gradient run(s) stopped early on a "
              f"plateau. The doubling control runs\n  UNCONSTRAINED, so the gap "
              f"above already answers whether that cost anything.")

    dead_rows = [r for r in rows if int(r.get("never_trained", 0))]
    if dead_rows:
        print(f"\n{len(dead_rows)} run(s) NEVER TRAINED -- zero iterations "
              f"recorded, circuit left at identity.\n  These report the classical "
              f"MPO error and a 0.00% doubling gap, so they would read as the\n"
              f"  best-converged rows in the table. Affected: "
              + ", ".join(sorted({f"{r['regime']}@chi={r['chi']}"
                                  for r in dead_rows})) + ".")
    bad = []
    for regime in dict.fromkeys(r["regime"] for r in rows):
        sel = [r for r in rows if r["regime"] == regime]
        gaps = [r["gap"] for r in sel]
        nconv = sum(r["converged"] for r in sel)
        tail = sum(r["still_improving"] for r in sel)
        nd = sum(int(r.get("never_trained", 0)) for r in sel)
        print(f"{regime:>22}{f'{nconv}/{len(sel)}':>12}{st.median(gaps):>12.2%}"
              f"{max(gaps):>12.2%}{f'{tail}/{len(sel)}':>18}"
              f"{f'{nd}/{len(sel)}':>16}")
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

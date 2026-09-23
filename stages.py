#!/usr/bin/env python3
"""Stages 1-3 of the C-versus-Q study: ansatz, training regime, objective.

The question underneath all three is the same one ``hybrid_vs_mpo.py`` answered for
a handful of configurations: fix a target error, and ask whether ``C(chi') + Q``
beats the ``C(chi)`` the MPO alone would need. What changes between stages is which
axis is being searched.

    stage1  ansatz        brickwall k x D, the pairings and their matched null,
                          head-block, all-to-all. Explicit sweep, Frobenius, wide.
    stage2  regime        explicit / gradient / explicit+gradient, and the two
                          Frobenius gradient objectives, on the stage-1 survivors.
    stage3  objective     activation-MSE against Frobenius, and train-both against
                          train-V-only (the cheap mixed scheme). Needs H.

Guards, each of which corresponds to a result this project had to retract:

* every circuit is trained **at the chi' it is served at** -- optimising for one
  bond and scoring at another manufactures wins;
* the classical baseline is swept at **every** chi, never a log grid, because gaps
  in the comparison family are free wins;
* the identity circuit is asserted to reproduce the classical point **exactly**, so
  a win cannot be a different measurement;
* results are reported **banded by the accuracy they were achieved at**, never as a
  maximum -- the maximum lives at chi'=1 where the error is 0.9998 and says nothing;
* the pairings ship with a **cost-matched, locality-matched null** (adjacent-pair),
  because the obvious null (random-pair) is confounded by qubit locality.

    python stages.py stage1 --layers-dir models/SmolLM2-135M/layers --out s1.csv
    python stages.py stage2 --layers-dir ... --ansatze brickwall-k4D2 rope-pair
    python stages.py stage3 --layers-dir ... --cov cov/
"""
import argparse
import csv
import statistics as st
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from qllm.compactifai import log_spaced_ints, relative_error
from qllm.activation_stats import output_relative_error
from qllm.disentangler import classical_param_count, disentangle, hybrid_weight
from qllm.qubit_mpo import make_plan, plan_compress
from qllm.checkpoint import ResumableCSV
from qllm.trace_log import TraceWriter

# --------------------------------------------------------------------------- #
# Ansatz catalogue. Stage 1 searches gate width x depth because that, not Q, was
# what separated the configurations in the pilot: k4D2 (Q=960) beat k2D8 (Q=432).
ANSATZE: dict[str, dict] = {}
for _k in (2, 3, 4, 5):
    for _d in (1, 2, 4, 8, 16, 32, 64, 128):
        ANSATZE[f"brickwall-k{_k}D{_d}"] = dict(gate_size=_k, depth=_d,
                                                ansatz="brickwall")
for _name, _kw in (
    ("rope-pair", dict(ansatz="rope-pair", share_heads=True)),
    ("rope-pair-untied", dict(ansatz="rope-pair", share_heads=False)),
    ("adjacent-pair", dict(ansatz="adjacent-pair", share_heads=True)),
    ("adjacent-pair-untied", dict(ansatz="adjacent-pair", share_heads=False)),
    ("random-pair", dict(ansatz="random-pair", share_heads=True)),
    ("head-block", dict(ansatz="head-block")),
):
    ANSATZE[_name] = dict(gate_size=0, depth=0, head_dim=64, **_kw)
ANSATZE["all-to-all"] = dict(gate_size=64, depth=1, ansatz="brickwall")

BANDS = [(0.99, 1.01), (0.95, 0.99), (0.80, 0.95), (0.40, 0.80), (0.0, 0.40)]


def load_layers(args):
    man = list(csv.DictReader(open(Path(args.layers_dir) / "manifest.csv")))
    out = []
    for row in man:
        lt, dep = row["layer_type"], int(row["depth"])
        if args.types and not any(t in lt for t in args.types):
            continue
        if args.depths and dep not in args.depths:
            continue
        W = next(iter(load_file(Path(args.layers_dir) / row["file"]).values())).float()
        out.append((row["param_name"], lt, dep, W))
    if not out:
        raise SystemExit("No layer matched --types/--depths.")
    return out


def classical_curve(W, cov=None):
    """(chi, C, err) at EVERY chi. Sampled exhaustively on purpose: this is the
    family the hybrid has to beat, and a sparse grid would hand it a free win."""
    plan = make_plan(W, "qubit", 0)
    out = []
    for chi in range(1, plan.max_chi + 1):
        approx, c = plan_compress(W, plan, chi)
        err = (float(relative_error(W, approx)) if cov is None
               else output_relative_error(W, approx, cov))
        out.append((chi, int(c), err))
    return out


def cheapest(curve, target):
    ok = [p for _k, p, e in curve if e <= target]
    return min(ok) if ok else None


def assert_identity_is_classical(W, chi, cov=None):
    """The hybrid family must CONTAIN the classical one. Without this the whole
    comparison could be measuring two different things and calling it a saving."""
    r = disentangle(W, gate_size=0, depth=0, target_chi=chi, n_sites=2, sweeps=0,
                    tensorization="qubit")
    a, _ = hybrid_weight(r, chi)
    b, _ = plan_compress(W, make_plan(W, "qubit", 0), chi)
    dev = float((a - b).abs().max())
    if dev > 0.0:
        raise SystemExit(f"identity circuit does not reproduce the classical point "
                         f"at chi={chi} (max|diff| = {dev:.3e}); the families are "
                         f"not nested and no ratio from this run is meaningful.")


def run_point(W, chi, name, sweeps, cov=None, writer=None, key=None, **extra):
    """One trained configuration, scored at the chi' it was trained at."""
    kw = dict(ANSATZE[name]); kw.update(extra)
    t0 = time.time()
    r = disentangle(W, target_chi=chi, n_sites=2, sweeps=sweeps,
                    tensorization="qubit", cov=cov, **kw)
    if writer:
        writer.add(r, **(key or {}))
    approx, _ = hybrid_weight(r, chi)
    err = (float(relative_error(W, approx)) if cov is None
           else output_relative_error(W, approx, cov))
    return dict(chi=chi, c=int(classical_param_count(r, chi)),
                q=int(r.quantum_params), err=round(err, 6),
                seconds=round(time.time() - t0, 2))


def banded(rows, curves, group_key):
    """Mean C_alone/(C+Q) per (group, accuracy band). The only honest reading:
    the max is set by the chi'=1 corner where errors are ~0.999."""
    out = {}
    for lo, hi in BANDS:
        for r in rows:
            if not (lo <= r["err"] < hi):
                continue
            alone = cheapest(curves[(r["layer_type"], r["depth"])], r["err"])
            if not alone:
                continue
            out.setdefault((r[group_key], (lo, hi)), []).append(
                alone / (r["c"] + r["q"]))
    return out


def print_bands(rows, curves, group_key, groups):
    tab = banded(rows, curves, group_key)
    print(f"\n{'band':>14}" + "".join(f"{g:>22}" for g in groups))
    for lo, hi in BANDS:
        cells = [tab.get((g, (lo, hi))) for g in groups]
        print(f"  [{lo:.2f},{hi:.2f})".rjust(14) + "".join(
            ("                   n/a" if not c
             else f"{st.mean(c):>16.3f}x n={len(c):<3}") for c in cells))
    print("\n(ratio > 1 = the circuit bought more bond dimension than its angles "
          "cost.\n n is how many (layer, chi') points landed in that band.)")
    # Rank by the bands that could matter, NOT by the maximum. The [0.99, 1.01)
    # band is the chi'=1 corner: huge ratios on tens of parameters at an error of
    # 0.9998. Ranking on it would pick the ansatz that wins where nobody operates.
    acc = {}
    for (g, (lo, hi)), v in tab.items():
        if hi > 0.99:
            continue
        acc.setdefault(g, []).extend(v)
    return {g: st.mean(v) for g, v in acc.items() if v}


# --------------------------------------------------------------------------- #
def _ratio(ck, cc, lt, dep, ansatz, regime, objective, seed, chi):
    """C_alone/(C+Q) for a row already in the checkpoint, so a resumed run's
    progress lines report the same numbers a fresh one would."""
    for r in ck.rows:
        if (r["layer_type"] == lt and int(r["depth"]) == dep
                and r["ansatz"] == ansatz and r["regime"] == regime
                and r["objective"] == objective and int(r["seed"]) == seed
                and int(r["chi"]) == chi):
            alone = cheapest(cc, float(r["err"]))
            tot = int(r["c"]) + int(r["q"])
            return alone / tot if alone and tot else float("nan")
    return float("nan")


def _checkpoint(args, stage):
    """One resumable table per stage, keyed by everything that identifies a point."""
    cfg = {k: getattr(args, k, None) for k in
           ("layers_dir", "types", "depths", "points", "sweeps", "ansatze",
            "seeds", "gd_steps", "gd_lr", "cov")}
    cfg["stage"] = stage
    return ResumableCSV(args.out, ("layer_type", "depth", "family", "ansatz", "regime", "objective", "seed", "chi"), cfg,
                        resume=not args.no_resume, numeric=("chi", "c", "q", "total", "err", "depth", "seed"))


def cmd_stage1(args):
    """Ansatz screening: which circuit shape buys the most bond per parameter."""
    layers = load_layers(args)
    names = args.ansatze or list(ANSATZE)
    rows, curves = [], {}
    ck = _checkpoint(args, 1)
    tracer = TraceWriter(args.trace, args.trace_every, append=bool(ck.n_resumed))
    for pname, lt, dep, W in layers:
        curves[(lt, dep)] = cc = classical_curve(W)
        assert_identity_is_classical(W, min(8, cc[-1][0]))
        grid = sorted(set(log_spaced_ints(1, cc[-1][0], args.points)))
        print(f"\n{lt} d{dep} {tuple(W.shape)}: chi=1..{cc[-1][0]}, "
              f"{len(grid)} chi' x {len(names)} ansatze", flush=True)
        for name in names:
            got = []
            for chi in grid:
                if ck.done(layer_type=lt, depth=dep, family="hybrid",
                           ansatz=name, regime="explicit",
                           objective="frobenius", seed=0, chi=chi):
                    got.append(_ratio(ck, cc, lt, dep, name, "explicit",
                                      "frobenius", 0, chi))
                    continue              # already measured by an earlier run
                p = run_point(W, chi, name, args.sweeps, writer=tracer,
                              key=dict(layer_type=lt, depth=dep, chi=chi,
                                       ansatz=name, regime="explicit",
                                       objective="frobenius", seed=0))
                ck.add(dict(stage=1, layer_type=lt, depth=dep, family="hybrid",
                            ansatz=name, regime="explicit",
                            objective="frobenius", seed=0, **p))
                a = cheapest(cc, p["err"])
                if a:
                    got.append(a / (p["c"] + p["q"]))
            print(f"  {name:<22} Q={ck.rows[-1]['q'] if ck.rows else 0:<7} "
                  f"median ratio {st.median(got) if got else float('nan'):.3f}x "
                  f"({sum(1 for g in got if g > 1)}/{len(got)} win)", flush=True)
    tracer.close()
    rows = ck.rows
    ck.close()
    order = print_bands(rows, curves, "ansatz", names)
    top = sorted(order, key=lambda k: -order[k])[:args.top]
    print(f"\nTop {args.top} ansatze by mean ratio below err 0.99 "
          f"(the chi'=1 corner is excluded on purpose): " + ", ".join(
              f"{t} ({order[t]:.3f}x)" for t in top))
    print("Carry these into stage 2 with --ansatze " + " ".join(top))


def cmd_stage2(args):
    """Training regime: does Adam, or Adam on top of the sweep, beat the sweep?"""
    layers = load_layers(args)
    names = args.ansatze or ["brickwall-k4D2", "brickwall-k2D4", "rope-pair"]
    regimes = [("explicit", dict(optimizer="explicit")),
               ("gradient-dl", dict(optimizer="gradient",
                                    gradient_objective="disentangle-loss")),
               ("gradient-re", dict(optimizer="gradient",
                                    gradient_objective="relative-error")),
               ("explicit+gradient-re", dict(optimizer="explicit+gradient",
                                             gradient_objective="relative-error"))]
    rows, curves = [], {}
    ck = _checkpoint(args, 2)
    tracer = TraceWriter(args.trace, args.trace_every, append=bool(ck.n_resumed))
    for pname, lt, dep, W in layers:
        curves[(lt, dep)] = cc = classical_curve(W)
        assert_identity_is_classical(W, min(8, cc[-1][0]))
        grid = sorted(set(log_spaced_ints(1, cc[-1][0], args.points)))
        print(f"\n{lt} d{dep}: {len(grid)} chi' x {len(names)} ansatze x "
              f"{len(regimes)} regimes x {args.seeds} seed(s)", flush=True)
        for name in names:
            for rname, rkw in regimes:
                got = []
                # The sweep from identity is deterministic, so seeds only buy
                # anything where Adam's initialisation actually varies.
                seeds = [0] if rname == "explicit" else range(args.seeds)
                for seed in seeds:
                    for chi in grid:
                        if ck.done(layer_type=lt, depth=dep, family="hybrid",
                                   ansatz=name, regime=rname,
                                   objective="frobenius", seed=seed, chi=chi):
                            got.append(_ratio(ck, cc, lt, dep, name, rname,
                                              "frobenius", seed, chi))
                            continue
                        p = run_point(W, chi, name, args.sweeps, seed=seed,
                                      gd_steps=args.gd_steps, gd_lr=args.gd_lr,
                                      fast_gradient=True, writer=tracer,
                                      key=dict(layer_type=lt, depth=dep, chi=chi,
                                               ansatz=name, regime=rname,
                                               objective="frobenius", seed=seed),
                                      **rkw)
                        ck.add(dict(stage=2, layer_type=lt, depth=dep,
                                    family="hybrid", ansatz=name, regime=rname,
                                    objective="frobenius", seed=seed, **p))
                        a = cheapest(cc, p["err"])
                        if a:
                            got.append(a / (p["c"] + p["q"]))
                print(f"  {name:<20} {rname:<22} median "
                      f"{st.median(got) if got else float('nan'):.3f}x", flush=True)
    tracer.close()
    rows = ck.rows
    ck.close()
    print_bands(rows, curves, "regime", [r for r, _ in regimes])


def cmd_stage3(args):
    """Objective: does training on the activation MSE change the verdict?

    Everything up to here is Frobenius. The metric that decides capability is
    ``tr[D H D^T]``, and it is measured here on BOTH sides -- the classical
    baseline is re-swept under H too, so the comparison stays internal.
    """
    covdir = Path(args.cov)
    cov = {}
    for r in csv.DictReader(open(covdir / "manifest.csv")):
        cov[r["param_name"]] = next(iter(load_file(covdir / r["file"]).values())).float()
    layers = [(p, lt, d, W) for p, lt, d, W in load_layers(args) if p in cov]
    if not layers:
        raise SystemExit("No layer had a matching covariance in --cov.")
    names = args.ansatze or ["brickwall-k4D2", "rope-pair"]
    setups = [("frobenius/both", dict(gradient_objective="relative-error",
                                      train_side="both")),
              ("activation/both", dict(gradient_objective="activation-mse",
                                       train_side="both")),
              ("activation/V-only", dict(gradient_objective="activation-mse",
                                         train_side="v"))]
    rows, curves = [], {}
    ck = _checkpoint(args, 1)
    tracer = TraceWriter(args.trace, args.trace_every, append=bool(ck.n_resumed))
    for pname, lt, dep, W in layers:
        H = cov[pname]
        curves[(lt, dep)] = cc = classical_curve(W, H)
        grid = sorted(set(log_spaced_ints(1, cc[-1][0], args.points)))
        print(f"\n{lt} d{dep}: {len(grid)} chi' x {len(names)} ansatze x "
              f"{len(setups)} objectives (scored under H)", flush=True)
        for name in names:
            for sname, skw in setups:
                got = []
                for seed in range(args.seeds):
                    for chi in grid:
                        if ck.done(layer_type=lt, depth=dep, family="hybrid",
                                   ansatz=name, regime="explicit+gradient",
                                   objective=sname, seed=seed, chi=chi):
                            got.append(_ratio(ck, cc, lt, dep, name,
                                              "explicit+gradient", sname,
                                              seed, chi))
                            continue
                        p = run_point(W, chi, name, args.sweeps, cov=H, seed=seed,
                                      optimizer="explicit+gradient",
                                      gd_steps=args.gd_steps, gd_lr=args.gd_lr,
                                      fast_gradient=True, writer=tracer,
                                      key=dict(layer_type=lt, depth=dep, chi=chi,
                                               ansatz=name,
                                               regime="explicit+gradient",
                                               objective=sname, seed=seed),
                                      **skw)
                        ck.add(dict(stage=3, layer_type=lt, depth=dep,
                                    family="hybrid", ansatz=name,
                                    regime="explicit+gradient",
                                    objective=sname, seed=seed, **p))
                        a = cheapest(cc, p["err"])
                        if a:
                            got.append(a / (p["c"] + p["q"]))
                print(f"  {name:<20} {sname:<20} median "
                      f"{st.median(got) if got else float('nan'):.3f}x", flush=True)
    tracer.close()
    rows = ck.rows
    ck.close()
    print_bands(rows, curves, "objective", [s for s, _ in setups])
    print("\n(errors and the classical baseline are BOTH under H here, so the "
          "ratio stays\n internal to the tensor-network family.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("stage1", cmd_stage1), ("stage2", cmd_stage2), ("stage3", cmd_stage3)):
        p = sub.add_parser(nm)
        p.add_argument("--layers-dir", required=True)
        p.add_argument("--types", nargs="+", default=None)
        p.add_argument("--depths", type=int, nargs="+", default=None)
        p.add_argument("--ansatze", nargs="+", default=None)
        p.add_argument("--points", type=int, default=10, help="chi' grid points")
        p.add_argument("--sweeps", type=int, default=12)
        p.add_argument("--out", default=f"{nm}.csv")
        p.add_argument("--no-resume", action="store_true",
                       help="discard any existing output and start over")
        p.add_argument("--trace", default=None,
                       help="write every optimisation's per-iteration trace to "
                            "this CSV (one tidy long-format file)")
        p.add_argument("--trace-every", type=int, default=5,
                       help="keep every Nth Adam step (endpoints always kept)")
        if nm != "stage1":
            p.add_argument("--seeds", type=int, default=2)
            p.add_argument("--gd-steps", type=int, default=150)
            p.add_argument("--gd-lr", type=float, default=0.05)
        if nm == "stage1":
            p.add_argument("--top", type=int, default=4)
        if nm == "stage3":
            p.add_argument("--cov", required=True)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

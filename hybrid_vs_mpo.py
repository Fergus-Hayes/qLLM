#!/usr/bin/env python3
"""Can a disentangler ever pay for itself against the MPO alone?

The question is internal to the tensor-network method and does not involve any
other compression family: fix a target reconstruction error. The CompactifAI layer
reaches it with ``C(chi)`` parameters at some bond dimension. The hybrid layer
reaches it with ``C(chi') + Q`` -- a smaller bond, plus the circuit's angles. Is the
sum ever smaller?

This is the one comparison where the circuit gets a fair hearing. The MPO's cost
grows roughly as ``chi^2``, so near a working bond dimension one unit of ``chi`` is
worth hundreds of parameters, while a tied RoPE gate costs 32. The circuit does not
have to be impressive -- it only has to buy a fraction of one bond.

Two things decide whether the answer is trustworthy:

* **The circuit is trained for the bond it is served at.** Optimising the
  disentangler for one ``chi`` and evaluating at another is a target/evaluation
  mismatch that manufactures wins; every hybrid point here is trained at its own
  ``chi'``.
* **The classical baseline is sampled at every ``chi``**, not on a log grid. A
  coarse classical curve would hand the hybrid a win purely by leaving gaps in the
  family it is being compared against.

    python hybrid_vs_mpo.py --layers-dir models/SmolLM2-135M/layers \
        --types q_proj v_proj --depths 20 --out hybrid_vs_mpo.csv
"""
import argparse
import csv
import math
import statistics as st
from pathlib import Path

import torch
from safetensors.torch import load_file

from qllm.compactifai import relative_error
from qllm.disentangler import classical_param_count, disentangle, hybrid_weight
from qllm.qubit_mpo import make_plan, plan_compress

# (label, kwargs for disentangle). Q is reported from the gates actually built.
ANSATZE = {
    "rope-pair":      dict(gate_size=0, depth=0, ansatz="rope-pair", head_dim=64),
    "adjacent-pair":  dict(gate_size=0, depth=0, ansatz="adjacent-pair", head_dim=64),
    "brickwall-k2D4": dict(gate_size=2, depth=4, ansatz="brickwall"),
    "brickwall-k2D8": dict(gate_size=2, depth=8, ansatz="brickwall"),
    "brickwall-k4D2": dict(gate_size=4, depth=2, ansatz="brickwall"),
}


def classical_curve(W):
    """(chi, C, err) at EVERY bond dimension -- the family the hybrid must beat."""
    plan = make_plan(W, "qubit", 0)
    out = []
    for chi in range(1, plan.max_chi + 1):
        approx, c = plan_compress(W, plan, chi)
        out.append((chi, int(c), float(relative_error(W, approx))))
    return out


def cheapest(curve, target):
    """Fewest parameters in a family that reaches `target` error, or None."""
    ok = [p for _k, p, e in curve if e <= target]
    return min(ok) if ok else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers-dir", required=True)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--depths", type=int, nargs="+", default=None)
    ap.add_argument("--chis", type=int, nargs="+", default=None,
                    help="chi' values to train a circuit at "
                         "(default: a log grid over the layer's range)")
    ap.add_argument("--points", type=int, default=12,
                    help="chi' grid points when --chis is not given")
    ap.add_argument("--ansatze", nargs="+", default=list(ANSATZE))
    ap.add_argument("--sweeps", type=int, default=12)
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.99, 0.98, 0.95, 0.90, 0.80])
    ap.add_argument("--out", default="hybrid_vs_mpo.csv")
    args = ap.parse_args()

    man = list(csv.DictReader(open(Path(args.layers_dir) / "manifest.csv")))
    layers = []
    for row in man:
        lt, dep = row["layer_type"], int(row["depth"])
        if args.types and not any(t in lt for t in args.types):
            continue
        if args.depths and dep not in args.depths:
            continue
        W = next(iter(load_file(Path(args.layers_dir) / row["file"]).values())).float()
        layers.append((lt, dep, W))
    if not layers:
        raise SystemExit("No layer matched --types/--depths.")

    rows, per_layer, per_point = [], {}, {}
    for lt, dep, W in layers:
        cc = classical_curve(W)
        for chi, c, e in cc:
            rows.append(dict(layer_type=lt, depth=dep, family="classical", ansatz="-",
                             chi=chi, c=c, q=0, total=c, err=round(e, 6)))
        print(f"\n{lt} d{dep} {tuple(W.shape)}: classical curve over "
              f"chi=1..{cc[-1][0]} (C={cc[0][1]}..{cc[-1][1]})", flush=True)

        from qllm.compactifai import log_spaced_ints
        grid = (args.chis if args.chis else
                sorted(set(log_spaced_ints(1, cc[-1][0], args.points))))
        fam = {"classical": cc}
        for name in args.ansatze:
            pts = []
            for chi in grid:
                # Trained AT the bond it is served at -- no target/eval mismatch.
                r = disentangle(W, target_chi=chi, n_sites=2, sweeps=args.sweeps,
                                tensorization="qubit", **ANSATZE[name])
                approx, _cp = hybrid_weight(r, chi)
                c = int(classical_param_count(r, chi))
                q = int(r.quantum_params)
                e = float(relative_error(W, approx))
                pts.append((chi, c + q, e))
                rows.append(dict(layer_type=lt, depth=dep, family="hybrid", ansatz=name,
                                 chi=chi, c=c, q=q, total=c + q, err=round(e, 6)))
            fam[name] = pts
            # THE measure. For each trained hybrid point, what would the MPO alone
            # have to spend to reach that same error? Ratio > 1 means the circuit
            # paid for itself: it bought more bond than its angles cost.
            ratios = [(cheapest(cc, e) / tot, chi, e)
                      for chi, tot, e in pts if cheapest(cc, e)]
            qv = next(r["q"] for r in rows
                      if r["ansatz"] == name and r["layer_type"] == lt
                      and r["depth"] == dep)
            if ratios:
                best, bchi, berr = max(ratios)
                nwin = sum(1 for r_, _c, _e in ratios if r_ > 1.0)
                print(f"  {name:<16} Q={qv:<7} best C_alone/(C+Q) = {best:.3f}x "
                      f"at chi'={bchi} (err {berr:.4f}); wins at "
                      f"{nwin}/{len(ratios)} chi'", flush=True)
            per_point.setdefault((lt, dep), {})[name] = ratios
        per_layer[(lt, dep)] = fam

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {Path(args.out).resolve()}")

    print("\nFewest total parameters to reach a target error "
          "(classical = MPO alone; others = C(chi') + Q)")
    names = ["classical"] + args.ansatze
    wins = []
    for (lt, dep), fam in per_layer.items():
        print(f"\n{lt} d{dep}")
        print(f"{'target':>8}" + "".join(f"{n:>17}" for n in names))
        for t in args.targets:
            cells = [cheapest(fam[n], t) for n in names]
            base = cells[0]
            line = f"{t:>8.3f}"
            for n, c in zip(names, cells):
                if c is None:
                    line += f"{'n/a':>17}"
                elif n == "classical":
                    line += f"{c:>17,}"
                else:
                    gain = base / c if base else float("nan")
                    if gain == gain and gain > 1.0:
                        wins.append((lt, dep, t, n, gain))
                    line += f"{c:>11,} {gain:>4.2f}x" if gain == gain else f"{c:>17,}"
            print(line)

    # The max ratio is set by the chi'=1 corner, where the errors are ~0.999 and
    # the parameter counts are tens. Banding by the hybrid point's own error is the
    # only reading that says whether the saving survives to a usable accuracy.
    print("\nSaving by the accuracy of the hybrid point (mean over layers)")
    bands = [(0.99, 1.01), (0.95, 0.99), (0.80, 0.95), (0.0, 0.80)]
    print(f"{'err band':>16}" + "".join(f"{n:>17}" for n in args.ansatze))
    for lo, hi in bands:
        cells = []
        for n in args.ansatze:
            v = []
            for (lt, dep), by in per_point.items():
                cc2 = per_layer[(lt, dep)]["classical"]
                for ratio, _chi, e in (by.get(n) or []):
                    if lo <= e < hi:
                        v.append(ratio)
            cells.append(st.mean(v) if v else float("nan"))
        print(f"  [{lo:.2f},{hi:.2f})".rjust(16) + "".join(
            ("              n/a" if c != c else f"{c:>16.3f}x") for c in cells))

    print("\nBest parameter saving of a circuit over the MPO alone, at matched error")
    print(f"{'layer':<18}{'depth':>6}" + "".join(f"{n:>17}" for n in args.ansatze))
    overall = []
    for (lt, dep), by in per_point.items():
        cells = []
        for n in args.ansatze:
            rs = by.get(n) or []
            cells.append(max(rs)[0] if rs else float("nan"))
            if rs:
                overall.append((max(rs)[0], lt, dep, n))
        print(f"{lt:<18}{dep:>6}" + "".join(
            ("              n/a" if c != c else f"{c:>16.3f}x") for c in cells))
    if overall:
        b = max(overall)
        print(f"\nbest overall: {b[0]:.3f}x ({b[3]} on {b[1]} d{b[2]})")
        print("(ratio > 1 means the circuit bought more bond dimension than its "
              "angles cost.)")

    if wins:
        best = max(wins, key=lambda w: w[4])
        print(f"\nVERDICT: the circuit pays for itself. Best case {best[4]:.2f}x fewer "
              f"total\n  parameters ({best[3]} on {best[0]} d{best[1]} at error "
              f"{best[2]:.3f}); {len(wins)} winning\n  (layer, target, ansatz) "
              f"combinations in all.")
    else:
        print("\nVERDICT: the circuit never pays for itself. At every target error and "
              "every\n  layer, the MPO alone reaches it with fewer total parameters "
              "than any circuit\n  plus a smaller bond -- Q always costs more than "
              "the bond reduction it buys.")


if __name__ == "__main__":
    raise SystemExit(main())

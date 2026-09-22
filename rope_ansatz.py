#!/usr/bin/env python3
"""Does the RoPE pairing make a better disentangler than an equally sized rotation?

RoPE is the one place in these weights where the index ordering carries meaning
rather than being an artefact of writing a row number in binary: it rotates
coordinate ``i`` of an attention head against ``i + head_dim/2``, with an angle that
depends on ``i`` and is shared across heads. That is exactly a uniformly-controlled
``SO(2)`` -- a block-diagonal orthogonal on strided index pairs -- and it costs
``head_dim/2`` parameters for the whole register.

The claim to test is not "a rotation helps". Any orthogonal circuit has some
freedom, and freedom bought with parameters is not structure. The claim is that the
*RoPE* pairing helps more than an equally sized rotation of the same shape, so the
deciding control is ``random-pair``: identical group count, identical group size,
identical tie structure, identical parameter count -- only which coordinates are
paired is destroyed. ``adjacent-pair`` is a second control with a different stride.

The layers without RoPE (v_proj, o_proj) are the negative control: the pairing is
meaningless there, so a gain that shows up on them too is a gain from the ansatz
shape and not from RoPE.

    python rope_ansatz.py --layers-dir models/SmolLM2-135M/layers \
        --chis 1 4 16 --null-reps 8 --out rope_ansatz.csv
"""
import argparse
import csv
import statistics as st
from pathlib import Path

import torch
from safetensors.torch import load_file

from qllm.disentangler import classical_param_count, disentangle, hybrid_weight
from qllm.layer_analysis import parse_layer_info

ROPE_TYPES = ("q_proj", "k_proj")          # the only layers RoPE acts on


def _rel_err(W, result, chi):
    approx, _c = hybrid_weight(result, chi)
    return float(torch.linalg.norm(W - approx) / torch.linalg.norm(W))


def _run(W, chi, sweeps, **kw):
    r = disentangle(W, target_chi=chi, n_sites=2, sweeps=sweeps,
                    tensorization="qubit", **kw)
    return dict(retained=float(r.retained), err=_rel_err(W, r, chi),
                q=int(r.quantum_params), c=int(classical_param_count(r, chi)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers-dir", required=True)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--depths", type=int, nargs="+", default=None)
    ap.add_argument("--chis", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--sweeps", type=int, default=8)
    ap.add_argument("--null-reps", type=int, default=8,
                    help="random-pair draws (the deciding control)")
    ap.add_argument("--brickwall", type=int, nargs=2, metavar=("K", "D"),
                    default=[2, 4], help="reference brickwall gate size and depth")
    ap.add_argument("--out", default="rope_ansatz.csv")
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

    rows = []
    for lt, dep, W in layers:
        is_rope = any(t in lt for t in ROPE_TYPES)
        for chi in args.chis:
            base = _run(W, chi, args.sweeps, gate_size=0, depth=0, ansatz="brickwall")
            k, d = args.brickwall
            cand = {
                "classical": base,
                "brickwall": _run(W, chi, args.sweeps, gate_size=k, depth=d,
                                  ansatz="brickwall"),
                "rope-pair": _run(W, chi, args.sweeps, gate_size=0, depth=0,
                                  ansatz="rope-pair", head_dim=args.head_dim),
                "adjacent-pair": _run(W, chi, args.sweeps, gate_size=0, depth=0,
                                      ansatz="adjacent-pair", head_dim=args.head_dim),
            }
            nulls = [_run(W, chi, args.sweeps, gate_size=0, depth=0,
                          ansatz="random-pair", head_dim=args.head_dim, pair_seed=s)
                     for s in range(max(1, args.null_reps))]
            for name, r in cand.items():
                rows.append(dict(layer_type=lt, depth=dep, chi=chi, ansatz=name,
                                 rope_layer=int(is_rope), seed=0, **r))
            for s, r in enumerate(nulls):
                rows.append(dict(layer_type=lt, depth=dep, chi=chi,
                                 ansatz="random-pair", rope_layer=int(is_rope),
                                 seed=s, **r))

            ne = [r["err"] for r in nulls]
            mu, sd = st.mean(ne), (st.pstdev(ne) if len(ne) > 1 else 0.0)
            z = (mu - cand["rope-pair"]["err"]) / sd if sd > 0 else float("nan")
            print(f"  {lt:<18} d{dep:<3} chi={chi:<3} "
                  f"classical {base['err']:.5f} | rope {cand['rope-pair']['err']:.5f} "
                  f"| random-null {mu:.5f}+-{sd:.5f} | z={z:+.2f} "
                  f"| brickwall {cand['brickwall']['err']:.5f} "
                  f"(Q {cand['rope-pair']['q']} vs {cand['brickwall']['q']})",
                  flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {Path(args.out).resolve()}")

    # The verdict is the z of rope-pair against its own matched null, split by
    # whether the layer actually has RoPE. A pairing effect that shows up on
    # v_proj/o_proj as well is not a RoPE effect.
    print("\nRoPE pairing vs the matched random-pair null (z = nulls better by ...sd)")
    print(f"{'layer':<18}{'depth':>6}{'chi':>5}{'rope?':>7}"
          f"{'rope err':>11}{'null mean':>11}{'z':>8}{'err/param gain':>16}")
    zs = {1: [], 0: []}
    for lt, dep, W in layers:
        for chi in args.chis:
            sel = [r for r in rows if r["layer_type"] == lt and r["depth"] == dep
                   and r["chi"] == chi]
            rope = next(r for r in sel if r["ansatz"] == "rope-pair")
            base = next(r for r in sel if r["ansatz"] == "classical")
            brick = next(r for r in sel if r["ansatz"] == "brickwall")
            ne = [r["err"] for r in sel if r["ansatz"] == "random-pair"]
            mu = st.mean(ne)
            sd = st.pstdev(ne) if len(ne) > 1 else 0.0
            z = (mu - rope["err"]) / sd if sd > 0 else float("nan")
            zs[rope["rope_layer"]].append(z)
            # error removed per quantum parameter, rope vs brickwall
            gr = (base["err"] - rope["err"]) / max(1, rope["q"])
            gb = (base["err"] - brick["err"]) / max(1, brick["q"])
            ratio = gr / gb if gb > 0 else float("nan")
            print(f"{lt:<18}{dep:>6}{chi:>5}{'yes' if rope['rope_layer'] else 'no':>7}"
                  f"{rope['err']:>11.5f}{mu:>11.5f}{z:>8.2f}"
                  + ("             n/a" if ratio != ratio else f"{ratio:>15.2f}x"))

    def _fmt(v):
        return f"{st.mean(v):+.2f}" if v else "n/a"
    live = [z for z in zs[1] if z == z]
    ctrl = [z for z in zs[0] if z == z]
    print(f"\nmean z on RoPE layers (q/k): {_fmt(live)}   "
          f"on non-RoPE layers (v/o): {_fmt(ctrl)}")
    if live and st.mean(live) > 2.0 and (not ctrl or st.mean(live) > st.mean(ctrl) + 2.0):
        print("\nVERDICT: the RoPE pairing beats an identically shaped random pairing "
              "on the\n  layers RoPE acts on, and does not on the layers it does not. "
              "That is a real\n  structural ansatz, not a parameter count.")
    else:
        print("\nVERDICT: no RoPE-specific effect. Whatever the pair ansatz buys, an "
              "identically\n  shaped random pairing buys too, so it is the shape of "
              "the rotation and not the\n  RoPE structure that is doing the work.")


if __name__ == "__main__":
    raise SystemExit(main())

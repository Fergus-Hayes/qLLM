#!/usr/bin/env python3
"""Does the RoPE pairing make a better disentangler than an equally sized rotation?

RoPE is the one place in these weights where the index ordering carries meaning
rather than being an artefact of writing a row number in binary: it rotates
coordinate ``i`` of an attention head against ``i + head_dim/2``, with an angle that
depends on ``i`` and is shared across heads. That is exactly a uniformly-controlled
``SO(2)`` -- a block-diagonal orthogonal on strided index pairs -- and it costs
``head_dim/2`` parameters for the whole register.

The claim to test is not "a rotation helps". Any orthogonal circuit has some
freedom, and freedom bought with parameters is not structure.

``random-pair`` matches the group count, group size, tie structure and parameter
count, destroying only which coordinates pair -- but it is **not** the deciding
control, because it is confounded. RoPE joins ``i`` with ``i + head_dim/2``, indices
differing in exactly one bit, so the RoPE gate is qubit-*local* by construction; a
random pairing joins indices differing in many bits and is not. Scoring against it
measures "local vs non-local pairing" as much as "RoPE vs not".

The deciding control is therefore ``adjacent-pair``: ``(2i, 2i+1)`` is also a
single-bit pairing at the same cost, so it differs from RoPE only in *which* bit is
paired. Both are reported, and the verdict reads the unconfounded one.

Effect size is reported next to significance on purpose. The null's spread here is
tiny, so a difference far below any practical relevance still produces a large z --
in the first run a z of +21 corresponded to removing 0.14% of the error.

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
    print("\nRoPE pairing vs its controls. 'vs adj' is the deciding column: both are\n"
          "single-bit pairings at the same cost, differing only in which bit.")
    print(f"{'layer':<18}{'depth':>6}{'chi':>5}{'rope?':>7}"
          f"{'rope err':>11}{'z vs rand':>11}{'vs rand':>10}{'vs adj':>10}"
          f"{'err/param':>12}")
    zs = {1: [], 0: []}
    adv = {1: [], 0: []}
    for lt, dep, W in layers:
        for chi in args.chis:
            sel = [r for r in rows if r["layer_type"] == lt and r["depth"] == dep
                   and r["chi"] == chi]
            rope = next(r for r in sel if r["ansatz"] == "rope-pair")
            base = next(r for r in sel if r["ansatz"] == "classical")
            brick = next(r for r in sel if r["ansatz"] == "brickwall")
            adj = next(r for r in sel if r["ansatz"] == "adjacent-pair")
            ne = [r["err"] for r in sel if r["ansatz"] == "random-pair"]
            mu = st.mean(ne)
            sd = st.pstdev(ne) if len(ne) > 1 else 0.0
            z = (mu - rope["err"]) / sd if sd > 0 else float("nan")
            zs[rope["rope_layer"]].append(z)
            # Effect sizes: error removed relative to each control, as a percent.
            d_rand = (mu - rope["err"]) / mu * 100 if mu > 0 else float("nan")
            d_adj = ((adj["err"] - rope["err"]) / adj["err"] * 100
                     if adj["err"] > 0 else float("nan"))
            adv[rope["rope_layer"]].append(d_adj)
            # error removed per quantum parameter, rope vs brickwall
            gr = (base["err"] - rope["err"]) / max(1, rope["q"])
            gb = (base["err"] - brick["err"]) / max(1, brick["q"])
            ratio = gr / gb if gb > 0 else float("nan")
            print(f"{lt:<18}{dep:>6}{chi:>5}{'yes' if rope['rope_layer'] else 'no':>7}"
                  f"{rope['err']:>11.5f}{z:>11.2f}{d_rand:>+9.3f}%{d_adj:>+9.3f}%"
                  + ("         n/a" if ratio != ratio else f"{ratio:>11.2f}x"))

    live_z = [z for z in zs[1] if z == z]
    ctrl_z = [z for z in zs[0] if z == z]
    live = [d for d in adv[1] if d == d]
    ctrl = [d for d in adv[0] if d == d]

    def _f(v, fmt="{:+.2f}"):
        return fmt.format(st.mean(v)) if v else "n/a"
    print(f"\nmean z vs the (confounded) random null -- RoPE layers {_f(live_z)}, "
          f"controls {_f(ctrl_z)}")
    print(f"mean advantage over the adjacent pairing  -- RoPE layers "
          f"{_f(live, '{:+.3f}%')}, controls {_f(ctrl, '{:+.3f}%')}")

    sep = (st.mean(live) - st.mean(ctrl)) if (live and ctrl) else float("nan")
    # Significance and size are separate questions and both have to pass. The null's
    # spread is small enough that a negligible difference still clears any z bar.
    if sep == sep and sep > 0 and st.mean(live) > 0.5:
        print(f"\nVERDICT: the RoPE pairing beats an equally local, equally priced "
              f"pairing by\n  {st.mean(live):+.3f}% on the layers RoPE acts on "
              f"against {st.mean(ctrl):+.3f}% on the layers it does\n  not. Real "
              f"structure, and large enough to matter.")
    elif sep == sep and sep > 0:
        print(f"\nVERDICT: directionally RoPE-specific but negligible. The pairing "
              f"beats an\n  equally local one by {st.mean(live):+.3f}% on RoPE "
              f"layers against {st.mean(ctrl):+.3f}% on the controls --\n  the right "
              f"sign, replicated, and far too small to change any compression "
              f"decision.\n  Large z against the random null reflects that null's "
              f"tiny spread, not effect size.")
    else:
        print("\nVERDICT: no RoPE-specific effect. An equally local pairing at the "
              "same cost does\n  as well, so it is the shape of the rotation and not "
              "the RoPE structure doing\n  the work.")


if __name__ == "__main__":
    raise SystemExit(main())

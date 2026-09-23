#!/usr/bin/env python3
"""Phase A: which per-layer metric predicts perplexity?

Stages 1-3 rank compressions by a per-layer error against the MPO baseline. That
presupposes the proxy rather than testing it. This measures it: perturb one layer
many different ways, score every candidate metric on the result, swap the layer
into the model, and read the perplexity it actually costs.

Two design points carry the whole thing.

**The perturbations must be diverse.** Sweep one family's one knob and a metric
looks excellent merely because both it and perplexity move with that knob. The
library spans MPO, hybrid, low-rank, whitened low-rank, sparse, sparse+low-rank
and -- the control -- a structureless random perturbation at matched Frobenius
error. A metric blind to structure is exposed by that row and by nothing else.

**One metric is a prediction, not a correlate.** ``pred_dnll`` is the first-order
change in mean NLL, ``mean_t g_t . (D x_t)``, in the units of the thing being
measured. Since ``d(ppl) ~ ppl_0 * d(NLL)`` it can be checked against the measured
perplexity change on a calibration line of slope 1, which is a far stronger test
than a rank correlation: a metric can rank perfectly and still be useless for
choosing a threshold.

Needs the FULL model, not an extracted-layers directory: the whole point is to
swap a perturbed layer back in and read the perplexity, which a loose weight file
cannot do.

    python proxy_study.py models/SmolLM2-135M --local-files-only \
        --token-ids ids.pt --types q_proj v_proj --depths 20 --out proxy.csv
"""
import argparse
import csv
import math
import statistics as st
from pathlib import Path

import torch

from qllm.activation_stats import input_covariance
from qllm.checkpoint import ResumableCSV
from qllm.layer_analysis import parse_layer_info, quick_perplexity
from qllm.perturb import METRICS, capture_xg, perturbations
from ppl_correlation import _pearson, _spearman

CANDIDATES = ["frobenius", "activation", "activation_val", "grad_weighted"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--types", nargs="+", default=["q_proj", "v_proj"])
    ap.add_argument("--depths", type=int, nargs="+", default=None)
    ap.add_argument("--per-family", type=int, default=6,
                    help="points per perturbation family")
    ap.add_argument("--hybrid-ansatz", default="brickwall-k4D2",
                    help="hybrid family to include ('' to skip; it is the only "
                         "family that needs an optimiser)")
    ap.add_argument("--sweeps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="tokens kept per layer for the (x, g) pairs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--local-files-only", "--offline", action="store_true",
                    dest="local_files_only")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--token-ids", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--calib-tokens", type=int, default=16384)
    ap.add_argument("--ppl-tokens", type=int, default=16384)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--holdout", type=float, default=0.4,
                    help="fraction of calibration batches reserved for the "
                         "held-out H and for the (x, g) pairs")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--figs", default="figs")
    ap.add_argument("--out", default="proxy.csv")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from activation_aware import _calibration_ids

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        local_files_only=args.local_files_only).to(args.device).eval()

    ids = _calibration_ids(args)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    ppl_ids = ids[:, :args.ppl_tokens]
    win = [ids[:, i:i + args.window]
           for i in range(0, min(ids.shape[1], args.calib_tokens), args.window)]
    win = [b for b in win if b.shape[1] >= 2]
    cut = max(1, int(len(win) * (1.0 - args.holdout)))
    if cut >= len(win):
        raise SystemExit("--holdout leaves no held-out batches; raise --calib-tokens.")
    train_b, val_b = win[:cut], win[cut:]
    print(f"calibration: {len(train_b)} train batch(es), {len(val_b)} held out")

    targets = []
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        lt, dep = parse_layer_info(name)
        if dep < 0 or not any(t in lt for t in args.types):
            continue
        if args.depths and dep not in args.depths:
            continue
        targets.append((name, lt, dep))
    if not targets:
        raise SystemExit("No layer matched --types/--depths.")
    names = [n for n, _l, _d in targets]

    print(f"capturing H (train and held-out) for {len(names)} layer(s) ...")
    H_tr, _ = input_covariance(model, names, train_b, device=args.device)
    H_va, _ = input_covariance(model, names, val_b, device=args.device)
    print(f"capturing token-aligned (x, g) pairs on the held-out batches ...")
    XG = capture_xg(model, names, val_b, device=args.device,
                    max_tokens=args.max_tokens)

    with torch.no_grad():
        base_ppl = quick_perplexity(model, ppl_ids, args.device, args.window,
                                    args.batch_size)
    print(f"baseline perplexity {base_ppl:.5f}\n")

    hyb = None
    if args.hybrid_ansatz:
        from stages import ANSATZE
        hyb = (args.hybrid_ansatz,
               dict(ANSATZE[args.hybrid_ansatz], sweeps=args.sweeps))

    ck = ResumableCSV(args.out, ("layer_type", "depth", "family", "knob"),
                      dict(model=args.model, per_family=args.per_family,
                           hybrid=args.hybrid_ansatz, ppl_tokens=args.ppl_tokens,
                           max_tokens=args.max_tokens, holdout=args.holdout),
                      resume=not args.no_resume,
                      numeric=("depth", "ppl", "dppl", "pred_dnll",
                               *CANDIDATES))
    params = dict(model.named_parameters())
    for name, lt, dep in targets:
        p = params[name]
        W = p.detach().clone().float()
        X, G = XG[name]
        ctx = dict(cov=H_tr[name].float(), X=X, G=G)
        print(f"{lt} d{dep} {tuple(W.shape)}", flush=True)
        for fam, knob, Wp in perturbations(W, H_tr[name].float(),
                                           args.per_family, hybrid=hyb):
            if ck.done(layer_type=lt, depth=dep, family=fam, knob=knob):
                continue
            vals = {}
            for mname in CANDIDATES:
                cov = H_va[name].float() if mname == "activation_val" \
                    else ctx["cov"]
                vals[mname] = METRICS[mname](W, Wp, cov=cov, X=X, G=G)
            pred = METRICS["pred_dnll"](W, Wp, X=X, G=G)
            with torch.no_grad():
                p.copy_(Wp.to(dtype=p.dtype, device=p.device))
                ppl = quick_perplexity(model, ppl_ids, args.device, args.window,
                                       args.batch_size)
                p.copy_(W.to(dtype=p.dtype, device=p.device))
            ck.add(dict(layer_type=lt, depth=dep, family=fam, knob=knob,
                        ppl=round(float(ppl), 5),
                        dppl=round(float(ppl) - base_ppl, 5),
                        pred_dnll=round(pred, 8),
                        **{k: round(v, 8) for k, v in vals.items()}))
            print(f"  {fam:<18} {str(knob):>8}  " + "  ".join(
                f"{k[:4]}={vals[k]:.4f}" for k in CANDIDATES)
                + f"  ppl={ppl:.4f}", flush=True)
    rows = ck.rows
    ck.close()
    report(rows, base_ppl, args)


def report(rows, base_ppl, args):
    for r in rows:
        for k in ("ppl", "dppl", "pred_dnll", *CANDIDATES):
            r[k] = float(r[k])
        r["depth"] = int(r["depth"])
    keys = sorted({(r["layer_type"], r["depth"]) for r in rows})

    print("\nRANK CORRELATION with perplexity (Spearman; 1.0 = perfect ranking)")
    print(f"{'layer':<18}{'depth':>6}" + "".join(f"{c:>16}" for c in CANDIDATES)
          + f"{'n':>5}")
    pooled = {c: [] for c in CANDIDATES}
    struct_only = {c: [] for c in CANDIDATES}
    for lt, dep in keys:
        sel = [r for r in rows if r["layer_type"] == lt and r["depth"] == dep]
        if len(sel) < 5:
            continue
        line = f"{lt:<18}{dep:>6}"
        for c in CANDIDATES:
            rho = _spearman([r[c] for r in sel], [r["ppl"] for r in sel])
            pooled[c].append(rho)
            line += f"{rho:>16.3f}"
        print(line + f"{len(sel):>5}")
        # The same correlation over the rows that are actually compression
        # CHOICES: no control, no exact reconstruction. This is the number that
        # answers "which metric should rank my candidates", and it can differ a
        # lot -- the control is a separate population, and pooling populations
        # flatters whichever metric best separates them.
        real = [r for r in sel if r["family"] != "random" and r["frobenius"] > 1e-9]
        if len(real) >= 5:
            for c in CANDIDATES:
                struct_only[c].append(
                    _spearman([r[c] for r in real], [r["ppl"] for r in real]))
    print(f"{'MEAN':<18}{'':>6}" + "".join(
        f"{st.mean(pooled[c]):>16.3f}" if pooled[c] else f"{'n/a':>16}"
        for c in CANDIDATES))
    if any(struct_only.values()):
        print(f"{'MEAN, real options':<18}{'':>6}" + "".join(
            f"{st.mean(struct_only[c]):>16.3f}" if struct_only[c] else f"{'n/a':>16}"
            for c in CANDIDATES))
        print("  (second row drops the control and the exact rows -- rank your "
              "candidates by it)")

    # Is the experiment resolvable at all? Some families reach zero error exactly
    # (a full-rank "low-rank" factorisation reproduces the weight), and those rows
    # MUST land on dppl = 0. Whatever they show instead is the measurement floor,
    # and a rank correlation computed over damage smaller than that floor is
    # ranking noise. This is printed before the verdict because it can invalidate
    # it -- a clean-looking rho over an unresolvable spread is the trap here.
    exact = [r for r in rows if r["frobenius"] < 1e-9]
    spread = max(abs(r["dppl"]) for r in rows) if rows else 0.0
    if exact:
        floor = max(abs(r["dppl"]) for r in exact)
        print(f"\nRESOLUTION: {len(exact)} zero-error row(s) give a perplexity "
              f"floor of {floor:.3g};\n  the full spread of |dppl| is {spread:.3g} "
              f"({spread / floor:.0f}x the floor)." if floor > 0 else
              f"\nRESOLUTION: {len(exact)} zero-error row(s) reproduce the "
              f"baseline exactly; spread of |dppl| is {spread:.3g}.")
        if floor > 0 and spread < 20 * floor:
            print("  WARNING: the damage is within ~20x of the noise floor. The "
                  "correlations\n  below are not resolvable -- raise --ppl-tokens "
                  "or use a trained model.")
    else:
        print(f"\nRESOLUTION: no zero-error row to calibrate the floor against; "
              f"spread of |dppl| is {spread:.3g}.")
    # A floor of exactly zero proves the swap-and-restore is exact but says
    # nothing about whether the damage is large enough to rank. This does: a
    # near-total destruction of one layer should move perplexity by a visible
    # fraction of the baseline, and if the whole spread is a rounding error on it,
    # the model is not actually using the layer.
    if spread < 1e-3 * base_ppl:
        print(f"  WARNING: the whole spread is {spread / base_ppl:.1e} of the "
              f"baseline perplexity ({base_ppl:.4g}).\n  Destroying a layer "
              f"barely moved the model -- the correlations below rank noise. "
              f"Use a\n  trained model and more --ppl-tokens.")

    # The control row. A metric that cannot separate structured damage from
    # structureless damage of the same size is not measuring structure.
    print("\nSTRUCTURE BLINDNESS: at matched Frobenius error, does the metric"
          "\nseparate a random perturbation from a structured one?")
    for lt, dep in keys:
        sel = [r for r in rows if r["layer_type"] == lt and r["depth"] == dep]
        rnd = [r for r in sel if r["family"] == "random"]
        oth = [r for r in sel if r["family"] != "random"]
        if not rnd or not oth:
            continue
        for r in rnd:
            cand = [o for o in oth if o["frobenius"] > 1e-9]
            if not cand:
                continue
            near = min(cand, key=lambda o: abs(o["frobenius"] - r["frobenius"]))
            # Relative, not absolute: at frob 0.05 an absolute 0.05 tolerance will
            # happily match an exact reconstruction and call it a comparison.
            if abs(near["frobenius"] - r["frobenius"]) > 0.15 * max(
                    near["frobenius"], r["frobenius"]):
                continue
            print(f"  {lt} d{dep}  frob~{r['frobenius']:.3f}: "
                  f"random dppl={r['dppl']:+.4f} vs {near['family']} "
                  f"dppl={near['dppl']:+.4f}   " + "  ".join(
                      f"{c[:4]} {r[c]:.4f}/{near[c]:.4f}" for c in CANDIDATES[1:]))

    # Calibration: pred_dnll is a PREDICTION, so check it against the measurement.
    print("\nCALIBRATION of the first-order prediction  (d(ppl) ~ ppl_0 * d(NLL))")
    # Sign agreement FIRST, over every row. The magnitude check below conditions
    # on both being positive, which is exactly the subset where they agree, so
    # quoting its correlation alone would be selection dressed up as a result.
    # Near a trained minimum the first-order term is small and of either sign
    # while the damage is second order and always positive, so a coin-flip here
    # is the expected outcome, not a bug -- and it means pred_dnll cannot be used
    # as a signed predictor for perturbations of this size.
    nz = [r for r in rows if r["dppl"] != 0 and r["pred_dnll"] != 0]
    if nz:
        agree = sum(1 for r in nz if r["dppl"] * r["pred_dnll"] > 0)
        print(f"  sign agrees on {agree}/{len(nz)} rows ({agree / len(nz):.0%})"
              + ("  -- no better than chance; the first-order term is not "
                 "predictive\n  at these perturbation sizes, and the fit below is "
                 "conditioned on agreement."
                 if agree < 0.7 * len(nz) else ""))
    pts = [(base_ppl * r["pred_dnll"], r["dppl"]) for r in rows
           if r["pred_dnll"] > 0 and r["dppl"] > 0]
    if len(pts) >= 5:
        lx = [math.log(a) for a, _b in pts]
        ly = [math.log(b) for _a, b in pts]
        rr = _pearson(lx, ly)
        sx = st.mean([b - a for a, b in zip(lx, ly)])
        print(f"  of the {len(pts)}/{len(rows)} rows where both are positive: "
              f"log-log Pearson r = {rr:+.3f}")
        print(f"  mean log ratio (measured / predicted) = {sx:+.3f}  "
              f"-> measured is {math.exp(sx):.2f}x the prediction")
        print("  (r near 1 with a ratio near 1 would mean the first-order term "
              "alone\n   predicts the damage, not merely ranks it.)")
    else:
        print("  too few points with a positive predicted and measured change.")

    if args.figs:
        from qllm.opt_plots import plot_proxy_scatter, plot_proxy_rho
        plot_proxy_scatter(rows, CANDIDATES, args.figs)
        plot_proxy_rho(rows, CANDIDATES, keys, args.figs)

    if not any(pooled.values()):
        print("\nVERDICT: no layer carried >= 5 perturbations, so no correlation "
              "was computed.")
        return
    # Judge on the real options when there are enough of them: that is the number
    # the ranking is actually for.
    score = struct_only if any(struct_only.values()) else pooled
    best = max(CANDIDATES, key=lambda c: st.mean(score[c]) if score[c] else -9)
    bv = st.mean(score[best])
    if not (bv >= 0.9):
        print(f"\nVERDICT: no metric reaches rho 0.9 (best {best} at {bv:.3f}). "
              f"Ranking\n  compressions by a per-layer proxy is unsound at this "
              f"resolution; optimise\n  against perplexity directly on a small "
              f"layer subset instead.")
    else:
        print(f"\nVERDICT: {best} ranks perplexity best (rho {bv:.3f}). Read the "
              f"calibration\n  line before using it as a threshold -- ranking well "
              f"and predicting well are\n  different properties.")


if __name__ == "__main__":
    raise SystemExit(main())

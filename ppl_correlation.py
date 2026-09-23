#!/usr/bin/env python3
"""Does activation MSE predict perplexity, for MPO and for hybrid layers?

Stages 1-3 score compressions by a per-layer error, because measuring perplexity
inside a search over thousands of configurations is not affordable. That is only
legitimate if the per-layer error tracks the thing anyone actually cares about.
This measures whether it does, and -- the decision that matters -- whether the
activation-weighted error tracks it *better* than the Frobenius error the pipeline
was originally built on.

Method: compress one layer at a time, swap it into the model, score perplexity over
a fixed token block, restore. One layer at a time on purpose -- compressing several
at once confounds each layer's contribution, and the question here is per layer.

Reported per layer and pooled, for the MPO-only and the hybrid families separately:

* **Spearman rho** between the per-layer error and ``ppl``. Rank-based, so it
  answers the question a search actually asks ("is configuration A better than
  B?") without assuming any functional form.
* **Pearson r** on ``log(ppl - ppl_0)`` against ``log(error)``, which is the
  relationship to expect if damage compounds multiplicatively.
* **the head-to-head**: activation MSE against Frobenius, on the same points. If
  the two are equal, the extra machinery to optimise the H-weighted objective is
  not earning anything, and that is worth knowing before stage 3 runs.

    python ppl_correlation.py models/SmolLM2-135M --local-files-only \
        --layers-dir models/SmolLM2-135M/layers --token-ids ids.pt \
        --types q_proj v_proj --depths 20 --out ppl_correlation.csv
"""
import argparse
import csv
import math
import statistics as st
from pathlib import Path

import torch

from qllm.activation_stats import input_covariance, output_relative_error
from qllm.compactifai import log_spaced_ints, relative_error
from qllm.disentangler import classical_param_count, disentangle, hybrid_weight
from qllm.checkpoint import ResumableCSV
from qllm.layer_analysis import quick_perplexity
from qllm.opt_plots import plot_ppl_rho, plot_ppl_scatter
from qllm.qubit_mpo import make_plan, plan_compress
from stages import ANSATZE


def _spearman(x, y):
    """Rank correlation, average ranks for ties."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return _pearson(rank(x), rank(y))


def _pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx <= 0 or sy <= 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--layers-dir", default=None,
                    help="unused for weights (they come from the model) but kept "
                         "so the same --types/--depths select the same layers")
    ap.add_argument("--types", nargs="+", default=["q_proj", "v_proj"])
    ap.add_argument("--depths", type=int, nargs="+", default=None)
    ap.add_argument("--ansatz", default="brickwall-k4D2",
                    help="hybrid ansatz to pair against the MPO-only family")
    ap.add_argument("--points", type=int, default=8, help="chi grid points")
    ap.add_argument("--sweeps", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--local-files-only", "--offline", action="store_true",
                    dest="local_files_only")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--token-ids", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--calib-tokens", type=int, default=65536)
    ap.add_argument("--ppl-tokens", type=int, default=16384)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--no-resume", action="store_true",
                    help="discard any existing output and start over")
    ap.add_argument("--out", default="ppl_correlation.csv")
    ap.add_argument("--figs", default="figs",
                    help="directory for the figures (--figs '' to skip plotting)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from activation_aware import _calibration_ids
    from qllm.layer_analysis import parse_layer_info

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        local_files_only=args.local_files_only).to(args.device).eval()

    ids = _calibration_ids(args)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    ppl_ids = ids[:, :args.ppl_tokens]
    cal = [ids[:, i:i + args.window]
           for i in range(0, min(ids.shape[1], args.calib_tokens), args.window)]
    cal = [b for b in cal if b.shape[1] >= 2]

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

    print(f"Capturing H for {len(targets)} layer(s) over {len(cal)} batch(es) ...")
    Hs, _cnt = input_covariance(model, [n for n, _l, _d in targets], cal,
                                device=args.device)

    with torch.no_grad():
        base_ppl = quick_perplexity(model, ppl_ids, args.device, args.window,
                                    args.batch_size)
    print(f"baseline perplexity {base_ppl:.4f}\n")

    params = dict(model.named_parameters())
    ck = ResumableCSV(args.out, ("layer_type", "depth", "family", "chi"),
                      dict(model=args.model, ansatz=args.ansatz,
                           points=args.points, sweeps=args.sweeps,
                           ppl_tokens=args.ppl_tokens, window=args.window),
                      resume=not args.no_resume,
                      numeric=("chi", "c", "q", "total", "frob", "act", "ppl",
                               "dppl", "depth"))
    for name, lt, dep in targets:
        p = params[name]
        W = p.detach().clone().float()
        H = Hs[name].float()
        plan = make_plan(W, "qubit", 0)
        grid = sorted(set(log_spaced_ints(1, plan.max_chi, args.points)))
        print(f"{lt} d{dep} {tuple(W.shape)}: {len(grid)} chi x 2 families",
              flush=True)
        for chi in grid:
            for family in ("mpo", "hybrid"):
                if ck.done(layer_type=lt, depth=dep, family=family, chi=chi):
                    continue              # already measured by an earlier run
                if family == "mpo":
                    approx, c, q = *plan_compress(W, plan, chi), 0
                else:
                    r = disentangle(W, target_chi=chi, n_sites=2,
                                    sweeps=args.sweeps, tensorization="qubit",
                                    **ANSATZE[args.ansatz])
                    approx, _ = hybrid_weight(r, chi)
                    c, q = int(classical_param_count(r, chi)), int(r.quantum_params)
                with torch.no_grad():
                    p.copy_(approx.to(dtype=p.dtype, device=p.device))
                    ppl = quick_perplexity(model, ppl_ids, args.device,
                                           args.window, args.batch_size)
                    p.copy_(W.to(dtype=p.dtype, device=p.device))
                ck.add(dict(
                    layer_type=lt, depth=dep, family=family, chi=chi,
                    c=int(c), q=int(q), total=int(c) + int(q),
                    frob=round(float(relative_error(W, approx)), 6),
                    act=round(output_relative_error(W, approx, H), 6),
                    ppl=round(float(ppl), 5),
                    dppl=round(float(ppl) - base_ppl, 5)))
                print(f"  chi={chi:<4} {family:<7} frob {ck.rows[-1]['frob']:.4f}"
                      f"  act {ck.rows[-1]['act']:.4f}  ppl {ppl:.4f}", flush=True)

    rows = ck.rows
    ck.close()
    if not rows:
        raise SystemExit("No rows measured or resumed.")

    # ---- correlations --------------------------------------------------------
    def corr(sel):
        if len(sel) < 3:
            return None
        out = {}
        for key in ("act", "frob"):
            xs = [r[key] for r in sel]
            out[f"rho_{key}"] = _spearman(xs, [r["ppl"] for r in sel])
            pos = [r for r in sel if r["dppl"] > 0 and r[key] > 0]
            out[f"r_{key}"] = (_pearson([math.log(r[key]) for r in pos],
                                        [math.log(r["dppl"]) for r in pos])
                               if len(pos) >= 3 else float("nan"))
            out[f"n_{key}"] = len(pos)
        return out

    print("\nCorrelation of per-layer error with perplexity "
          "(rho = Spearman on ppl; r = Pearson on log-log)")
    print(f"{'layer':<18}{'depth':>6}{'family':>9}{'rho act':>10}{'rho frob':>10}"
          f"{'r act':>9}{'r frob':>9}{'n':>5}{'n>0':>5}")
    pooled, per_layer_rho = {}, []
    for family in ("mpo", "hybrid"):
        for lt, dep in dict.fromkeys((r["layer_type"], r["depth"]) for r in rows):
            sel = [r for r in rows if r["family"] == family
                   and r["layer_type"] == lt and r["depth"] == dep]
            c = corr(sel)
            if not c:
                continue
            pooled.setdefault(family, []).append(c)
            per_layer_rho.append((lt, dep, family, c))
            print(f"{lt:<18}{dep:>6}{family:>9}{c['rho_act']:>10.3f}"
                  f"{c['rho_frob']:>10.3f}{c['r_act']:>9.3f}{c['r_frob']:>9.3f}"
                  f"{len(sel):>5}{c['n_act']:>5}")

    if args.figs:
        allc = corr(rows) or {}
        plot_ppl_scatter(rows, args.figs,
                         {"act": allc.get("rho_act"), "frob": allc.get("rho_frob")})
        plot_ppl_rho(per_layer_rho, args.figs)

    print("\nPooled")
    verdict = {}
    for family, cs in pooled.items():
        ra = st.mean([c["rho_act"] for c in cs if c["rho_act"] == c["rho_act"]])
        rf = st.mean([c["rho_frob"] for c in cs if c["rho_frob"] == c["rho_frob"]])
        verdict[family] = (ra, rf)
        print(f"  {family:<7} mean Spearman: activation {ra:+.3f}   "
              f"frobenius {rf:+.3f}   (activation better on "
              f"{sum(1 for c in cs if c['rho_act'] > c['rho_frob'])}/{len(cs)} layers)")

    # SIGNED, not absolute. A proxy anti-correlated with perplexity is the worst
    # possible case -- it would rank every configuration backwards -- so scoring it
    # by |rho| would call the most dangerous outcome a success.
    lo = min(v for pair in verdict.values() for v in pair)
    if lo < 0:
        print(f"\nVERDICT: the per-layer error is ANTI-correlated with perplexity "
              f"somewhere (rho = {lo:+.2f}).\n  Ranking configurations by it would "
              f"order them backwards there. Do not run the\n  stages on this "
              f"evidence: find out why before trusting any per-layer score.")
    elif lo < 0.5:
        print(f"\nVERDICT: the per-layer error is a WEAK proxy for perplexity "
              f"(rho as low as {lo:+.2f}).\n  Stage results ranked by it should not "
              f"be read as rankings of model quality;\n  re-score the finalists "
              f"against perplexity directly.")
    else:
        better = all(a > f for a, f in verdict.values())
        print(f"\nVERDICT: the per-layer error tracks perplexity (all rho >= "
              f"{lo:+.2f}), so ranking\n  configurations by it is sound. The "
              f"activation-weighted error is "
              + ("the better proxy in every family, which is what justifies the "
                 "stage-3 objective." if better else
                 "NOT uniformly better than the\n  plain Frobenius error here -- "
                 "the H-weighted objective is not earning its extra cost."))


if __name__ == "__main__":
    raise SystemExit(main())

"""Analyze the statistical error of a perplexity estimate vs. the number of
tokens evaluated, for a given model.

Perplexity is ``PPL = exp(mean NLL)`` over the scored tokens, so estimating it on
a finite sample of ``N`` tokens carries a sampling error. By the delta method the
relative error of ``PPL`` equals the standard error of the mean NLL, and for
weakly dependent text that scales as ``~ 1/sqrt(N)``: quadrupling the tokens
halves the error. The *constant* (and the effective sample size, which is smaller
than ``N`` because tokens within a window share context) is model- and
corpus-specific, so this tool measures it empirically.

It scores a long corpus **once**, keeping the per-window ``(nll_sum, n_tokens)``
sufficient statistics (the scored tokens are disjoint across windows), then:

* the **running** perplexity over the first ``k`` tokens (a convergence trace);
* the **error vs. tokens** curve, from (a) a block bootstrap over windows and
  (b) direct disjoint ``N``-token chunks -- two independent estimates of the
  standard error at each budget;
* a fit ``rel_err ~ C * N^p`` (``p`` should be near ``-0.5``) and the tokens a
  target relative error would need; plus the variance-inflation factor over the
  i.i.d.-token ideal (how much within-window correlation costs).

With ``--compare-chi`` it also scores the model with one layer MPO-compressed and
reports the error of the **perplexity ratio** (compressed / dense) on the *same*
windows -- the paired quantity a compression sweep actually compares, whose error
is smaller than either absolute PPL because the shared corpus noise cancels.

    # Dense-model PPL error vs. tokens (up to 131072 tokens), with plots + CSV
    python perplexity_error.py HuggingFaceTB/SmolLM2-135M --max-eval-tokens 131072

    # Also the paired ratio error for v_proj @ chi=8 (what a sweep compares)
    python perplexity_error.py HuggingFaceTB/SmolLM2-135M --max-eval-tokens 131072 \
        --compare-chi 8 --block 10 --layer-type v_proj --tensorization qubit
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Perplexity sampling error vs. number of tokens evaluated.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--dtype", default="float32",
                   choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--revision", default=None)
    p.add_argument("--threads", type=int, default=None)

    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="test")
    p.add_argument("--text-column", default="text")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--stride", type=int, default=1024,
                   help="Window stride (default: non-overlapping, so windows are "
                        "independent blocks).")
    p.add_argument("--max-eval-tokens", type=int, default=131072,
                   help="Total tokens to score once (the analysis budget).")
    p.add_argument("--ppl-batch-size", type=int, default=4)

    p.add_argument("--num-budgets", type=int, default=14,
                   help="Number of log-spaced token budgets on the error curve.")
    p.add_argument("--token-grid", type=int, nargs="+", default=None,
                   help="Explicit token budgets (overrides --num-budgets).")
    p.add_argument("--bootstrap", type=int, default=500,
                   help="Bootstrap resamples per budget.")
    p.add_argument("--target-errors", type=float, nargs="+", default=[0.01, 0.005, 0.001],
                   help="Relative-error targets to solve for the required tokens.")
    p.add_argument("--seed", type=int, default=0)

    # Optional paired ratio analysis: compress one layer and compare on the same
    # windows (the quantity a compression sweep reports).
    p.add_argument("--compare-chi", type=int, default=None,
                   help="Also analyze the PPL-ratio error with this layer MPO-"
                        "compressed at bond dimension chi.")
    p.add_argument("--block", type=int, default=10)
    p.add_argument("--layer-type", default="v_proj")
    p.add_argument("--tensorization", default="qubit", choices=["balanced", "qubit"])
    p.add_argument("--qubit-align", default="msb", choices=["msb", "lsb"])
    p.add_argument("--mpo-sites", type=int, default=2)

    p.add_argument("--results-dir", default="results/llms")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--csv-name", default="perplexity_error.csv")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Estimators over per-window sufficient statistics
# --------------------------------------------------------------------------- #
def _ppl(nll: np.ndarray, ntok: np.ndarray) -> float:
    t = float(ntok.sum())
    return float(np.exp(nll.sum() / t)) if t > 0 else float("nan")


def _bootstrap_ppl(nll, ntok, m, reps, rng):
    """Bootstrap the PPL from ``m`` windows resampled with replacement."""
    w = len(nll)
    idx = rng.integers(0, w, size=(reps, m))
    sums = nll[idx].sum(axis=1)
    toks = ntok[idx].sum(axis=1)
    return np.exp(sums / np.maximum(toks, 1))


def _disjoint_chunk_std(vals_nll, vals_ntok, m):
    """Std of the PPL over disjoint, in-order groups of ``m`` windows (>=4 groups)."""
    w = len(vals_nll)
    n_groups = w // m
    if n_groups < 4:
        return float("nan"), n_groups
    ppls = []
    for g in range(n_groups):
        sl = slice(g * m, (g + 1) * m)
        ppls.append(_ppl(vals_nll[sl], vals_ntok[sl]))
    return float(np.std(ppls, ddof=1)), n_groups


def _iid_sigma_nll(nll, ntok) -> float:
    """Per-token NLL std under an i.i.d.-within-window model (the ideal baseline)."""
    mu = nll.sum() / ntok.sum()
    # (nll_w - ntok_w*mu) is the centered window sum; its variance is ntok_w*sigma^2.
    num = ((nll - ntok * mu) ** 2 / np.maximum(ntok, 1)).sum()
    return float(np.sqrt(num / len(nll)))


def analyze(nll, ntok, budgets, reps, rng):
    """Error-vs-tokens rows from per-window stats. Returns (rows, full_ppl, sigma)."""
    full_ppl = _ppl(nll, ntok)
    mean_ntok = float(ntok.mean())
    sigma = _iid_sigma_nll(nll, ntok)
    rows = []
    for N in budgets:
        m = max(1, min(len(nll), round(N / mean_ntok)))
        boot = _bootstrap_ppl(nll, ntok, m, reps, rng)
        actual = m * mean_ntok
        std = float(np.std(boot, ddof=1))
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        chunk_std, n_groups = _disjoint_chunk_std(nll, ntok, m)
        iid_rel = sigma / math.sqrt(actual)                      # delta-method ideal
        rows.append(dict(
            tokens=int(round(actual)), windows=m,
            ppl_mean=round(float(boot.mean()), 6), ppl_std=round(std, 6),
            ppl_rel_err=round(std / full_ppl, 8),
            ci95_lo=round(lo, 6), ci95_hi=round(hi, 6),
            chunk_std=round(chunk_std, 6) if chunk_std == chunk_std else "",
            chunk_rel_err=round(chunk_std / full_ppl, 8) if chunk_std == chunk_std else "",
            n_chunks=n_groups,
            iid_rel_err=round(iid_rel, 8),
            inflation=round((std / full_ppl) / iid_rel, 4) if iid_rel > 0 else ""))
    return rows, full_ppl, sigma


def _fit_power(tokens, rel_err):
    """Fit rel_err = C * tokens^p by least squares in log-log. Returns (C, p)."""
    x = np.log(np.asarray(tokens, float))
    y = np.log(np.asarray(rel_err, float))
    ok = np.isfinite(x) & np.isfinite(y) & (np.asarray(rel_err) > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    p, a = np.polyfit(x[ok], y[ok], 1)
    return float(np.exp(a)), float(p)


def _log_budgets(total_tokens, max_length, num):
    lo = max(2 * max_length, 512)
    hi = max(lo * 2, total_tokens)
    vals = np.unique(np.round(np.geomspace(lo, hi, num)).astype(int))
    return [int(v) for v in vals if v <= total_tokens]


# --------------------------------------------------------------------------- #
def _load(args):
    from .benchmark import BenchmarkConfig, load_model_and_tokenizer, resolve_device
    dev = resolve_device(args.device)
    cfg = BenchmarkConfig(model_id=args.model, device=args.device, dtype=args.dtype,
                          trust_remote_code=args.trust_remote_code, revision=args.revision,
                          dataset=args.dataset, dataset_config=args.dataset_config,
                          split=args.split, text_column=args.text_column)
    model, tok = load_model_and_tokenizer(cfg, dev)
    return model, tok, dev, cfg


def _compare_layer_nlls(model, tok, dev, args):
    """window_nlls for the model with one layer MPO-compressed at --compare-chi."""
    from .benchmark import BenchmarkConfig, tokenize_corpus, window_nlls
    from .layer_analysis import parse_layer_info
    from .qubit_mpo import make_plan, plan_compress
    match = None
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        lt, depth = parse_layer_info(name)
        if depth == args.block and args.layer_type in lt:
            match = (name, param); break
    if match is None:
        raise RuntimeError(f"No weight matched block {args.block}/'{args.layer_type}'.")
    name, param = match
    orig = param.detach().to("cpu", torch.float32, copy=True)
    plan = make_plan(orig, args.tensorization, args.mpo_sites, align=args.qubit_align)
    approx, _ = plan_compress(orig, plan, args.compare_chi)
    load_cfg = BenchmarkConfig(model_id=args.model, dataset=args.dataset,
                               dataset_config=args.dataset_config, split=args.split,
                               text_column=args.text_column)
    ids = tokenize_corpus(tok, load_cfg)
    with torch.no_grad():
        param.copy_(approx.to(dtype=param.dtype, device=param.device))
    print(f"Scoring with {name} @ chi={args.compare_chi} ({args.tensorization}) ...")
    nll, ntok = window_nlls(model, ids, dev, args.max_length, args.stride,
                            batch_size=args.ppl_batch_size,
                            max_eval_tokens=args.max_eval_tokens, progress=True)
    with torch.no_grad():
        param.copy_(orig.to(dtype=param.dtype, device=param.device))
    return name, nll, ntok


def _out_dir(args) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    name = args.model.rstrip("/").split("/")[-1]
    return Path(args.results_dir) / name / "perplexity_error"


def plot(rows, ratio_rows, cum_tokens, cum_ppl, out_dir, full_ppl):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                              # noqa: BLE001
        print(f"(Skipping plots: matplotlib unavailable: {exc})")
        return
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 9))
    ax0.plot(cum_tokens, cum_ppl, color="#1f77b4")
    ax0.axhline(full_ppl, ls="--", color="gray", lw=1, label=f"full-corpus PPL {full_ppl:.4f}")
    ax0.set_xscale("log"); ax0.set_xlabel("tokens evaluated (running)")
    ax0.set_ylabel("running perplexity"); ax0.set_title("Convergence of the PPL estimate")
    ax0.grid(True, which="both", alpha=0.25); ax0.legend(fontsize="small")

    tok = [r["tokens"] for r in rows]
    ax1.plot(tok, [r["ppl_rel_err"] for r in rows], marker="o", label="bootstrap")
    ct = [(r["tokens"], r["chunk_rel_err"]) for r in rows if r["chunk_rel_err"] != ""]
    if ct:
        ax1.plot([t for t, _ in ct], [e for _, e in ct], marker="s", ls="--",
                 label="disjoint chunks")
    ax1.plot(tok, [r["iid_rel_err"] for r in rows], ls=":", color="gray",
             label="i.i.d. ideal ($\\propto N^{-1/2}$)")
    if ratio_rows:
        ax1.plot([r["tokens"] for r in ratio_rows],
                 [r["ratio_rel_err"] for r in ratio_rows], marker="^", color="crimson",
                 label="PPL ratio (paired)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("tokens evaluated N"); ax1.set_ylabel("relative error of PPL")
    ax1.set_title("Perplexity error vs. tokens"); ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize="small")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "perplexity_error.png"
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved plot {path}")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    from .benchmark import tokenize_corpus, window_nlls
    rng = np.random.default_rng(args.seed)

    model, tok, dev, cfg = _load(args)
    ids = tokenize_corpus(tok, cfg)
    total = min(ids.size(1), args.max_eval_tokens)
    print(f"Scoring {total} tokens once (window {args.max_length}, stride {args.stride}) ...")
    nll, ntok = window_nlls(model, ids, dev, args.max_length, args.stride,
                            batch_size=args.ppl_batch_size,
                            max_eval_tokens=args.max_eval_tokens, progress=True)
    if len(nll) < 8:
        print("error: too few windows for an error analysis; raise --max-eval-tokens "
              "or lower --max-length/--stride.")
        return 2

    total_tokens = int(ntok.sum())
    budgets = (sorted({b for b in args.token_grid if b <= total_tokens})
               if args.token_grid else _log_budgets(total_tokens, args.max_length, args.num_budgets))
    rows, full_ppl, sigma = analyze(nll, ntok, budgets, args.bootstrap, rng)

    # Running (cumulative) perplexity trace.
    cum_tokens = np.cumsum(ntok).tolist()
    cum_ppl = np.exp(np.cumsum(nll) / np.maximum(np.cumsum(ntok), 1)).tolist()

    # Optional paired ratio error.
    ratio_rows = []
    if args.compare_chi:
        _name, nll_c, ntok_c = _compare_layer_nlls(model, tok, dev, args)
        n = min(len(nll), len(nll_c))
        # Same windows -> paired. Ratio PPL = exp(mean dNLL) where dNLL is per-token.
        dnll = nll_c[:n] - nll[:n]
        nt = ntok[:n]
        base_ppl = _ppl(nll[:n], nt)
        comp_ppl = _ppl(nll_c[:n], nt)
        full_ratio = comp_ppl / base_ppl
        mean_ntok = float(nt.mean())
        for N in budgets:
            m = max(1, min(n, round(N / mean_ntok)))
            idx = rng.integers(0, n, size=(args.bootstrap, m))
            r = np.exp(dnll[idx].sum(axis=1) / np.maximum(nt[idx].sum(axis=1), 1))
            actual = m * mean_ntok
            ratio_rows.append(dict(
                tokens=int(round(actual)), ratio_mean=round(float(r.mean()), 8),
                ratio_std=round(float(np.std(r, ddof=1)), 8),
                ratio_rel_err=round(float(np.std(r, ddof=1)) / full_ratio, 8)))
        print(f"\nPaired ratio (compressed/dense) full-corpus = {full_ratio:.6f} "
              f"(base {base_ppl:.4f} -> {comp_ppl:.4f} on {total_tokens} tokens)")

    # Fit the scaling and solve for target token budgets.
    C, pwr = _fit_power([r["tokens"] for r in rows], [r["ppl_rel_err"] for r in rows])
    print(f"\nFull-corpus perplexity = {full_ppl:.6f} on {total_tokens} tokens "
          f"({len(nll)} windows).")
    print(f"Per-token NLL sigma (i.i.d. model) = {sigma:.4f}; "
          f"mean variance-inflation over i.i.d. = "
          f"{np.mean([r['inflation'] for r in rows if r['inflation'] != '']):.2f}x.")
    print(f"Error scaling: rel_err ~= {C:.4g} * N^({pwr:.3f})  "
          f"(N^-0.5 is the i.i.d. ideal).")
    if math.isfinite(pwr) and pwr < 0:
        for te in args.target_errors:
            n_need = (te / C) ** (1.0 / pwr)
            print(f"  for rel_err <= {te:.3%}: ~{n_need:,.0f} tokens")

    print(f"\n{'tokens':>10} {'ppl_mean':>10} {'rel_err':>10} {'chunk_rel':>10} "
          f"{'iid_rel':>10} {'inflation':>9}")
    for r in rows:
        cr = f"{r['chunk_rel_err']:.6f}" if r["chunk_rel_err"] != "" else "-"
        inf = f"{r['inflation']:.2f}" if r["inflation"] != "" else "-"
        print(f"{r['tokens']:>10} {r['ppl_mean']:>10.4f} {r['ppl_rel_err']:>10.6f} "
              f"{cr:>10} {r['iid_rel_err']:>10.6f} {inf:>9}")

    out_dir = _out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / args.csv_name
    fields = list(rows[0].keys()) + (["ratio_tokens", "ratio_mean", "ratio_std",
                                      "ratio_rel_err"] if ratio_rows else [])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(rows):
            out = dict(r)
            if ratio_rows and i < len(ratio_rows):
                out.update(ratio_tokens=ratio_rows[i]["tokens"],
                           ratio_mean=ratio_rows[i]["ratio_mean"],
                           ratio_std=ratio_rows[i]["ratio_std"],
                           ratio_rel_err=ratio_rows[i]["ratio_rel_err"])
            w.writerow(out)
    print(f"\nCSV: {path}")
    if not args.no_plots:
        plot(rows, ratio_rows, cum_tokens, cum_ppl, out_dir, full_ppl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

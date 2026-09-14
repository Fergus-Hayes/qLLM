#!/usr/bin/env python
"""Correlate per-layer MPO compressibility with normalized layer_analysis metrics.

    # All eligible layers
    python analyze_compressibility.py \
        results/llms/SmolLM2-135M/compactifai/compactifai_per_layer.csv \
        results/llms/SmolLM2-135M/layer_analysis/layer_analysis.csv

    # Self-attention layers only (q/k/v/o projections)
    python analyze_compressibility.py PER_LAYER.csv LAYER_ANALYSIS.csv --attention-only

Only the shape-normalized / scale-free metrics are considered (entropies / log k,
ranks / k, relative spectral gap, log / RMT-normalized condition number, and the
already-relative perplexity sensitivity). Reports Spearman correlations of each
against a compressibility scalar (default: negated log perplexity-ratio at the
median chi; higher = more compressible) and writes a scatter-plot grid.
"""

import argparse
from pathlib import Path

from qllm.compressibility_metrics import correlate, make_plots, print_report

# Substrings identifying attention / MLP projections across naming schemes.
ATTENTION_TOKENS = ["self_attn", "attention", "attn"]
MLP_TOKENS = ["mlp", "feed_forward", "ffn"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("per_layer_csv", help="compactifai_per_layer.csv")
    ap.add_argument("layer_analysis_csv", help="layer_analysis.csv")
    ap.add_argument("--target", default="log_ppl_ratio_ref",
                    choices=["log_ppl_ratio_ref", "chi_at_budget", "max_compression"],
                    help="Compressibility scalar to correlate against.")
    ap.add_argument("--ref-chi", type=int, default=None,
                    help="Reference bond dimension (default: median chi in the sweep).")
    ap.add_argument("--budget", type=float, default=1.10,
                    help="ppl_ratio budget for chi_at_budget / max_compression.")

    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--attention-only", action="store_true",
                       help="Restrict to self-attention layers (q/k/v/o).")
    scope.add_argument("--mlp-only", action="store_true",
                       help="Restrict to MLP layers.")
    ap.add_argument("--layer-types", nargs="+", default=None,
                    help="Custom substrings to keep (e.g. q_proj o_proj); "
                         "overrides --attention-only/--mlp-only.")

    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    if args.layer_types:
        layer_filter = args.layer_types
    elif args.attention_only:
        layer_filter = ATTENTION_TOKENS
    elif args.mlp_only:
        layer_filter = MLP_TOKENS
    else:
        layer_filter = None

    res = correlate(args.per_layer_csv, args.layer_analysis_csv,
                    target=args.target, ref_chi=args.ref_chi, budget=args.budget,
                    layer_filter=layer_filter)
    if res["n_layers"] < 3:
        print(f"Only {res['n_layers']} layers matched — need >=3 to correlate. "
              "Check the layer filter and that both CSVs cover the same layers.")
        return 1
    print_report(res)
    if not args.no_plots:
        make_plots(res, Path(args.per_layer_csv).resolve().parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

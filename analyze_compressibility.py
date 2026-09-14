#!/usr/bin/env python
"""Correlate per-layer MPO compressibility with layer_analysis metrics.

    python analyze_compressibility.py \
        results/llms/SmolLM2-135M/compactifai/compactifai_per_layer.csv \
        results/llms/SmolLM2-135M/layer_analysis/layer_analysis.csv

Reports Spearman correlations of each spectral / sensitivity metric with a
compressibility scalar (default: negated log perplexity-ratio at the median chi;
higher = more compressible), and writes a scatter-plot grid next to the first CSV.
"""

import argparse
from pathlib import Path

from qllm.compressibility_metrics import correlate, make_plots, print_report


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
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    res = correlate(args.per_layer_csv, args.layer_analysis_csv,
                    target=args.target, ref_chi=args.ref_chi, budget=args.budget)
    print_report(res)
    if not args.no_plots:
        make_plots(res, Path(args.per_layer_csv).resolve().parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

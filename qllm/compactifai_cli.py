"""CLI for the CompactifAI MPO bond-dimension sweep.

Examples
--------
    # Default model (SmolLM2-135M), 12 log-spaced bond dimensions
    python compactify.py

    # Explicit model and bond dimensions
    python compactify.py HuggingFaceTB/SmolLM2-360M --chi 2 4 8 16 32 64

    # Fast sweep: small corpus, big eval batch, threaded SVDs
    python compactify.py --max-eval-tokens 10000 --ppl-batch-size 16 --svd-workers 4

    # Follow the paper's advice and leave the block-output MLP layer dense
    python compactify.py --exclude-down-proj

Re-running resumes from the CSV checkpoint: bond dimensions already recorded
are skipped.
"""

from __future__ import annotations

import argparse

from .compactifai_sweep import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    CompactifaiConfig,
    _read_rows,
    compactifai_dir,
    per_layer_csv_path,
    plot_global_sweep,
    plot_per_layer,
    run_per_layer_sweep,
    run_sweep,
    sweep_csv_path,
)

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress LLM layers with CompactifAI-style MPO tensor "
                    "networks and measure perplexity vs. bond dimension.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                        help=f"Model id/path (default: {DEFAULT_MODEL}).")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="float32",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads for torch intra-op parallelism.")

    # Compression / bond dimension.
    parser.add_argument("--mpo-sites", type=int, default=2,
                        help="Number of MPO tensors per weight matrix "
                             "(2 = one SVD, enables the cached fast path).")
    parser.add_argument("--chi-min", type=int, default=2,
                        help="Smallest bond dimension in the sweep.")
    parser.add_argument("--chi-max", type=int, default=None,
                        help="Largest bond dimension (default: largest chi that "
                             "still compresses, computed from the layer shapes).")
    parser.add_argument("--num-chi", type=int, default=12,
                        help="Number of logarithmically spaced bond dimensions.")
    parser.add_argument("--chi", type=int, nargs="+", default=None,
                        help="Explicit bond dimensions (overrides the log spacing).")
    parser.add_argument("--include", default=DEFAULT_INCLUDE,
                        help="Regex of parameter names eligible for compression.")
    parser.add_argument("--exclude", default=DEFAULT_EXCLUDE,
                        help="Regex of parameter names to never compress.")
    parser.add_argument("--exclude-down-proj", action="store_true",
                        help="Leave each block's output MLP layer dense (the "
                             "paper reports it is the most compression-sensitive).")
    parser.add_argument("--min-depth", type=int, default=None,
                        help="Only compress decoder blocks at this depth or deeper "
                             "(the paper finds early blocks most sensitive).")
    parser.add_argument("--force-all", action="store_true",
                        help="Apply the MPO even when it stores more parameters "
                             "than the dense matrix.")
    parser.add_argument("--svd-workers", type=int, default=1,
                        help="Threads used to build the cached layer SVDs.")
    parser.add_argument("--no-svd-cache", action="store_true",
                        help="Do not cache SVDs (lower memory, much slower sweep).")

    # Evaluation corpus.
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Sliding-window context length.")
    parser.add_argument("--stride", type=int, default=512,
                        help="Sliding-window stride.")
    parser.add_argument("--max-eval-tokens", type=int, default=20000,
                        help="Tokens used per perplexity evaluation (the sweep "
                             "re-scores the corpus once per bond dimension; use "
                             "0 for the whole split).")
    parser.add_argument("--ppl-batch-size", type=int, default=8,
                        help="Sliding windows scored per forward pass.")

    # What to run.
    parser.add_argument("--mode", default="global",
                        choices=["global", "per-layer", "both"],
                        help="'global': compress all layers together at each chi. "
                             "'per-layer': compress ONE layer at a time to get a "
                             "perplexity-vs-chi curve per layer (n_layers x n_chi "
                             "evaluations). 'both' runs each in turn.")
    parser.add_argument("--profile-depths", type=int, nargs="+", default=None,
                        help="Restrict per-layer profiling to these block indices "
                             "(the paper profiles a handful, e.g. 0 5 15 31).")
    parser.add_argument("--per-layer-eval-tokens", type=int, default=4096,
                        help="Token budget per evaluation in per-layer mode "
                             "(many more evaluations, so keep it small).")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate plots from existing CSVs without "
                             "loading the model or recomputing anything.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Do not write plots.")

    # Output.
    parser.add_argument("--results-dir", default="results/llms")
    parser.add_argument("--csv-name", default="compactifai_sweep.csv")
    parser.add_argument("--layer-csv-name", default="compactifai_layers.csv")
    parser.add_argument("--no-layer-csv", action="store_true",
                        help="Skip the per-layer detail CSV.")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore the checkpoint and recompute every chi.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    config = CompactifaiConfig(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        mpo_sites=args.mpo_sites,
        chi_min=args.chi_min,
        chi_max=args.chi_max,
        num_chi=args.num_chi,
        chi_values=args.chi,
        include_pattern=args.include,
        exclude_pattern=args.exclude,
        exclude_down_proj=args.exclude_down_proj,
        min_depth=args.min_depth,
        force_all=args.force_all,
        svd_workers=args.svd_workers,
        svd_cache=not args.no_svd_cache,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        max_length=args.max_length,
        stride=args.stride,
        max_eval_tokens=args.max_eval_tokens or None,
        ppl_batch_size=args.ppl_batch_size,
        profile_depths=args.profile_depths,
        per_layer_eval_tokens=args.per_layer_eval_tokens or None,
        results_dir=args.results_dir,
        csv_name=args.csv_name,
        layer_csv_name=args.layer_csv_name,
        write_layer_csv=not args.no_layer_csv,
        make_plots=not args.no_plots,
        force_recompute=args.recompute,
    )

    if args.plot_only:
        out = compactifai_dir(config)
        print(f"Regenerating plots in {out} ...")
        plot_global_sweep(_read_rows(sweep_csv_path(config)), out)
        plot_per_layer(_read_rows(per_layer_csv_path(config)), out)
        return 0

    if args.mode in ("global", "both"):
        run_sweep(config)
    if args.mode in ("per-layer", "both"):
        run_per_layer_sweep(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

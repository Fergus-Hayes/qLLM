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

# Presets for per-layer profiling: a representative subset of blocks, a
# log-spaced chi range, and a token budget per evaluation. Cost is roughly
# (num_depths x layer_types x num_chi) evaluations, each over the token budget.
PRESETS = {
    #          blocks  chi points  tokens/eval   ~evaluations (7 layer types)
    "quick":    dict(num_depths=4,  num_chi=6,  per_layer_eval_tokens=4096),
    "standard": dict(num_depths=6,  num_chi=8,  per_layer_eval_tokens=8192),
    "thorough": dict(num_depths=10, num_chi=10, per_layer_eval_tokens=16384),
}
# Values used when neither a preset nor an explicit flag supplies them.
FALLBACKS = dict(num_depths=None, num_chi=12, per_layer_eval_tokens=8192)


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
    parser.add_argument("--num-chi", type=int, default=None,
                        help="Number of logarithmically spaced bond dimensions "
                             "(default 12, or the --preset value).")
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
    parser.add_argument("--preset", choices=sorted(PRESETS),
                        help="Sensible per-layer profiling configuration: "
                             "quick (4 blocks x 6 chi @ 4k tokens), "
                             "standard (6 blocks x 8 chi @ 8k tokens), "
                             "thorough (10 blocks x 10 chi @ 16k tokens). "
                             "Explicit flags override the preset.")
    parser.add_argument("--num-depths", type=int, default=None,
                        help="Profile this many evenly spaced decoder blocks, "
                             "chosen automatically from the model's actual depth "
                             "(always includes the first and last block).")
    parser.add_argument("--profile-depths", type=int, nargs="+", default=None,
                        help="Explicit block indices to profile (overrides "
                             "--num-depths).")
    parser.add_argument("--layer-types", nargs="+", default=None,
                        help="Restrict profiling to layer types containing these "
                             "strings, e.g. q_proj down_proj (default: all).")
    parser.add_argument("--per-layer-eval-tokens", type=int, default=None,
                        help="Token budget per evaluation in per-layer mode "
                             "(default 8192, or the --preset value).")
    parser.add_argument("--per-layer-stride", type=int, default=None,
                        help="Probe stride (default: non-overlapping windows, "
                             "i.e. equal to --max-length).")

    # Healing (brief retraining of the compressed layer).
    parser.add_argument("--heal", action="store_true",
                        help="After truncating each layer, briefly retrain its "
                             "MPO tensors (all other weights frozen) and record "
                             "the healed perplexity alongside the raw one.")
    parser.add_argument("--heal-steps", type=int, default=40,
                        help="Adam steps of healing per (layer, chi) point.")
    parser.add_argument("--heal-lr", type=float, default=1e-3,
                        help="Healing learning rate.")
    parser.add_argument("--heal-tokens", type=int, default=16384,
                        help="Tokens drawn from the healing split per point.")
    parser.add_argument("--heal-batch", type=int, default=2,
                        help="Windows per healing step (batch size).")
    parser.add_argument("--heal-split", default="train",
                        help="Dataset split used for healing (disjoint from the "
                             "perplexity split; default: train).")
    parser.add_argument("--heal-dataset", default=None,
                        help="Healing dataset (default: same as --dataset).")
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


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Fill unset preset-controlled options; explicit flags always win."""
    preset = PRESETS.get(args.preset, {}) if args.preset else {}
    for key, fallback in FALLBACKS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, preset.get(key, fallback))
    if args.preset:
        print(f"Preset '{args.preset}': {args.num_depths} blocks, "
              f"{args.num_chi} bond dimensions, "
              f"{args.per_layer_eval_tokens} tokens per evaluation.")
    return args


def main(argv=None) -> int:
    args = apply_preset(parse_args(argv))
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
        num_depths=args.num_depths,
        layer_types=args.layer_types,
        per_layer_eval_tokens=args.per_layer_eval_tokens or None,
        per_layer_stride=args.per_layer_stride,
        heal=args.heal,
        heal_steps=args.heal_steps,
        heal_lr=args.heal_lr,
        heal_tokens=args.heal_tokens,
        heal_batch=args.heal_batch,
        heal_split=args.heal_split,
        heal_dataset=args.heal_dataset,
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

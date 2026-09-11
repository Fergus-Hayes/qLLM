"""Command-line interface for per-layer spectral & sensitivity analysis.

Examples
--------
    # Analyze the default model (SmolLM2-135M)
    python analyze_layers.py

    # Any model, quick sensitivity budget
    python analyze_layers.py HuggingFaceTB/SmolLM2-360M --sensitivity-eval-tokens 2048

    # Spectral metrics only (fast: skips the perplexity-sensitivity probe)
    python analyze_layers.py gpt2 --skip-sensitivity

Re-running resumes from the CSV checkpoint: layers already recorded are skipped.
"""

from __future__ import annotations

import argparse

from .layer_analysis import LayerAnalysisConfig, run_layer_analysis

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-layer Shannon/Renyi-2 entropy, spectral gap, condition "
                    "number, and perplexity sensitivity vs. depth for any causal LM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                        help=f"Model id/path to analyze (default: {DEFAULT_MODEL}).")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="float32",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", default=None)

    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")

    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip the perplexity-sensitivity probe (spectral only).")
    parser.add_argument("--sensitivity-eval-tokens", type=int, default=4096,
                        help="Token budget for the sensitivity probe.")
    parser.add_argument("--sensitivity-window", type=int, default=1024,
                        help="Context window for the sensitivity probe.")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Relative weight-perturbation size (||dW||/||W||).")
    parser.add_argument("--sensitivity-samples", type=int, default=1,
                        help="Noise draws averaged per layer.")
    parser.add_argument("--sensitivity-seed", type=int, default=0)

    parser.add_argument("--results-dir", default="results/llms")
    parser.add_argument("--csv-name", default="layer_analysis.csv")
    parser.add_argument("--no-plots", action="store_true",
                        help="Do not write depth plots.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = LayerAnalysisConfig(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        run_sensitivity=not args.skip_sensitivity,
        sensitivity_eval_tokens=args.sensitivity_eval_tokens,
        sensitivity_window=args.sensitivity_window,
        epsilon=args.epsilon,
        sensitivity_samples=args.sensitivity_samples,
        sensitivity_seed=args.sensitivity_seed,
        results_dir=args.results_dir,
        csv_name=args.csv_name,
        make_plots=not args.no_plots,
    )
    run_layer_analysis(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

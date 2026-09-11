"""Command-line interface for benchmarking one or many language models.

Examples
--------
    # Single model (SmolLM2-135M is the default target)
    python -m qllm

    # An explicit model
    python -m qllm HuggingFaceTB/SmolLM2-360M

    # Sweep several models in one run
    python -m qllm HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M Qwen/Qwen2.5-0.5B

    # Read the model list from a file (one id per line, '#' comments allowed)
    python -m qllm --models-file models.txt

    # GPU, half precision, quick perplexity subset
    python -m qllm gpt2 --device cuda --dtype float16 --max-eval-tokens 20000
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .benchmark import (
    BenchmarkConfig,
    format_result,
    model_result_path,
    run_benchmark,
    write_result_csv,
)

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def read_models_file(path: Path) -> list[str]:
    """Read model ids from a text file (one per line; '#' starts a comment)."""
    models = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            models.append(line)
    return models


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark perplexity, inference time, and memory footprint "
                    "of any Hugging Face causal LM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("models", nargs="*",
                        help=f"One or more model ids/paths to benchmark "
                             f"(default: {DEFAULT_MODEL}).")
    parser.add_argument("--models-file", default=None,
                        help="Path to a file listing model ids (one per line).")

    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to run on (default: auto-detect).")
    parser.add_argument("--dtype", default="float32",
                        choices=["auto", "float32", "float16", "bfloat16"],
                        help="Model weight/compute dtype ('auto' uses the "
                             "checkpoint's own dtype).")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow loading models that ship custom code.")
    parser.add_argument("--revision", default=None,
                        help="Specific model revision/commit/branch to load.")

    # Perplexity options.
    parser.add_argument("--dataset", default="Salesforce/wikitext",
                        help="Dataset name for perplexity evaluation (use a "
                             "namespaced id, e.g. 'Salesforce/wikitext').")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1",
                        help="Dataset configuration/subset.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--text-column", default="text",
                        help="Column holding the raw text.")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Sliding-window context length (clamped per model "
                             "to its native maximum).")
    parser.add_argument("--stride", type=int, default=512,
                        help="Sliding-window stride in tokens.")
    parser.add_argument("--max-eval-tokens", type=int, default=None,
                        help="Cap on tokens used for perplexity "
                             "(default: whole split; set e.g. 20000 for a quick run).")

    # Inference-timing options.
    parser.add_argument("--prompt",
                        default="The history of artificial intelligence began",
                        help="Prompt used for the inference-timing benchmark.")
    parser.add_argument("--gen-tokens", type=int, default=64,
                        help="Number of tokens to generate when timing.")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warm-up generations before timing.")

    parser.add_argument("--skip-perplexity", action="store_true",
                        help="Skip the perplexity evaluation.")
    parser.add_argument("--skip-timing", action="store_true",
                        help="Skip the inference-timing benchmark.")

    # Output options.
    parser.add_argument("--results-dir", default="results/llms",
                        help="Base directory for per-model CSVs; each is written "
                             "to <results-dir>/<model-name>/benchmark.csv.")
    parser.add_argument("--summary-csv", default="results/llms/summary.csv",
                        help="Combined CSV that every run is appended to for "
                             "cross-model comparison (set '' to disable).")
    parser.add_argument("--no-csv", action="store_true",
                        help="Do not write any CSV output.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep going if a model fails to load or benchmark.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip models already recorded (same model/dtype/device) "
                             "in their per-model CSV — checkpoints a multi-model sweep.")
    return parser.parse_args(argv)


def config_from_args(model_id: str, args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        model_id=model_id,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        max_length=args.max_length,
        stride=args.stride,
        max_eval_tokens=args.max_eval_tokens,
        run_perplexity=not args.skip_perplexity,
        prompt=args.prompt,
        gen_tokens=args.gen_tokens,
        warmup=args.warmup,
        run_timing=not args.skip_timing,
    )


def already_benchmarked(model_id: str, args: argparse.Namespace) -> bool:
    """True if the per-model CSV already holds a matching model/dtype/device row."""
    csv_path = model_result_path(args.results_dir, model_id)
    if not csv_path.exists():
        return False
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("model_id") == model_id
                    and row.get("dtype", "").endswith(args.dtype.replace("auto", ""))
                    and (args.device == "auto" or row.get("device") == args.device)):
                return True
    return False


def main(argv=None) -> int:
    args = parse_args(argv)

    models = list(args.models)
    if args.models_file:
        models.extend(read_models_file(args.models_file))
    if not models:
        models = [DEFAULT_MODEL]
    # De-duplicate while preserving order.
    models = list(dict.fromkeys(models))

    failures = []
    for i, model_id in enumerate(models, start=1):
        if args.resume and already_benchmarked(model_id, args):
            print(f"\n[{i}/{len(models)}] Skipping {model_id} "
                  f"(already benchmarked; --resume).")
            continue
        print(f"\n[{i}/{len(models)}] Benchmarking {model_id} ...")
        config = config_from_args(model_id, args)
        try:
            result = run_benchmark(config)
        except Exception as exc:  # noqa: BLE001 - report and optionally continue
            failures.append((model_id, str(exc)))
            print(f"ERROR benchmarking {model_id}: {exc}")
            if args.continue_on_error or len(models) > 1:
                continue
            return 1

        print("\n" + format_result(result, config))

        if not args.no_csv:
            per_model = model_result_path(args.results_dir, model_id)
            write_result_csv(result, per_model)
            print(f"\nSaved results to {per_model}")
            if args.summary_csv:
                write_result_csv(result, Path(args.summary_csv))
                print(f"Appended to summary  {args.summary_csv}")

    if failures:
        print(f"\n{len(failures)} model(s) failed: "
              + ", ".join(m for m, _ in failures))
        return 1 if len(failures) == len(models) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

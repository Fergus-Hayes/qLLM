"""Fetch SmolLM2-135M and measure its perplexity and inference time.

This script downloads the SmolLM2-135M causal language model from the Hugging
Face Hub, evaluates its perplexity on a text corpus using a sliding-window
approach, and benchmarks its inference latency and throughput.

Example
-------
    python benchmark_smollm2.py
    python benchmark_smollm2.py --device cuda --dtype float16
    python benchmark_smollm2.py --max-eval-tokens 20000 --gen-tokens 128

The default configuration runs comfortably on CPU; pass ``--device cuda`` (and
optionally ``--dtype float16``/``bfloat16``) to use a GPU.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
BYTES_PER_MB = 1024 ** 2


@dataclass
class BenchmarkResult:
    timestamp: str
    model_id: str
    device: str
    dtype: str
    num_parameters: int
    param_memory_mb: float
    buffer_memory_mb: float
    total_memory_mb: float
    perplexity: float
    eval_tokens: int
    dataset: str
    perplexity_seconds: float
    prefill_latency_ms: float
    generation_tokens_per_second: float
    generated_tokens: int


def compute_memory_footprint(model) -> tuple[int, float, float, float]:
    """Return (#parameters, param MB, buffer MB, total MB) for the loaded model.

    Memory is measured from the actual tensors, so it reflects the model's
    loaded dtype (e.g. float16 halves the footprint versus float32). Buffers
    (non-trainable tensors such as rotary-embedding caches) are reported
    separately and folded into the total.
    """
    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (
        n_params,
        param_bytes / BYTES_PER_MB,
        buffer_bytes / BYTES_PER_MB,
        (param_bytes + buffer_bytes) / BYTES_PER_MB,
    )


def resolve_device(requested: str) -> str:
    """Turn a requested device string into a concrete, available device."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_model_and_tokenizer(model_id: str, device: str, dtype: torch.dtype):
    """Fetch the model and tokenizer from the Hugging Face Hub."""
    print(f"Fetching '{model_id}' from the Hugging Face Hub ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model with {n_params / 1e6:.1f}M parameters on {device} ({dtype}).")
    return model, tokenizer


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    device: str,
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_column: str,
    max_length: int,
    stride: int,
    max_eval_tokens: int | None,
) -> tuple[float, int, float]:
    """Compute perplexity with a sliding window over a text corpus.

    The window slides by ``stride`` tokens; only the newly revealed tokens in
    each window contribute to the loss so that every target token is scored with
    the maximum available left-context and never counted twice. This is the
    standard fixed-length-model perplexity estimate.
    """
    print(f"Loading dataset '{dataset_name}/{dataset_config}' [{split}] ...")
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(dataset[text_column])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids_full = encodings.input_ids
    seq_len = input_ids_full.size(1)
    if max_eval_tokens is not None:
        seq_len = min(seq_len, max_eval_tokens)
    print(f"Evaluating perplexity over {seq_len} tokens "
          f"(window={max_length}, stride={stride}) ...")

    nll_sum = torch.tensor(0.0)
    n_tokens = 0
    prev_end = 0
    start_time = time.perf_counter()

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end  # tokens newly scored in this window
        input_ids = input_ids_full[:, begin:end].to(device)

        target_ids = input_ids.clone()
        target_ids[:, :-target_len] = -100  # ignore already-scored context

        outputs = model(input_ids, labels=target_ids)

        # outputs.loss is the mean NLL over scored tokens. The model shifts
        # labels internally (predicting token i from tokens < i), so the exact
        # number of scored positions is the count of non-ignored labels after
        # dropping the first one. Weighting each window's mean loss by this
        # count yields a correctly token-averaged perplexity.
        num_valid = int((target_ids[:, 1:] != -100).sum().item())
        nll_sum += outputs.loss.detach().cpu().float() * num_valid
        n_tokens += num_valid

        prev_end = end
        if end == seq_len:
            break

    elapsed = time.perf_counter() - start_time
    avg_nll = nll_sum / n_tokens
    perplexity = torch.exp(avg_nll).item()
    return perplexity, n_tokens, elapsed


@torch.no_grad()
def benchmark_inference(
    model,
    tokenizer,
    device: str,
    prompt: str,
    gen_tokens: int,
    warmup: int,
) -> tuple[float, float, int]:
    """Measure prefill latency and generation throughput."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    # Warm-up runs (kernel autotuning, cache allocation, lazy init).
    for _ in range(warmup):
        model.generate(**inputs, max_new_tokens=8, do_sample=False)
    sync()

    # Prefill latency: time for a single forward pass over the prompt.
    sync()
    t0 = time.perf_counter()
    model(**inputs)
    sync()
    prefill_latency_ms = (time.perf_counter() - t0) * 1000.0

    # Generation throughput: greedy decode of gen_tokens new tokens.
    sync()
    t0 = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=gen_tokens,
        min_new_tokens=gen_tokens,
        do_sample=False,
    )
    sync()
    gen_seconds = time.perf_counter() - t0

    generated = output.shape[1] - inputs.input_ids.shape[1]
    tokens_per_second = generated / gen_seconds if gen_seconds > 0 else float("nan")
    return prefill_latency_ms, tokens_per_second, generated


def write_result_csv(result: BenchmarkResult, csv_path: Path) -> None:
    """Append the result as a row to ``csv_path`` (creating it with a header)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(result)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nSaved results to {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL,
                        help="Hugging Face model id to fetch.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to run on (default: auto-detect).")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model weight/compute dtype.")

    # Perplexity options.
    parser.add_argument("--dataset", default="Salesforce/wikitext",
                        help="Dataset name for perplexity evaluation. Use a "
                             "namespaced id (e.g. 'Salesforce/wikitext'); recent "
                             "huggingface_hub versions reject bare ids like 'wikitext'.")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1",
                        help="Dataset configuration/subset.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--text-column", default="text",
                        help="Column holding the raw text.")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Sliding-window context length.")
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
                        help="Base directory for CSV results; the file is written "
                             "to <results-dir>/<model-name>/benchmark.csv.")
    parser.add_argument("--csv-path", default=None,
                        help="Explicit CSV output path (overrides --results-dir).")
    parser.add_argument("--no-csv", action="store_true",
                        help="Do not write a CSV file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    torch.manual_seed(0)

    model, tokenizer = load_model_and_tokenizer(args.model_id, device, dtype)
    n_params, param_mb, buffer_mb, total_mb = compute_memory_footprint(model)

    perplexity = float("nan")
    eval_tokens = 0
    ppl_seconds = 0.0
    if not args.skip_perplexity:
        perplexity, eval_tokens, ppl_seconds = compute_perplexity(
            model, tokenizer, device,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            text_column=args.text_column,
            max_length=args.max_length,
            stride=args.stride,
            max_eval_tokens=args.max_eval_tokens,
        )

    prefill_ms = float("nan")
    tok_per_s = float("nan")
    generated = 0
    if not args.skip_timing:
        print("Benchmarking inference time ...")
        prefill_ms, tok_per_s, generated = benchmark_inference(
            model, tokenizer, device,
            prompt=args.prompt,
            gen_tokens=args.gen_tokens,
            warmup=args.warmup,
        )

    dataset_ref = f"{args.dataset}/{args.dataset_config}:{args.split}"
    result = BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_id=args.model_id,
        device=device,
        dtype=args.dtype,
        num_parameters=n_params,
        param_memory_mb=round(param_mb, 3),
        buffer_memory_mb=round(buffer_mb, 3),
        total_memory_mb=round(total_mb, 3),
        perplexity=perplexity,
        eval_tokens=eval_tokens,
        dataset=dataset_ref,
        perplexity_seconds=round(ppl_seconds, 2),
        prefill_latency_ms=round(prefill_ms, 2),
        generation_tokens_per_second=round(tok_per_s, 2),
        generated_tokens=generated,
    )

    print("\n" + "=" * 60)
    print(f"Model:               {result.model_id}")
    print(f"Device / dtype:      {result.device} / {result.dtype}")
    print(f"Parameters:          {result.num_parameters:,} "
          f"({result.num_parameters / 1e6:.1f}M)")
    print(f"Parameter memory:    {result.param_memory_mb:.2f} MB")
    print(f"Buffer memory:       {result.buffer_memory_mb:.2f} MB")
    print(f"Total model memory:  {result.total_memory_mb:.2f} MB "
          f"({result.total_memory_mb / 1024:.3f} GB)")
    if not args.skip_perplexity:
        print(f"Perplexity:          {result.perplexity:.4f} "
              f"({result.eval_tokens} tokens on "
              f"{args.dataset}/{args.dataset_config}:{args.split})")
        print(f"Perplexity eval:     {result.perplexity_seconds:.2f} s")
    if not args.skip_timing:
        print(f"Prefill latency:     {result.prefill_latency_ms:.2f} ms "
              f"({tokenizer(args.prompt, return_tensors='pt').input_ids.shape[1]} prompt tokens)")
        print(f"Generation speed:    {result.generation_tokens_per_second:.2f} tokens/s "
              f"({result.generated_tokens} new tokens)")
    print("=" * 60)

    if not args.no_csv:
        if args.csv_path is not None:
            csv_path = Path(args.csv_path)
        else:
            model_name = args.model_id.split("/")[-1]
            csv_path = Path(args.results_dir) / model_name / "benchmark.csv"
        write_result_csv(result, csv_path)


if __name__ == "__main__":
    main()

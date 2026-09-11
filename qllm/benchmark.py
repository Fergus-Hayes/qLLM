"""Core benchmarking logic for arbitrary causal language models.

This module is model-agnostic: any Hugging Face causal LM (``AutoModelForCausalLM``)
can be benchmarked by passing its Hub id (or a local path) via
:class:`BenchmarkConfig`. It measures:

* **Perplexity** on a text corpus using a sliding window.
* **Inference time** — prefill latency and greedy-decode throughput.
* **Memory footprint** — parameter, buffer, and total memory of the loaded model.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BYTES_PER_MB = 1024 ** 2

DTYPE_MAP = {
    "auto": "auto",
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class BenchmarkConfig:
    """Everything needed to benchmark a single model.

    Only ``model_id`` is required; every other field has a sensible default so
    the same config shape works across arbitrary models.
    """

    model_id: str
    device: str = "auto"          # auto | cpu | cuda | mps
    dtype: str = "float32"        # auto | float32 | float16 | bfloat16
    trust_remote_code: bool = False
    revision: str | None = None

    # Perplexity.
    dataset: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "test"
    text_column: str = "text"
    max_length: int = 1024        # sliding-window size (clamped to model max)
    stride: int = 512
    max_eval_tokens: int | None = None
    ppl_batch_size: int = 8       # windows scored per forward pass (parallelism)
    run_perplexity: bool = True

    # Inference timing.
    prompt: str = "The history of artificial intelligence began"
    gen_tokens: int = 64
    warmup: int = 1
    run_timing: bool = True


@dataclass
class BenchmarkResult:
    """A single row of benchmark output."""

    timestamp: str
    model_id: str
    device: str
    dtype: str
    num_parameters: int
    param_memory_mb: float
    buffer_memory_mb: float
    total_memory_mb: float
    context_window: int
    perplexity: float
    eval_tokens: int
    dataset: str
    perplexity_seconds: float
    prompt_tokens: int
    prefill_latency_ms: float
    generation_tokens_per_second: float
    generated_tokens: int
    error: str = ""


# --------------------------------------------------------------------------- #
# Device / dtype helpers
# --------------------------------------------------------------------------- #
def resolve_device(requested: str) -> str:
    """Turn a requested device string into a concrete, available device."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str):
    try:
        return DTYPE_MAP[name]
    except KeyError:
        raise ValueError(f"Unknown dtype '{name}'. Choose from {sorted(DTYPE_MAP)}.")


# --------------------------------------------------------------------------- #
# Model loading and introspection
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(config: BenchmarkConfig, device: str):
    """Fetch a model and tokenizer from the Hub (or a local path)."""
    print(f"Fetching '{config.model_id}' from the Hugging Face Hub ...")
    dtype = resolve_dtype(config.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
        revision=config.revision,
    )
    # Many base-model tokenizers have no pad token; generation needs one.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        trust_remote_code=config.trust_remote_code,
        revision=config.revision,
    )
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {config.model_id}: {n_params / 1e6:.1f}M parameters "
          f"on {device} ({model.dtype}).")
    return model, tokenizer


def get_context_window(model, tokenizer, requested_max_length: int) -> int:
    """Pick a sliding-window size valid for this model.

    Uses the smaller of the requested window and the model's native maximum
    position count, so the same ``max_length`` request works for models with
    short or long contexts without overflowing positional embeddings.
    """
    candidates = []
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            candidates.append(value)
    model_max = min(candidates) if candidates else requested_max_length

    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 10 ** 8:
        model_max = min(model_max, tok_max)

    return max(1, min(requested_max_length, model_max))


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


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    device: str,
    config: BenchmarkConfig,
    max_length: int,
) -> tuple[float, int, float]:
    """Compute perplexity with a sliding window over a text corpus.

    The window slides by ``stride`` tokens; only the newly revealed tokens in
    each window contribute to the loss so that every target token is scored with
    the maximum available left-context and never counted twice. This is the
    standard fixed-length-model perplexity estimate.
    """
    print(f"Loading dataset '{config.dataset}/{config.dataset_config}' [{config.split}] ...")
    dataset = load_dataset(config.dataset, config.dataset_config, split=config.split)
    text = "\n\n".join(dataset[config.text_column])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids_full = encodings.input_ids
    seq_len = input_ids_full.size(1)
    if config.max_eval_tokens is not None:
        seq_len = min(seq_len, config.max_eval_tokens)
    # Enumerate the sliding windows as (begin, end, target_len) specs; target_len
    # is the count of newly revealed tokens scored in that window.
    specs = []
    prev_end = 0
    for begin in range(0, seq_len, config.stride):
        end = min(begin + max_length, seq_len)
        specs.append((begin, end, end - prev_end))
        prev_end = end
        if end == seq_len:
            break

    batch_size = max(1, getattr(config, "ppl_batch_size", 1))
    total_windows = len(specs)
    print(f"Evaluating perplexity over {seq_len} tokens "
          f"(window={max_length}, stride={config.stride}, {total_windows} windows, "
          f"batch={batch_size}) ...")

    nll_sum = torch.tensor(0.0)
    n_tokens = 0
    processed = 0
    last_print = 0
    start_time = time.perf_counter()

    # Batch windows of identical length into one forward pass (data parallelism
    # across windows). Only the final window can be shorter, so a length change
    # forces a flush. The batched mean-NLL times the batch's valid-token count is
    # exactly the summed NLL, so this is numerically equivalent to the per-window
    # loop (up to floating-point associativity), just far better at saturating
    # the CPU/GPU matmul units.
    i = 0
    while i < total_windows:
        window_len = specs[i][1] - specs[i][0]
        batch = []
        while (i < total_windows and len(batch) < batch_size
               and (specs[i][1] - specs[i][0]) == window_len):
            batch.append(specs[i])
            i += 1

        input_ids = torch.stack(
            [input_ids_full[0, b:e] for (b, e, _) in batch]
        ).to(device)                                   # [B, L]
        target_ids = input_ids.clone()
        for r, (_, _, target_len) in enumerate(batch):
            target_ids[r, :window_len - target_len] = -100

        outputs = model(input_ids, labels=target_ids)
        num_valid = int((target_ids[:, 1:] != -100).sum().item())
        nll_sum += outputs.loss.detach().cpu().float() * num_valid
        n_tokens += num_valid
        processed += len(batch)

        if processed - last_print >= max(batch_size, total_windows // 20) or processed == total_windows:
            last_print = processed
            elapsed = time.perf_counter() - start_time
            running_ppl = float(torch.exp(nll_sum / max(1, n_tokens)))
            eta = elapsed / processed * (total_windows - processed)
            print(f"    window {processed}/{total_windows}  "
                  f"running_ppl={running_ppl:.4f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    elapsed = time.perf_counter() - start_time
    if n_tokens == 0:
        return float("nan"), 0, elapsed
    perplexity = torch.exp(nll_sum / n_tokens).item()
    return perplexity, n_tokens, elapsed


@torch.no_grad()
def benchmark_inference(
    model,
    tokenizer,
    device: str,
    config: BenchmarkConfig,
) -> tuple[int, float, float, int]:
    """Measure prompt length, prefill latency, and generation throughput."""
    inputs = tokenizer(config.prompt, return_tensors="pt").to(device)
    prompt_tokens = int(inputs.input_ids.shape[1])
    gen_kwargs = dict(do_sample=False, pad_token_id=tokenizer.pad_token_id)

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    # Warm-up runs (kernel autotuning, cache allocation, lazy init).
    for _ in range(config.warmup):
        model.generate(**inputs, max_new_tokens=8, **gen_kwargs)
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
        max_new_tokens=config.gen_tokens,
        min_new_tokens=config.gen_tokens,
        **gen_kwargs,
    )
    sync()
    gen_seconds = time.perf_counter() - t0

    generated = int(output.shape[1] - inputs.input_ids.shape[1])
    tokens_per_second = generated / gen_seconds if gen_seconds > 0 else float("nan")
    return prompt_tokens, prefill_latency_ms, tokens_per_second, generated


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Fetch a model and run the configured measurements, returning one result."""
    device = resolve_device(config.device)
    model, tokenizer = load_model_and_tokenizer(config, device)

    n_params, param_mb, buffer_mb, total_mb = compute_memory_footprint(model)
    context_window = get_context_window(model, tokenizer, config.max_length)

    perplexity, eval_tokens, ppl_seconds = float("nan"), 0, 0.0
    if config.run_perplexity:
        perplexity, eval_tokens, ppl_seconds = compute_perplexity(
            model, tokenizer, device, config, context_window,
        )

    prompt_tokens, prefill_ms, tok_per_s, generated = 0, float("nan"), float("nan"), 0
    if config.run_timing:
        print("Benchmarking inference time ...")
        prompt_tokens, prefill_ms, tok_per_s, generated = benchmark_inference(
            model, tokenizer, device, config,
        )

    return BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_id=config.model_id,
        device=device,
        dtype=str(model.dtype).replace("torch.", ""),
        num_parameters=n_params,
        param_memory_mb=round(param_mb, 3),
        buffer_memory_mb=round(buffer_mb, 3),
        total_memory_mb=round(total_mb, 3),
        context_window=context_window,
        perplexity=round(perplexity, 4) if perplexity == perplexity else perplexity,
        eval_tokens=eval_tokens,
        dataset=f"{config.dataset}/{config.dataset_config}:{config.split}",
        perplexity_seconds=round(ppl_seconds, 2),
        prompt_tokens=prompt_tokens,
        prefill_latency_ms=round(prefill_ms, 2) if prefill_ms == prefill_ms else prefill_ms,
        generation_tokens_per_second=round(tok_per_s, 2) if tok_per_s == tok_per_s else tok_per_s,
        generated_tokens=generated,
    )


def format_result(result: BenchmarkResult, config: BenchmarkConfig) -> str:
    """Render a result as a human-readable block."""
    lines = [
        "=" * 60,
        f"Model:               {result.model_id}",
        f"Device / dtype:      {result.device} / {result.dtype}",
        f"Parameters:          {result.num_parameters:,} ({result.num_parameters / 1e6:.1f}M)",
        f"Parameter memory:    {result.param_memory_mb:.2f} MB",
        f"Buffer memory:       {result.buffer_memory_mb:.2f} MB",
        f"Total model memory:  {result.total_memory_mb:.2f} MB ({result.total_memory_mb / 1024:.3f} GB)",
        f"Context window:      {result.context_window} tokens",
    ]
    if config.run_perplexity:
        lines.append(f"Perplexity:          {result.perplexity:.4f} "
                     f"({result.eval_tokens} tokens on {result.dataset})")
        lines.append(f"Perplexity eval:     {result.perplexity_seconds:.2f} s")
    if config.run_timing:
        lines.append(f"Prefill latency:     {result.prefill_latency_ms:.2f} ms "
                     f"({result.prompt_tokens} prompt tokens)")
        lines.append(f"Generation speed:    {result.generation_tokens_per_second:.2f} tokens/s "
                     f"({result.generated_tokens} new tokens)")
    lines.append("=" * 60)
    return "\n".join(lines)


def write_result_csv(result: BenchmarkResult, csv_path: Path) -> None:
    """Append the result as a row to ``csv_path`` (creating it with a header)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(result)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def model_result_path(results_dir: Path, model_id: str) -> Path:
    """Default per-model CSV path: <results-dir>/<model-name>/benchmark.csv."""
    model_name = model_id.rstrip("/").split("/")[-1]
    return Path(results_dir) / model_name / "benchmark.csv"
